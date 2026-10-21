"""Real Chromium, fake media/network: startup, cancellation and mic privacy.

Run with Playwright installed, e.g.:
uv run --no-project --with playwright python browser_lifecycle_checks.py
No GPU, hosted API, physical microphone, camera or real WebSocket is used.
"""
from functools import partial
import http.server
import json
import os
from pathlib import Path
import threading

from playwright.sync_api import sync_playwright


class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *_args):
        pass


CHECKS = r"""async () => {
  const {RoboClient,playbackStart} = await import('/client.js');
  const realWS = window.WebSocket;
  const realGUM = navigator.mediaDevices.getUserMedia;
  const sockets = [];
  const results = [];
  const tick = () => new Promise(r => setTimeout(r, 0));
  const assert = (yes, text) => { if (!yes) throw Error(text); };
  function context(addModule = async () => {}) {
    return {audioWorklet: {addModule}, close: async () => {},
      createMediaStreamSource: () => ({connect() {}, disconnect() {}})};
  }
  function client(addModule) {
    const c = new RoboClient();
    c.ctx = context(addModule);
    c.clock = {stop() {}, inputLatency: null};
    return c;
  }
  function media() {
    let stopped = 0;
    const track = {stop() { stopped++; }, getSettings() { return {}; }};
    return {stream: {getTracks: () => [track], getAudioTracks: () => [track]},
      get stopped() { return stopped; }};
  }
  class FakeWS {
    constructor() {
      this.readyState = 0; sockets.push(this);
      setTimeout(() => { if (this.readyState === 0) {
        this.readyState = 1; this.onopen?.();
      }}, 15);
    }
    close() { this.readyState = 3; this.onclose?.(); }
    send() {}
  }
  try {
    const timing={currentTime:1,sampleRate:48000,baseLatency:.01};
    assert(Math.abs(playbackStart(timing,0,true)-1.4625)<1e-6,'First reply missing jitter reserve');
    assert(playbackStart(timing,2,false)===2,'Continuous chunks gained an extra delay');
    assert(Math.abs(playbackStart(timing,0,false)-1.0125)<1e-6,'Backchannel was delayed by reply reserve');
    results.push('initial_reply_reserve_without_per_chunk_or_backchannel_delay');
    window.WebSocket = FakeWS;

    const c = client();
    c.metrics.mic_blocks=3168;c.sentMicSamples=100000;c.captureClock={audio:6,perf:0};
    const before = sockets.length;
    await Promise.all([c.start(false), c.start(false), c.start(false)]);
    assert(sockets.length-before === 1, 'Concurrent start opened multiple sockets');
    assert(c.metrics.mic_blocks===0&&c.sentMicSamples===0&&c.captureClock===null,'New connection retained previous microphone counters');
    results.push('connection_resets_microphone_clock_and_counters');
    c.stop(); results.push('concurrent_start_uses_one_websocket');

    const failedMedia = media();
    navigator.mediaDevices.getUserMedia = async () => failedMedia.stream;
    const f = client(async () => { throw Error('Injected worklet loading failure'); });
    let failed = false;
    try { await f.start(true); } catch { failed = true; }
    assert(failed, 'Injected worklet failure did not reject');
    assert(failedMedia.stopped >= 1 && f.stream === null && f.ctx === null,
      'Microphone leaked after worklet failure');
    results.push('worklet_failure_stops_microphone');

    let resolveLate;
    const late = media();
    navigator.mediaDevices.getUserMedia = () => new Promise(r => { resolveLate = r; });
    const l = client();
    const pending = l.start(true).then(() => false, () => true);
    await tick(); l.stop(); resolveLate(late.stream);
    assert(await pending, 'Stopped startup unexpectedly succeeded');
    assert(late.stopped >= 1 && l.stream === null && l.ctx === null,
      'Late media permission left microphone running');
    results.push('late_get_user_media_after_stop_is_released');

    let resolveOld;
    const oldMedia = media();
    navigator.mediaDevices.getUserMedia = () => new Promise(r => { resolveOld = r; });
    const r = client();
    const oldStart = r.start(true).then(() => false, () => true);
    await tick(); r.stop();
    const newContext = context(); r.ctx = newContext; r.clock = {stop() {}};
    await r.start(false); const newSocket = r.ws;
    resolveOld(oldMedia.stream); await oldStart;
    assert(oldMedia.stopped >= 1 && r.ws === newSocket && r.ctx === newContext,
      'Cancelled old media startup damaged the new conversation');
    r.stop(); results.push('old_media_result_cannot_close_new_session');

    const s = client(); await s.start(false); const oldSocket = s.ws;
    s.stop(); s.ctx = context(); s.clock = {stop() {}};
    await s.start(false); const currentSocket = s.ws;
    oldSocket.onclose();
    assert(s.ws === currentSocket && currentSocket.readyState === 1,
      'Stale close callback stopped a new WebSocket');
    s.stop(); results.push('stale_websocket_close_is_ignored');

    const resumed = client();let resumes=0;
    resumed.ctx.state='suspended';resumed.ctx.resume=async()=>{resumes++;resumed.ctx.state='running';};
    await resumed.start(false);
    assert(resumes===1 && resumed.ctx.state==='running','Existing suspended context did not resume');
    resumed.stop();results.push('existing_suspended_audio_context_resumes');

    let resolveModule;
    const workletMedia = media();
    navigator.mediaDevices.getUserMedia = async () => workletMedia.stream;
    const w = client(() => new Promise(r => { resolveModule = r; }));
    const workletStart = w.start(true).then(() => false, () => true);
    await tick(); w.stop(); resolveModule(); await workletStart;
    assert(workletMedia.stopped >= 1 && w.stream === null && w.ctx === null,
      'Stopped pending worklet reattached a microphone');
    results.push('late_worklet_cannot_reattach_microphone');

    const stoppedReply=client();stoppedReply.queued[17]=24000;stoppedReply.stopAudio();
    const latePacket=new ArrayBuffer(4+640);new DataView(latePacket).setInt32(0,17,true);
    stoppedReply.ctx.createBuffer=()=>{throw Error('Cancelled audio reached playback allocation');};
    stoppedReply.play(latePacket);
    assert(stoppedReply.metrics.reply_chunks===0,'Late packet restarted an interrupted reply');
    assert(stoppedReply.cancelledReplies.has(17),'Reply ID was not cancelled');
    stoppedReply.stop();assert(stoppedReply.cancelledReplies.size===0,'Reply IDs leaked across sessions');
    results.push('late_audio_cannot_restart_a_stopped_reply');

    const cancelledBeforeAudio=client();await cancelledBeforeAudio.start(false);
    cancelledBeforeAudio.ws.onmessage({data:JSON.stringify({type:'reply_cancelled',reply_id:18})});
    new DataView(latePacket).setInt32(0,18,true);
    cancelledBeforeAudio.ctx.createBuffer=()=>{throw Error('Pre-audio cancellation was ignored');};
    cancelledBeforeAudio.play(latePacket);cancelledBeforeAudio.stop();
    results.push('server_cancellation_blocks_first_in_flight_audio_packet');

    return {passed: results.length, checks: results,
      physical_microphone_used: false, hosted_api_calls: 0, real_websockets: 0};
  } finally {
    window.WebSocket = realWS;
    navigator.mediaDevices.getUserMedia = realGUM;
  }
}"""


def main():
    static = Path(__file__).resolve().parent / 'static'
    server = http.server.ThreadingHTTPServer(('127.0.0.1', 0),
        partial(QuietHandler, directory=str(static)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(executable_path=os.environ.get(
                'ROBO_CHROME', '/usr/bin/google-chrome'), headless=True,
                args=['--no-sandbox'])
            try:
                page = browser.new_page()
                # A directory index is unnecessary: only establish a secure-local origin.
                page.goto(f'http://127.0.0.1:{server.server_port}/clock.js')
                errors = []
                page.on('pageerror', lambda e: errors.append(str(e)))
                report = page.evaluate(CHECKS)
                assert not errors, errors
                target=Path(__file__).resolve().parent/'logs/browser-lifecycle-checks.json'
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps(report,indent=2))
                print(json.dumps(report, indent=2))
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()


if __name__ == '__main__':
    main()
