"""Rebuild the English onboarding-fix report from non-private test receipts."""
import datetime,html,json
from pathlib import Path
from final_verify import runtime_signature

ROOT=Path(__file__).parent

def main():
    def read(name):
        path=ROOT/'logs'/name
        return json.loads(path.read_text()) if path.exists() else {}
    live=read('visual-onboarding-checks.json');browser=read('multimodal-browser-checks.json')
    unit=read('unit-checks.json');lifecycle=read('browser-lifecycle-checks.json');gaze=read('follow-pipeline-checks.json')
    signature=runtime_signature();e=html.escape
    current=live.get('passed') and live.get('runtime_source_sha256')==signature
    rows=[]
    for case in live.get('cases',[]):
        state=case.get('state',{});endpoints=[x for x in case.get('events',[]) if x.get('type')=='endpoint']
        rows.append((case['mode'],str(case.get('passed',False)),str(len(endpoints)),str(state.get('moving_frames','—')),
            str(state.get('metrics',{}).get('reply_chunks',0)),json.dumps(case.get('recovery',{}))))
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robby · visible-speaker onboarding fix</title><style>body{max-width:1000px;margin:40px auto;padding:0 20px;font:17px/1.55 system-ui;color:#203839;background:#f6f8f4}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border:1px solid #becdc7;padding:8px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere}a{color:#21665f}.scroll{overflow:auto}</style>
<h1>Visible-speaker onboarding fix</h1><p><strong>Latest default: face VAD OFF and webcam Face ID OFF.</strong> Both are independent opt-in checkboxes next to the camera controls. Position tracking and acoustic speaker embeddings remain active. Old v1 camera-gating preferences migrate OFF; subsequent explicit choices persist. The audio_default test below exercises enrollment without mouth-motion or face-identity evidence.</p><p><a href="/desk-robot/">Open Robby</a> · <a href="/static/worklog.html">Task log</a></p>'''
    out+='<p>Generated '+e(datetime.datetime.now(datetime.timezone.utc).isoformat())+'. Current runtime live regression: <strong>'+str(bool(current))+'</strong>.</p>'
    out+='''<h2>What was wrong?</h2><p>“No time-aligned visible speaking face” rejected the sentence before final acoustic identity resolution. It did not mean that the name Christoph failed to parse. The 10-Hz visual detector required unusually large per-frame lip movement and multiple reversals in a 700-ms window. A modest 2-Hz articulation fixture was rejected by that rule. The browser also accumulated its packet counter across reconnects while the server reset its seconds counter.</p>
<h2>Changes</h2><ul><li>800-ms temporal window, motion per second, hysteresis and excursion/reversal checks; a secondary normalized lip-width cue. No extra model, GPU use or face upload.</li><li>Static mouths, small jitter, one opening held open, slow non-speech movement and rigid translation/scale remain negative controls. This is a heuristic, not proof that sound came from the face.</li><li>Missing timestamps, missing faces and insufficient mouth activity have different recovery messages. Invalid timestamps cannot keep a camera observation fresh.</li><li>The recovery button automatically appears in the expanded microphone panel. It disables only the camera/offscreen filter, preserving the voice recognition setting, camera and gaze. It asks for a new introduction, without replaying a rejected turn or guessing a face binding.</li><li>Per-connection microphone counters now match the server session. Voice-match threshold remains 0.32; no identity check was bypassed.</li></ul>
<h2>End-to-end browser checks</h2><p>Actual public HTTPS page, Chrome capture worklet, ASR, speaker enrollment, Luna and streamed TTS. Facial landmarks are synthetic and processed by the actual JavaScript mouth-motion detector; speaking flags are not injected. Synthetic movement follows the fixture's captured speech, not an assumed file start time. The session is opened before starting the fake microphone so the fixture's wake phrase is not lost during connection establishment. A 35-second deadline prevents success by automatic fixture looping.</p><div class="scroll"><table><tr><th>Condition</th><th>Passed</th><th>Submitted turns</th><th>Moving frames</th><th>TTS chunks</th><th>Recovery state</th></tr>'''
    out+=''.join('<tr>'+''.join('<td>'+e(v)+'</td>' for v in row)+'</tr>' for row in rows)+'</table></div>'
    out+='<h2>Detector and regression results</h2><pre>'+e(json.dumps({'geometry':browser.get('geometry'),
        'unit_tests':unit.get('tests'),'unit_passed':unit.get('passed'),'microphone_lifecycle':lifecycle,
        'camera_to_head_passed':gaze.get('passed')},indent=2))+'</pre>'
    out+='''<h2>Limits and retained failures</h2><p>No physical recording of the user was available, so human webcam recognition accuracy, lighting robustness and true A/V offset are not claimed. Real MediaPipe and face-api CPU smoke tests use public still imagery; the live onboarding test uses artificial moving landmarks. Chewing and other plausible mouth movements can resemble speech. Multiple overlapping faces remain uncertain. An initial harness assumed the fake file started with the WebSocket clock and failed; another run only enrolled on the file's second loop. A separate failed browser run exposed the recovery button hidden inside a collapsed panel, which is now automatically opened. Those receipts are retained locally rather than counted as passes.</p>
<p>To try the fix: reload the page, start talking, wait for readiness, face the camera and say “Hi Robbie, I am Christoph.” If the visual filter still rejects the sentence, use <strong>Use voice recognition without camera filtering</strong> and repeat. This leaves acoustic recognition active, rather than using “Talk without speaker filter.”</p>'''
    out+='<p>Runtime SHA-256: <code>'+signature+'</code></p></html>'
    (ROOT/'static/visual-onboarding-update.html').write_text(out)
    print(json.dumps({'report':'static/visual-onboarding-update.html','live_current':bool(current)}))

if __name__=='__main__':main()
