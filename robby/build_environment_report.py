"""Publish measured receipts, including failed attempts, without private audio."""
import datetime,html,json,statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parent;STATIC=ROOT/'static'
def read(path):return json.loads(path.read_text())
def esc(value):return html.escape(str(value))
def main():
    bench=read(STATIC/'environment-benchmark.json')
    browsers=sorted(STATIC.glob('environment-browser-*.json'))
    voices=sorted(STATIC.glob('environment-voices-*.json'))
    music_runs=sorted(STATIC.glob('environment-music-*.json'))
    music=read(music_runs[-1]) if music_runs else {}
    browser=read(browsers[-1]) if browsers else {}
    voice=read(voices[-1]) if voices else {}
    units=read(ROOT/'logs/unit-checks.json')
    memory=read(STATIC/'two-stage-memory-browser.json')
    general=read(STATIC/'two-stage-browser.json')
    rows=bench['rows'];med=lambda values:round(statistics.median(values),1)
    sounds={name:med([r['http_ms'] for r in rows if r.get('fixture')==name]) for name in ('impacts','beeps','quiet','tone_chord')}
    stages={mode:med([r['wall_ms'] for r in rows if r.get('mode')==mode]) for mode in ('asr_only','serial','parallel')}
    impact=[r['result'] for r in rows if r.get('fixture')=='impacts']
    permitted=next((r for r in browser.get('cases',[]) if r['mode']=='permitted'),{})
    checked={'unit_tests':units,'browser_receipt':browsers[-1].name if browsers else None,
        'browser_passed':browser.get('passed'), 'voice_receipt':voices[-1].name if voices else None,
        'voice_passed':voice.get('passed'),'memory_regression_passed':memory.get('passed'),
        'conversation_regression_passed':general.get('passed')}
    if music_runs:checked.update(music_receipt=music_runs[-1].name,music_passed=music.get('passed'))
    (STATIC/'environment-release-checks.json').write_text(json.dumps(checked,separators=(',',':')))
    checks=[('Offline contracts',f"{units['tests']} tests; {units['failures']} failures, {units['errors']} errors",units['passed']),
        ('Real HTTPS automatic startup, microphone and sound event','Real GPU / local LLM / API / TTS; fake-device microphone',browser.get('passed')),
        ('Unfamiliar voice and quiet TV','Synthetic multi-speaker digital injection; no unauthorized enrollment',voice.get('passed')),
        ('Wiki-memory recall after reconnect','Bob / Alice with no recent-history budget; memory off/reset also checked',memory.get('passed')),
        ('Existing conversation and bridges','Typed explanations, joke, four-line poem, cancellation and mobile layout',general.get('passed'))]
    if music_runs:checks.append(('Prominent music regression','Quiet music ignored; real GPU music caption and spoken response; repeated-onset cooldown',music.get('passed')))
    table=''.join(f'<tr><td>{esc(a)}</td><td>{"PASS" if passed else "NOT PASSED"}</td><td>{esc(b)}</td></tr>' for a,b,passed in checks)
    timing=[('Sound caption: impacts',sounds['impacts'],'Median of three warm HTTP requests; two seconds of authored audio'),
        ('Sound caption: beeps',sounds['beeps'],'Median of three warm HTTP requests'),
        ('Sound encoder (impacts)',med([r['encoder_gpu_ms'] for r in impact]),'CUDA event interval; one encoder pass'),
        ('Sound decoder (impacts)',med([r['decoder_gpu_ms'] for r in impact]),'Autoregressive decoder, including host launch gaps on the CUDA timeline'),
        ('ASR only',stages['asr_only'],'Parakeet v3; four-second synthetic utterance, three repeats'),
        ('ASR + sound + vocal tone, serial',stages['serial'],'Same services requested sequentially'),
        ('ASR + sound + vocal tone, parallel',stages['parallel'],'Concurrent HTTP requests; total wall time, not their sum'),
        ('Caption received → first reply PCM',permitted.get('caption_to_first_pcm_ms','not measured'),'Latest permitted-autoplay browser run; includes LLM + TTS + transport'),
        ('Sound onset → first reply PCM, estimate',permitted.get('onset_to_first_pcm_estimate_ms','not measured'),'Includes about 1.04 s capture; excludes measured physical speaker latency')]
    times=''.join(f'<tr><td>{esc(a)}</td><td>{esc(b)} ms</td><td>{esc(c)}</td></tr>' for a,b,c in timing)
    receipts=[*browsers,*voices,*music_runs,STATIC/'environment-benchmark.json',STATIC/'environment-reactions-before.json',STATIC/'environment-reactions.json',STATIC/'environment-release-checks.json']
    links=''.join(f'<li><a href="{esc(p.name)}">{esc(p.name)}</a></li>' for p in receipts if p.exists())
    examples=''.join(f'<tr><td>{esc(r["fixture"])}</td><td>{esc(r["result"]["caption"])}</td></tr>' for r in rows if r.get('repeat')==0 and 'fixture' in r)
    music_section=('''<section><h2>Music recognition correction</h2><p>The initial version explicitly vetoed captions classified as music. This was too restrictive. Prominent music is now eligible, including captions mentioning musical vocals; negated phrases such as “no voices” no longer override a positive music description. A separate three-minute music-category cooldown prevents repeated comments when wording changes. Quiet music and playback echo remain suppressed.</p><p>The regression uses an authored keyboard melody, bass and drums through Chrome’s fake-device microphone, the actual caption model, LLM and TTS. It checks quiet input, a prominent onset, a spoken music response and a second onset after the detector cooldown. No song recording or private audio is published. Speech-like vocals may still be gated by the speech detector; this instrumental test does not certify every song or physical microphone.</p><p>Latest spoken response: <q>'''+esc(music.get('reply','not measured'))+'''</q>. The source and raw result are in the music receipt below.</p></section>''') if music_runs else ''
    text=f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Robby: proactive sounds, startup and voice awareness</title><style>
body{{font:16px/1.65 system-ui,sans-serif;color:#253b40;background:#f4f7f5;max-width:1100px;margin:auto;padding:24px}}h1,h2{{line-height:1.2}}section{{background:white;border-radius:16px;padding:22px;margin:20px 0}}table{{border-collapse:collapse;width:100%;font-size:15px}}td,th{{padding:12px;border:1px solid #d4e0dc;text-align:left;vertical-align:top}}th{{background:#e9f2ed}}.scroll{{overflow:auto}}a{{color:#28695e}}code{{overflow-wrap:anywhere}}.flow{{padding:18px;background:#e9f2ed;border-radius:12px}}@media(max-width:600px){{body{{padding:14px}}section{{padding:14px}}}}
</style><nav><a href="https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/">Live Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/blob/main/robby/PROACTIVE_ENVIRONMENT.md">Setup and exact protocol</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/robby-before-proactive-2026-10-27">Restore previous version</a></nav>
<h1>Robby notices prominent sounds</h1><p>Updated {datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')}. An additive extension to the existing conversational robot, with actual GPU and browser measurements.</p>
<section><h2>Verification at a glance</h2><div class="scroll"><table><tr><th>Check</th><th>Result</th><th>What it establishes</th></tr>{table}</table></div><p>These are reproducible synthetic tests. They do not certify every microphone, room, accent or television program.</p></section>
<section><h2>What changes for the user</h2><p>Automatic start and prominent-sound sensing are enabled for a fresh browser. When permissions allow, the microphone starts and Robby speaks a short welcome. If browser autoplay or microphone permission blocks this, a visible enable button explains what to do. Turning Robby off stops capture and playback and remembers that choice. Camera and persistent wiki memory remain off by default.</p><p>The lower settings provide independent controls for sound sensing, unfamiliar-voice questions, optional uncertain vocal impressions and a 6–24 dB relative prominence slider. A new timing card displays the sound-caption request latency. The underlying caption is inspectable as an unconfirmed hypothesis.</p></section>
<section><h2>One audio stream, parallel understanding</h2><p class="flow">Existing microphone → ASR / speaker focus<br>Same microphone → salience + echo guard → GPU sound caption<br>Uncertain observation → existing conversation context or an idle proactive turn → local reaction + main LLM → streaming TTS</p><p>The caption job never delays ordinary ASR. Existing owner focus and Stop detection remain separate. Quiet TV speech is not an invitation. A sufficiently different, salient, repeatedly unmatched voice can cause one cautious question, but cannot enroll itself or take over the conversation.</p><p>Normal user requests keep the memory-informed reaction, informed transition and full answer. An optional environmental turn is short: a supported tiny reaction, if available, then the main question. Adding a second transition to such a brief turn was unnecessary.</p></section>
<section><h2>Measured latency on this server</h2><p>Two RTX 3090 GPUs. Sound captioning and ASR share GPU 0; existing Humaneness Ears and TTS use GPU 1. FP16 Whisper Small, SDPA, greedy decoding, up to 48 tokens, stop after two sentences. These measurements are observations from three repetitions, not confidence intervals or latency guarantees.</p><div class="scroll"><table><tr><th>Stage</th><th>Measured time</th><th>Definition</th></tr>{times}</table></div><p>Before the sentence-stop optimization, the impact example required about 369 ms of worker time. The optimized decoder stops at 28 tokens instead of continuing to 48 tokens. Decoding dominates the remaining caption latency. Parallel ASR + sound + tone reduced the observed median wall time from {stages['serial']} to {stages['parallel']} ms. Caption latency is not microphone-to-ear latency: the detector needs event context, and the conversational reply also needs LLM/TTS/transport and browser buffering.</p><p>Latest browser sound response: <q>{esc(permitted.get('sound_reply','not measured'))}</q></p></section>
{music_section}<section><h2>Uncertainty is essential</h2><div class="scroll"><table><tr><th>Authored test input</th><th>Actual model output, unverified hypothesis</th></tr>{examples}</table></div><p>The direct quiet-input probe hallucinated a sound event. In the application, the outer level/onset detector prevents this quiet input from reaching the captioner. The model also described several impacts as a single event. It cannot be trusted as an event counter, alarm, identity detector or source of demographic facts.</p><p>Prompts require ordinary tentative questions and treat captions as data, never instructions or persistent user facts. The small reaction model can abstain. A noun-support check rejects unsupported tiny-model word slips; it does not substitute canned phrases. Model captions and generated dialogue can still be inaccurate or awkward.</p></section>
<section><h2>Bounded behavior and known tradeoffs</h2><p>Default gating: 12 dB above the learned floor, minimum −30 dBFS, 40-ms onset, roughly one second of context, 25-second detection cooldown and 30-second proactive-reply cooldown. A welcome does not consume the sound cooldown. Repeated caption prefixes are suppressed for three minutes; observations expire after eight seconds. Quiet speech, ongoing replies and pending user turns take priority.</p><p>Capture is suppressed throughout Robby's playback and for one second afterward, intentionally missing some simultaneous environmental sounds to avoid self-triggering. Real echo delay beyond that guard, microphone automatic gain and noise suppression need physical-room validation. A sound event is never a reason to cancel a normal answer.</p><p>When an unfamiliar voice continues while an optional question is speaking, the optional turn may yield. The multi-speaker receipt retains such a cancellation; no unrelated main-partner turn was tested away or counted as a full heard question. Speaker matching remains approximate.</p></section>
<section><h2>Reproduce and inspect</h2><p>The source includes <code>bootstrap_sound_caption.py</code>, <code>sound_caption_worker.py</code>, <code>ambient_audio.py</code>, <code>environment_session.py</code>, UI controls and all test scripts. No new heavyweight dependency, training dataset, private recording, biometric profile or API key is bundled. Model revision: <code>{esc(bench['worker']['revision'])}</code>. The <a href="https://huggingface.co/laion/sound-effect-captioning-whisper">model card</a> documents its training and limitations.</p><p>The first blocked-autoplay test was invalidated by Playwright granting synthetic user activation during an early JavaScript evaluation. The corrected test lets browser policy act before any evaluation; both attempts are retained. The pre-feature source checkpoint was pushed and tagged before implementation began.</p><ul>{links}</ul></section></html>'''
    (STATIC/'environment-report.html').write_text(text)
    print(json.dumps(checked,default=str))

if __name__=='__main__':main()
