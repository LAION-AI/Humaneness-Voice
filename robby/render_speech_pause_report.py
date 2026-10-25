"""English standalone handoff from the real pause regression measurements."""
import html,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):
    p=ROOT/'static'/name
    return json.loads(p.read_text()) if p.exists() else {}
def esc(x):return html.escape(str(x))
def main():
    audio=read('speech-pause-check.json');browser=read('speech-pause-browser.json')
    unit_path=ROOT/'logs/speech-pause-unit.log'
    unit=json.loads(unit_path.read_text().splitlines()[-1]) if unit_path.exists() else {}
    rows=''.join('<tr>'+''.join('<td>'+esc(v)+'</td>' for v in [r['case'],str(r['pause_ms'])+' ms',str(r['gap_s'])+' s',r['noise'],r['passed'],
        f"{r['submission_after_clip_end_ms']:.0f} ms" if 'submission_after_clip_end_ms' in r else 'See event record'])+'</tr>' for r in audio.get('rows',[]))
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — wait for a speech pause</title>
<style>body{font:17px/1.65 system-ui;max-width:1120px;margin:30px auto;padding:20px;color:#20333a;background:#f6f8fa}h1,h2{line-height:1.25}.scroll{overflow:auto}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #bfcbd0;padding:10px;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:16px;font-size:13px}a{color:#166a78}.note{padding:18px;background:#e2f1ec}</style>
<p><a href="https://arrive-locations-parade-research.trycloudflare.com/desk-robot/">Open Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source and setup</a></p>
<h1>Robby sends your complete turn after a speech pause</h1>
<p>Previously, recognized punctuation could send the first sentence immediately, even while the user continued speaking. The fallback pause timer also treated sufficient sound energy as activity, allowing background noise to hold a turn open.</p>
<p class="note">The input path now waits for a configurable neural speech pause: <strong>400 ms by default</strong>, adjustable from 200 to 2,000 ms with the prominent <strong>Speech pause before replying</strong> slider. Consecutive sentences before that pause form one LLM request. Punctuation remains a preview. Output sentence-to-TTS streaming is unchanged.</p>
<h2>Real audio regression checks</h2><div class="scroll"><table><tr><th>Scenario</th><th>Pause setting</th><th>Gap between sentences</th><th>Added noise</th><th>Passed</th><th>Submission after final clip end</th></tr>'''+rows+'''</table></div>
<p>These checks send authored synthetic German microphone audio in real time through the actual websocket, Nemotron ASR, local LLM and TTS. Noise is continuous Gaussian noise plus a 120-Hz hum, loud enough to exceed the former energy gate. Short gaps must retain both sentences in one request; longer gaps must produce two turns. Noise alone must produce none. Each committed event must report at least the selected neural quiet duration. Transport acknowledgments do not simulate a physical loudspeaker.</p>
<h2>Browser and unit checks</h2>'''
    out+='<p>HTTPS Chrome browser passed: <strong>'+esc(browser.get('passed','Pending'))+'</strong>. It checks the visible 400-ms default, saves 950 ms, reloads, confirms server restoration, changes live to 1,000 ms, speaks two sentences with a 500-ms gap and continuous noise, receives one complete turn and real TTS audio, then changes live to 300 ms. CPU browser, authored fake microphone; no physical camera or private microphone recording.</p>'
    out+='<p>Unit checks: '+esc(unit.get('tests','Pending'))+'; passed: '+esc(unit.get('passed','Pending'))+'. Dedicated regressions cover native endpoint/punctuation buffering, grouped multi-sentence submission, short farewell, resume-during-ASR races, early speech candidates, bounded settings, immediate Stop and avoiding old words in the next turn’s pre-roll.</p>'
    out+='<h2>Implementation and measured CPU cost</h2><p>A 1.29-MB Silero ONNX model classifies 32-ms audio windows on one CPU inference thread, retaining recurrent state only for this session. Two positive frames establish speech; probability hysteresis prevents flicker. The same microphone packets and timestamp-aligned playback reference are reused. Inference runs off the event loop. Mean measured detector work per 32-ms frame: <strong>'+esc(round(audio.get('vad_ms_per_32ms_frame',0),3))+' ms</strong>. This is actual local CPU inference, not a vendor speed claim.</p>'
    out+='''<p>Native ASR and TYGER remain unchanged. The final transcript check uses a bounded full-window ASR pass; if speech resumes while that pass is running, its old pause cannot submit a turn. An accepted turn resets its input segment so subsequent pre-roll cannot re-submit old words. Explicit Stop while Robby speaks bypasses the pause delay. The browser persists only the selected setting; voice memory remains session RAM.</p>
<h2>Scope and remaining limits</h2><p>The selected pause is a minimum quiet interval according to a neural detector, not exact ground-truth word timing. Model onset/tail uncertainty, packet scheduling and ASR work add latency. The table measures time after the authored clip end; it is not physical mouth-to-speaker latency. Background speech can still be detected as speech: this detector is not source separation or a guarantee against TV voices. Strong aligned bot echo is suppressed for the pause clock, but real-room echo needs its own test. Passing these synthetic noise cases does not cover every environment.</p>
<p>A request already decoding in the language model cannot be edited in place. Speech before the configured pause is grouped; speech after a submitted turn becomes the next ordered request. True mutable-context input streaming would require a different generation/control protocol. The older strict speaker-gated mode retains its own endpoint policy.</p>
<p>Reproduce with <code>python run_unit_checks.py</code>, <code>python validate_speech_pause.py</code>, then the Playwright <code>browser_speech_pause_check.py</code>. Do not run audio tests concurrently: the demo has one ASR session slot. Model weights are fetched only at bootstrap/startup and hash-checked; live microphone inference does not download files.</p>
<p>Model revision: <code>066409f3d9db45a30d21dfcbf9147826ffdd7052</code>. SHA256: <code>7ed98ddbad84ccac4cd0aeb3099049280713df825c610a8ed34543318f1b2c49</code>. Upstream MIT license and interface: <a href="https://github.com/snakers4/silero-vad">Silero VAD</a>. No extra GPU model is loaded.</p><details><summary>Raw measurements and test receipts</summary><pre>'''
    out+=esc(json.dumps({'audio':audio,'browser':browser,'unit':unit},indent=2,ensure_ascii=False))+'</pre></details><p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/speech-pause-update.html').write_text(out)
    (ROOT.parent/'robby-speech-pause-report.html').write_text(out)
    print(json.dumps({'audio_passed':audio.get('passed'),'browser_passed':browser.get('passed'),'unit_passed':unit.get('passed')}))
if __name__=='__main__':main()
