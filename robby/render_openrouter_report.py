"""English rollout report with real provider and HTTPS-browser test receipts."""
import html,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def esc(v):return html.escape(str(v))
def read(path):return json.loads(path.read_text()) if path.exists() else {}

def main():
    checked=read(ROOT/'static/openrouter-check.json');browser=read(ROOT/'static/openrouter-browser.json')
    p=ROOT/'logs/openrouter-unit.log';unit=json.loads(p.read_text().splitlines()[-1]) if p.exists() else {}
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — OpenRouter, private keys and visible stage timings</title>
<style>body{font:17px/1.65 system-ui;max-width:1180px;margin:30px auto;padding:20px;background:#f6f8fa;color:#20333a}h1,h2{line-height:1.25}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #bfcbd0;text-align:left;padding:10px}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:18px;font-size:14px}a{color:#166a78}.note{padding:18px;background:#e3f0ed}</style>
<p><a href="https://arrive-locations-parade-research.trycloudflare.com/desk-robot/">Open Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source and setup</a></p>
<h1>Choose the language model; keep the conversation and timing in view</h1>
<p class="note">Robby can use Qwen 3.8 27B through the selected OpenRouter endpoint, with Cerebras as the configured default, or the already resident local Nemotron. The administrator's key stays encrypted outside the repository. The last recognized/user/robot turns sit beside the avatar, and measured component timings appear underneath it.</p>
<h2>Actual language-model checks</h2><p>These are individual engineering observations, not a statistical provider speed ranking. Provider load, geography, prompt size and API transit affect them. LLM first sentence is when enough generated words exist for the first TTS request; the model still emits only compact acting cues and speech.</p><div class="scroll"><table><tr><th>Requested route</th><th>Actual provider</th><th>First token</th><th>First sentence</th><th>Task</th><th>Passed</th></tr>'''
    for row in checked.get('rows',[]):
        values=[row['model']+' / '+row['provider'],row['actual_provider'],str(row.get('first_token_ms'))+' ms',str(row.get('first_sentence_ms'))+' ms',row['input'],row['passed']]
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    page+='''</table></div><h2>Real browser turns and stage timing</h2><p>CPU Chromium, actual public HTTPS/WSS, authored synthetic microphone audio, real local ASR and real speech synthesis. No private microphone recording was used. A dash means unmeasured/not applicable; typed input has no ASR operation.</p><div class="scroll"><table><tr><th>Turn</th><th>ASR</th><th>Speaker wait</th><th>LLM first sentence</th><th>TTS first PCM</th><th>Reply → first PCM</th><th>Playback lead</th></tr>'''
    def ms(v):return '—' if v is None else f'{v:.1f} ms'
    for row in browser.get('turns',[]):
        v=row['timings'];values=[row['case'],row['asr_card'],row['speaker_card'],ms(v.get('llm')),ms(v.get('tts')),ms(v.get('reply')),ms(v.get('buffer'))]
        page+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in values)+'</tr>'
    page+='''</table></div>
<h2>Read the timings correctly</h2><ul>
<li><b>ASR:</b> the final speech-pause recognition pass. Live preview and the selected pause itself are separate. Alternative recognition includes its loopback request; the native path measures the existing decoder call.</li>
<li><b>Speaker ID:</b> remaining ECAPA/lookup wait on the path to submission. An already prefetched embedding can make this below one millisecond; the background compute time remains visible in details.</li>
<li><b>LLM:</b> time to the first complete sentence or validated tool plan. TTFT and full stream wall time are additional details. Cloud measurements include network transit. Full stream wall time may include downstream queue backpressure.</li>
<li><b>TTS:</b> first decoded/resampled PCM, including server admission and transport. Completed-sentence GPU generation times and RTF are shown separately when supplied by the TTS backend.</li>
<li><b>Reply → first PCM:</b> server response pipeline start to first audio, excluding prior ASR/speaker processing. This is not microphone-to-ear latency.</li>
<li><b>Playback lead:</b> real scheduled audio start minus the browser AudioContext clock. It is a scheduling buffer, not a physical-device latency measurement.</li></ul>
<p>Stages overlap, especially LLM generation and TTS. Do not sum the cards. Each accepted reply has an ID so queued input and late measurements can be distinguished. Existing playback timestamps, interruptions and turn-taking remain in place.</p>
<h2>Exact provider routing and tools</h2><p>The requested web endpoint corresponds to OpenRouter provider tag <code>cerebras/fp16</code>. Requests use <code>provider.only</code> with <code>allow_fallbacks=false</code>; an unavailable provider is reported instead of silently switching. Additional current endpoint tags come from the public model-endpoint API. The model list is allowlisted to Qwen and the local model; arbitrary provider URLs cannot receive the server key.</p>
<p>Reasoning is disabled/excluded for fast spoken replies. Provider-specific unsupported JSON-format parameters are omitted. The same acting catalogue, sentence framer, complete-task guard, avatar commands and tools are shared. Explicit interruption-rule requests require actual tools, preserving the banana-game setup rather than accepting a verbal promise. The independent decision model remains separate. One brief retry handles a rate-limit response before any speech; already emitted words are never replayed. Rate limits and provider outages can still interrupt a request.</p>
<h2>Credential handling</h2><p>The administrator enrolls the key once with <code>python credential_store.py</code>, using a hidden terminal prompt. Authenticated Fernet ciphertext and a separately generated master secret live in a private directory outside the checkout. Directory mode is 0700, file modes 0600. No read-key API exists. The browser sees only credential availability and never receives the administrator's plaintext or ciphertext. Source export and staged release checks exclude credential files.</p>
<p>Optional personal keys use a password field, are cleared on submission and kept encrypted only in that connection's server memory. They neither overwrite the shared administrator key nor enter browser storage. Disconnecting discards the override. Encryption at rest depends on a protected master secret and OS account: this is not an HSM or protection against server-account compromise. Cloud mode sends conversation text to OpenRouter and the chosen provider; audio processing stays on this server and camera inference stays in the browser.</p>
<h2>Interface changes</h2><p>The sticky desktop sidebar shows live ASR, the latest user turn and Robby's current reply. Full history remains available. The six timing cards sit directly under the avatar. Model selectors, microphone/camera controls and the main interruption checkbox stay visible; speech-pause and detailed interruption settings are grouped compactly. Mobile stacks the panels without removing controls. Preferences save model/provider IDs only.</p>'''
    page+='<p>Automatic checks: '+esc(unit.get('tests','Pending'))+'; passed='+esc(unit.get('passed'))+'. Real route/API checks: '+esc(checked.get('passed'))+'. HTTPS browser checks: '+esc(browser.get('passed'))+'.</p>'
    page+='''<h2>Reproduce</h2><pre>pip install -r requirements.txt
python credential_store.py  # hidden prompt; secret lives outside the checkout
# Start/restart the documented Robby server and FastAPI facade.
python run_unit_checks.py
python validate_openrouter.py
uv run --no-project --with playwright --python /path/to/venv/bin/python python browser_openrouter_check.py</pre>
<p>The validation calls actual paid OpenRouter routes using the configured server credential, with authored requests. Historical local-model browser checks explicitly select local mode. No key is included in this report, its JSON receipts, source archive or repository. The source contains the exact shared system prompt, transport adapter, key-store code, per-session settings and measurement definitions.</p>
<p>Primary references: <a href="https://openrouter.ai/docs/guides/routing/provider-selection">OpenRouter exact-provider routing</a>, <a href="https://openrouter.ai/docs/guides/best-practices/reasoning-tokens">reasoning controls</a>, <a href="https://openrouter.ai/qwen/qwen3.8-27b">Qwen model/endpoints</a>, and <a href="https://cryptography.io/en/latest/fernet/">Fernet authenticated encryption</a>.</p>
<details><summary>Machine-readable receipts</summary><pre>'''
    page+=esc(json.dumps({'api':checked,'https_browser':browser,'unit':unit},ensure_ascii=False,indent=2))+'</pre></details>'
    page+='<p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/openrouter-update.html').write_text(page);(ROOT.parent/'robby-openrouter-report.html').write_text(page)
    print(json.dumps({'api_passed':checked.get('passed'),'browser_passed':browser.get('passed'),'unit_passed':unit.get('passed')}))
if __name__=='__main__':main()
