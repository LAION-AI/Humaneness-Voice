"""English report; retain both successful and difficult acoustic probes."""
import html,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
def read(name):
    p=ROOT/name
    return json.loads(p.read_text()) if p.exists() else {}
def esc(x):return html.escape(str(x))
def main():
    check=read('static/conversation-focus-check.json');browser=read('static/conversation-focus-browser.json')
    unit=read('logs/unit-checks.json')
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Robby: keeping the conversation partner</title>
<style>body{font:16px/1.65 system-ui;max-width:1100px;margin:30px auto;padding:20px;background:#f6f8f4;color:#263c35}h1,h2{line-height:1.25}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #cbd6cc;text-align:left}pre{white-space:pre-wrap;font-size:12px}section{background:white;padding:18px;margin:18px 0;border-radius:14px}.scroll{overflow:auto}a{color:#226356}</style>
<nav><a href="/desk-robot/">Open Robby</a> · <a href="/static/openrouter-update.html">Language models, security and live timings</a> · <a href="/downloads/robby-source.zip">Source</a></nav>
<h1>Keep the conversation partner, ignore the background</h1>
<p>The sentence-streaming mode used to admit every intelligible voice. More seriously, native TYGER BARGE events and literal Stop previews could bypass speaker confirmation entirely. Lowering one global similarity threshold would have increased background interruptions. This update introduces session-local conversation ownership and acoustic admission before both conversation turns and playback cancellation.</p>
<div class="scroll"><table><tr><th>Behavior</th><th>Previous sentence-stream mode</th><th>Current default</th></tr>
<tr><td>Native interruption</td><td>Direct cancellation on a TYGER proposal</td><td>Current reply ID, fresh speech, active-person ECAPA match, bot-echo/rival margins and available native-hop student features. Nonlexical interruption requires stronger agreement.</td></tr>
<tr><td>A different background voice</td><td>Could cancel playback or be assigned the next turn</td><td>Ignored while the conversation is focused; no new profile is created from a rejected turn.</td></tr>
<tr><td>Changing intonation</td><td>New identity below 0.38 cosine; adaptation needed 0.70</td><td>Owner continuity at 0.32, with rival margin. Confirmed moderate variants can be retained from 0.48; the original anchor stays fixed.</td></tr>
<tr><td>Another person wants to join</td><td>Every voice could join at any time</td><td>Say Hey Robbie while he is listening, release the focus, disable the focus, or wait for 3 minutes of inactivity.</td></tr>
<tr><td>Background speech continues after the user stops</td><td>Global speech activity could keep the pause open</td><td>Rolling 1.28-s ECAPA checks can hold the owner's pause clock when a clearly different voice continues. Final ASR excludes the rejected tail where possible.</td></tr>
</table></div>
<h2>Controls and identity memory</h2><p><b>Stay with the current conversation partner</b> is on by default near the main controls. Its adjustable partner cosine starts at <b>0.32</b>, independently of the 0.38 new-voice label threshold. Lower values tolerate more variation but can merge similar voices. The separate 0.55 floor for a nonlexical TYGER interruption avoids treating moderate acoustic similarity as permission to cancel. The label threshold and fast-cue legacy settings are separate controls.</p>
<p>The first clear voice can begin without an introduction. An explicit self-correction such as “Ich bin auch Christoph” links another acoustic mode to the named person. Up to eight variants per person are retained; weak matches never train the reference. Names, embeddings and audio buffers are RAM-only and cleared when the session ends. Embeddings are approximate identity cues, not authentication. A manual Stop button always works independently of voice matching.</p>
<h2>Fast and longer evidence</h2><p>The frozen Nemotron encoder and existing V4 student already supply 320-ms speaker features. The update reuses those features and compares them to prototypes from accepted turns. Longer 3.04-s ECAPA anchors improve the voice reference. A separate 1.28-s recent window, sampled every 640 ms on CPU, checks whether current activity belongs to the partner. This reuses already heard audio and does not wait for a future three-second recording.</p>
<p>Literal Stop requests use the same guard. Weak short-word ECAPA evidence can be supported by temporally consistent native-hop features, with rival/echo separation retained. Very strong recent student evidence can trigger a second, longer ECAPA view. Merely keeping one's mouth open, making noise or matching the previous speaker's long context is not sufficient. Checks are single-flight and revalidate the reply ID before cancelling, so a late result cannot stop a newer answer.</p>'''
    page+='<h2>Measured checks</h2><p>Offline implementation contracts: '+esc(unit.get('tests'))+'; passed='+esc(unit.get('passed'))+'. Public HTTPS CPU-browser test: '+esc(browser.get('passed'))+'. Public synthetic voice diagnostics: '+esc(check.get('summary'))+'.</p>'
    page+='<p>Warm CPU ECAPA extraction: median '+esc(check.get('teacher_cpu_ms',{}).get('median'))+' ms across '+esc(check.get('teacher_cpu_ms',{}).get('n'))+' short reference probes. This is inference compute, not the time to hear the utterance or perform ASR.</p>'
    page+='<div class="scroll"><table><tr><th>Real short-command guard</th><th>Expected owner</th><th>Playback cancelled</th><th>Teacher cosine</th><th>Fast cosine</th><th>Guard wall time</th></tr>'
    for r in check.get('short_command_probes',[]):
        page+='<tr>'+''.join('<td>'+esc(x)+'</td>' for x in [r['case'],r['expected_owner'],r['cancelled'],r.get('teacher_similarity'),r.get('fast_similarity'),str(r['wall_ms'])+' ms'])+'</tr>'
    page+='</table></div><h2>Important limits revealed by the tests</h2>'
    page+='''<p>These are deliberately small, correlated synthetic probes, not a false-acceptance-rate estimate for people. In the difficult unknown-voice test, the synthetic Robert Stop can resemble the Christoph anchor in both ECAPA and the feature student. This remains an observed false acceptance; it is not hidden by the functional test result. Registering Robert as a distinct known speaker supplies the rival margin and rejects that probe. Similar or overlapping voices cannot be reliably separated by a threshold alone.</p>
<p>The emotion-profile test is also mixed: the Mira anchor retained all six selected variants, whereas the fixed Idris source did not match six synthesized profile variants at 0.32. A nominal dataset identity does not guarantee that synthesis preserved its acoustic voice. Use actual session speech as the anchor, speak a clear sentence when joining, and explicitly correct a changed voice when necessary. No general speaker accuracy or physical-room guarantee is claimed.</p>
<p>Rolling ownership can stop another voice from extending a pause after a clearly recognized speaker switch. It cannot cleanly unmix two people speaking simultaneously. Strong overlap, a very short unfamiliar Stop, microphone/device artifacts or a failed embedding can remain uncertain. The system then keeps the conversation rather than inventing a new identity; use the manual Stop when needed. No extra GPU model or heavy library was added.</p>
<h2>Browser scenarios</h2><ul>'''
    for r in browser.get('cases',[]):page+='<li>'+esc(r['case'])+': '+esc(r['passed'])+'</li>'
    page+='</ul><p>Controls, reload persistence, chat/metrics clearing, desktop adjacency and mobile overflow are checked in the same script. Only public authored synthetic audio is submitted; no private microphone recording or enrolled voiceprint is published.</p>'
    page+='''<h2>Files and reproduction</h2><p><code>conversation_focus.py</code> is the pure ownership policy. <code>sentence_session.py</code> owns async evidence, pause filtering, bounded variants and guarded cancellation. <code>server.py</code> routes native proposals and settings through it. Both director prompts preserve the current speaker/thread without repetitive name questions. The existing camera, avatar, TTS and TYGER network remain available.</p>
<pre>python run_unit_checks.py
python prepare_focus_fixtures.py
python validate_conversation_focus.py
uv run --no-project --with playwright --python /path/to/venv/bin/python python browser_conversation_focus.py</pre>
<p>The audio diagnostics expect the documented existing public-profile asset cache. Source export contains code and result receipts, not the fixture audio or session embeddings. The browser test needs the configured running ASR/TTS/local-LLM services and one free ASR session.</p>
<details><summary>Exact receipts and limitations</summary><pre>'''
    page+=esc(json.dumps({'diagnostics':check,'browser':browser,'unit':unit},indent=2,ensure_ascii=False))+'</pre></details>'
    page+='<p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/conversation-focus-update.html').write_text(page);(ROOT.parent/'robby-conversation-focus-report.html').write_text(page)
    print(json.dumps({'browser_passed':browser.get('passed'),'short_commands_all_correct':check.get('short_commands_all_correct'),'unit_passed':unit.get('passed')}))
if __name__=='__main__':main()
