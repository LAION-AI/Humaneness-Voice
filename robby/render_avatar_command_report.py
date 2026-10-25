"""English report of authored command regressions, real speech and CPU UI checks."""
import html
import json
import statistics
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def read(path,default=None):
    return json.loads(path.read_text()) if path.exists() else default

def esc(value):return html.escape(str(value))

def main():
    checked=read(ROOT/'static/avatar-command-check.json',{})
    browser=read(ROOT/'static/avatar-command-browser.json',{})
    before=read(ROOT/'static/avatar-command-before.json',[])
    unit_log=ROOT/'logs/avatar-command-unit.log'
    unit=json.loads(unit_log.read_text().splitlines()[-1]) if unit_log.exists() else {}
    rows=checked.get('rows',[])
    def median(key):
        values=[r[key] for r in rows if key in r]
        return f'{statistics.median(values):.2f} ms' if values else 'Not measured'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — reliable requested faces and spoken reactions</title>
<style>body{font:17px/1.65 system-ui;max-width:1120px;margin:30px auto;padding:20px;background:#f6f8fa;color:#20333a}h1,h2{line-height:1.25}table{border-collapse:collapse;background:white;width:100%}td,th{border:1px solid #bfcbd0;text-align:left;padding:10px}.scroll{overflow:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;background:#eaf0f4;padding:18px;font-size:14px}a{color:#166a78}.note{padding:18px;background:#e3f0ed}</style>
<p><a href="https://arrive-locations-parade-research.trycloudflare.com/desk-robot/">Open Robby</a> · <a href="https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby">Source and setup</a></p>
<h1>Requested faces now have a direct animation path</h1>
<p class="note">A clear request such as “Mach eine Grimasse” dispatches a validated screen animation before the language model starts. Robby generates its own spoken reaction alongside that animation. A missing model tool call or a failed speech request no longer silently erases the requested face.</p>
<h2>Verification summary</h2><div class="scroll"><table><tr><th>Check</th><th>Result</th><th>What was measured</th></tr>'''
    summary=[
        ('Local real-model repeats',f"{sum(bool(r.get('passed')) for r in rows)}/{len(rows)} passed",'Five German/English face requests, twice each; valid ordered actions and nonempty spoken text; no tool-JSON fallback.'),
        ('HTTPS CPU Chrome',f"{len(browser.get('turns',[]))} commands; passed={browser.get('passed')}",'Real WebSocket, local LLM and TTS PCM. Renderer frame outputs verified the requested action kinds, expressive controls and variant changes.'),
        ('Paced synthetic microphone',f"passed={checked.get('spoken_command',{}).get('passed')}",'Authored speech sent in real-time 20-ms packets through Nemotron ASR; recognized grimace command, action event, spoken response and actual TTS PCM.'),
        ('Disabled preference',str(browser.get('disabled_control_explained')),'Expressive actions stay disabled; a prominent notice explains why, while speech still works.'),
        ('Automatic regressions',f"{unit.get('tests','Pending')} tests; passed={unit.get('passed')}",'Parser, bounded retries, ordering, negative/quoted commands, 40 emotion slugs, silent requests and retained application tests.'),
        ('Server action-event dispatch',median('action_dispatch_ms'),'Median from entering the Python pipeline to yielding the event; excludes browser/network delay. Includes one first-use import.'),
        ('First generated spoken text',median('first_spoken_text_ms'),'Median local pipeline latency with TTS off; not time to physical sound.')]
    for row in summary:page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in row)+'</tr>'
    page+='''</table></div>
<h2>Why the old request could do nothing</h2><ol>
<li>The small model sometimes replied “Schau mal” without calling perform_avatar. No action reached the browser even though its words implied a performance.</li>
<li>Other responses contained malformed inline acting cues. The planning parser raised “Acting cue must precede its sentence” before speech or animation began.</li>
<li>A later end-to-end check caught a separate standalone <code>*[grimace]*</code> line after a valid spoken sentence. It was incorrectly sent as another TTS request, containing no literal words. Stage-only lines are now discarded before TTS.</li></ol>
<p>Technical production errors and source code were inspected. The examples below are independently authored reproductions; no private user conversation is included.</p>
<h2>How the fix works</h2>
<p><code>avatar_commands.py</code> recognizes clear German/English requests and calls the existing bounded action validator. Faces, eye rolls, nods, head shakes, winks and all 40 catalogue emotion slugs reuse the existing renderer. Generic grimaces cycle through eight variants. The conservative direct parser excludes quotes, conditional games, explanations and negated actions. The existing LLM tools remain available for complex choreography, knowledge and conversation rules.</p>
<p>The pipeline emits <code>avatar_action</code> before starting the LLM. A unique ID keeps subsequent sentence metadata from replaying the same action. Normal spoken words still come from the local model and enter sentence TTS immediately. Pure requested rotations retain their previous audio-playback anchor. Explicitly silent actions skip LLM/TTS. Older clients still receive the validated plan in ordinary reply metadata.</p>
<p>If a direct request gets a wholly empty or malformed unspoken completion, one bounded fresh LLM retry asks for actual spoken words. Already spoken sentences are never replayed. Standalone renderer notes are ignored; real words and ordinary acting cues remain. A real speech-service error stays visible but does not clear a valid requested face. Explicit Stop, interruption or cancellation still clears it.</p>
<h2>Real HTTPS browser turns</h2><div class="scroll"><table><tr><th>Authored request</th><th>Rendered actions</th><th>Action event before first audio</th><th>PCM chunks</th><th>Generated reply</th></tr>'''
    for row in browser.get('turns',[]):
        values=[row['request'],', '.join(row['rendered_kinds']),str(row['action_before_audio_ms'])+' ms',row['pcm_chunks'],row['reply']]
        page+='<tr>'+''.join('<td>'+esc(v)+'</td>' for v in values)+'</tr>'
    page+='''</table></div><p>The browser observed actual renderer-frame outputs, not merely the arrival of an action label. These deltas compare the action event with receipt of the first real audio block; physical loudspeaker latency and human perception were not measured.</p>
<h2>Before-fix authored reproductions</h2>'''
    for row in before:
        page+='<details><summary>'+esc(row['input'])+'</summary><pre>'+esc(json.dumps(row,indent=2,ensure_ascii=False))+'</pre></details>'
    page+='''<h2>Reproduce and inspect</h2><pre>python run_unit_checks.py
python validate_avatar_commands.py
uv run --no-project --with playwright --python /path/to/venv/bin/python python browser_avatar_commands_check.py
python render_avatar_command_report.py</pre>
<p>Run audio/browser checks serially: the live ASR engine has one session slot. The validation script uses the local services; the browser script has a configurable source constant for this deployment's public HTTPS hostname. The source archive includes the command parser, validators, exact prompt changes, bridge/UI handling and regression scripts. Test output contains authored synthetic speech only.</p>
<h2>Limits</h2><p>This is a targeted engineering regression, not a guarantee for every possible wording. Ambiguous and complex requests retain the model tool path. ASR can misrecognize a real microphone utterance. Device-room echo, human interruption and unrelated background speakers were not certified by these checks. Camera tracking, speaker identity, speech-pause settings, idle play, petting and existing model choices remain in place.</p>
<p>Animation preferences are respected. Hidden tabs, eye-contact calibration or reduced-motion settings may intentionally suppress motion. The browser test disables camera/idle events to isolate explicit commands. One additional clearly labeled injected speech error confirms action survival; a synthetic BARGE confirms cancellation. Those injected cases are separate from the real-model/TTS turns. No private microphone recording, face profile or enrolled voiceprint is published.</p>
<details><summary>Machine-readable test receipts</summary><pre>'''
    page+=esc(json.dumps({'local_model_and_asr':checked,'https_cpu_browser':browser,'unit':unit},indent=2,ensure_ascii=False))+'</pre></details>'
    page+='<p>Generated '+esc(time.strftime('%Y-%m-%d %H:%M UTC',time.gmtime()))+'.</p></html>'
    (ROOT/'static/avatar-command-update.html').write_text(page)
    (ROOT.parent/'robby-avatar-command-report.html').write_text(page)
    print(json.dumps({'model_and_asr_passed':checked.get('passed'),'browser_passed':browser.get('passed'),'unit_passed':unit.get('passed')}))

if __name__=='__main__':main()
