"""Publish explicit synthetic receipts only, including unsuccessful attempts."""
import datetime,html,json,shutil
from pathlib import Path
from final_verify import runtime_signature

ROOT=Path(__file__).resolve().parent
LIVE=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')

def main():
    folder=ROOT/'static/stop-reading-data';folder.mkdir(exist_ok=True)
    files=[LIVE/'stop-commands-live.json',LIVE/'streamed-echo-live.json',*sorted(LIVE.glob('stop-attempt-179261*.json')),
        *[ROOT/'logs'/n for n in ('unit-checks.json','stop-controls-browser.json','browser-lifecycle-checks.json',
            'reliability-ui-checks.json','open-speakers-browser.json','knowledge-browser.json','speaker-variant-live.json',
            'reading-feedback-live.json','reading-final-regressions.json','director-reading-regression.json','knowledge-director-live.json')]]
    for path in files:
        if path.exists():shutil.copyfile(path,folder/path.name)
    def read(p):return json.loads(p.read_text()) if p.exists() else {}
    current=read(LIVE/'stop-commands-live.json');signature=runtime_signature()
    e=lambda x:html.escape(str(x))
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby: interruptible reading update</title><style>body{font:16px/1.55 system-ui;max-width:1050px;margin:30px auto;padding:0 18px;background:#faf8f4;color:#293637}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border:1px solid #ccc;text-align:left}.scroll{overflow:auto}.note{padding:15px;background:#fff0c8}a{color:#25608a}code{overflow-wrap:anywhere}</style><p><a href="/desk-robot/">Open Robby</a> · <a href="/static/worklog.html">Task log</a> · <a href="/downloads/robby-source.zip">Source bundle</a></p><h1>Interruptible long reading: measured results and limits</h1>'''
    out+='<p>Built '+e(datetime.datetime.now(datetime.timezone.utc).isoformat())+'. Test inputs are public synthetic voices, not private recordings.</p>'
    out+='<h2>Latest long-reading replay</h2><p>Actual Nemotron ASR, speaker models, director, Wikipedia lookup and streamed Small RL TTS. A command starts about six seconds into playback of a ≥70-word answer. Virtual DAC buffering: 240 ms; injected echo: 5% gain and 120-ms delay. Each case has a fresh session. Another enrolled person and an unknown background voice are negative controls. No physical microphone or calibrated human identity accuracy.</p>'
    out+='<p>Current-source match: <strong>'+e(current.get('runtime_source_sha256')==signature)+'</strong>. Whole latest replay passed: <strong>'+e(current.get('passed',False))+'</strong>.</p>'
    out+='<div class="scroll"><table><tr><th>Command fixture</th><th>Authorized</th><th>Words being read</th><th>Stopped</th><th>Latency from clip start</th><th>Passed / limitation</th></tr>'
    for c in current.get('cases',[]):
        values=[c['command'],c['authorized'],c.get('reading_words','—'),c.get('interrupted','—'),
            '—' if c.get('interruption_latency_s') is None else f"{c['interruption_latency_s']:.2f} s",c.get('error') or c['passed']]
        out+='<tr>'+''.join('<td>'+e(v)+'</td>' for v in values)+'</tr>'
    out+='</table></div><p class="note">A failed identity check remains a failure, not a success because ASR heard the words. Silence is the default after a stop. Very short synthetic commands can have unstable voice cues and can require repetition or the manual Stop reply button. Development cases tune the implementation and are not held-out human calibration. A too-short answer is invalid test setup, not an interruption pass.</p>'
    out+='<h2>What changed</h2><ul><li>Separate Stop task phase prevents an old proactive decision marker from cancelling the next pending stop decision.</li><li>Every 320-ms hop can schedule bounded stop work during playback; at most one ASR probe and one semantic request are active. This is not a 320-ms reaction-time guarantee.</li><li>Native partial candidates, current reply text and actual playback position reach the decision controller. Stale decisions, changed replies, uncertain speaker identity, rivals and echo are guarded.</li><li>Bounded German/English wait commands and a narrow ASR homophone are consumed as commands. Statements about waiting, quotations and unrelated person names are not direct stops.</li><li>Compact commands require agreement from two acoustic ASR crops plus teacher/fast speaker consensus. Thresholds and their development-only status are explicit in README and source. General turn thresholds remain unchanged.</li><li>Browser cancellation stops scheduled sources, cancels generation and rejects late packets from the old reply ID.</li><li>Optional, default-OFF short acknowledgments rotate four local phrases in German/English without a director call. Wait/quiet commands stay silent; new speech, pending turns and changed sessions suppress feedback.</li><li>News/Wikipedia/long explanation prompts explicitly request calm, gentle, unhurried articulation and pauses, without changing spoken facts. No fixed words-per-minute improvement is claimed.</li><li>Wikipedia entity extraction excludes trailing reading instructions while preserving names containing “und”.</li></ul>'
    out+='<h2>Regression receipts</h2><ul>'
    for name in ('unit-checks','stop-controls-browser','browser-lifecycle-checks','reliability-ui-checks','open-speakers-browser','knowledge-browser','speaker-variant-live','reading-feedback-live','reading-final-regressions'):
        d=read(ROOT/'logs'/(name+'.json'))
        if d:out+='<li>'+e(name)+': '+e(d.get('passed',False))+'; source match where signed: '+e(not d.get('runtime_source_sha256') or d['runtime_source_sha256']==signature)+'.</li>'
    out+='</ul><h2>Retained development failures</h2><p>The first long-reading runs accidentally searched Wikipedia for the whole “Bananen und lies mir …” instruction. Another multi-user setup lost the beginning of its request during diarization, so the answer was too short; the revised test re-establishes the actual addressee first. Some short commands were recognized but rejected by speaker checks, and one homophone reached the slower semantic fallback too late. All retained attempts below remain inspectable. No claim of flawless real-room recognition is made.</p><h2>Reproduce and inspect</h2><p>Run <code>python run_unit_checks.py</code>, then <code>python -m speaker_live.stop_test --long-reading --continue-on-failure</code> with the commands listed in README. Run live tests sequentially: one ASR slot is available. Browser tests use Playwright/Chrome and fake devices. The source bundle includes the harness, prompts and settings; weights and credentials are excluded.</p><ul>'
    for path in files:
        if path.exists():out+='<li><a href="/static/stop-reading-data/'+e(path.name)+'">'+e(path.name)+'</a></li>'
    out+='</ul></html>'
    (ROOT/'static/stop-reading-update.html').write_text(out)
    print('Built stop-reading-update.html')

if __name__=='__main__':main()
