"""Current correction receipts; explicit synthetic allowlist, no debug audio."""
import datetime,html,json,shutil
from pathlib import Path
from final_verify import runtime_signature

ROOT=Path(__file__).resolve().parent
CURRENT=('stop-phase-regressions','stop-enrolled-regressions','stop-negative-regressions','reading-feedback-live',
         'reading-final-regressions','unit-checks','stop-controls-browser','browser-spoken-stop','shared-cancellation-live')
FILES=CURRENT+('stop-phase-before-dual-view','stop-phase-before-asr-padding','stop-asr-padding-ablation','stop-phase-before-short-consensus',
              'stop-phase-before-span-guard','stop-negative-before-span-guard','feedback-before-backend-cancellation',
              'regressions-before-backend-cancellation','browser-stop-first-attempt','browser-stop-long-loopback')


def main():
    signature=runtime_signature();folder=ROOT/'static/stop-fix-data';folder.mkdir(exist_ok=True)
    data={}
    for name in FILES:
        path=ROOT/'logs'/(name+'.json')
        if path.exists():
            data[name]=json.loads(path.read_text());shutil.copyfile(path,folder/path.name)
    e=lambda value:html.escape(str(value))
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby Stop correction</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:30px auto;padding:0 18px;background:#faf8f4;color:#25363a}table{border-collapse:collapse;width:100%;background:white}th,td{padding:9px;border:1px solid #ddd;text-align:left}.scroll{overflow:auto}.note{background:#fff0ca;padding:16px}a{color:#125a7c}</style><p><a href="/desk-robot/">Robby demo</a> · <a href="/static/worklog.html">Task log</a> · <a href="/static/stop-reading-update.html">Earlier results</a></p><h1>Short Stop commands and spoken feedback: correction</h1>'''
    out+='<p>Built '+e(datetime.datetime.now(datetime.timezone.utc).isoformat())+'. Public synthetic voices, real ASR/TTS and fake browser devices. No private microphone recordings or voiceprints are published.</p>'
    out+='<div class="scroll"><table><tr><th>Test</th><th>Passed</th><th>Current source</th></tr>'
    for name in CURRENT:
        d=data.get(name,{})
        out+='<tr><td>'+e(name)+'</td><td>'+e(d.get('passed','not run'))+'</td><td>'+e(d.get('runtime_source_sha256')==signature if 'runtime_source_sha256' in d else 'offline / unsigned')+'</td></tr>'
    out+='</table></div><h2>Long-reading commands and controls</h2><div class="scroll"><table><tr><th>Fixture</th><th>Timing shift</th><th>Authorized</th><th>Stopped</th><th>Latency from clip start</th><th>Pass / error</th></tr>'
    for name in ('stop-phase-regressions','stop-enrolled-regressions','stop-negative-regressions'):
        for c in data.get(name,{}).get('cases',[]):
            latency=c.get('interruption_latency_s')
            cells=[c['command'],c.get('phase_offset_s',0),c['authorized'],c.get('interrupted','—'),
                   '—' if latency is None else f'{latency:.2f} s',c.get('error') or c['passed']]
            out+='<tr>'+''.join('<td>'+e(v)+'</td>' for v in cells)+'</tr>'
    out+='</table></div><p class="note">A pass requires an actual interruption event during a real ≥70-word streamed answer, not merely a recognized transcript. Negative controls require no interruption and no new spoken reply. Every declared case is retained; these are development regression checks, not held-out human accuracy estimates.</p>'
    out+='<h2>Corrections</h2><ul><li>Quiet consonants survive acoustic cropping after foreground admission. Echo residuals use a bounded, held-out-validated nine-tap filter when scalar gain is insufficient; native microphone/TYGER inputs remain unchanged.</li><li>Short Stop can be corroborated by two acoustic ASR views or by two consecutive distinct decode hops. Old replies, a different addressee and expired audio cannot borrow this evidence.</li><li>When ordinary short-word verification is inconclusive, the existing frozen encoder and CPU student recheck the command at three fixed frame alignments. The embeddings are averaged, not selected by a favorable score. Existing verified recent voice examples preserve phonetic diversity.</li><li>Current-person, rival and bot checks remain mandatory. A strong joint-score route or agreement from both live and phase-stable student views is required alongside the classic speaker model. This is not unconditional keyword interruption or authentication.</li><li>Optional acknowledgment waits up to a bounded quiet-floor deadline for cancelled generation and consumed-command processing. New turns, replies, owner changes and disconnect still suppress it. Stop tails no longer become erroneous new identity requests.</li></ul>'
    out+='<p>The isolated ASR views now use different fixed leading-silence contexts (0 and 1.28 seconds). This is immediate padding, not additional waiting. The padding ablation records cases where the previous 200-ms context decoded a clean Stop as empty. A very short word with teacher cosine 0.20–0.25 requires both live and phase-stable student cues ≥0.45 plus the stronger rival/bot margin. These correlated cues are not independent probabilities.</p>'
    out+='<p>The initial enrolled-rival setups in the phase run are retained, including failures before a sufficiently long reply was produced. Separate enrolled-rival controls explicitly speak the current user’s introduction again before the long request; ASR and voice matching still run normally. This checks stop authorization, not general wake-free multi-user request accuracy. An invalid setup is not a successful negative control.</p>'
    browser=data.get('browser-spoken-stop',{})
    cancel=data.get('shared-cancellation-live',{})
    out+='<h2>Shared GPU worker cancellation</h2><p>Robby opts only its own request into cooperative disconnect cancellation. Ordinary Playground calls keep their prior default. The busy flag is held until the GPU thread unwinds, not released by canceling an asyncio wrapper. Measured disconnect-to-idle: <strong>'+e(cancel.get('disconnect_to_idle_s','not measured'))+' seconds</strong>; passed: '+e(cancel.get('passed',False))+'.</p>'
    out+='<h2>Actual browser audio playback</h2><p>HTTPS UI, synthetic MediaStream, real ASR and TTS, with delayed bot echo. Passed: <strong>'+e(browser.get('passed',False))+'</strong>; acknowledgment playback-complete received: <strong>'+e(browser.get('acknowledgment_played',False))+'</strong>. The test also checks that old audio is canceled and microphone capture remains active. No physical microphone or human listening is claimed.</p>'
    feedback=data.get('reading-feedback-live',{})
    out+='<h2>Optional acknowledgment</h2><p>Actual matching reply text AND non-empty streamed PCM observed: <strong>'+e(feedback.get('acknowledgment_observed',False))+'</strong>. Silence remains the default; the checkbox enables four rotating local phrases. Explicit wait/quiet commands remain silent.</p>'
    out+='<h2>Reproduce</h2><p>Run sequentially with the single ASR slot free: <code>python -m speaker_live.stop_regressions</code>, <code>python -m speaker_live.stop_negative_regressions</code>, <code>python -m speaker_live.feedback_check</code>, and <code>python reading_final_regressions.py</code>. Runtime hashes separate current evidence from the retained failed development run. Neither a finite synthetic suite nor a secret-pattern scan proves flawless real-room behavior.</p><ul>'
    for name in data:out+='<li><a href="/static/stop-fix-data/'+e(name)+'.json">'+e(name)+'.json</a></li>'
    out+='</ul></html>'
    (ROOT/'static/stop-fix-update.html').write_text(out)
    print('Built stop-fix-update.html')


if __name__=='__main__':main()
