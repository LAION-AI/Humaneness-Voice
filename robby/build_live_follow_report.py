"""Current bugfix evidence. Synthetic data only; retain failed controls."""
import datetime,html,json
from pathlib import Path
from final_verify import runtime_signature

ROOT=Path(__file__).parent
LIVE=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')
def read(p):return json.loads(p.read_text()) if p.exists() else {}
def table(headers,rows):
    esc=lambda v:html.escape(str(v))
    return '<div class="scroll"><table><tr>'+''.join('<th>'+esc(h)+'</th>' for h in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+esc(c)+'</td>' for c in row)+'</tr>' for row in rows)+'</table></div>'
def main():
    signature=runtime_signature();visual=read(LIVE/'proactive-visual-live.json');audio=read(LIVE/'proactive-live.json')
    follow=read(ROOT/'logs/follow-pipeline-checks.json');unit=read(ROOT/'logs/unit-checks.json')
    semantic=read(ROOT/'logs/semantic-decision-checks.json');gaze=read(ROOT/'logs/gaze-rendering-checks.json');mic=read(ROOT/'logs/browser-capture-desk.json');stop=read(LIVE/'stop-commands-live.json')
    current=lambda r:r.get('runtime_source_sha256')==signature
    out='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby: live decisions and eye-contact repair</title><style>body{font:16px/1.55 system-ui;max-width:1100px;margin:auto;padding:24px;background:#faf9f5;color:#28363c}td,th{border:1px solid #ccc;padding:9px;text-align:left}table{border-collapse:collapse;width:100%}.scroll{overflow:auto}code{overflow-wrap:anywhere}section{margin:24px 0}a{color:#276d79}</style><nav><a href="/desk-robot/">Robby</a> · <a href="/static/worklog.html">Task log</a> · <a href="/downloads/robby-source.zip">Source</a></nav><h1>Live decisions and eye-contact repair</h1>'''
    out+='<p>Updated '+datetime.datetime.now(datetime.timezone.utc).isoformat()+'. No private microphone recordings or user face images are published. Synthetic live tests are not a guarantee of real-room recognition.</p>'
    rows=[('Unit contracts',unit.get('passed'),str(unit.get('tests'))+' tests'),('Actual CPU camera → landmarks → selection → 3D head',follow.get('passed'),'Public NASA image on a moving canvas camera; four directions, stale capture clock, frozen-calibration recovery'),('Keyboard calibration/rendering',gaze.get('passed'),'Synthetic positions; physical perceptual gaze accuracy not measured'),('Browser microphone → ASR → TTS',mic.get('passed'),'Fake 48-kHz input, both speaker-filter settings')]
    for name,r in [('Visual-assisted live interruptions',visual),('Audio-only live interruptions',audio)]:rows.append((name,bool(r.get('passed') and current(r)),str(len(r.get('cases',[])))+' conditions; current source = '+str(current(r))))
    rows.append(('Spoken Stop during actual streamed playback',bool(stop.get('passed') and current(stop)),str(len(stop.get('cases',[])))+' conditions; current source = '+str(current(stop))))
    out+=table(['Check','Passed','Scope'],rows)
    out+='''<section><h2>Root causes and repairs</h2><ul>
<li>The scheduler advertised 320 ms but throttled proactive checks to 640 ms. It now evaluates every 320-ms audio hop. This is a scheduling interval, NOT a promise of a model/API response or audible speech within 320 ms.</li>
<li>A fired rule remained latched until silence. Fresh post-playback mentions can now re-arm it after four seconds, even during continuous speech. Old trigger words in cumulative ASR do not by themselves re-arm a rule.</li>
<li>An unsuccessful stop-identity check could permanently consume an unchanged ASR candidate. It now retries while the audio remains fresh.</li>
<li>The first repeated-topic replay exposed a Laya false positive: ordinary continuing speech canceled the interjection as though it were Stop. Incoming semantic checks now require a stop-related cue; ambiguous requests use Luna, not repeated uncalibrated Laya guesses. This does not authorize anyone without the existing acoustic verification.</li>
<li>Literal rules preempt a pending contextual API request. There is one outstanding decision task and no queue of stale decisions. Unchanged semantic text reuses its decision; speaker/visual admission remains fresh.</li>
<li>A native teacher/echo check previously awaited inside the microphone hop. It is now a single-flight background task with reply-ID and identity-epoch rechecks. The shared GPU lock prioritizes waiting live encoder hops over another ASR decode; an in-flight decode cannot be preempted.</li>
<li>Diarization can split a continuous monologue after an interjection. A bounded, same-person, same-literal-rule tail no longer produces an extra ordinary reply after the pause. Explicit cancellation and new requests are not swallowed.</li>
<li>A stopped camera presentation timestamp could make current landmarks appear stale. Fresh captured frames now get monotonic bounded timestamps when presentation callbacks stop advancing. The display reports actual pan/tilt or manual-override status.</li>
<li><b>Resume eye contact</b> exits frozen calibration, clears manual/tour overrides, recenters the inspection camera and resumes camera tracking. Optional gallery/recognition UI failures no longer prevent single-face gaze updates.</li>
</ul></section>'''
    out+='''<section><h2>Useful interjections versus exact agreements</h2><p>The new default-on, separately switchable useful-interjection option asks Luna to decide from interim words and recent conversation. It may briefly say “Darf ich kurz einhaken?” / “May I jump in briefly?” when invited or when intervention is clearly useful. It does not invent a new long answer. A 12-second cooldown prevents chatter. Ordinary questions, requests to finish, uncertainty and ambiguous identities should be left alone. The main interruption checkbox disables both proactive paths.</p><p>Laya remains available for incoming Stop decisions. Its contextual paired classifier returned near-chance scores on our invitation controls, so it was NOT made more permissive. Contextual decisions use the existing Luna provider instead; exact agreed topic rules use local keywords, without another LLM call. This is slower for open-ended meaning decisions. Below are actual API latencies, separately from audio recognition and TTS.</p>'''
    out+=table(['Text-only control','Passed','Decision','Measured ms'],[(r['case'],r['passed'],r.get('result',{}).get('action'),r.get('ms')) for r in semantic.get('cases',[])])
    out+='<p>A failure stays a failure: garbled echo-prefix Stop and very short physical commands remain limitations if their rows fail. Do not interpret synthetic unit passes as physical-room speaker accuracy.</p></section>'
    out+='<section><h2>Incoming Stop: real audio controls</h2>'+table(['Fixture','Authorized','Passed','Stopped','Clip duration s','Stop latency s','Failure'],[(c.get('command'),c.get('authorized'),c.get('passed'),c.get('stopped',c.get('interrupted')),c.get('command_duration_s'),round(c['interruption_latency_s'],3) if c.get('interruption_latency_s') is not None else '—',c.get('error','—')) for c in stop.get('cases',[])])+'<p>Only the listed fixtures were rechecked; short single-word commands and all real-room voices are not covered by two long-phrase controls. The manual Stop reply button remains available.</p></section>'
    rows=[]
    for route,r in [('Visual-assisted',visual),('Audio only',audio)]:
        for c in r.get('cases',[]):
            events=c.get('events',[]);begin=c.get('user_start_s',0)
            interruptions=[e for e in events if e.get('phase')=='probe' and e.get('type')=='endpoint' and e.get('by')=='agreed_interruption']
            mentions=[e['at_s'] for e in events if e.get('phase')=='probe' and e.get('type')=='asr_preview' and 'banan' in e.get('text','').lower() and e['at_s']>=begin]
            first=min(mentions) if mentions else None;decision=c.get('decision_at_s');sound=c.get('first_new_audio_s')
            rows.append((route,c['condition'],c['passed'],len(interruptions),round(decision-first,3) if first and decision else '—',round(sound-first,3) if first and sound else '—',round(decision-begin,3) if decision else '—',current(r)))
    out+='<section><h2>Real ASR / rule tools / TTS tests</h2><p>Public synthetic speech, actual models and virtual playback with 240-ms buffering / 120-ms delayed echo. Visual-assisted cases inject a known-face/lip-activity signal; they do not prove camera recognition accuracy. Initial ASR can miss the first keyword, so both delay from clip start and from matching ASR are reported. Earlier failed attempts remain archived.</p>'
    out+=table(['Route','Condition','Pass','Interjections','ASR → decision s','ASR → audio s','Clip → decision s','Current source'],rows)+'</section>'
    out+='<section><h2>Camera measurements</h2>'+table(['Camera image x/y','Head pan °','Head tilt °','Gaze active'],[(str(s['position']),round(s['pan'],2),round(s['tilt'],2),s['active']) for s in follow.get('samples',[])])
    out+='<p>Stalled-clock recovery: '+html.escape(str(follow.get('stalled_capture_clock')))+'. Frozen calibration recovery: '+str(follow.get('resume_exits_frozen_calibration'))+'. No new face model was added. The existing CPU MediaPipe worker remains capped at 10 Hz; movement follows eye position, not head-roll imitation. Browser permission and explicit camera opt-out are respected.</p></section>'
    out+='''<section><h2>Try and reproduce</h2><p>Reload Robby. Click <b>Resume eye contact</b> and allow camera access. Introduce yourself, then agree the exact interruption condition. Inspect the installed rule, not only the spoken acknowledgment. For unrestricted usefulness decisions enable the second checkbox. The live status distinguishes words, identity and visible-speech gating.</p><pre>python run_unit_checks.py
uv run --no-project --with playwright --python /path/to/python browser_follow_checks.py
python decision_checks.py
python -m speaker_live.proactive_test --visual --conditions positive repeated unknown disabled unrelated
python -m speaker_live.proactive_test --conditions positive unknown
python build_live_follow_report.py</pre><p>Run live audio checks sequentially: one ASR session slot. Changed runtime files: live_decisions.py, interaction_policy.py, server.py, speaker_session.py, gpu_guard.py, laya_policy.py, static/ui.js and static/index.html. Tests, README and report builders are included in the source ZIP. No model weights were changed.</p>'''
    out+='<p>Runtime SHA-256: <code>'+signature+'</code></p></section></html>'
    (ROOT/'static/live-follow-update.html').write_text(out)
    print(json.dumps({'report':'static/live-follow-update.html','runtime_source_sha256':signature,'visual_current':bool(visual.get('passed') and current(visual)),'audio_current':bool(audio.get('passed') and current(audio))}))
if __name__=='__main__':main()
