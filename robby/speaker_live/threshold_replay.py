"""Re-score existing public synthetic receipts; no new audio or private data.

This is a cosine/margin operating-point check, not an end-to-end replay or a
human false-acceptance estimate. Other production gates remain independent.
"""
import hashlib,html,json,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent.parent

def main():
    paths=[ROOT/'logs'/n for n in ('speaker-context-lengths.json','background-gallery-checks.json')]
    context,background=[json.loads(p.read_text()) for p in paths]
    rows=[]
    for seconds in (1.,2.,3.04,6.):
        cases=next(c['rows'] for c in context['results'] if c['max_context_s']==seconds)
        for threshold in (.32,.35,.40,.45,.50,.55,.60):
            correct=wrong=0
            for case in cases:
                scores=sorted(case['scores'].items(),key=lambda item:item[1],reverse=True)
                accepted=scores[0][1]>=threshold and scores[0][1]-scores[1][1]>=.08
                correct+=bool(accepted and scores[0][0]==case['target'])
                wrong+=bool(accepted and scores[0][0]!=case['target'])
            admitted_background=0
            for case in background['rows']:
                scores=sorted(case['scores'].values(),reverse=True)
                admitted_background+=bool(scores[0]>=threshold and scores[0]-scores[1]>=.08)
            rows.append({'context_s':seconds,'threshold':threshold,'correct':correct,'wrong':wrong,
                'rejected':len(cases)-correct-wrong,'cases':len(cases),
                'background_admitted':admitted_background,'background_cases':len(background['rows'])})
    report={'finished':time.time(),'public_synthetic_only':True,'real_person_calibration':False,
        'method':'Re-score stored ECAPA cosines with fixed 0.08 rival margin. No ASR, fast-cue, view-agreement, energy, playback or timing gates are simulated.',
        'source_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'rows':rows,
        'limitations':'Three development synthetic identities, repeated prompts, correlated noise conditions and one unrelated background identity. Genuine fixed-length references and multi-duration background galleries differ. Not held-out real-human validation or an end-to-end success rate.'}
    (ROOT/'logs/threshold-replay.json').write_text(json.dumps(report,indent=2))
    headers=['Context (s)','Cosine threshold','Correct / cases','Wrong','Rejected','Background admitted / probes']
    cells=[[r['context_s'],r['threshold'],f"{r['correct']} / {r['cases']}",r['wrong'],r['rejected'],f"{r['background_admitted']} / {r['background_cases']}"] for r in rows]
    table='<table><tr>'+''.join('<th>'+x+'</th>' for x in headers)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in row)+'</tr>' for row in cells)+'</table>'
    page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby voice-match sensitivity</title><style>body{font:16px/1.5 system-ui;max-width:1050px;margin:30px auto;padding:15px}table{border-collapse:collapse}td,th{border:1px solid #ddd;padding:8px}pre{white-space:pre-wrap}</style><a href="/desk-robot/">Open Robby</a> · <a href="/static/worklog.html">Current worklog and defaults</a><h1>Voice-match threshold replay</h1><p>The current user-requested normal-turn default is 0.32, previously 0.45. This permissive setting is not human-calibrated and can accept unrelated voices. The visible slider supports 0.30–0.90; Reset adopts 0.32 while custom saved settings are preserved. Fast-cue strictness is separate.</p><p>The configured normal cosine also applies when a known person gets a new diarization channel; that path still requires a 0.12 rival gap and an independently agreeing strong fast cue below 0.50. Ordinary turns retain the 0.08 gap. View consistency, transcript provenance and playback-echo rejection remain enabled. Voice interruptions have additional current-addressee checks; their short-command experiments are documented separately in the worklog.</p><p>The existing default combines streaming Nemotron diarization, Nemotron-feature speaker prediction and full-sentence ECAPA references. Diarization channel numbers are session-local, not persistent identities. Voiceprints stay in RAM. The table below re-scores earlier development cosines only: it is NOT a live success rate for the current source.</p>'''+table+'<h2>Method and limitations</h2><p>'+html.escape(report['method'])+'</p><p>'+html.escape(report['limitations'])+'</p><details><summary>Auditable synthetic-only receipt</summary><pre>'+html.escape(json.dumps(report,indent=2))+'</pre></details></html>'
    receipt_paths={
        'unit':ROOT/'logs/unit-checks.json',
        'ui':ROOT/'logs/browser-ui-status.json',
        'desk':ROOT/'logs/desk-robot-checks.json',
        'microphone_controls':ROOT/'logs/browser-capture-desk.json',
        'default_pause':ROOT/'logs/browser-pause-sensitivity-045-corroborated.json',
        'echo':Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/streamed-echo-live.json'),
        'multi_speaker':Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/sensitivity-corroborated-045.json')}
    checks={name:json.loads(path.read_text()) for name,path in receipt_paths.items() if path.exists()}
    summary=[]
    if 'unit' in checks:
        c=checks['unit'];summary.append(['Offline contracts',c['passed'],f"{c['tests']} unit tests"])
    if 'ui' in checks:
        c=checks['ui'];summary.append(['Browser settings/status',c['passed'],'Visible slider, reload persistence, reset and score display'])
    if 'microphone_controls' in checks:
        c=checks['microphone_controls'];summary.append(['Actual browser microphone path and settings',c.get('passed',False),'Synthetic fake-device getUserMedia → ASR → Luna → TTS; backend acknowledgments for both sliders and model switching'])
    if 'desk' in checks:
        c=checks['desk'];summary.append(['Software 3D',c['passed'],f"Initial gaze error {c.get('initial_eye_contact_error_degrees',float('nan')):.3f}°, 40 distinct expressions; reset/front/Home, mechanical limits and mobile layout"])
    if 'default_pause' in checks:
        c=checks['default_pause'];summary.append(['Returning speaker with continuous background',c['passed'],str([{'submit_s':r.get('pause_to_accepted_turn_s'),'first_audio_s':r.get('pause_to_first_audio_s'),'wer':r.get('question_wer')} for r in c.get('cases',[])])])
    if 'echo' in checks:
        c=checks['echo'];summary.append(['Streamed playback echo and interruption',c.get('passed',False),f"False echo barges: {c.get('echo_only_false_barges')}; real interruption {c.get('interruption_latency_s')} s"])
    if 'multi_speaker' in checks:
        c=checks['multi_speaker'];summary.append(['Three-speaker live replay',c.get('passed',False),f"{len(c.get('replies',[]))} replies, {len(c.get('explicit_repeats',[]))} explicit repeats; max one repeat permitted per request"])
    live_table='<h2>Targeted implementation checks</h2><p>These checks use public synthetic audio, fake browser microphone devices and simulated echo, not a physical microphone or held-out human voices. They are targeted regressions, not a rerun of every historical integration study.</p><table><tr><th>Check</th><th>Passed</th><th>Scope / observations</th></tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(c))+'</td>' for c in r)+'</tr>' for r in summary)+'</table>'
    live_table+='<p>Historical 0.45 development run: an unconditionally relaxed new-channel recovery floor failed, binding a weak background candidate to the known person. That failed receipt is retained as browser-pause-sensitivity-default-045.json. The corrected path requires independent strong fast corroboration below 0.50. These historical live receipts do not validate the current 0.32 setting; see the worklog for current-source checks.</p>'
    from final_verify import runtime_signature
    verification={'generated':time.time(),'runtime_source_sha256':runtime_signature(),
        'physical_microphone_used':False,'scope':'Targeted sensitivity, browser, 3D and live synthetic regressions; not full historical sweep',
        'checks':summary,'all_selected_checks_passed':len(checks)==len(receipt_paths) and all(row[1] for row in summary),
        'receipt_sha256':{name:hashlib.sha256(path.read_bytes()).hexdigest() for name,path in receipt_paths.items() if path.exists()},
        'known_failed_exploration':'browser-pause-sensitivity-default-045.json; unconditional new-channel floor relaxation; corrected with independent fast corroboration'}
    (ROOT/'logs/sensitivity-verification.json').write_text(json.dumps(verification,indent=2))
    live_table+='<details><summary>Targeted verification receipt</summary><pre>'+html.escape(json.dumps(verification,indent=2))+'</pre></details>'
    page=page.replace('<h2>Method and limitations</h2>',live_table+'<h2>Method and limitations</h2>')
    (ROOT/'static/threshold-replay.html').write_text(page)
    print(json.dumps([r for r in rows if r['context_s'] in (2.,3.04)]))

if __name__=='__main__':main()
