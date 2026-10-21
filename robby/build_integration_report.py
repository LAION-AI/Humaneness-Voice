"""Build the public English overview from explicit synthetic QA receipts only.

Never reads arbitrary logs, microphone recordings or session voiceprints.
Missing/failed checks remain visible, not silently converted to success.
"""
import base64,datetime,hashlib,html,json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
LIVE=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results')

def main():
    receipts={}
    def read(key,path):
        if not path.exists():receipts[key]={'available':False};return {}
        data=json.loads(path.read_text());receipts[key]={'available':True,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'modified_utc':datetime.datetime.fromtimestamp(path.stat().st_mtime,datetime.timezone.utc).isoformat()}
        return data
    unit=read('unit',ROOT/'logs/unit-checks.json')
    lifecycle=read('lifecycle',ROOT/'logs/browser-lifecycle-checks.json')
    clock=read('clock',ROOT/'logs/browser-checks.json')
    desk=read('desk',ROOT/'logs/desk-robot-checks.json')
    fixture=read('fixture',LIVE/'fixture-clean-qa.json')
    multi=read('three_people',LIVE/'three-user-live.json')
    multi_first=read('three_people_unshifted',LIVE/'three-user-unshifted.json')
    capture=read('browser_capture',ROOT/'logs/browser-capture-checks.json')
    capture_desk=read('browser_capture_desk',ROOT/'logs/browser-capture-desk.json')
    capture_original=read('browser_capture_original',ROOT/'logs/browser-capture-original.json')
    echo=read('streamed_echo',LIVE/'streamed-echo-live.json')
    sweep=read('frozen_source_sweep',ROOT/'logs/final-verification.json')
    from final_verify import runtime_signature
    sweep['passed']=bool(sweep.get('passed') and sweep.get('runtime_source_sha256')==runtime_signature())
    incident=read('host_gpu_incident',ROOT/'logs/host-gpu-incident.json')
    pause=read('speaker_pause_browser',ROOT/'logs/browser-pause-checks.json')
    prompt=read('compact_prompt',ROOT/'logs/prompt-quality-checks.json')
    ui_status=read('ui_pending_turn',ROOT/'logs/browser-ui-status.json')
    valid_multi=multi.get('protocol_version',0)>=3 and multi.get('passed',False)
    capture_rows=capture.get('results',[])
    valid_capture=all(d.get('passed') and len(d.get('results',[]))==2 for d in (capture_desk,capture_original))
    recovery=multi.get('protocol_version',0)>=4
    repeats=multi.get('explicit_repeats',[])
    multi_detail=('Protocol 4, bounded recovery: ' if recovery else 'Strict protocol 3, no repeats: ')+f'real-time DAC tags, three enrollments/returns, six named replies, continuous unrelated speech, nonverbal negative and voice interruption; {len(repeats)} explicit repeat(s) in shifted scenario'
    checks=[('Pending answer UI status',bool(ui_status.get('passed')),'Actual HTTPS DOM: background rejection/ASR preview cannot overwrite an accepted pending answer; interruption and error restore normal feedback'),
        ('One-second pause under background speech',bool(pause.get('passed')),'Repeated default-context HTTPS Playwright capture, 800-ms mid-sentence hesitation, correct IDs, actual TTS and no background duplicates; separate context comparison and current latency report linked below'),
        ('Compact response instructions',bool(prompt.get('passed')),'Seven explicit real-Luna regression checks: arithmetic, constraints, story, follow-up, verbatim, current identity and unsupported device actions'),
        ('Frozen-source final sweep',bool(sweep.get('passed')),'Sequential end-to-end checks, two background timings, both HTTPS avatars, UI settings, camera/clock, native feature parity and selective-dataset safety; unchanged runtime-source hash; repeat policy recorded separately'),
        ('Unit/regression checks',bool(unit.get('passed')),f'{unit.get("tests",0)} tests, {unit.get("failures",0)} failures, {unit.get("errors",0)} errors'),
        ('Microphone lifecycle',lifecycle.get('passed')==7,'Seven real-browser lifecycle cases, including suspended AudioContext, permission/worklet cancellation and stale socket cleanup'),
        ('HTTPS voice-to-voice',valid_capture,'Browser fake getUserMedia → actual 48→16 kHz worklet → WebSocket → ASR → Luna → actual TTS reply chunks; filtered and explicit unfiltered modes'),
        ('Three named voices',bool(valid_multi and multi_first.get('passed')),multi_detail),
        ('Streamed reply echo + interruption',bool(echo.get('passed')),'Actual streamed TTS is played on a simulated DAC, leaks into input with 120-ms delay and playback tags; no false echo barge, then an enrolled user interrupts'),
        ('Playback clock',clock.get('scenarios')==60 and not clock.get('page_errors'),'60 ideal-clock latency scenarios; real-device latency remains unmeasured'),
        ('CPU 3D robot',bool(desk.get('passed')),f'{desk.get("unique_expression_renders",0)} distinct expression renders; {desk.get("observed_render_fps",0):.1f} fps; render p95 {desk.get("render_p95_ms",0):.1f} ms; GPU disabled')]
    report={'generated_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'synthetic_only':True,
        'physical_microphone_or_room_test':False,'checks':[{'name':name,'passed':passed,'detail':detail} for name,passed,detail in checks],
        'all_selected_checks_passed':all(p for _,p,_ in checks),'receipts':receipts,
        'limitations':['Not authentication','No reliable source separation of simultaneous speakers','One live browser session','No physical device synchronization measurement','Synthetic repeated-identity benchmark, not real-human validation']}
    (ROOT/'logs/integration-results.json').write_text(json.dumps(report,indent=2))
    e=lambda x:html.escape(str(x))
    rows=''.join(f'<tr><td>{e(n)}</td><td class="{"pass" if p else "pending"}">{"PASS" if p else "NOT VERIFIED"}</td><td>{e(d)}</td></tr>' for n,p,d in checks)
    turns=''
    if valid_multi:
        turns='<h2>Observed conversation</h2><div class="scroll"><table><tr><th>Time (s)</th><th>Verified name</th><th>Accepted transcript</th></tr>'+''.join(
            f'<tr><td>{e(t.get("at_s"))}</td><td>{e(t.get("speaker_name"))}</td><td>{e(t.get("transcript"))}</td></tr>' for t in multi['endpoints'])+'</table></div>'
        # The policy's one-second clock is not the measured final submission
        # latency: native framing, diarization and ASR also take time.
        boundaries=[];seen=set()
        for b in multi.get('fixture_boundaries',[]):
            if 'nonverbal' in b['fixture'] or b['fixture'] in seen:continue
            seen.add(b['fixture']);boundaries.append(b)
        boundaries=boundaries[:6]
        latencies=[round(t['at_s']-b['end_s'],3) for t,b in zip(multi['endpoints'][:6],boundaries)]
        if len(latencies)==6 and min(latencies)>=0:
            import statistics
            report['measured_synthetic_endpoint_latency_s']={'samples':latencies,'median':statistics.median(latencies),
                'min':min(latencies),'max':max(latencies),'from':'End of public fixture to accepted transcript; not physical hardware'}
            turns+=f'<p>Measured final transcript submission after the six initial fixture endings: median {statistics.median(latencies):.2f} s, range {min(latencies):.2f}–{max(latencies):.2f} s, including any explicitly recorded repetition. The one-second silence policy is only part of this delay; diarizer framing and ASR add latency. This is synthetic real-time pacing, not a physical microphone measurement.</p>'
            (ROOT/'logs/integration-results.json').write_text(json.dumps(report,indent=2))
        if recovery:
            report['conversation_first_attempts']={'target_turns':12,
                'accepted_without_repeat':12-len(multi_first.get('explicit_repeats',[]))-len(repeats),
                'unshifted_repeats':multi_first.get('explicit_repeats',[]),'shifted_repeats':repeats,
                'scope':'Two controlled synthetic conversations, not a statistical real-user accuracy estimate'}
            turns+='<h3>First attempts versus recovery (not a perfect-recognition claim)</h3><p class="notice">Protocol 4 permits one explicit repetition of the same utterance after nine seconds without acceptance. An already accepted request is never repeated just because TTS is queued. Speaker IDs, names, content, background rejection and real voice interruption remain strict. This tests safe conversational recovery, not 100% first-pass recognition.</p><table><tr><th>Background timing</th><th>First-attempt turns / 6</th><th>Explicit repeats</th></tr>'
            for label,d in [('Unshifted',multi_first),('Shifted by 2.3 s',multi)]:
                r=d.get('explicit_repeats',[]);turns+=f'<tr><td>{e(label)}</td><td>{6-len(r)} / 6</td><td>{len(r)}</td></tr>'
            turns+='</table><pre>'+e(json.dumps({'unshifted_repeats':multi_first.get('explicit_repeats',[]),'shifted_repeats':repeats},indent=2))+'</pre>'
    image='';picture=ROOT/'logs/desk-robot-40-expressions.png'
    if picture.exists():image='<h2>Forty expressions on the same 3D head</h2><img alt="Forty authored robot expressions" src="data:image/png;base64,'+base64.b64encode(picture.read_bytes()).decode()+'">'
    content='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Robby — integration, verification and limitations</title><style>body{font:16px/1.65 system-ui;max-width:1100px;margin:35px auto;padding:0 20px;background:#f8f6f0;color:#35332f}a{color:#615238}table{border-collapse:collapse;width:100%;font-size:14px}td,th{padding:10px;border-bottom:1px solid #d9d1c5;text-align:left;vertical-align:top}.notice{background:#eee6d8;padding:18px;border-radius:12px}.pass{color:#247344}.pending{color:#9b4500}.scroll{overflow:auto}img{max-width:100%;height:auto}code{background:#eee9df;padding:2px 5px}pre{white-space:pre-wrap;font-size:13px}li{margin-bottom:8px}</style>
<nav><a href="/">Original avatar</a> · <a href="/desk-robot/">3D desk robot</a> · <a href="https://huggingface.co/laion/tyger-speaker-predictor">Permanent source and model snapshot</a></nav><h1>Robby: voice, speaker memory and a two-axis desk robot</h1>'''
    if not report['all_selected_checks_passed']:content+='<p class="notice"><strong>Current revision is still under validation.</strong> There is no complete unchanged-runtime-source pass yet. Individual receipts below may belong to earlier exploratory runs; do not treat them as a completed current release.</p>'
    content+=f'<p>Updated {e(report["generated_utc"])}. Tests below use public synthetic speech and actual models/APIs where specified, never private microphone recordings.</p><div class="scroll"><table><tr><th>Check</th><th>Result</th><th>What was actually exercised</th></tr>{rows}</table></div>'
    content+='''<h2>How to use it</h2><p>Reload an old tab (Ctrl+Shift+R), press <b>Start talking</b> and grant microphone access. Say “Hey Robbie, ich bin Christoph” or “Hey Robby, I am Christoph”, preferably as one complete sentence. Each new person introduces themselves the same way. Subsequent speech is matched against session references and the answer addresses the current speaker by name. After about one second without that person's voice, their turn closes even if the quieter background keeps talking. Wake-only enrollment gets a short 2.5-second grace for a following introduction. <b>Talk without speaker filter</b> explicitly bypasses identity gating; typed messages also bypass enrollment.</p><p>The microphone panel distinguishes local capture, server receipt, ASR preview and accepted turns. A visible preview is deliberately not proof of submission: ambiguous or unrelated speech must not trigger a reply. Clear or disconnect forgets enrolled voices. One live browser session is supported; several people can share it.</p>
<h2>What was fixed</h2><ul><li>Existing suspended browser audio contexts resume when microphone capture begins. Cancelled startup, late permissions and stale socket events cannot leave a microphone attached or stop a newer session.</li><li>Speaker-owned one-second clocks replace room-wide silence for the filtered mode. Silence does not refresh a person's clock merely because a diarizer channel remains bound. Diarizer channels are reusable activity tracks, never permanent identity labels.</li><li>Chronological pre-roll preserves clipped wake words. Independently cropped owned-word ASR must agree with raw sentence views; a nonverbal foreground cannot borrow background-TV words. Teacher embeddings use full and ASR-trimmed acoustic views, not an accidentally shortened reference. Async identity corrections rebind open continuations atomically.</li><li>Only native TYGER/second-ear interruption proposals stop playback; enrollment gates them and can replay a recent proposal after speaker verification. No speaker-activity-only bypass can turn bot echo into an interruption.</li><li>DAC position, capture timestamps, playback gaps/cancellation and explicit latency estimates drive the second-ear reference. Software clock results are not a claim of zero physical-device error.</li></ul>
<h2>Speaker models and completed research</h2><p>The frozen Nemotron ASR encoder supplies live hidden states, so speaker prediction does not run a second encoder. The fast V4 Query-GRU mixes layers 4/8/12/15 (one-based), has 648,071 trainable parameters and pools four 80-ms frames. Its matched synthetic test identification/EER is 71.07%/14.28%, versus 67.38%/15.48% for the baseline. The completed V5 duration study ran 45 jobs; its selected utterance-target recipe reaches 70.63%/14.25% at 2,000 ms and 71.05%/14.18% at 3,040 ms. The three-second identification advantage is only +0.43 pp (identity-bootstrap interval −0.28 to +1.07 pp). Live fast cues use V4; longer references default to completed V5 two-second features, with the three-second option available.</p><p>Both studies use 480/10/10 synthetic voice identities with no split overlap, but reuse earlier benchmark test identities. The 320-ms figure is a pooling/output interval with causal encoder history, not a cold-start 320-ms-only guarantee. Full-utterance frozen ECAPA verification complements, rather than magically fixes, the student. Default full-utterance cosine is 0.50 with nearest-reference margin 0.08; conservative bounded continuity relaxation is documented in the README. These are prototype operating points, not security calibration.</p><p><a href="/speaker-training/320ms-study/">V4 layers and hard-pair ablations</a> · <a href="/speaker-training/context-study/">Completed V5 context study</a> · <a href="/speaker-training/report.html">Larger-corpus baseline</a> · <a href="/speaker-study/">Historical small pilot (failed reliability gate)</a></p>
<h2>3D design and animation</h2><p>The browser prototype follows the supplied desk-robot proposal: 190-mm white head, 150-mm round display bezel with 127.6-mm active area, top camera opening, side caps, pan/tilt neck and 220×70-mm base; nominal height 315 mm. There are no arms, chest heart or cross. Original lightweight procedural surfaces replace a supplied 1.89-million-triangle mesh. Canvas 2D projects/depth-sorts 2,788 quads, without WebGL, CDN or mesh download. Pan is limited to ±90°, tilt to ±15°, and roll is disabled. The head has optional local face-following; orbit, manual controls and an exploded inspection view are available.</p><p>Both avatars reuse the same Luna/ASR/TTS pipeline and 40 authored emotion poses. Luna's explicit sentence-level TTS acting instructions drive the face; microphone volume does not classify emotion. Actual playback positions anchor the animation, but sentence boundaries remain prompt-derived estimates rather than forced alignment. Fake-camera startup was tested; real gaze-tracking accuracy was not measured. This is a visual prototype, not manufacturing CAD, collision checking or servo control.</p>'''
    content+='<h2>Interruption operating point and latency</h2><p>The original frozen native GRU remains the source of interruption proposals. Filtered mode uses its supported sensitive setting (yield threshold 0.20, one-frame debounce), but every proposal must pass full ECAPA matching against enrolled users and the actual streamed bot voice. Default cosine is 0.50 with margin 0.08 against other users and bot playback. Strong native proposals (yield ≥0.8) can retain a recent, independently fast-matched known track using the bounded continuity floor max(0.40, threshold −0.08) and margin ≥0.12. There is no enrollment or speaker-activity-only bypass. Unfiltered mode restores the original native settings. The actual streamed-echo test initially lost an Idris interruption; the more sensitive, acoustically guarded operating point corrected it. Its first successful interruption took about 2.8 seconds. A 320-ms prediction interval is not a promise of 320-ms identity-confirmed interruption latency.</p>'
    content+='<p>A medium short-window speaker cosine alone is insufficient: ordinary confirmation also requires recent independently diarized ownership. A previously unseen channel needs a substantially stronger known-reference match (cosine ≥0.70, margin ≥0.20). This guards background male voices that can otherwise obtain misleading short ECAPA scores. No new identity is enrolled by this exception.</p><h2>Sentence boundaries and reference stability</h2><p>Up to 800 ms of already-received post-roll preserves quiet word endings without adding a second endpoint timer. Identity enrollment excludes this trailing room audio. Independent owned/raw recognition rejects ambiguous text; a multiword TV suffix after an independently recognized sentence is discarded. This remains a conservative heuristic, not true source separation.</p><p>Timestamped, same-name wake evidence can recover a greeting clipped by the diarizer. Incremental name guesses are provisional and can be corrected by the completed introduction without orphan profiles; the UI says joining… until a full reference exists. Two sufficiently long, acoustically agreeing and separated follow-up embeddings may form a small bounded centroid, weighted at half the original introduction anchor. The anchor is immutable, updates cannot recursively drift it, and actual playback is also a competitor at full-turn verification. All identity memory remains session RAM.</p>'
    content+=turns+image+'''<h2>Limitations and privacy</h2><p class="notice">Real rooms, loud simultaneous speakers, distant microphones and very short utterances remain uncertain. Diarization is not source separation; the safe response to an unresolved mixture is to request repetition, not confidently invent a speaker. The synthetic test passing does not establish real-person accuracy or secure identification. Test fixtures are synthetic and a malformed Selma fixture was explicitly retained as a nonverbal rejection case rather than counted as an intelligible turn.</p><p>Live audio, speaker references and names are session RAM only. The server sends recognized text to the configured Luna API and uses the existing Playground's temporary synthesized-output storage. No private recordings or enrolled voiceprints are in the source/model release. Camera frames stay in the browser. Public synthetic test traces are opt-in on loopback only, never enabled through Cloudflare. User-supplied design PDFs/images/mesh are not redistributed or relicensed.</p>
<h2>Reproduce and inspect</h2><p><a href="/downloads/robby-source.zip">Audited source ZIP</a> · <a href="https://huggingface.co/laion/tyger-speaker-predictor/tree/main">Published code, small student weights, provenance and reports</a> · <a href="/static/sync-report.html">Earlier synchronization measurements</a>. The README contains pinned upstream revisions, dependencies, architecture, data-generation/training commands and test commands. The snapshot is verified on this running host; it is not a claim of a tested clean-machine installer.</p><pre>python run_unit_checks.py
python -m speaker_live.fixture_qa
python -m speaker_live.multi_test --extended --background-shift 2.3
python -m speaker_live.echo_test
python final_verify.py --url https://YOUR-HOST/
uv run --no-project --with playwright python browser_mic_checks.py --url https://YOUR-HOST/desk-robot/
uv run --no-project --with playwright python desk_robot_checks.py
uv run --no-project --with playwright python browser_lifecycle_checks.py</pre>'''
    content+='<h2>Receipt hashes</h2><pre>'+e(json.dumps(receipts,indent=2))+'</pre></html>'
    content=content.replace('Live audio, speaker references and names are session RAM only.',
        'Raw microphone audio and enrolled speaker references stay in session RAM; names and words also follow the configured Luna API and shared Playground temporary-output policy.')
    content=content.replace('<h2>Limitations and privacy</h2>',
        '<h2>Shared-model admission and cancellation</h2><p>A cancelled shared TTS request may remain busy briefly. The bridge now waits up to 45 seconds with visible waiting/preparing status and supports Stop throughout. Explicit HTTP 409/503 responses are retried only before admission. An accepted stream or ambiguous network failure is never retried, preventing duplicate speech. Other Playground requests are never evicted. Newly confirmed enrolled speech can cancel a pending reply before audio arrives; evidence starts after reply submission, so a preceding utterance cannot cancel its own answer.</p><h2>Limitations and privacy</h2>')
    if incident:
        content=content.replace('<h2>Limitations and privacy</h2>',
            '<h2>Observed host GPU incident</h2><p>A separate failed run accepted six named turns and then lost the native ASR GPU during interruption testing. NVIDIA kernel locks and nvidia-smi were observed stalled while ordinary HTTP health still answered. The five-second microphone-backlog guard closed the call. After the driver responded again, only the owned Robby worker was restarted; no shared GPU reset, host reboot or termination of unrelated jobs was performed. The failed run was not counted as a pass. The final receipt above is a subsequent rerun. Application fixes cannot guarantee availability during a shared-host driver stall; the UI now explains the backlog/reconnect error.</p><h2>Limitations and privacy</h2>')
    content=content.replace('<h2>Sentence boundaries and reference stability</h2>',
        '<p>For close known voices, full/cropped consensus permits a smaller 0.04 nearest-user margin only with cosine ≥max(0.55, configured threshold), ≥0.90 acoustic agreement, ≥1.5 s voiced and ≥0.12 separation from playback. Close-score decisions cannot update the reference. New self-introductions always use the original full identity span, even when the diarizer reused another person’s channel. Shifted acoustic views are compared by their samples, not merely their lengths.</p><h2>Sentence boundaries and reference stability</h2>')
    content=content.replace('<h2>Sentence boundaries and reference stability</h2>',
        '<p>A low-energy background candidate cannot activate using a louder person’s greeting, even if its clipped ASR invents a full introduction. Eligible new foreground tracks may independently re-decode up to 1.6 s of preceding original audio to verify a real same-name greeting across a channel transition. No wake text is fabricated. When independently owned-word recognition verifies a substantial ASR prefix trim and at least 1.28 s remains, identity combines 80% crop and 20% whole-span embeddings; otherwise the weights are equal. Views with cosine below 0.50 are still rejected, and divergent views cannot adapt the original reference. This avoids giving discarded TV pre-roll equal weight without removing the whole-span check.</p><h2>Sentence boundaries and reference stability</h2>')
    content=content.replace('<h2>Sentence boundaries and reference stability</h2>',
        '<p>For a near-tied teacher winner, the separately calibrated Nemotron student can corroborate that same known person: fast centered cosine ≥max(0.70, fast threshold), combined margin ≥0.10, full ECAPA above threshold, ≥0.85 acoustic-view agreement, ≥1.28 s voiced, and ≥0.12 separation from bot playback. The student cannot overrule a different teacher winner or enroll an unknown voice through this path. Close-score corroboration never adapts the reference; disagreement requests repetition.</p><h2>Sentence boundaries and reference stability</h2>')
    content=content.replace('<h2>Sentence boundaries and reference stability</h2>',
        '<p>When bounded prefix crops retain exactly the same independently verified word sequence, the latest such crop avoids unnecessary TV pre-roll. The background-suffix guard tolerates one ASR insertion/deletion in the best matching owned prefix, such as a dropped article, while still permitting one quiet missing final word. A multiword unowned TV suffix cannot bypass the guard merely because one article differs.</p><h2>Sentence boundaries and reference stability</h2>')
    content=content.replace('longer references default to completed V5 two-second features, with the three-second option available.',
        'longer references now default to completed V5 three-second features, with the two-second option available; no extra waiting is added to collect a full window.')
    content=content.replace('<h2>Speaker models and completed research</h2>',
        '<h2>Current latency update</h2><p><a href="/static/turn-latency.html">One-second pauses, before/after latency, actual browser tests and longer speaker-window comparison</a>. The current live path uses one compact Luna call, prepares ASR/identity during the pause and validates name metadata locally. It reuses the same shared TTS model through its literal manual endpoint; no second director runs.</p><h2>Speaker models and completed research</h2>')
    content=content.replace('Views with cosine below 0.50 are still rejected, and divergent views cannot adapt the original reference.',
        'Ordinary views require cosine 0.50. Only an independently proven quiet other-voice prefix allows 0.35, with full retained-speaker matching, no worse owned-word transcript agreement and the original view retained at 20% weight. Divergent views cannot adapt the original reference.')
    content=content.replace('max(0.40, threshold −0.08)','max(0.42, threshold −0.08)')
    content=content.replace('<h2>Current latency update</h2>',
        '<h2>Current latency update</h2><p>Up to three frozen enrollment templates (2, 3.04 and 4 s) capture phonetic variation in the same verified introduction. Scores use the best template per identity; margins remain against other people. The new report includes duration and background-impostor checks and their limited synthetic scope. A high-energy candidate is no longer marked as background merely for missing a wake or having too little embedding context.</p>')
    content=content.replace('<h2>Current latency update</h2>',
        '<h2>Current latency update</h2><p>The latest path uses monotonic native timestamps to bridge delayed diarizer rows and time-aligned native enrollment centroids. Observed background channels establish a relative 2x-amplitude foreground guard, so quiet matching background cannot renew/reassign a user clock. Actual DAC-aligned playback-waveform evidence complements the timbre-based echo competitor; native BARGE proposals and full known-user checks are still mandatory. Shared encoder calls are serialized defensively, and fatal CUDA-worker failures now report not ready rather than appearing microphone-ready. Physical microphone/room validation remains outstanding.</p>')
    (ROOT/'static/report.html').write_text(content)
    (ROOT/'logs/integration-results.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({'report':'static/report.html','all_selected_checks_passed':report['all_selected_checks_passed'],'checks':report['checks']}))

if __name__=='__main__':main()
