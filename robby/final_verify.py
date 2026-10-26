"""Sequential, fail-closed final QA. One live slot; never stop other sessions.

Uses only public synthetic fixtures/fake browser devices. Does not start/stop
services, publish files remotely, or change model weights.
"""
import argparse,hashlib,json,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def runtime_signature():
    # Visual gate is executable server logic, not only a static UI feature.
    files=[ROOT/n for n in ('server.py','speaker_session.py','speaker_turns.py','speaker_runtime.py','speaker_teacher.py',
        'speaker_names.py','identity.py','audio_sync.py','feature_tap.py','paths.py','tts_bridge.py','shared_turn.py',
        'asr_priority.py','gpu_guard.py','enrollment_cues.py','recovery_rules.py','stop_commands.py','interaction_policy.py','live_decisions.py','laya_policy.py','laya_decision_server.py','conversation_prompt.py','conversation_director.py','performance.py','diarization_client.py')]
    files.append(ROOT/'visual_identity.py')
    files.extend(ROOT/n for n in ('conversation_focus.py','sentence_session.py','llm_backends.py','llm_session.py','credential_store.py'))
    files.extend([ROOT/'open_speakers.py',ROOT/'knowledge_tools.py'])
    files.append(ROOT/'stop_feedback.py')
    files.append(ROOT/'stop_voice.py')
    files.append(ROOT/'avatar_actions.py')
    shared=ROOT.parent/'humaneness-small'
    if not shared.exists():shared=ROOT.parent/'playground/runtime'
    for name in ('small_app.py','turn_cancellation.py'):
        if (shared/name).exists():files.append(shared/name)
    for pattern in ('*.js','*.css','emotion-catalog.json','index.html','robby.svg'):files+=sorted((ROOT/'static').glob(pattern))
    config=ROOT/'config.local.json'
    values=json.loads(config.read_text()) if config.exists() else {}
    decision_settings=json.dumps({k:values.get(k) for k in ('ROBO_DECISION_BASE','ROBO_DECISION_MODEL','ROBO_DIRECTOR_MODE','ROBO_DIRECTOR_MODEL','ROBO_DIRECTOR_BASE')},sort_keys=True).encode()
    return hashlib.sha256(b''.join(p.read_bytes() for p in files)+decision_settings).hexdigest()

def main(url,max_repeats=0,continue_on_failure=False):
    py=sys.executable
    browser=['uv','run','--no-project','--with','playwright','--python',py]
    stages=[('unit',[py,'run_unit_checks.py'],60),
        ('fixture_asr',[py,'-m','speaker_live.fixture_qa'],120),
        ('three_people',[py,'-m','speaker_live.multi_test','--extended','--background-shift','0','--max-repeats',str(max_repeats),'--label','latency-after'],360),
        ('three_people_shifted',[py,'-m','speaker_live.multi_test','--extended','--background-shift','2.3','--max-repeats',str(max_repeats)],360),
        ('streamed_echo',[py,'-m','speaker_live.echo_test'],180),
        ('browser_original',browser+['browser_mic_checks.py','--url',url],180),
        ('browser_desk',browser+['browser_mic_checks.py','--url',url.rstrip('/')+'/desk-robot/'],180),
        ('browser_background_pause',['uv','run','--no-project','--with','playwright','--with','numpy==1.26.4','--with','scipy','--with','soundfile','--with','aiohttp','--python',py,
            'browser_pause_checks.py','--url',url.rstrip('/')+'/desk-robot/','--modes','v4-long3','v4-long3'],180),
        ('microphone_lifecycle',browser+['browser_lifecycle_checks.py'],60),
        ('ui_turn_status',browser+['browser_ui_status_checks.py','--url',url.rstrip('/')+'/desk-robot/'],90),
        ('desk_cpu',browser+['desk_robot_checks.py','--url',url.rstrip('/')+'/desk-robot/'],90),
        ('clock_camera',['uv','run','--no-project','--with','playwright','--with','numpy==1.26.4','--python',py,
            'browser_checks.py','--url',url.rstrip('/')+'/desk-robot/','--camera'],90),
        ('native_features',[py,'-m','speaker_v4.preflight'],90),
        ('selective_dataset_safety',[py,'-m','pytest','-q','speaker_v2/test_dataset_pipeline.py'],60)]
    source_hash=runtime_signature()
    report={'started':time.time(),'synthetic_only':True,'physical_microphone_used':False,
        'runtime_source_sha256':source_hash,'url':url,'max_explicit_repeats_per_turn':max_repeats,
        'first_pass_guaranteed':False,'stages':[],'passed':False}
    report['continue_on_failure']=continue_on_failure
    target=ROOT/'logs/final-verification.json'
    def save():target.write_text(json.dumps(report,indent=2))
    save()
    for name,command,timeout in stages:
        print('START '+name,flush=True);start=time.monotonic()
        record={'name':name,'passed':False,'started':time.time()};report['stages'].append(record);save()
        with (ROOT/'logs'/('verify-'+name+'.log')).open('w') as output:
            try:
                result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
                record.update(exit_code=result.returncode,passed=result.returncode==0)
            except subprocess.TimeoutExpired:record.update(exit_code=None,timed_out=True)
        record['seconds']=time.monotonic()-start;save()
        print(json.dumps(record),flush=True)
        if not record['passed'] and not continue_on_failure:raise SystemExit('Final QA stopped at '+name+'; see its synthetic-only test log')
        if name=='three_people' and record['passed']:
            # Keep both independently timed conversation receipts, not only
            # the shifted-background rerun that uses the same output filename.
            import shutil
            source=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/three-user-live.json')
            shutil.copyfile(source,source.with_name('three-user-unshifted.json'))
    final_hash=runtime_signature()
    report.update(finished=time.time(),passed=source_hash==final_hash and all(s['passed'] for s in report['stages']),source_unchanged=source_hash==final_hash);save()
    if not report['passed']:raise SystemExit('Final QA has failed stages or changed runtime source; inspect the receipt')
    print('All final checks passed on unchanged runtime source.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/')
    p.add_argument('--max-repeats',type=int,choices=(0,1),default=0,help='0: strict first pass; 1: explicitly reported bounded-recovery test')
    p.add_argument('--continue-on-failure',action='store_true',help='Collect all checks; any failed stage still makes the final result fail')
    args=p.parse_args();main(args.url,args.max_repeats,args.continue_on_failure)
