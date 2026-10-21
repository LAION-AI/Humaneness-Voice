"""Frozen-source QA for the current UI/eye-contact/interruption task batch.

Public synthetic audio and fake browser devices only. Never restarts services,
disconnects another user, changes weights, or republishes private voiceprints.
Historical research sweeps remain in final_verify.py.
"""
import argparse,json,subprocess,sys,time
from pathlib import Path
import requests
from final_verify import runtime_signature

ROOT=Path(__file__).resolve().parent
STOP=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/stop-commands-live.json')

def main(url,wait_for_stop=False):
    signature=runtime_signature()
    if wait_for_stop:
        print('Waiting for the current eight-case stop replay; no live slot opened.',flush=True)
        deadline=time.monotonic()+600
        while time.monotonic()<deadline:
            stop=json.loads(STOP.read_text())
            if stop.get('finished'):break
            time.sleep(.5)
        else:raise SystemExit('Stop replay did not finish within ten minutes.')
    stop=json.loads(STOP.read_text())
    if not (stop.get('passed') and len(stop.get('cases',[]))>=8 and stop.get('source_unchanged')
            and stop.get('runtime_source_sha256')==signature):
        raise SystemExit('Current-source eight-case stop replay is not passing; final verification not claimed.')
    proactive=json.loads((STOP.parent/'proactive-live.json').read_text())
    if not (proactive.get('passed') and len(proactive.get('cases',[]))>=4 and proactive.get('runtime_source_sha256')==signature):
        raise SystemExit('Current-source four-case proactive replay is not passing; final verification not claimed.')
    py=sys.executable;browser=['uv','run','--no-project','--with','playwright','--python',py]
    stages=[('unit',[py,'run_unit_checks.py'],60),
        ('real_luna_tools',[py,'interaction_checks.py'],60),
        ('semantic_decisions',[py,'decision_checks.py'],60),
        ('fixture_asr',[py,'-m','speaker_live.fixture_qa'],120),
        ('streamed_echo',[py,'-m','speaker_live.echo_test'],180),
        ('browser_desk_microphone',browser+['browser_mic_checks.py','--url',url],180),
        ('browser_background_pause',['uv','run','--no-project','--with','playwright','--with','numpy==1.26.4',
            '--with','scipy','--with','soundfile','--with','aiohttp','--python',py,
            'browser_pause_checks.py','--url',url,'--modes','v4-long3','v4-long3'],180),
        ('microphone_lifecycle',browser+['browser_lifecycle_checks.py'],60),
        ('ui_status',browser+['browser_ui_status_checks.py','--url',url],60),
        ('desk_cpu',browser+['desk_robot_checks.py','--url',url],90),
        ('gaze_camera_tutorial',browser+['browser_gaze_checks.py','--url',url],90),
        ('real_camera_and_clock',['uv','run','--no-project','--with','playwright','--with','numpy==1.26.4','--python',py,
            'browser_checks.py','--url',url,'--camera'],90),
        ('luna_instructions',[py,'prompt_quality_checks.py'],120)]
    target=ROOT/'logs/active-todos-verification.json'
    if target.exists():
        previous=json.loads(target.read_text())
        (ROOT/'logs'/f'active-todos-attempt-{int(time.time())}.json').write_text(json.dumps(previous,indent=2))
    report={'started':time.time(),'passed':False,'runtime_source_sha256':signature,
        'public_synthetic_only':True,'physical_microphone_or_camera_used':False,
        'scope':'Current two-way conversational interruptions plus prior UI, eye-contact, rendering, naming and microphone regressions; not every historical research experiment',
        'stop_cases':[{k:c.get(k) for k in ('command','authorized','passed','interruption_latency_s','new_spoken_replies')} for c in stop['cases']],
        'stages':[]}
    def save():target.write_text(json.dumps(report,indent=2))
    save()
    for name,command,timeout in stages:
        print('START '+name,flush=True);start=time.monotonic()
        record={'name':name,'passed':False};report['stages'].append(record);save()
        with (ROOT/'logs'/('active-'+name+'.log')).open('w') as output:
            try:
                result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
                record.update(exit_code=result.returncode,passed=result.returncode==0)
            except subprocess.TimeoutExpired:record.update(exit_code=None,timed_out=True)
        record['seconds']=time.monotonic()-start;save();print(json.dumps(record),flush=True)
        if not record['passed']:raise SystemExit('Targeted QA stopped at '+name+'; failure retained.')
        if signature!=runtime_signature():raise SystemExit('Runtime source changed during QA; result invalid.')
    public=requests.get(url,timeout=15);public.raise_for_status()
    state=requests.get('http://127.0.0.1:8793/api/state',timeout=10).json()
    ready=bool(state.get('ready') and state.get('identity_available') and not state.get('audio_worker_error'))
    report.update(finished=time.time(),source_unchanged=signature==runtime_signature(),
        public_http=public.status_code,backend_ready=ready)
    report['passed']=bool(report['source_unchanged'] and ready and all(s['passed'] for s in report['stages']))
    save()
    if not report['passed']:raise SystemExit('Final readiness/source validation failed.')
    print('All current-task checks passed on unchanged runtime source.',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',default='https://diane-asylum-understood-invoice.trycloudflare.com/desk-robot/')
    p.add_argument('--wait-for-stop',action='store_true',help='Wait for the in-progress replay, without rerunning failed cases')
    args=p.parse_args();main(args.url,args.wait_for_stop)
