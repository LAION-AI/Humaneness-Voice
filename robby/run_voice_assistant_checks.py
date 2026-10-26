"""Serial, versioned voice-assistant regression runs. See the test protocol.

Never restarts production or evicts a browser. All output stays in a fresh
logs/voice-assistant-runs directory; no environment or credential dump.
"""
import argparse, ast, datetime, fcntl, hashlib, importlib.metadata, json, os
from pathlib import Path
import platform, shutil, subprocess, sys, time, urllib.request, uuid

ROOT=Path(__file__).resolve().parent
FIXTURES=['christoph-intro','christoph-followup','christoph-stop','christoph-wait',
    'christoph-halt','christoph-pause-clear','christoph-bananas','background',
    'selma-intro','selma-followup-clear','focus-invite-alice',
    'focus-ignore-onlyme','focus-background-invite']

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def source_files():
    return sorted([*ROOT.glob('*.py'),*ROOT.glob('*.md'),ROOT/'voice_assistant_test_matrix.json',
        *[p for p in (ROOT/'static').iterdir() if p.suffix in ('.js','.css','.html')],
        *[p for folder in ('speaker_live','speaker_v2','speaker_v3','speaker_v4','speaker_v5') for p in (ROOT/folder).glob('*.py')]])

def deployed_source_hash():
    tree=ast.parse((ROOT/'server.py').read_text())
    loop=next(n for n in tree.body if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='_name')
    sha=hashlib.sha256()
    for name in ast.literal_eval(loop.iter):sha.update(name.encode());sha.update((ROOT/name).read_bytes())
    return sha.hexdigest()

def public_state(url):
    with urllib.request.urlopen(url+'/api/state',timeout=10) as r:raw=json.load(r)
    # Allowlist, never copy arbitrary service configuration.
    return {k:raw[k] for k in ('ready','phase','source_sha256','speech_pause_default_ms',
        'sentence_stream','decision_interval_ms','identity_models','asr_model_selection','tts','llm_default') if k in raw}

def model_inventory():
    """Read-only local asset pins; no private configuration or recordings."""
    import paths
    from speaker_runtime import registry
    from speaker_teacher import TEACHER_REVISION,resolve_model_root,REQUIRED_FILES
    result={'tyger_source_revision':paths.TYGER_REVISION,'tyger_bundle_revision':paths.BUNDLE_REVISION,
        'ecapa_revision':TEACHER_REVISION,'speaker_students':{}}
    for key,spec in registry().items():
        path=Path(spec['path'])
        result['speaker_students'][key]={'file':path.name,'sha256':digest(path),
            'frames':spec['frames'],'mix_layers':spec['mix_layers'],'label':spec['label']}
    teacher=resolve_model_root()
    result['ecapa_files']={name:digest(teacher/name) for name in REQUIRED_FILES}
    with urllib.request.urlopen('http://127.0.0.1:8792/api/state',timeout=10) as response:tts=json.load(response)
    checkpoint=tts.get('checkpoint') or {}
    result['tts']={k:checkpoint[k] for k in ('id','repo','revision','checkpoint','sha256','adapter_sha256','policy_dtype','compute_precision') if k in checkpoint}
    result['tts_service_defaults']={k:tts[k] for k in ('temperature_default','guidance_default','chunk_frames_default','reference_seconds_max','speaker_default') if k in tts}
    try:
        result['gpu']=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv,noheader'],text=True,timeout=10).strip().splitlines()
    except (OSError,subprocess.SubprocessError):result['gpu']='Unavailable; record manually'
    result['concurrency_note']='Shared host. This inventory does not prove an idle GPU; record concurrent workloads separately.'
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suites',default='unit',help='Comma list: unit,decision,stress,browser,stops,permissions,continuity,dialogue')
    p.add_argument('--url',default='http://127.0.0.1:8793')
    p.add_argument('--threshold',type=float,default=None);p.add_argument('--pause-ms',type=int,default=None)
    p.add_argument('--speakers',default='christoph,selma');p.add_argument('--seeds',default='42')
    p.add_argument('--fixtures',default=os.environ.get('ROBBY_TEST_FIXTURES','/mnt/nvme/robo-speaker-studies/speaker-live-fixtures'))
    p.add_argument('--keep-going',action='store_true',help='Archive failures and continue independent suites, without retries')
    a=p.parse_args();suites=a.suites.split(',')
    if set(suites)-{'unit','decision','stress','browser','stops','permissions','continuity','dialogue'}:p.error('Unknown suite')
    threshold=.28 if a.threshold is None else a.threshold;pause=700 if a.pause_ms is None else a.pause_ms
    if not .20<=threshold<=.65 or not 200<=pause<=2000:p.error('Settings outside supported UI ranges')
    speakers=a.speakers.split(',');seeds=list(map(int,a.seeds.split(',')))
    if set(speakers)-{'christoph','selma'}:p.error('Unknown fixture speaker')
    work=ROOT/'logs/voice-assistant-runs';work.mkdir(parents=True,exist_ok=True)
    lock=(work/'.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Another protocol runner holds the local test lock; nothing was interrupted.')
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')+'-'+uuid.uuid4().hex[:8]
    output=work/stamp;output.mkdir()
    manifest={'schema_version':1,'run_id':stamp,'started_utc':stamp.split('-')[0],
        'scope':'Synthetic digital injection; no physical microphone/DAC or held-out accuracy claim',
        'configuration':{'threshold':threshold,'pause_ms':pause,'browser_default_migration':a.threshold is None and a.pause_ms is None,
            'speakers':speakers,'seeds':seeds,'url':a.url},
        'python':platform.python_version(),'platform':platform.platform(),'packages':{},
        'source_sha256':{str(x.relative_to(ROOT)):digest(x) for x in source_files()},
        'fixture_sha256':{},'cases':[],'passed':False}
    for name in ('numpy','scipy','torch','torchaudio','speechbrain','transformers','httpx','playwright'):
        try:manifest['packages'][name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:manifest['packages'][name]=None
    fixture_root=Path(a.fixtures)
    if set(suites)&{'stress','browser','stops','permissions'}:
        missing=[name for name in FIXTURES if not (fixture_root/(name+'.mp3')).is_file()]
        if missing:raise SystemExit('Missing synthetic fixtures: '+', '.join(missing)+'. Restore the pinned fixture pack; see protocol.')
        manifest['fixture_sha256']={name:digest(fixture_root/(name+'.mp3')) for name in FIXTURES}
        pin=json.loads((ROOT/'static/voice-test-fixtures.json').read_text())
        pinned={row['name']:row['sha256'] for row in pin['cases']}
        drift=[name for name in FIXTURES if manifest['fixture_sha256'][name]!=pinned.get(name)]
        if drift:raise SystemExit('Fixture hashes differ from the pinned corpus: '+', '.join(drift)+'. Version new inputs explicitly before comparing configurations.')
    manifest_path=output/'manifest.json'
    def save():manifest_path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    save()
    env=os.environ.copy();env['ROBBY_TEST_FIXTURES']=str(fixture_root.resolve());env['ROBBY_TEST_URL']=a.url
    env.pop('ROBBY_TEST_QUICK_STOP',None);env.pop('ROBBY_TEST_PERMISSIONS',None)
    if a.threshold is not None or a.pause_ms is not None:
        env.update(ROBBY_TEST_THRESHOLD=str(threshold),ROBBY_TEST_PAUSE_MS=str(pause))
    else:
        env.pop('ROBBY_TEST_THRESHOLD',None);env.pop('ROBBY_TEST_PAUSE_MS',None)
    jobs=[]
    for suite in suites:
        if suite=='unit':jobs.append((suite,['run_unit_checks.py'],'logs/unit-checks.json',{},120))
        elif suite=='decision':jobs.append((suite,['validate_live_decision_route.py'],'static/rebalance-decision-check.json',{},120))
        elif suite=='continuity':jobs.append((suite,['validate_conversation_continuity.py'],'static/conversation-continuity-check.json',{},600))
        elif suite=='dialogue':jobs.append((suite,['browser_conversation_continuity.py'],'static/conversation-continuity-browser.json',{},900))
        elif suite=='stress':
            for speaker in speakers:
                for seed in seeds:
                    jobs.append((f'stress-{speaker}-{seed}',['conversation_stress.py','--asr','--speaker',speaker,'--seed',str(seed),'--threshold',str(threshold),'--pause-ms',str(pause)],
                        'static/conversation-stress-check.json' if speaker=='christoph' else 'static/conversation-stress-selma.json',{},240))
        else:
            mode={'stops':'ROBBY_TEST_QUICK_STOP','permissions':'ROBBY_TEST_PERMISSIONS'}.get(suite)
            jobs.append((suite,['browser_rebalance_check.py'],
                'static/'+{'stops':'rebalance-stop-browser.json','permissions':'rebalance-permissions-browser.json','browser':'rebalance-browser.json'}[suite],
                {mode:'1'} if mode else {},900))
    try:
        if any(s!='unit' for s in suites):
            manifest['server_before']=public_state(a.url.rstrip('/'));save()
            if not manifest['server_before'].get('ready'):raise RuntimeError('ASR service is not ready; no production restart attempted')
            manifest['expected_deployed_source_sha256']=deployed_source_hash()
            if manifest['server_before'].get('source_sha256')!=manifest['expected_deployed_source_sha256']:
                raise RuntimeError('The running ASR/UI source differs from this checkout; test the deployed checkout or explicitly deploy the reviewed change first')
            manifest['model_inventory_before']=model_inventory();save()
        for number,(name,args,receipt,extra,limit) in enumerate(jobs,1):
            receipt_path=ROOT/receipt;previous=receipt_path.stat().st_mtime_ns if receipt_path.exists() else None
            start=time.monotonic();entry={'id':name,'command':[sys.executable,*args],'attempt':1,'passed':False}
            print(f'[{number}/{len(jobs)}] {name}',flush=True)
            with (output/(name+'.log')).open('w') as log:
                try:
                    result=subprocess.run([sys.executable,*args],cwd=ROOT,env={**env,**extra},stdout=log,stderr=subprocess.STDOUT,timeout=limit)
                    entry['exit_code']=result.returncode
                except subprocess.TimeoutExpired:entry['error']='timeout';entry['exit_code']=124
            fresh=receipt_path.exists() and receipt_path.stat().st_mtime_ns!=previous
            if fresh:
                copied=output/(name+'.json');shutil.copy2(receipt_path,copied)
                try:record=json.loads(copied.read_text());entry['receipt_passed']=record.get('passed') is True
                except (ValueError,OSError):entry['receipt_passed']=False
                entry['receipt']=copied.name;entry['receipt_sha256']=digest(copied)
            else:entry['error']=entry.get('error','No fresh receipt; an older pass was not reused')
            entry.update(seconds=round(time.monotonic()-start,3),passed=entry.get('exit_code')==0 and entry.get('receipt_passed',False))
            manifest['cases'].append(entry);save()
            if not entry['passed'] and not a.keep_going:break
        manifest['source_changed_during_run']=(set(manifest['source_sha256'])!={str(x.relative_to(ROOT)) for x in source_files()} or any(not (ROOT/name).exists() or digest(ROOT/name)!=value for name,value in manifest['source_sha256'].items()))
        if any(s!='unit' for s in suites):
            manifest['server_after']=public_state(a.url.rstrip('/'))
            manifest['server_changed_during_run']=manifest['server_after'].get('source_sha256')!=manifest['server_before'].get('source_sha256')
            manifest['model_inventory_after']=model_inventory()
            manifest['models_changed_during_run']=manifest['model_inventory_before']!=manifest['model_inventory_after']
        manifest['passed']=(len(manifest['cases'])==len(jobs) and all(x['passed'] for x in manifest['cases'])
            and not manifest['source_changed_during_run'] and not manifest.get('server_changed_during_run',False) and not manifest.get('models_changed_during_run',False))
    except Exception as exc:
        # Controlled errors only; no environment/config contents serialized.
        manifest['runner_error']=type(exc).__name__+': '+str(exc)[:400]
    finally:
        manifest['finished_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();save()
        print(json.dumps({'run':str(output),'passed':manifest['passed'],'completed_suites':len(manifest['cases'])}))
    if not manifest['passed']:raise SystemExit(1)

if __name__=='__main__':main()
