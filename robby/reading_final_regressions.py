"""Sequential current-update QA, one live ASR slot, no private input.

Failure collection never turns failed tests into passes. Does not restart any
service, delete data, publish anything or retry failed live test stages.
"""
import json,subprocess,sys,time
from pathlib import Path
from final_verify import runtime_signature

def main():
    py=sys.executable;b=['uv','run','--no-project','--with','playwright','--python',py]
    stages=[('unit',[py,'run_unit_checks.py'],90),
        ('browser_stop',b+['browser_stop_controls.py'],150),
        ('lifecycle',b+['browser_lifecycle_checks.py'],90),
        ('gaze_privacy_ui',b+['browser_reliability_checks.py'],120),
        ('wake_free_and_strict',b+['browser_open_speakers_checks.py'],240),
        ('public_sources_browser',b+['browser_knowledge_checks.py'],150),
        ('explicit_voice_variants',[py,'-m','speaker_live.variant_check'],240),
        ('public_tools',[py,'speed_study/knowledge_live.py'],180),
        ('director_contract',[py,'speed_study/benchmark.py','--models','gpt-5.6-luna','--repetitions','1','--output','logs/director-reading-regression.json'],180),
        ('streamed_echo',[py,'-m','speaker_live.echo_test'],240)]
    report={'passed':False,'synthetic_only':True,'runtime_source_sha256':runtime_signature(),'started':time.time(),'stages':[]}
    target=Path('logs/reading-final-regressions.json')
    for name,command,timeout in stages:
        start=time.monotonic();print('START '+name,flush=True)
        row={'stage':name,'passed':False}
        with Path('logs/reading-qa-'+name+'.log').open('w') as output:
            try:
                result=subprocess.run(command,stdout=output,stderr=subprocess.STDOUT,timeout=timeout)
                row.update(exit_code=result.returncode,passed=result.returncode==0)
                if name=='director_contract':
                    rows=json.loads(Path('logs/director-reading-regression.json').read_text())['rows']
                    row.update(passed=row['passed'] and len(rows)==14 and all(r['passed'] for r in rows),checks_passed=sum(r['passed'] for r in rows),checks=len(rows))
                if name=='public_tools':row['passed']=row['passed'] and json.loads(Path('logs/knowledge-director-live.json').read_text())['passed']
            except subprocess.TimeoutExpired:row.update(timed_out=True)
        row['seconds']=round(time.monotonic()-start,2);report['stages'].append(row)
        target.write_text(json.dumps(report,indent=2));print(json.dumps(row),flush=True)
    report.update(finished=time.time(),passed=all(r['passed'] for r in report['stages']) and runtime_signature()==report['runtime_source_sha256'])
    target.write_text(json.dumps(report,indent=2))
    if not report['passed']:raise SystemExit('Retained regression failures: inspect logs/reading-final-regressions.json')

if __name__=='__main__':main()
