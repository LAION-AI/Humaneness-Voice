"""Finish frozen-source two-way QA after an explicitly started proactive replay.

No retries, service restarts, private recordings or replacement model outputs.
"""
import json,subprocess,sys,time
from pathlib import Path
from final_verify import runtime_signature

def main():
    root=Path(__file__).resolve().parent
    path=Path('/mnt/nvme/robo-speaker-studies/speaker-live-results/proactive-live.json')
    signature=runtime_signature();deadline=time.monotonic()+600
    print('Waiting for the current-source four-condition live replay; no extra audio slot opened.',flush=True)
    while time.monotonic()<deadline:
        report=json.loads(path.read_text())
        if report.get('runtime_source_sha256')!=signature or runtime_signature()!=signature:
            raise SystemExit('Source changed; no stale verification claimed.')
        if any(not r.get('passed') for r in report.get('cases',[])):
            raise SystemExit('Proactive replay failed; failure retained, no hidden retry.')
        if report.get('finished'):break
        time.sleep(.5)
    else:raise SystemExit('Proactive replay did not finish within ten minutes.')
    if not (report.get('passed') and {r['condition'] for r in report['cases']}=={'positive','unknown','disabled','unrelated'}):
        raise SystemExit('Incomplete proactive verification; stop.')
    py=sys.executable
    commands=['christoph-stop','christoph-halt','christoph-quiet','christoph-stop-speaking',
        'robert-stop','background-stop','robert-stop-short','background-stop-short','christoph-stop-repeat']
    stages=[('Nine actual streamed stop cases',[py,'-m','speaker_live.stop_test','--commands',*commands]),
        ('Full current-task regression suite',[py,'verify_active_todos.py']),
        ('Publish the English local worklog',[py,'build_worklog.py']),
        ('Build audited text-only source download',[py,'export_source.py'])]
    for name,command in stages:
        if runtime_signature()!=signature:raise SystemExit('Runtime changed; stop before '+name)
        print('START '+name,flush=True)
        subprocess.run(command,cwd=root,check=True)
    print('Two-way interaction QA and current regressions passed; live services were not restarted by this verifier.',flush=True)

if __name__=='__main__':main()
