"""Coordinate only this study's two bounded GPU workers, then publish report."""
import json
from pathlib import Path
import subprocess
import sys
import time

import seed_scaling_study as study


def run_stage(phase):
    jobs=[]
    for voice,device in zip(study.VOICES,('cuda:0','cuda:1')):
        log=(study.ROOT.parent/'logs'/f'seed-scaling-{phase}-{voice}.log').open('ab')
        jobs.append(subprocess.Popen([sys.executable,str(study.ROOT.parent/'seed_scaling_study.py'),
                     phase,'--voice',voice,'--device',device],stdout=log,stderr=subprocess.STDOUT))
    while any(p.poll() is None for p in jobs):
        print(phase,'takes',len(study.read_rows(phase=phase)),flush=True)
        time.sleep(10)
    if any(p.returncode for p in jobs):
        raise RuntimeError('Worker failed in '+phase)


def main():
    study.freeze()
    # The initial collection workers were already launched by the parent.
    while len(study.read_rows(phase='collection'))<1000:
        rows=study.read_rows(phase='collection')
        if any(r.get('failed') for r in rows): raise RuntimeError('Collection failure')
        print('collection takes',len(rows),flush=True);time.sleep(10)
    print('Deriving vectors',flush=True);study.derive()
    run_stage('screen')
    print('Selection',json.dumps(study.select()),flush=True)
    run_stage('confirm')
    study.report()
    print('Pipeline complete',flush=True)


if __name__=='__main__':main()
