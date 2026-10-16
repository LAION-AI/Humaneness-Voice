"""Run the user's second experiment only after the first pipeline completes."""
import json
import argparse
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.request

import strict_tail_study as study


def run_stage(phase):
    jobs=[]
    try:
        for voice,device in zip(study.VOICES,('cuda:0','cuda:1')):
            log=(study.ROOT.parent/'logs'/f'strict-tail-{phase}-{voice}.log').open('ab')
            jobs.append(subprocess.Popen([sys.executable,str(study.ROOT.parent/'strict_tail_study.py'),
                phase,'--voice',voice,'--device',device],stdout=log,stderr=subprocess.STDOUT))
        while any(p.poll() is None for p in jobs):
            if any(p.poll() not in (None,0) for p in jobs):
                raise RuntimeError('Worker failed in '+phase)
            print(phase,'takes',len(study.runner.read_rows(phase=phase)),flush=True)
            time.sleep(10)
        if any(p.returncode for p in jobs):raise RuntimeError('Worker failed in '+phase)
    finally:
        for job in jobs:
            if job.poll() is None:
                job.terminate()
                try:job.wait(timeout=15)
                except subprocess.TimeoutExpired:job.kill();job.wait()


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--first-pipeline-pid',type=int,required=True)
    parser.add_argument('--first-scorer-pid',type=int,required=True)
    args=parser.parse_args()
    study.freeze()
    first_log=study.ROOT.parent/'logs/seed-scaling-pipeline.log'
    while 'Pipeline complete' not in first_log.read_text():
        first_process=Path(f'/proc/{args.first_pipeline_pid}/cmdline')
        if not first_process.exists() or b'seed_scaling_pipeline.py' not in first_process.read_bytes():
            raise RuntimeError('First study stopped without completion; do not start second')
        print('Waiting for first study completion',flush=True);time.sleep(10)
    # Stop only the local scoring process started for this experiment, not TTS.
    first_scorer=Path(f'/proc/{args.first_scorer_pid}/cmdline')
    if first_scorer.exists() and b'seed_scaling_study:score_app' in first_scorer.read_bytes():
        os.kill(args.first_scorer_pid,signal.SIGTERM)
        for _ in range(30):
            if not first_scorer.exists() or not first_scorer.read_bytes():break
            time.sleep(1)
        if first_scorer.exists() and first_scorer.read_bytes():raise RuntimeError('Old study scorer did not stop safely')
    log=(study.ROOT.parent/'logs/strict-tail-scorer.log').open('ab')
    scorer=subprocess.Popen([sys.executable,'-m','uvicorn','strict_tail_study:score_app',
        '--host','127.0.0.1','--port','8795'],stdout=log,stderr=subprocess.STDOUT)
    try:
        ready=False
        for _ in range(180):
            if scorer.poll() is not None:raise RuntimeError('Follow-up scorer failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8795/state',timeout=2) as response:
                    ready=bool(json.load(response).get('ready'))
            except (OSError,ValueError):pass
            if ready:break
            time.sleep(1)
        if not ready:raise RuntimeError('Scorer startup timeout')
        run_stage('collection');print('Deriving strict-tail vectors',flush=True);study.derive()
        run_stage('screen');print('Frozen follow-up selection',json.dumps(study.select()),flush=True)
        run_stage('confirm');study.report();print('Strict-tail pipeline complete',flush=True)
    finally:
        if scorer.poll() is None:
            scorer.terminate()
            try:scorer.wait(timeout=15)
            except subprocess.TimeoutExpired:scorer.kill();scorer.wait()


if __name__=='__main__':main()
