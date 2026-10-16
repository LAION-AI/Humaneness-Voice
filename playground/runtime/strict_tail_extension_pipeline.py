"""Prospective large-tail confirmation after the original study pipeline."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request

import strict_tail_study as study
from strict_tail_pipeline import run_stage


def main():
    log_path=study.ROOT.parent/'logs/strict-tail-pipeline.log'
    while 'Strict-tail pipeline complete' not in log_path.read_text():
        previous=Path('/proc/2720516/cmdline')
        if not previous.exists() or b'strict_tail_pipeline.py' not in previous.read_bytes():
            raise RuntimeError('Original follow-up pipeline stopped without completion')
        print('Waiting for original confirmation',flush=True);time.sleep(10)
    previous_scorer=Path('/proc/2720574/cmdline')
    for _ in range(30):
        if not previous_scorer.exists() or not previous_scorer.read_bytes():break
        time.sleep(1)
    if previous_scorer.exists() and previous_scorer.read_bytes():
        raise RuntimeError('Previous owned scorer has not exited')
    log=(study.ROOT.parent/'logs/strict-tail-extension-scorer.log').open('ab')
    scorer=subprocess.Popen([sys.executable,'-m','uvicorn','strict_tail_study:score_app',
        '--host','127.0.0.1','--port','8795'],stdout=log,stderr=subprocess.STDOUT)
    try:
        ready=False
        for _ in range(180):
            if scorer.poll() is not None:raise RuntimeError('Extension scorer failed')
            try:
                with urllib.request.urlopen('http://127.0.0.1:8795/state',timeout=2) as response:
                    ready=bool(json.load(response).get('ready'))
            except (OSError,ValueError):pass
            if ready:break
            time.sleep(1)
        if not ready:raise RuntimeError('Extension scorer startup timed out')
        run_stage('confirm')
        study.report()
        print('Large-tail confirmation complete',flush=True)
    finally:
        if scorer.poll() is None:
            scorer.terminate()
            try:scorer.wait(timeout=15)
            except subprocess.TimeoutExpired:scorer.kill();scorer.wait()


if __name__=='__main__':main()
