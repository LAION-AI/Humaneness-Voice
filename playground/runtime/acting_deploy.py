"""Restart only the explicitly identified idle Playground, retaining its env."""
import argparse
import os
from pathlib import Path
import signal
import subprocess
import time
import urllib.request
import json
import socket

ROOT=Path(__file__).resolve().parent


def start(environment=None):
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8792))==0:
            raise RuntimeError('Port 8792 is already occupied; refusing a duplicate server')
    environment=dict(environment or os.environ)
    environment.update(ACTING_DEVICE='cuda:1',SMALL_ALIGN_DEVICE='cuda:1',PYTHONUNBUFFERED='1')
    environment.pop('LD_LIBRARY_PATH',None)
    with (ROOT/'logs/app-acting.log').open('ab') as log:
        process=subprocess.Popen(['/home/c4r33u19/moss15v2/.venv/bin/python','-m','uvicorn',
            'small_app:app','--host','127.0.0.1','--port','8792','--timeout-graceful-shutdown','5'],
            cwd=ROOT,env=environment,stdin=subprocess.DEVNULL,stdout=log,stderr=log,start_new_session=True)
    (ROOT/'logs/app.pid').write_text(str(process.pid))
    print('Playground PID:',process.pid,'log: logs/app-acting.log',flush=True)


def restart(pid):
    cmd=Path(f'/proc/{pid}/cmdline').read_bytes()
    if b'uvicorn\0small_app:app\0' not in cmd or Path(f'/proc/{pid}/cwd').resolve()!=ROOT:
        raise RuntimeError('Refusing to signal a process that is not this Playground')
    with urllib.request.urlopen('http://127.0.0.1:8792/api/state',timeout=10) as response:
        if json.load(response)['busy']:raise RuntimeError('Playground is busy; retry after generation')
    environment=dict(item.decode().split('=',1) for item in Path(f'/proc/{pid}/environ').read_bytes().split(b'\0') if b'=' in item)
    os.kill(pid,signal.SIGTERM)
    deadline=time.monotonic()+30
    while Path(f'/proc/{pid}').exists():
        if time.monotonic()>deadline:raise RuntimeError('Old Playground did not exit; no force kill')
        time.sleep(.25)
    start(environment)


if __name__=='__main__':
    parser=argparse.ArgumentParser();mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--pid',type=int);mode.add_argument('--start',action='store_true')
    args=parser.parse_args()
    start() if args.start else restart(args.pid)
