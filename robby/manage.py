"""Own-process-only Robo server lifecycle. Never stops the shared TTS or image service."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parent
RECORD=ROOT/'.runtime/server.json'
CONFIG=ROOT/'config.local.json'


def owned(pid):
    try:
        process=Path('/proc')/str(pid)
        command=(process/'cmdline').read_bytes().split(b'\0')
        return (process/'cwd').resolve()==ROOT and b'server.py' in command
    except (OSError,RuntimeError):return False


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status']);args=parser.parse_args()
    saved=json.loads(RECORD.read_text()) if RECORD.exists() else {}
    pid=saved.get('pid');running=bool(pid and owned(pid))
    if args.action=='status':print(json.dumps({'running':running,'pid':pid,'url':'http://127.0.0.1:8793'}));return
    if args.action=='stop':
        if not running:print('No validated owned Robo process; nothing stopped.');return
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not owned(pid):break
            time.sleep(.05)
        if owned(pid):raise SystemExit('Robo still exiting; no force-kill issued.')
        RECORD.unlink(missing_ok=True);print('Stopped only Robo server PID',pid);return
    if running:print('Robo already running, PID',pid);return
    if sys.version_info<(3,10):
        raise SystemExit('Robby requires Python >=3.10. Activate its documented virtual environment before starting; no process was launched.')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8793))==0:raise SystemExit('Port8793 occupied by an unrecorded process; inspect ownership instead of stopping it.')
    (ROOT/'logs').mkdir(exist_ok=True);RECORD.parent.mkdir(exist_ok=True)
    config=json.loads(CONFIG.read_text()) if CONFIG.exists() else {}
    allowed={'CUDA_VISIBLE_DEVICES','ROBO_ECAPA_ROOT','ROBO_TTS_BASE','ROBO_MAX_SESSION_S',
        'ROBO_TYGER_SOURCE','ROBO_TYGER_BUNDLE','ROBO_TYGER_CACHE','SB_DISABLE_QUIRKS','ROBO_SPEAKER_MODEL_DIR','ROBO_DIRECTOR_MODE'}
    allowed.update({'ROBO_DECISION_BASE','ROBO_DECISION_MODEL'})
    allowed.update({'ROBO_DIRECTOR_MODEL','ROBO_DIRECTOR_BASE'})
    if set(config)-allowed:raise SystemExit('Only documented non-secret runtime settings belong in config.local.json')
    if config.get('ROBO_DECISION_BASE')=='http://127.0.0.1:8797':
        subprocess.run([sys.executable,'manage_laya.py','start'],cwd=ROOT,check=True)
    environment={**{k:str(v) for k,v in config.items()},**os.environ}
    environment.setdefault('CUDA_VISIBLE_DEVICES','0')
    with (ROOT/'logs/server.log').open('ab') as output:
        process=subprocess.Popen([sys.executable,'-u','server.py'],cwd=ROOT,env=environment,
            stdout=output,stderr=output,stdin=subprocess.DEVNULL,start_new_session=True)
    RECORD.write_text(json.dumps({'pid':process.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    print('Started only Robo server PID',process.pid)


if __name__=='__main__':main()
