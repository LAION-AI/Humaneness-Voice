"""Owned-process lifecycle for the isolated diarizer (GPU default, CPU opt-in)."""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RECORD=ROOT/'.runtime/diarization.json'

def owned(pid):
    try:
        p=Path('/proc',str(pid));return (p/'cwd').resolve()==ROOT and b'diarization_service.py' in (p/'cmdline').read_bytes().split(b'\0')
    except (FileNotFoundError,ProcessLookupError):return False

def main(action):
    saved=json.loads(RECORD.read_text()) if RECORD.exists() else {};pid=saved.get('pid',0)
    running=owned(pid)
    if action=='status':print(json.dumps({'running':running,'pid':pid,'url':'http://127.0.0.1:8796','device':saved.get('device','unknown')}));return
    if action=='stop':
        if not running:print('No validated owned diarization process.');return
        os.kill(pid,signal.SIGTERM)
        for _ in range(200):
            if not owned(pid):break
            time.sleep(.05)
        if owned(pid):raise RuntimeError('Diarization process still exiting; not force-killed')
        RECORD.unlink(missing_ok=True);print('Stopped isolated diarization PID',pid);return
    if running:print('Diarization already running, PID',pid);return
    if sys.version_info<(3,10):raise SystemExit('Diarization requires Python >=3.10; activate the documented virtual environment. No process was launched.')
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8796))==0:raise RuntimeError('Port8796 belongs to an unrecorded process; not stopped')
    deps=Path(os.environ.get('ROBO_DIARIZATION_DEPS','/mnt/nvme/robo-speaker-studies/diarization-deps'))
    if not (deps/'transformers/models/nemotron3_diarization').is_dir():raise RuntimeError('Pinned isolated diarization dependencies are missing')
    (ROOT/'logs').mkdir(exist_ok=True)
    with (ROOT/'logs/diarization.log').open('ab') as out:
        p=subprocess.Popen([sys.executable,'-u','diarization_service.py'],cwd=ROOT,
            env={**os.environ,'PYTHONPATH':str(deps),'CUDA_VISIBLE_DEVICES':os.environ.get('ROBO_DIARIZATION_GPU','1'),
                'ROBO_DIARIZATION_DEVICE':os.environ.get('ROBO_DIARIZATION_DEVICE','cuda')},stdout=out,stderr=out,stdin=subprocess.DEVNULL,start_new_session=True)
    print('Started isolated diarization PID',p.pid)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','stop','status']);main(p.parse_args().action)
