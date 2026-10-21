"""Validated own-process lifecycle for the isolated local CPU decision worker."""
import argparse,json,os,signal,socket,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
RECORD=ROOT/'.runtime/laya.json'

def owned(pid):
    try:
        root=Path('/proc')/str(pid)
        return (root/'cwd').resolve()==ROOT and b'laya_decision_server.py' in (root/'cmdline').read_bytes().split(b'\0')
    except OSError:return False

def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','stop','status']);args=parser.parse_args()
    record=json.loads(RECORD.read_text()) if RECORD.exists() else {};pid=record.get('pid');running=bool(pid and owned(pid))
    if args.action=='status':print(json.dumps({'running':running,'pid':pid,'url':'http://127.0.0.1:8797'}));return
    if args.action=='stop':
        if not running:print('No validated owned Laya process; nothing stopped.');return
        os.kill(pid,signal.SIGTERM)
        for _ in range(100):
            if not owned(pid):break
            time.sleep(.05)
        if owned(pid):raise SystemExit('Laya still exiting; no force kill issued.')
        RECORD.unlink(missing_ok=True);print('Stopped only local Laya worker',pid);return
    if running:print('Local Laya already running',pid);return
    with socket.socket() as probe:
        if probe.connect_ex(('127.0.0.1',8797))==0:raise SystemExit('Port 8797 occupied by an unrecorded process; nothing stopped.')
    (ROOT/'logs').mkdir(exist_ok=True);RECORD.parent.mkdir(exist_ok=True)
    with (ROOT/'logs/laya.log').open('ab') as output:
        process=subprocess.Popen([sys.executable,'-u','laya_decision_server.py'],cwd=ROOT,
            env={**os.environ,'USE_TF':'0'},stdin=subprocess.DEVNULL,stdout=output,stderr=output,start_new_session=True)
    RECORD.write_text(json.dumps({'pid':process.pid,'started_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())}))
    print('Started only local CPU Laya worker',process.pid)

if __name__=='__main__':main()
