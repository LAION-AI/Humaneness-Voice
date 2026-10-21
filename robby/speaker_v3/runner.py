"""Bounded resumable overnight supervisor; never kills or restarts live services."""
import json,os,subprocess,sys,time
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def main():
    statefile=ROOT/'runner.json'
    if statefile.exists():
        old=json.loads(statefile.read_text());pid=old.get('pid',0)
        try:
            command=Path('/proc',str(pid),'cmdline').read_bytes()
            if b'speaker_v3.runner' in command:raise SystemExit('An owned runner is already active')
        except FileNotFoundError:pass
    statefile.write_text(json.dumps({'pid':os.getpid(),'started':time.time()}))
    with (ROOT/'overnight.log').open('ab',buffering=0) as log:
        for attempt in range(4):
            if (ROOT/'STOP').exists():return
            child=subprocess.Popen([sys.executable,'-u','-m','speaker_v3.overnight'],cwd=ROOT.parent,stdout=log,stderr=log)
            statefile.write_text(json.dumps({'pid':os.getpid(),'child_pid':child.pid,'attempt':attempt+1,'started':time.time()}))
            code=child.wait()
            if code==0:return
            status=json.loads((ROOT/'progress.json').read_text());error=status.get('error','')
            if any(s in error for s in ('Storage','space','budget','STOP','Nonfinite')):return
            time.sleep(60*(attempt+1))

if __name__=='__main__':main()
