"""Replace this Playground's explicit quick-tunnel process, keeping its logs."""
import argparse
import json
from pathlib import Path
import re
import signal
import subprocess
import time
import urllib.request

ROOT=Path(__file__).resolve().parent
BINARY='/home/c4r33u19/.local/bin/cloudflared'


def restart(pid,new_pid=None):
    args=Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    if not args or args[0]!=BINARY.encode() or b'http://127.0.0.1:8792' not in args:
        raise RuntimeError('Refusing to stop a different tunnel')
    # Unique log file preserves the old tunnel diagnostics and prevents an
    # old quick URL from being mistaken for the new one.
    if new_pid:
        new_args=Path(f'/proc/{new_pid}/cmdline').read_bytes().split(b'\0')
        if new_args[0]!=BINARY.encode() or b'http://127.0.0.1:8792' not in new_args:
            raise RuntimeError('Replacement process is not this Playground tunnel')
        path=Path(f'/proc/{new_pid}/fd/1').resolve()
        if not path.is_relative_to(ROOT/'logs'):raise RuntimeError('Unexpected replacement log')
        process=None
    else:
        path=ROOT/'logs'/f'tunnel-{time.time_ns()}.log'
        with path.open('wb') as log:
            process=subprocess.Popen([BINARY,'tunnel','--url','http://127.0.0.1:8792',
                '--no-autoupdate','--protocol','http2'],cwd=ROOT,stdout=log,stderr=log,
                stdin=subprocess.DEVNULL,start_new_session=True)
        new_pid=process.pid
    deadline=time.monotonic()+90
    while time.monotonic()<deadline:
        if (process is not None and process.poll() is not None) or not Path(f'/proc/{new_pid}').exists():
            raise RuntimeError('New tunnel exited; old tunnel was left intact')
        matches=re.findall(r'https://[a-z0-9-]+\.trycloudflare\.com',path.read_text())
        if matches and 'Registered tunnel connection' in path.read_text():
            url=matches[-1]
            try:
                with urllib.request.urlopen(url+'/api/models/catalog',timeout=10) as response:
                    data=json.load(response)
            except (OSError,ValueError):
                time.sleep(1)
                continue # DNS and edge routes may lag tunnel registration.
            if data['default']!='small_acting':raise RuntimeError('New tunnel reached the wrong service')
            if Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')!=args:
                raise RuntimeError('Old tunnel identity changed; refusing to signal it')
            import os
            os.kill(pid,signal.SIGTERM)
            (ROOT/'logs/tunnel.pid').write_text(str(new_pid))
            (ROOT/'logs/tunnel-url.txt').write_text(url+'\n')
            print(url,flush=True)
            return
        time.sleep(.5)
    raise RuntimeError('New tunnel did not connect; old tunnel was left intact')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--pid',type=int,required=True)
    parser.add_argument('--new-pid',type=int)
    args=parser.parse_args();restart(args.pid,args.new_pid)
