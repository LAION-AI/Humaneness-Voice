"""Generate, audit and export an episode with one command. No service management."""
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,add_help=False)
    p.add_argument('--output',default=str(HERE/'outputs/podcast'));p.add_argument('--asr-url')
    a,rest=p.parse_known_args(argv)
    if '-h' in rest or '--help' in rest:
        print('One-command generation + audit + MP3/HTML export. --output is the episode directory. Optional --asr-url supplies an existing raw-float32 Parakeet service.',flush=True)
        subprocess.run([sys.executable,str(HERE/'speaker_run.py'),'--help'],check=True);return
    out=Path(a.output).resolve();runs=out/'generation'
    subprocess.run([sys.executable,str(HERE/'speaker_run.py'),*rest,'--output',str(runs)],check=True)
    device=rest[rest.index('--device')+1] if '--device' in rest else 'cuda:0'
    args=[sys.executable,str(HERE/'speaker_finish.py'),'--runs',str(runs),'--output',str(out),'--device',device]
    if a.asr_url:args+=['--asr-url',a.asr_url]
    subprocess.run(args,check=True)
if __name__=='__main__':main()
