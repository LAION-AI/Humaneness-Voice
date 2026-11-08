"""Generate N=2 and N=1 with one model load; audit/export each run afterwards."""
import argparse,json,time,gc
from pathlib import Path
from recipe import HERE,dump
from run import setup
from speaker_run import arguments,main as generate
from inputs import load_inputs

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
    setup(a.device)
    import torch
    from generation import Generator
    from speaker_scoring import SpeakerScorer
    torch.set_num_threads(4);torch.cuda.set_device(a.device)
    root=Path(a.output).resolve();root.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    template=arguments(['--model',a.model,'--device',a.device])
    refs,_,_=load_inputs(template);scorer=SpeakerScorer(a.device,refs);g=Generator(a.model,refs,a.device)
    shared_load=time.perf_counter()-started
    runs=[]
    try:
        for n,groups in [(2,4),(1,8)]:
            out=root/f'n{n}'/'generation'
            args=arguments(['--model',a.model,'--device',a.device,'--n',str(n),'--groups',str(groups),'--output',str(out)])
            out.mkdir(parents=True,exist_ok=True);dump(out/'process-start.json',dict(created_unix=time.time()))
            generate(args,resources=(g,scorer))
            runs.append(dict(n=n,groups=groups,path=str(out)))
    finally:
        g.close();del scorer,g;gc.collect();torch.cuda.empty_cache()
    dump(root/'generation-complete.json',dict(runs=runs,shared_load_seconds=shared_load,seconds=time.perf_counter()-started,
        note='Two fresh candidate pools; same scripts/references/settings. Eight simultaneous candidates via N2×4 turns or N1×8 turns. Random draws/batch assignment differ from N4; not matched sample trajectories. N1 has no online ranking.'))
if __name__=='__main__':main()
