"""Audit/export all generated N modes with one reusable set of frozen judges."""
import argparse,gc,json,time
from pathlib import Path
from types import SimpleNamespace
from recipe import dump
from speaker_finish import main as finish
from run import setup

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',required=True);p.add_argument('--output');p.add_argument('--device',default='cuda:0');p.add_argument('--asr-url');a=p.parse_args();setup(a.device)
    import torch
    from speaker_scoring import SpeakerScorer
    from audit_alignment import BoundaryAligner
    torch.set_num_threads(4);root=Path(a.runs);out=Path(a.output or a.runs);ns=json.loads((root/'generation-complete.json').read_text())['n'];tick=time.perf_counter()
    config=json.loads((root/f'n{ns[0]}/generation/configuration.json').read_text())
    scorer=SpeakerScorer(a.device,config['references'],load_asr=True,asr_url=a.asr_url);aligner=BoundaryAligner(a.device)
    for n in ns:finish(SimpleNamespace(runs=str(root/f'n{n}/generation'),output=str(out/f'n{n}'),device=a.device,asr_url=a.asr_url),resources=(scorer,aligner))
    del scorer,aligner;gc.collect();torch.cuda.empty_cache()
    dump(root/'gpu-finished.json',dict(n=ns,seconds=time.perf_counter()-tick,finished_unix=time.time()))
    dump(out/'exports-complete.json',dict(n=ns,time=time.time()))
if __name__=='__main__':main()
