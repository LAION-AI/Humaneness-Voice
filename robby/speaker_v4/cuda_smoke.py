"""CUDA/AMP/pair-loss plumbing check; synthetic earlier layers, no quality claim.

Uses public training and validation features; no test optimization. Temporary
weights/results are removed. Does not change the real study's status/outputs.
"""
import json,tempfile
from pathlib import Path
import numpy as np
import torch
from . import train,data
from .common import WORK,OLD,atomic

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.10,0)
    resident=train.Resident(np.r_[np.arange(1024),np.arange(20000,22048)])
    # Validate the 4-layer tensor and gradients without writing a real feature
    # cache or pretending these copied/sign-flipped layers are research data.
    resident.q=resident.q.repeat(1,4,1,1);resident.q[:,0]*=-1
    resident.scales=resident.scales.repeat(1,4,1,1)
    pairs=train.Pairs(resident,data.stable_targets());results=[]
    with tempfile.TemporaryDirectory(prefix='tyger-v4-cuda-selfcheck-') as directory:
        train.WORK=Path(directory);train.update=lambda **fields:None
        for arm,mix,mode in [('cuda_layers',True,None),('cuda_both',True,'hard'),('cuda_random',False,'random')]:
            row=train.train_one(resident,arm,9981,.5,OLD/'models/query_gru-best.pt',mix_layers=mix,pair_mode=mode,pairs=pairs,epochs=1)
            assert row['history'] and np.isfinite(row['history'][0]['loss'])
            if mode=='random':assert row['full_transcript_negative_match_rate']==0
            results.append({'arm':arm,'finite_loss':True,'checkpoint_created':True})
    result={'purpose':'pipeline correctness only; earlier layers are synthetic copies/sign flips',
        'held_out_test_access':False,'temporary_artifacts_retained':False,'checks':results,
        'peak_cuda_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}
    atomic(WORK/'cuda_smoke.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
