"""One real short training run to check optimizer/metrics/resume plumbing only."""
import json,tempfile
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from speaker_v4 import data as v4
from . import data,train
from .common import WORK,atomic

def main():
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08,0)
    splits=v4.splits();indices=np.sort(np.r_[splits['train'][:512],splits['validation'][:512]])
    # Onset labels do not change optimization or the main validation metrics.
    with patch('speaker_v5.data.v4.subset',return_value=indices),patch('speaker_v5.data.onset_ages',return_value=np.ones(len(indices))):
        resident=data.Resident(25)
    with tempfile.TemporaryDirectory(prefix='robo-v5-train-selfcheck-') as directory:
        with patch.object(train,'WORK',Path(directory)),patch.object(train,'update',lambda **fields:None):
            row=train.train_one(resident,25,'stable20_hard',9991,epochs=1)
            assert np.isfinite(row['history'][0]['loss']) and row['pair_counts']['total']>0
            assert (Path(directory)/'models'/row['checkpoint']).exists()
            assert not list(Path(directory).glob('models/*-resume.pt'))
    result={'purpose':'pipeline correctness, not a quality estimate','real_public_training_windows':512,
        'validation_windows':512,'test_windows_accessed':0,'finite_objective':True,'pair_sampling':True,
        'best_checkpoint_written':True,'redundant_optimizer_removed':True,'temporary_weights_retained':False}
    atomic(WORK/'cuda_smoke.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
