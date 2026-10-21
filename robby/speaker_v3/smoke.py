"""Pipeline correctness/optimizer smoke test, not a speaker-quality evaluation.

Use a public training speaker only; no held-out identities for optimization.
No smoke weights or feature files are retained, and no live audio is recorded.
"""
import argparse,io,json,os,tempfile,time
from pathlib import Path
import numpy as np
import requests
import torch
from torch.nn import functional as F
from .overnight import WORK,ROOT,plan
from .train import Student,CONFIGS
from speaker_teacher import get_cached_teacher,preserve_encoder_precision
from speaker_v2.prepare_pilot import pcm

def main(dataset=False):
    preserve_encoder_precision();torch.set_num_threads(4)
    row=next(r for r in plan() if r['split']=='train' and (WORK/'sources'/r['voice']/'clips.json').exists())
    clips=json.loads((WORK/'sources'/row['voice']/'clips.json').read_text())
    clip=next(c for c in clips if c['role']=='prediction');wave=pcm(Path(clip['path']).read_bytes())
    teacher=get_cached_teacher('/mnt/nvme/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/0f99f2d0ebe89ac095bcc5903c4dd8f72b367286')
    start=time.perf_counter();target=teacher.embed(wave);teacher_seconds=time.perf_counter()-start
    waves=[]
    for end in np.linspace(2,min(len(wave)/16000,12),4):
        n=int(end*16000);part=wave[max(0,n-128000):n];waves.append(np.pad(part,(max(0,128000-len(part)),0)))
    begin=time.perf_counter();r=requests.post('http://127.0.0.1:8793/internal/context-features',data=np.stack(waves).astype('<f4').tobytes(),timeout=90)
    r.raise_for_status();z=np.load(io.BytesIO(r.content));http_seconds=time.perf_counter()-begin
    x=z['features'].astype(np.float32);scale=np.maximum(np.abs(x).max(-1,keepdims=True)/127,1e-6)
    quant=np.clip(np.rint(x/scale),-127,127).astype(np.int8).astype(np.float32)*scale
    cos=float(((x*quant).sum(-1)/(np.linalg.norm(x,axis=-1)*np.linalg.norm(quant,axis=-1)+1e-8)).mean())
    device='cuda:0';torch.cuda.set_per_process_memory_fraction(.10,0);results=[]
    a=torch.from_numpy(x).to(device);p=torch.from_numpy(z['pool'].astype(np.float32)).to(device);e=torch.from_numpy(z['energy'].astype(np.float32)).to(device)
    t=torch.from_numpy(target).to(device).expand(4,-1)
    for cfg in CONFIGS:
        torch.manual_seed(123);m=Student(cfg['architecture']).to(device);optimizer=torch.optim.AdamW(m.parameters(),lr=.001);losses=[]
        for step in range(12):
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                pred,logits=m(a[:,-[4,8,13,25][step%4]:],p,e[:,-[4,8,13,25][step%4]:],step%2==0)
                loss=(1-(pred*t).sum(-1)).mean()+.1*F.cross_entropy(logits,torch.zeros(4,dtype=torch.long,device=device))
            loss.backward()
            assert all(torch.isfinite(v.grad).all() for v in m.parameters() if v.grad is not None),'Nonfinite gradient'
            optimizer.step();losses.append(float(loss.detach()))
        assert np.isfinite(losses).all() and losses[-1]<losses[0],(cfg,losses)
        results.append({'name':cfg['name'],'parameters':sum(v.numel() for v in m.parameters()),'finite_gradients':True,
            'single_training_utterance_loss_start':losses[0],'single_training_utterance_loss_end':losses[-1]})
        del m,optimizer
    report={'kind':'correctness smoke test, NOT speaker accuracy or generalization','speaker_split':'train only','source_uid':clip['uid'],
        'teacher_full_utterance_seconds':teacher_seconds,'encoder_four_windows_seconds':float(z['encoder_seconds']),
        'http_batch_seconds':http_seconds,'int8_hidden_state_cosine':cos,'models':results,'peak_student_allocated_MiB':torch.cuda.max_memory_allocated()/1024**2}
    if dataset:
        # Encoder/label/storage correctness only; held-out identities are NOT
        # optimized and these temporary examples are not quality evaluation.
        from . import overnight
        selected=[]
        for split in ('train','validation','test'):
            candidates=[r for r in plan() if r['split']==split and (WORK/'sources'/r['voice']/'clips.json').exists()][:2]
            assert len(candidates)==2
            for entry in candidates:
                clips=json.loads((WORK/'sources'/entry['voice']/'clips.json').read_text())
                probes=[c.copy() for c in clips if c['role']=='prediction'][:2]
                for c in probes:c['target']=teacher.embed(pcm(Path(c['path']).read_bytes()))
                selected.append({**entry,'clips':probes})
        with tempfile.TemporaryDirectory(prefix='tyger-public-dataset-selfcheck-') as directory:
            overnight.WORK=Path(directory);overnight.update=lambda **kw:None
            arrays=overnight.corpus(selected,25)
            speech=arrays['activity']==0
            assert arrays['filled'].all() and speech.any()
            assert np.isfinite(arrays['pool']).all() and (np.linalg.norm(arrays['local_target'][speech].astype(np.float32),axis=-1)>.95).all()
            assert set(arrays['speaker'][speech].tolist())<=set(r['speaker_idx'] for r in selected)
            report['dataset_pipeline']={'temporary_windows':len(speech),'speakers':6,'scenarios':np.unique(arrays['scenario']).tolist(),
                'activity_counts':np.bincount(arrays['activity'],minlength=3).tolist(),'all_committed':True,'targets_nonzero_on_speech':True,
                'optimization_on_held_out_speakers':False,'temporary_files_retained':False}
    (ROOT/'smoke_results.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--dataset',action='store_true');main(parser.parse_args().dataset)
