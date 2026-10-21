"""Real public training-window replay; discarded CUDA gradients, no quality claims."""
import io,json,time
import numpy as np
import requests
from speaker_v4 import data as v4
from .common import WORK,atomic

def main():
    from speaker_v2 import prepare_pilot as core
    from speaker_v3.overnight import decoded
    core.source_signal=lambda c:decoded(c['path'],json.dumps(c['words'],sort_keys=True))
    def post(path,raw):
        for _ in range(60):
            r=requests.post('http://127.0.0.1:8793'+path,data=raw,timeout=60)
            if r.status_code in (423,503):time.sleep(1);continue
            r.raise_for_status();return r
        raise RuntimeError('Research encoder busy')
    rows=v4.rows();rownumber,anchor=next((i,r) for i,r in enumerate(rows) if r['split']=='train')
    group=[r for r in rows if r['split']==anchor['split']]
    rank=next(i for i,r in enumerate(group) if r['voice']==anchor['voice']);cohort=[group[(rank+k)%len(group)] for k in range(min(12,len(group)))]
    seed=7730000+anchor['speaker_idx']*10000;audio,*_=core.conversation(cohort,anchor,seed)
    steps=np.sort(np.random.default_rng(seed).choice(np.arange(5,75),5,replace=False));waves=[]
    for step in steps[:2]:
        end=int(step)*5120;x=audio[max(0,end-128000):end];waves.append(np.pad(x,(max(0,128000-len(x)),0)))
    raw=np.stack(waves).astype('<f4').tobytes()
    with np.load(io.BytesIO(post('/internal/context-features?frames=38',raw).content)) as z:x=z['features'].astype(np.float32);energy=z['energy'].astype(np.float32)
    with np.load(io.BytesIO(post('/internal/context-features',raw).content)) as z:original=z['features'].astype(np.float32)
    assert x.shape==(2,38,1024) and np.isfinite(x).all()
    np.testing.assert_allclose(x[:,-25:],original,atol=1e-5,rtol=1e-5)
    old=v4.arrays();indices=np.arange(rownumber*1000,rownumber*1000+2)
    quantized=old['q'][indices].astype(np.float32)*old['scale'][indices].astype(np.float32)
    similarity=(x[:,-25:]*quantized).sum(-1)/(np.linalg.norm(x[:,-25:],axis=-1)*np.linalg.norm(quantized,axis=-1)+1e-8)
    assert similarity.mean()>.995
    np.testing.assert_allclose(energy[:,-25:],old['energy'][indices],atol=.05)
    with np.load(io.BytesIO(post('/internal/context-features?frames=38',np.zeros(128000,'<f4').tobytes()).content)) as z:assert z['features'].shape==(1,38,1024) and np.isfinite(z['features']).all()
    import torch
    from speaker_v4.train import MixedStudent,regression
    torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1,0)
    m=MixedStudent(False).cuda();saved=torch.load(v4.OLD/'models/query_gru-best.pt',map_location='cuda',weights_only=False);m.load_state_dict(saved['model']);del saved
    # Worst-case pair arm: 256 anchors + 256 positives + 256 negatives.
    a=torch.tensor(x[:1],device='cuda').repeat(768,1,1).unsqueeze(1);e=torch.tensor(energy[:1],device='cuda').repeat(768,1)
    target=torch.randn(768,192,device='cuda')
    with torch.autocast('cuda',dtype=torch.bfloat16):
        emb,states=m(a,e);loss=regression(emb,target,m.teacher_mean)+states.square().mean()
    loss.backward();torch.cuda.synchronize()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
    peak=torch.cuda.max_memory_allocated();resident_bytes=116000*38*(1024+2)+116000*(192*2+38*2+16)
    assert resident_bytes+peak<.25*torch.cuda.get_device_properties(0).total_memory,'Projected resident + gradient peak exceeds 25% VRAM cap'
    result={'public_training_audio_only':True,'held_out_test_access':False,'native_25_frame_suffix_unchanged':True,
        'paired_cached_suffix_cosine':float(similarity.mean()),'zero_window_supported':True,'cuda_pair_batch_examples':768,
        'finite_amp_gradients':True,'peak_forward_backward_bytes':peak,'projected_resident_plus_peak_bytes':resident_bytes+peak,
        'note':'Repeated training audio for plumbing/memory check only; no speaker-quality estimate, weights discarded'}
    atomic(WORK/'preflight.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
