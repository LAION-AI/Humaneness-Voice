"""Real public-audio replay and native-layer consistency, no training/test selection."""
import io,json,time
import numpy as np
import requests
from speaker_v2 import prepare_pilot as core
from . import data
from .common import WORK,atomic

def main():
    def post(path,raw):
        for _ in range(30):
            r=requests.post('http://127.0.0.1:8793'+path,data=raw,timeout=60)
            if r.status_code in (423,503):time.sleep(1);continue
            r.raise_for_status();return r
        raise RuntimeError('Encoder unavailable or reserved for live conversation')
    rows=data.rows();anchor=rows[0];group=[r for r in rows if r['split']==anchor['split']]
    rank=next(i for i,r in enumerate(group) if r['voice']==anchor['voice']);cohort=[group[(rank+k)%len(group)] for k in range(min(12,len(group)))]
    seed=7730000+anchor['speaker_idx']*10000
    wave,*_=core.conversation(cohort,anchor,seed)
    steps=np.sort(np.random.default_rng(seed).choice(np.arange(5,75),5,replace=False))
    audio=[]
    for step in steps[:2]:
        end=int(step)*5120;x=wave[max(0,end-128000):end];audio.append(np.pad(x,(max(0,128000-len(x)),0)))
    raw=np.stack(audio).astype('<f4').tobytes()
    response=post('/internal/layer-features',raw)
    with np.load(io.BytesIO(response.content)) as z:layer=z['features'].astype(np.float32);seconds=float(z['encoder_seconds'])
    original=data.arrays();old=original['q'][:2,-4:].astype(np.float32)*original['scale'][:2,-4:].astype(np.float32)
    similarity=(layer[:,3]*old).sum(-1)/(np.linalg.norm(layer[:,3],axis=-1)*np.linalg.norm(old,axis=-1)+1e-8)
    assert layer.shape==(2,4,4,1024) and np.isfinite(layer).all() and similarity.mean()>.995
    response=post('/internal/context-features',raw)
    with np.load(io.BytesIO(response.content)) as z:native=z['features'].astype(np.float32)[:,-4:]
    np.testing.assert_allclose(layer[:,3],native,rtol=1e-5,atol=1e-5)
    response=post('/internal/layer-features',np.zeros(128000,'<f4').tobytes())
    with np.load(io.BytesIO(response.content)) as z:assert np.isfinite(z['features']).all()
    result={'public_audio_only':True,'optimization':False,'paired_l15_cosine':float(similarity.mean()),
        'native_l15_unchanged':True,'zero_window_supported':True,'encoder_seconds_two_windows':seconds}
    atomic(WORK/'preflight.json',result);print(json.dumps(result,indent=2))

if __name__=='__main__':main()
