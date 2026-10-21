"""Reuse 25 L15 states; replay only the missing 13 older states into tmpfs."""
import io,json,shutil,tempfile,time
from pathlib import Path
import numpy as np
import requests
from speaker_v4 import data as v4
from .common import WORK,OLD,atomic,guard,update,digest

def cache_path():
    path=WORK/'cache_pointer.json'
    if path.exists():
        p=Path(json.loads(path.read_text())['path'])
        if p.parent!=Path('/dev/shm') or not p.name.startswith('robo-v5-context-'):raise RuntimeError('Unexpected RAM cache pointer')
        if p.exists():return p
    required=116000*13*(1024+2)+116000*38*2+116000
    if shutil.disk_usage('/dev/shm').free<required+8*1024**3:raise RuntimeError('Insufficient tmpfs headroom')
    # tmpfs can exceed physically available RAM; check both before allocation.
    available=next(int(line.split()[1])*1024 for line in Path('/proc/meminfo').read_text().splitlines() if line.startswith('MemAvailable:'))
    if available<required+8*1024**3:raise RuntimeError('Insufficient available system RAM')
    p=Path(tempfile.mkdtemp(prefix='robo-v5-context-',dir='/dev/shm'))
    atomic(path,{'path':str(p),'planned_bytes':required,'ephemeral':True,'created':time.time()});return p

def onset_ages():
    path=WORK/'onset_age.npy'
    if path.exists():return np.load(path)
    import gzip
    keep=v4.subset();old=v4.arrays();ages=np.full(len(keep),-1,np.float32);lookup={int(g):i for i,g in enumerate(keep)}
    for rownumber,row in enumerate(v4.rows()):
        with gzip.open(OLD/'conversations'/(row['voice']+'.json.gz'),'rt') as stream:conversations=json.load(stream)
        cursor=rownumber*1000
        for conversation in conversations:
            for step in conversation['selected_320ms_steps']:
                if cursor in lookup and old['activity'][cursor]==0:
                    end=step*.32;who=int(old['speaker'][cursor])
                    event=next((e for e in reversed(conversation['events']) if e.get('speaker_idx')==who and e.get('onset_s',99)<end and e.get('offset_s',0)>end-.32),None)
                    if event is None:raise RuntimeError('Current speaker onset metadata mismatch')
                    ages[lookup[cursor]]=end-event['onset_s']
                cursor+=1
        if cursor!=(rownumber+1)*1000:raise RuntimeError('Conversation count changed')
    np.save(path,ages);return ages

def extract():
    cache=cache_path();guard(cache);keep=v4.subset();n=len(keep)
    specs={'q':(np.int8,(n,13,1024)),'scale':(np.float16,(n,13,1)),
           'energy':(np.float16,(n,38)),'filled':(np.uint8,(n,))}
    z={key:np.load(cache/(key+'.npy'),mmap_mode='r+') if (cache/(key+'.npy')).exists() else
       np.lib.format.open_memmap(cache/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,(dtype,shape) in specs.items()}
    if z['filled'].all():return cache
    from speaker_v2 import prepare_pilot as core
    from speaker_v3.overnight import decoded
    core.source_signal=lambda c:decoded(c['path'],json.dumps(c['words'],sort_keys=True))
    original=v4.arrays();people=v4.rows();groups={s:[r for r in people if r['split']==s] for s in ('train','validation','test')}
    lookup={int(g):i for i,g in enumerate(keep)};client=requests.Session();checks=[];timings=[];started=time.time()
    for rownumber,anchor in enumerate(people):
        relevant=keep[(keep>=rownumber*1000)&(keep<(rownumber+1)*1000)]
        if all(z['filled'][lookup[int(g)]] for g in relevant):continue
        group=groups[anchor['split']];rank=next(i for i,r in enumerate(group) if r['speaker_idx']==anchor['speaker_idx'])
        cohort=[group[(rank+k)%len(group)] for k in range(min(12,len(group)))];waves=[];locations=[];globals_=[]
        for repeat in sorted({int((g-rownumber*1000)//5) for g in relevant}):
            chosen=[int(g) for g in relevant if (g-rownumber*1000)//5==repeat and not z['filled'][lookup[int(g)]]]
            if not chosen:continue
            seed=7730000+anchor['speaker_idx']*10000+repeat
            audio,*_=core.conversation(cohort,anchor,seed);steps=np.sort(np.random.default_rng(seed).choice(np.arange(5,75),5,replace=False))
            for g in chosen:
                end=int(steps[(g-rownumber*1000)%5])*5120;wave=audio[max(0,end-128000):end]
                waves.append(np.pad(wave,(max(0,128000-len(wave)),0)));locations.append(lookup[g]);globals_.append(g)
        for offset in range(0,len(waves),4):
            guard(cache);wave=np.stack(waves[offset:offset+4]).astype('<f4');local=locations[offset:offset+4];global_idx=globals_[offset:offset+4]
            retries=0
            while True:
                guard(cache)
                try:
                    response=client.post('http://127.0.0.1:8793/internal/context-features?frames=38',data=wave.tobytes(),timeout=60)
                    if response.status_code in (423,429,503):time.sleep(2);continue
                    response.raise_for_status();break
                except requests.RequestException:
                    retries+=1
                    if retries>=12:raise
                    update(phase='Waiting for shared causal encoder',retry=retries);time.sleep(5)
            with np.load(io.BytesIO(response.content)) as encoded:
                x=encoded['features'].astype(np.float32);energy=encoded['energy'];seconds=float(encoded['encoder_seconds'])
            if x.shape!=(len(local),38,1024) or energy.shape!=(len(local),38):raise RuntimeError('38-frame research endpoint unavailable or shape changed')
            old=original['q'][global_idx].astype(np.float32)*original['scale'][global_idx].astype(np.float32)
            similarity=(x[:,-25:]*old).sum(-1)/(np.linalg.norm(x[:,-25:],axis=-1)*np.linalg.norm(old,axis=-1)+1e-8)
            if not np.isfinite(x).all() or not np.isfinite(energy).all() or float(similarity.mean())<.995:raise RuntimeError('3-s suffix mismatch; paired comparison invalid')
            if not np.allclose(energy[:,-25:],original['energy'][global_idx],atol=.05):raise RuntimeError('Replayed energy mismatch')
            checks.append(float(similarity.mean()));timings.append(seconds)
            x=x[:,:13];scale=np.maximum(np.abs(x).max(-1,keepdims=True)/127,1e-6)
            z['q'][local]=np.clip(np.rint(x/scale),-127,127).astype(np.int8);z['scale'][local]=scale;z['energy'][local]=energy
            # Atomic logical commit: features are flushed before the completion bitmap.
            for key in ('q','scale','energy'):z[key].flush()
            z['filled'][local]=1;z['filled'].flush()
            if offset%40==0:update(phase='Extracting 13 missing older L15 states',windows=int(z['filled'].sum()),total=n,
                elapsed_seconds=time.time()-started,paired_suffix_cosine=float(np.mean(checks)),encoder_ms_per_four=1000*float(np.mean(timings)))
    atomic(cache/'provenance.json',{'global_indices_sha256':digest(WORK.parent/'robo-v4/factorial_indices.npy'),
        'older_frames':13,'total_frames':38,'original_suffix_frames':25,'paired_suffix_cosine':float(np.mean(checks)) if checks else None,
        'encoder_window_seconds':8,'layer_one_based':15,'quantization':'per-frame symmetric int8 + fp16 scale'})
    return cache

class Resident:
    def __init__(self,frames,device='cuda:0'):
        import torch
        if frames not in (4,25,38):raise ValueError('Unsupported context')
        self.indices=v4.subset();self.device=device;self.arrays=v4.arrays();n=len(self.indices)
        self.q=torch.empty(n,1,frames,1024,dtype=torch.int8,device=device)
        self.scales=torch.empty(n,1,frames,1,dtype=torch.float16,device=device)
        if frames==38:
            cache=cache_path();extra={k:np.load(cache/(k+'.npy'),mmap_mode='r') for k in ('q','scale','energy','filled')}
            if not extra['filled'].all():raise RuntimeError('Incomplete context cache')
        for start in range(0,n,4096):
            end=min(n,start+4096);g=self.indices[start:end];original_frames=min(25,frames)
            self.q[start:end,0,-original_frames:]=torch.from_numpy(self.arrays['q'][g,-original_frames:].copy()).to(device)
            self.scales[start:end,0,-original_frames:]=torch.from_numpy(self.arrays['scale'][g,-original_frames:].copy()).to(device)
            if frames==38:
                self.q[start:end,0,:13]=torch.from_numpy(extra['q'][start:end].copy()).to(device)
                self.scales[start:end,0,:13]=torch.from_numpy(extra['scale'][start:end].copy()).to(device)
        energy=extra['energy'].copy() if frames==38 else self.arrays['energy'][self.indices,-frames:].copy()
        self.energy=torch.from_numpy(energy).to(device)
        self.target=torch.from_numpy(self.arrays['local_target'][self.indices].copy()).to(device)
        self.ids_cpu=np.asarray(self.arrays['speaker'][self.indices],np.int64)
        self.classes_cpu=np.asarray(self.arrays['activity'][self.indices],np.int64)
        self.emotions_cpu=np.asarray(self.arrays['emotion'][self.indices],np.int64)
        self.ids=torch.tensor(self.ids_cpu,device=device);self.classes=torch.tensor(self.classes_cpu,device=device)
        self.split={s:np.flatnonzero(np.isin(self.indices,g)) for s,g in v4.splits().items()}
        self.ages=onset_ages()
    def batch(self,indices):
        import torch
        i=torch.as_tensor(indices,dtype=torch.long,device=self.device)
        return self.q[i].float()*self.scales[i].float(),self.energy[i].float(),self.target[i].float(),self.classes[i],self.ids[i]
