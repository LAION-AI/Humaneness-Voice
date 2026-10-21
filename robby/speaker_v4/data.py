"""Reuse original immutable features; cache only three extra four-frame layers."""
import gzip,hashlib,io,json,re,time,unicodedata
from pathlib import Path
import numpy as np
import requests
from .common import OLD,WORK,FEATURES,atomic,guard,update

def arrays():return {p.stem:np.load(p,mmap_mode='r') for p in (OLD/'arrays').glob('*.npy')}
def splits():return {k:np.asarray(v,np.int64) for k,v in json.loads((OLD/'splits.json').read_text()).items()}
def rows():return json.loads((OLD/'sources_manifest.json').read_text())

def stable_targets():
    """Trim outliers and average emotion/language groups for TRAIN voices only.

    Prediction clips are disjoint from enrollment references. No validation or
    test probe embeddings are used to construct a training target or statistic.
    """
    saved=WORK/'stable_targets.npy'
    if saved.exists():return np.load(saved)
    vectors=np.zeros((500,192),np.float32);details=[]
    for row in rows():
        if row['split']!='train':continue
        with np.load(OLD/'sources'/row['voice']/'targets.npz') as z:x=z['vectors'].astype(np.float32)
        probes=[i for i,c in enumerate(row['clips']) if c['role']=='prediction'];y=x[probes]
        y/=np.maximum(np.linalg.norm(y,axis=-1,keepdims=True),1e-8)
        mean=y.mean(0);mean/=np.linalg.norm(mean)
        take=np.argsort(y@mean)[-max(2,int(np.ceil(.8*len(y)))):];groups={}
        for j in take:
            clip=row['clips'][probes[j]];emotion=clip['uid'].split('__E__')[-1].split('__')[0]
            groups.setdefault((emotion,clip.get('lang')),[]).append(y[j])
        centers=np.stack([np.mean(v,axis=0) for v in groups.values()]);center=centers.mean(0);center/=np.linalg.norm(center)
        vectors[row['speaker_idx']]=center
        details.append({'speaker':row['speaker_idx'],'prediction_clips':len(probes),'retained':len(take),'groups':len(groups)})
    np.save(saved,vectors);atomic(WORK/'centroid_protocol.json',{'training_identities':len(details),
        'rule':'top 80% by cosine to initial mean; equal emotion/language group weight; prediction clips only','voices':details})
    return vectors

def subset():
    path=WORK/'factorial_indices.npy'
    if path.exists():return np.load(path)
    keep=[]
    for i,row in enumerate(rows()):
        local=np.arange(i*1000,(i+1)*1000)
        if row['split']=='train':local=np.random.default_rng(9117+row['speaker_idx']).choice(local,200,replace=False)
        keep.extend(local.tolist())
    keep=np.sort(keep).astype(np.int64);np.save(path,keep)
    return keep

def metadata():
    """Map each original sampled speech window back to its public source clip."""
    saved=WORK/'clip_index.npy';infofile=WORK/'clip_info.json'
    if saved.exists() and infofile.exists():return np.load(saved),json.loads(infofile.read_text())
    original=arrays();ids=original['speaker'];info=[];lookup={}
    for row in rows():
        for clip in row['clips']:
            text=unicodedata.normalize('NFKC',' '.join(w['w'] for w in clip['words'])).casefold()
            text=re.sub(r'\W+',' ',text).strip();lookup[clip['uid']]=len(info)
            info.append({'uid':clip['uid'],'speaker':row['speaker_idx'],'role':clip['role'],
                'emotion':clip['uid'].split('__E__')[-1].split('__')[0],'text_hash':hashlib.sha256(text.encode()).hexdigest()[:24]})
    index=np.full(len(ids),-1,np.int32)
    for rownumber,row in enumerate(rows()):
        with gzip.open(OLD/'conversations'/(row['voice']+'.json.gz'),'rt') as stream:conversations=json.load(stream)
        cursor=rownumber*1000
        for conversation in conversations:
            for step in conversation['selected_320ms_steps']:
                who=int(ids[cursor]);end=step*.32
                if who>=0:
                    event=next((e for e in reversed(conversation['events']) if e.get('speaker_idx')==who and e.get('onset_s',99)<end and e.get('offset_s',0)>end-.32),None)
                    if event is None:raise RuntimeError('Source-label replay mismatch')
                    index[cursor]=lookup[event['uid']]
                    if info[index[cursor]]['speaker']!=who:raise RuntimeError('Speaker metadata mismatch')
                cursor+=1
        if cursor!=(rownumber+1)*1000:raise RuntimeError('Conversation sample count changed')
    np.save(saved,index);atomic(infofile,info);return index,info

def extract_layers():
    """Replay EXACT V3 seeds and windows for a paired layer comparison."""
    guard();keep=subset();directory=FEATURES;count=len(keep)
    specs={'q':(np.int8,(count,3,4,1024)),'scale':(np.float16,(count,3,4,1)), 'filled':(np.uint8,(count,))}
    remaining=sum(np.prod(shape)*np.dtype(dtype).itemsize for key,(dtype,shape) in specs.items() if not (directory/(key+'.npy')).exists())
    import shutil
    if shutil.disk_usage(directory).free-remaining<3*1024**3:raise RuntimeError('Layer cache would breach 3-GiB reserve')
    cache={key:np.load(directory/(key+'.npy'),mmap_mode='r+') if (directory/(key+'.npy')).exists() else
        np.lib.format.open_memmap(directory/(key+'.npy'),mode='w+',dtype=dtype,shape=shape) for key,(dtype,shape) in specs.items()}
    if cache['filled'].all():return
    from speaker_v2 import prepare_pilot as core
    from speaker_v3.overnight import decoded
    core.source_signal=lambda c:decoded(c['path'],json.dumps(c['words'],sort_keys=True))
    original=arrays();people=rows();groups={s:[r for r in people if r['split']==s] for s in ('train','validation','test')}
    lookup={int(g):i for i,g in enumerate(keep)};client=requests.Session();timings=[];checks=[];started=time.time()
    for rownumber,anchor in enumerate(people):
        relevant=keep[(keep>=rownumber*1000)&(keep<(rownumber+1)*1000)]
        if all(cache['filled'][lookup[int(g)]] for g in relevant):continue
        group=groups[anchor['split']];rank=next(i for i,r in enumerate(group) if r['speaker_idx']==anchor['speaker_idx'])
        cohort=[group[(rank+k)%len(group)] for k in range(min(12,len(group)))];waves=[];locations=[];global_ids=[]
        for repeat in sorted({int((g-rownumber*1000)//5) for g in relevant}):
            chosen=[int(g) for g in relevant if (g-rownumber*1000)//5==repeat and not cache['filled'][lookup[int(g)]]]
            if not chosen:continue
            seed=7730000+anchor['speaker_idx']*10000+repeat
            audio,*_=core.conversation(cohort,anchor,seed);steps=np.sort(np.random.default_rng(seed).choice(np.arange(5,75),5,replace=False))
            for g in chosen:
                step=steps[(g-rownumber*1000)%5];end=int(step)*5120;wave=audio[max(0,end-128000):end]
                waves.append(np.pad(wave,(max(0,128000-len(wave)),0)));locations.append(lookup[g]);global_ids.append(g)
        pending=[]
        for offset in range(0,len(waves),4):
            guard();wave=np.stack(waves[offset:offset+4]).astype('<f4');local=locations[offset:offset+4];global_idx=global_ids[offset:offset+4]
            retries=0
            while True:
                guard()
                try:
                    response=client.post('http://127.0.0.1:8793/internal/layer-features',data=wave.tobytes(),timeout=60)
                    if response.status_code in (423,429,503):time.sleep(2);continue
                    response.raise_for_status();break
                except requests.RequestException:
                    retries+=1
                    if retries>=12:raise
                    update(phase='Waiting for shared encoder',retry=retries);time.sleep(5)
            with np.load(io.BytesIO(response.content)) as z:x=z['features'].astype(np.float32);seconds=float(z['encoder_seconds'])
            # A mismatched waveform would invalidate a paired ablation.
            old=original['q'][global_idx,-4:].astype(np.float32)*original['scale'][global_idx,-4:].astype(np.float32)
            similarity=(x[:,3]*old).sum(-1)/(np.linalg.norm(x[:,3],axis=-1)*np.linalg.norm(old,axis=-1)+1e-8)
            if not np.isfinite(x).all() or float(similarity.mean())<.995:raise RuntimeError('Replayed L15 features do not match original windows')
            checks.append(float(similarity.mean()));timings.append(seconds)
            y=x[:,:3];scale=np.maximum(np.abs(y).max(-1,keepdims=True)/127,1e-6)
            cache['q'][local]=np.clip(np.rint(y/scale),-127,127).astype(np.int8);cache['scale'][local]=scale
            pending.extend(local)
            if len(pending)>=128 or offset+4>=len(waves):
                cache['q'].flush();cache['scale'].flush();cache['filled'][pending]=1;cache['filled'].flush();pending=[]
            if offset%40==0:update(phase='Extracting paired layers 4/8/12',layer_windows=int(cache['filled'].sum()),layer_total=count,
                layer_elapsed_seconds=time.time()-started,l15_replay_cosine=float(np.mean(checks)),encoder_ms_per_four=float(np.mean(timings)*1000))
    atomic(directory/'provenance.json',{'global_indices':str(WORK/'factorial_indices.npy'),'one_based_layers':[4,8,12],
        'last_frames':4,'paired_l15_cosine_mean':float(np.mean(checks)) if checks else None,'new_windows_this_run':len(timings)*4})
