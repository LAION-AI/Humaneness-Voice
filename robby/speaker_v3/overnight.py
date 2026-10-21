"""Bounded overnight dataset: 500 disjoint voices, 500k selected 320-ms steps.

Only selected 8-second causal encoder windows are computed, rather than all
frames of 100k full conversations. The live browser has priority. No microphone
audio, user embeddings, full TAR shards or raw conversation WAVs are stored.
"""
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import argparse,csv,gzip,hashlib,io,json,os,shutil,subprocess,threading,time
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq
import requests
from speaker_v2 import prepare_pilot as core
from speaker_v2.prepare_emotion_pilot import tar_pairs

ROOT=Path(__file__).resolve().parent
WORK=Path(os.environ.get('ROBO_V3_WORK',str(Path.home()/'.cache/robo-v3')))
LOCK=threading.Lock();DOWNLOAD_LIMIT=14_000_000_000;DISK_LIMIT=22_000_000_000
STATE={'phase':'initializing','started':time.time(),'speakers_total':500,'windows_target':500000}
EMOTIONS=['other','Contentment','Affection','Interest','Amusement','Elation','Anger','Fear','Sadness']


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,indent=2,allow_nan=False));temp.replace(path)


def update(**fields):
    with LOCK:
        STATE.update(fields,updated=time.time());atomic(ROOT/'progress.json',STATE)
        WORK.mkdir(parents=True,exist_ok=True)
        with (WORK/'events.jsonl').open('a') as out:out.write(json.dumps({'time':STATE['updated'],**fields},allow_nan=False)+'\n')


def guard():
    WORK.mkdir(parents=True,exist_ok=True)
    if (ROOT/'STOP').exists():raise RuntimeError('STOP requested; all completed work retained')
    if shutil.disk_usage(WORK).free<5*1024**3:raise RuntimeError('Storage floor: less than 5 GiB free; no user files deleted')


def byte_range(path,start,length):
    guard()
    with LOCK:
        ledger=WORK/'network.json';state=json.loads(ledger.read_text()) if ledger.exists() else {'reserved_bytes':0}
        state['reserved_bytes']+=length
        if state['reserved_bytes']>DOWNLOAD_LIMIT:raise RuntimeError('14 GB bounded-download budget reached; resume with reviewed budget only')
        atomic(ledger,state)
    url=core.BASE+path+f'?speaker_v3_range={start}-{start+length-1}'
    for attempt in range(8):
        try:
            with requests.get(url,headers={'Range':f'bytes={start}-{start+length-1}'},stream=True,timeout=90) as response:
                if response.status_code in (429,500,502,503,504):
                    time.sleep(min(60,float(response.headers.get('Retry-After',2**attempt))));continue
                if response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-'):
                    raise RuntimeError(f'HTTP Range not honored ({response.status_code}); refuse full archive')
                total=int(response.headers['Content-Range'].split('/')[-1]);raw=response.raw.read(length)
                return raw,total
        except requests.RequestException:
            if attempt==7:raise
            time.sleep(min(60,2**attempt))
    raise RuntimeError('Bounded download retry exhausted')


class RangeFile(io.RawIOBase):
    def __init__(self,path):self.path=path;self.pos=0;self.cache={};_,self.size=byte_range(path,0,1)
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset;return self.pos
    def read(self,n=-1):
        n=self.size-self.pos if n<0 else n;begin=self.pos;end=min(self.size,begin+n)
        if begin==end:return b''
        key=(begin,end)
        if key not in self.cache:self.cache[key]=byte_range(self.path,begin,end-begin)[0]
        self.pos=end;return self.cache[key]


def plan():
    saved=WORK/'global_split.json';legacy=core.HERE/'pilot/global_split.json'
    if saved.exists():split=json.loads(saved.read_text())['speakers']
    elif legacy.exists():split=json.loads(legacy.read_text())['speakers']
    else:
        # Reproduction does not depend on an unpublished local pilot folder.
        _,size=byte_range('stats/stats_profiles.csv',0,1)
        if size>4*1024**2:raise ValueError('Unexpectedly large profile index')
        raw,_=byte_range('stats/stats_profiles.csv',0,size)
        stats=list(csv.DictReader(io.StringIO(raw.decode())))
        voices=sorted({r['voice'] for r in stats if r['origin']=='original'})
        ordered=sorted(voices,key=lambda v:hashlib.sha256(('tyger-v2-split-v1:'+v).encode()).hexdigest())
        split={v:'validation' if i<10 else 'test' if i<20 else 'train' for i,v in enumerate(ordered)}
    ordered=sorted(split,key=lambda v:hashlib.sha256(('tyger-v2-split-v1:'+v).encode()).hexdigest())
    expected={v:'validation' if i<10 else 'test' if i<20 else 'train' for i,v in enumerate(ordered)}
    if split!=expected:raise ValueError('Global speaker split does not match immutable hash scheme')
    atomic(saved,{'speakers':split,'scheme':'sha256(tyger-v2-split-v1:voice)','dataset':core.REPO,'revision':core.REVISION})
    voices=sorted(split)
    rows=[{'voice':v,'split':split[v],'catalog_index':i,'speaker_idx':i} for i,v in enumerate(voices)]
    if len(rows)!=500 or {s:sum(r['split']==s for r in rows) for s in set(split.values())}!={'train':480,'validation':10,'test':10}:
        raise ValueError('Expected immutable 480/10/10 disjoint global speaker split')
    rows.sort(key=lambda r:({'validation':0,'test':1,'train':2}[r['split']],r['speaker_idx']))
    atomic(WORK/'plan.json',rows);return rows


def sources(row):
    directory=WORK/'sources'/row['voice'];directory.mkdir(parents=True,exist_ok=True)
    saved=directory/'clips.json'
    if saved.exists():return {**row,'clips':json.loads(saved.read_text())}
    indexes={}
    for local in (0,1):
        path=f'index/origin=original/part-{row["catalog_index"]*4+local:06d}.parquet'
        table=pq.read_table(RangeFile(path),columns=['uid','audio_bytes'])
        ids=table['uid'].to_pylist();sizes=table['audio_bytes'].to_numpy().astype(np.float64)+20000
        indexes[local]=(ids,np.cumsum(sizes)/sizes.sum())
    selected={}
    emotions=['Contentment','Affection','Interest','Amusement','Elation','Anger','Fear','Sadness']
    for emotion in emotions:
      for language in ('en','de'):
        found=None
        for local,(ids,frac) in indexes.items():
            matches=[i for i,u in enumerate(ids) if '__E__'+emotion+'__' in u and '__'+language+'.' in u]
            if matches:found=(local,float(frac[matches[len(matches)//2]]));break
        if found is None:continue
        local,fraction=found;tar=f'data/origin=original/vprof_base-{row["catalog_index"]*4+local:05d}.tar'
        _,total=byte_range(tar,0,1);length=1024*1024;start=max(0,min(int(total*fraction)//512*512,total-length)//512*512)
        blob,_=byte_range(tar,start,length);candidates=[]
        for uid,mp3,record in tar_pairs(blob,row['voice']):
            words=record.get('words') or [];meta=record.get('src_meta') or {}
            if not (6<=record.get('dur_s',0)<=25 and len(words)>=8 and float(meta.get('wer',1))<=.35):continue
            if uid in selected or (record.get('n_bursts') or 0)>1:continue
            match=('__E__'+emotion+'__' in uid and '__'+language+'.' in uid)
            quality=float((record.get('quality') or {}).get('overall_quality',0))
            candidates.append((match,quality,uid,mp3,record))
        for match,quality,uid,mp3,record in sorted(candidates,key=lambda x:(-x[0],-x[1],x[2]))[:4]:
            name=hashlib.sha256(uid.encode()).hexdigest()[:20]+'.mp3';path=directory/name;path.write_bytes(mp3)
            selected[uid]={'uid':uid,'path':str(path),'role':'prediction','duration_s':record['dur_s'],
                'words':record['words'],'lang':record.get('lang'),'quality':quality,'condition':uid.split('__')[2:-1],
                'source_tar':tar,'source_range':[start,start+length-1],'sha256':hashlib.sha256(mp3).hexdigest()}
    clips=list(selected.values())
    if len(clips)<20:raise RuntimeError('Insufficient diverse snippets for '+row['voice'])
    # Clean/soft, long, distinct enrollment utterances. Never used in soundscapes.
    clips.sort(key=lambda c:(not any(e in c['uid'] for e in ('Contentment','Affection','Interest')),-c['quality'],-c['duration_s']))
    teacher=[clips[0]]
    teacher.append(next((c for c in clips[1:] if c.get('lang')!=teacher[0].get('lang')),clips[1]))
    keep=teacher+[c for c in clips if c not in teacher][:48]
    for c in teacher:c['role']='teacher'
    atomic(saved,keep)
    with LOCK:
        STATE['sources_ready']=STATE.get('sources_ready',0)+1;STATE['updated']=time.time();atomic(ROOT/'progress.json',STATE)
    return {**row,'clips':keep}


def teacher_targets(rows):
    import torch
    from speaker_teacher import get_cached_teacher,preserve_encoder_precision
    preserve_encoder_precision();torch.set_num_threads(6)
    teacher=get_cached_teacher(os.environ.get('ROBO_ECAPA_ROOT',
        '/mnt/nvme/hf_cache/hub/models--speechbrain--spkrec-ecapa-voxceleb/snapshots/'+core.TEACHER_REVISION))
    prototypes=np.zeros((500,192),np.float32)
    for index,row in enumerate(rows):
        guard();path=WORK/'sources'/row['voice']/'targets.npz'
        if path.exists():data=np.load(path);vectors=data['vectors'];centroid=data['centroid']
        else:
            vectors=[]
            for clip in row['clips']:vectors.append(teacher.embed(core.pcm(Path(clip['path']).read_bytes())))
            vectors=np.stack(vectors);enroll=[i for i,c in enumerate(row['clips']) if c['role']=='teacher']
            centroid=vectors[enroll].mean(0);centroid/=max(np.linalg.norm(centroid),1e-9)
            np.savez_compressed(path,vectors=vectors.astype(np.float32),centroid=centroid.astype(np.float32))
        prototypes[row['speaker_idx']]=centroid
        for clip,vector in zip(row['clips'],vectors):clip['target']=vector
        update(phase='ECAPA clean full-utterance targets',teachers_ready=index+1)
    np.save(WORK/'prototypes.npy',prototypes)


@lru_cache(maxsize=1024)
def decoded(path,words):
    clip={'path':path,'words':json.loads(words)};return ORIGINAL_SIGNAL(clip)
ORIGINAL_SIGNAL=core.source_signal


def build_arrays(count):
    directory=WORK/'arrays';directory.mkdir(exist_ok=True)
    spec={'q':(np.int8,(count,25,1024)),'scale':(np.float16,(count,25,1)),
        'pool':(np.float16,(count,2,1024)),'energy':(np.float16,(count,25)),
        'local_target':(np.float16,(count,192)),'speaker':(np.int16,(count,)),
        'activity':(np.uint8,(count,)),'scenario':(np.uint8,(count,)),'emotion':(np.uint8,(count,)),'filled':(np.uint8,(count,))}
    expected=sum(np.prod(shape)*np.dtype(dtype).itemsize for dtype,shape in spec.values())
    if expected>DISK_LIMIT:raise RuntimeError('Array plan exceeds working budget')
    remaining=sum(np.prod(shape)*np.dtype(dtype).itemsize for name,(dtype,shape) in spec.items() if not (directory/(name+'.npy')).exists())
    if shutil.disk_usage(WORK).free-remaining<5*1024**3:
        raise RuntimeError('Not enough space for feature plan plus 5 GiB reserve; source clips retained')
    arrays={}
    for name,(dtype,shape) in spec.items():
        path=directory/(name+'.npy')
        arrays[name]=np.load(path,mmap_mode='r+') if path.exists() else np.lib.format.open_memmap(path,mode='w+',dtype=dtype,shape=shape)
        if arrays[name].shape!=shape:raise ValueError('Resume shape mismatch')
    atomic(WORK/'array_schema.json',{'count':count,'estimated_bytes':int(expected),'context_frames':25,
        'full_window_seconds':8,'new_frames':4,'quantization':'per-frame symmetric int8; float16 scale',
        'encoder':'frozen Nemotron L15','teacher':'clean source utterance ECAPA192 plus enrollment-only voice prototype','emotion_labels':EMOTIONS,
        'dataset_revision':core.REVISION,'teacher_revision':core.TEACHER_REVISION})
    return arrays


def corpus(rows,per_voice):
    arrays=build_arrays(len(rows)*per_voice);client=requests.Session()
    core.source_signal=lambda c:decoded(c['path'],json.dumps(c['words'],sort_keys=True))
    splits={s:[r for r in rows if r['split']==s] for s in ('train','validation','test')}
    scenarios=['clean','noise','reverb','telephone','echo']
    total_done=int(arrays['filled'].sum());initial_done=total_done;start_job=time.time();pending=[];encoder_total=0.;encoded_windows=0
    for row_number,anchor in enumerate(rows):
        guard();base=row_number*per_voice
        if arrays['filled'][base:base+per_voice].all():continue
        # Small overlapping cohorts improve decode-cache locality, while test
        # identity metrics compare against all ten held-out prototypes.
        group=splits[anchor['split']];rank=group.index(anchor);cohort=[group[(rank+k)%len(group)] for k in range(min(12,len(group)))]
        events_log=[];samples=[];metadata=[]
        for repeat in range((per_voice+4)//5):
            rng=np.random.default_rng(7730000+anchor['speaker_idx']*10000+repeat)
            seed=7730000+anchor['speaker_idx']*10000+repeat
            audio,playback,activity,people,events,scenario=core.conversation(cohort,anchor,seed)
            steps=np.sort(rng.choice(np.arange(5,75),size=5,replace=False))
            events_log.append({'seed':seed,'events':events,'selected_320ms_steps':steps.tolist()})
            for step in steps:
                index=base+len(metadata)
                if index>=base+per_voice:break
                end=int(step)*5120;begin=max(0,end-128000)
                wave=np.pad(audio[begin:end],(max(0,128000-end),0))
                activity_step=activity[:,end-5120:end];occupancy=activity_step.mean(1)
                overlap=float((activity_step.sum(0)>=2).mean());who=int(np.argmax(occupancy));speaker=-1;label=1;target=np.zeros(192,np.float32);emotion=0
                if overlap>=.2 or (occupancy>.3).sum()>1:label=2;speaker=-2
                elif occupancy[who]>=.5:
                    label=0;speaker=people[who]['speaker_idx']
                    event=next((e for e in reversed(events) if e.get('speaker_idx')==speaker and e.get('onset_s',99)<end/16000 and e.get('offset_s',0)>end/16000-.32),None)
                    if event is not None:
                        target=next(c['target'] for c in people[who]['clips'] if c['uid']==event['uid'])
                        name=event['uid'].split('__E__')[-1].split('__')[0]
                        emotion=EMOTIONS.index(name) if name in EMOTIONS else 0
                    else:label=2;speaker=-2
                metadata.append((index,speaker,label,scenarios.index(scenario),target,emotion))
                samples.append(wave)
        for offset in range(0,len(samples),4):
            ids=metadata[offset:offset+4]
            if all(arrays['filled'][item[0]] for item in ids):continue
            guard();wave=np.stack(samples[offset:offset+4]).astype('<f4')
            while True:
                guard()
                try:
                    response=client.post('http://127.0.0.1:8793/internal/context-features',data=wave.tobytes(),timeout=90)
                    if response.status_code in (423,429,503):time.sleep(2);continue
                    response.raise_for_status();break
                except requests.RequestException:
                    update(phase='Waiting for local encoder',windows_ready=total_done);time.sleep(10)
            z=np.load(io.BytesIO(response.content),allow_pickle=False);x=z['features'].astype(np.float32)
            encoder_total+=float(z['encoder_seconds']);encoded_windows+=len(ids)
            scales=np.maximum(np.max(np.abs(x),axis=-1,keepdims=True)/127,1e-6)
            q=np.clip(np.rint(x/scales),-127,127).astype(np.int8)
            for j,(index,speaker,label,scenario,target,emotion) in enumerate(ids):
                arrays['q'][index]=q[j];arrays['scale'][index]=scales[j];arrays['pool'][index]=z['pool'][j]
                arrays['energy'][index]=z['energy'][j];arrays['local_target'][index]=target
                arrays['speaker'][index]=speaker;arrays['activity'][index]=label;arrays['scenario'][index]=scenario
                arrays['emotion'][index]=emotion
            pending.extend(item[0] for item in ids)
            # Commit blocks, not every four rows: bounded fsync overhead. A
            # crash recomputes only the last uncommitted block (<=128 windows).
            if len(pending)>=128 or offset+4>=len(samples):
                for key,value in arrays.items():
                    if key!='filled':value.flush()
                arrays['filled'][pending]=1;arrays['filled'].flush()
                total_done=int(arrays['filled'].sum());pending=[]
            if offset%40==0:update(phase='Frozen L15 selected-step extraction',windows_ready=total_done,
                feature_seconds_last_batch=float(z['encoder_seconds']),elapsed_feature_seconds=time.time()-start_job,
                encoder_windows_per_second=encoded_windows/max(encoder_total,1e-6),
                estimated_remaining_encoder_hours=(len(arrays['filled'])-total_done)*encoder_total/max(encoded_windows,1)/3600,
                feature_wall_windows_per_second=(total_done-initial_done)/max(time.time()-start_job,1e-6))
            time.sleep(.015)
        folder=WORK/'conversations';folder.mkdir(exist_ok=True)
        with gzip.open(folder/(anchor['voice']+'.json.gz'),'wt') as out:json.dump(events_log,out)
        update(phase='Frozen L15 selected-step extraction',voices_featured=row_number+1,windows_ready=total_done)
    split_ids={s:[i for n,r in enumerate(rows) if r['split']==s for i in range(n*per_voice,(n+1)*per_voice)] for s in splits}
    atomic(WORK/'splits.json',split_ids)
    return arrays


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--per-voice',type=int,default=1000);parser.add_argument('--stage',default='all',choices=['all','sources','teacher','features','train']);args=parser.parse_args()
    guard();rows=plan();update(phase='Bounded selective source downloads',windows_target=len(rows)*args.per_voice)
    if args.stage in ('all','sources'):
        STATE['sources_ready']=sum((WORK/'sources'/r['voice']/'clips.json').exists() for r in rows)
        update(sources_ready=STATE['sources_ready'])
        with ThreadPoolExecutor(max_workers=3) as pool:rows=list(pool.map(sources,rows))
    else:rows=[{**r,'clips':json.loads((WORK/'sources'/r['voice']/'clips.json').read_text())} for r in rows]
    atomic(WORK/'sources_manifest.json',rows)
    if args.stage=='sources':update(phase='Sources complete');return
    teacher_targets(rows)
    if args.stage=='teacher':update(phase='Teacher targets complete');return
    if args.stage in ('all','features'):corpus(rows,args.per_voice)
    if args.stage=='features':update(phase='Features complete');return
    from .train import run_four
    run_four(WORK,update)


if __name__=='__main__':
    try:main()
    except Exception as exc:
        update(phase='Stopped safely; completed work is resumable',error=type(exc).__name__+': '+str(exc)[:600])
        raise
