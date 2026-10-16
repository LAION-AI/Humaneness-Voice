"""All-catalogue quality contrasts and paired reference/no-reference study.

All conditioning arms use identical pure acting cues and exact transcripts.
Ten seeded repetitions contain ten different emotion/language conditions each;
batches4/4/2 therefore give TEN distinct actual seeds per identical condition.
The common no-reference arm is reusable across voices, not regenerated per voice.
"""
import argparse
import asyncio
from contextlib import asynccontextmanager
import hashlib
import json
from pathlib import Path
import subprocess
import threading
import time
import traceback
import urllib.request

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import numpy as np
import soundfile as sf
import torch

import settings
import dynamic_references as references
import mira_activation_analysis as analysis
from mira_activation_study import Quality, EMOTION_KEYS, write_json
from activation_steering import ActivationCollector, research_stream_frames
from semantic_steering import SemanticVectorTransform
from prompting import build, parse_literal

ROOT = settings.HERE / 'catalogue-steering-study'
BANK = settings.HERE / 'speaker-steering'
POINTS = ('layer_16', 'semantic_global', 'semantic_local')
LOCK = threading.Lock()
SCORER = {}


def voice_catalog():
    return references.catalog()['voices']


def tasks(phase):
    source = 'screen' if phase == 'collection' else 'validation'
    repeats = 10 if phase == 'collection' else 1
    seed_base = 6100000 if phase == 'collection' else 7200000
    output = []
    for repeat in range(repeats):
        for i, emotion in enumerate(analysis.EMOTIONS):
            for language in ('en', 'de'):
                position = 2*i + (language == 'de')
                text = analysis.TEXTS[source][language][i]
                prompt = 'CAPTION: ' + analysis.EMOTIONS[emotion] + '\nTRANSCRIPT: ' + json.dumps(text, ensure_ascii=False)
                output.append({'block': repeat*10+position, 'condition_id':emotion+'-'+language,
                    'repeat':repeat, 'emotion':emotion, 'language':language, 'text':text,
                    'prompt':prompt, 'seed':seed_base + repeat*1009 + (position//4)*4,
                    'reference_identity':False})
    return output


def freeze():
    ROOT.mkdir(exist_ok=True)
    voices = voice_catalog()
    plan = {'version':1, 'checkpoint_sha256':settings.FULL_FT_SHA256,
        'voices':voices, 'collection_conditions':tasks('collection'),
        'validation_conditions':tasks('validation'),
        'collection_n_per_reference':100, 'common_no_reference_n':100,
        'pairing':'10 identical text/emotion/language conditions, ten actual seeds each. Each repeat batches DIFFERENT conditions 4/4/2. Same batch companions, seed and literal prompt in both reference arms.',
        'prompt':'No gender, timbre, age, identity description or reference-identity cue. Exact same pure emotion caption/transcript in reference/no-reference arms.',
        'settings':{'cfg':2.,'temperature':1.,'adapters':[],'alignment':False,
            'conversion':False,'max_frames':256,'batch_size':4,'reference_seconds':15},
        'capture':'Conditional predictor positions only, commit iff actual frame emitted, first final-prefill predictor included. Means layer16, postnormsemantic1024, projectedsemantic2560; equal sample then condition weight.',
        'two_directions':'Quality contrast = high-similarity minus low-similarity takes, with reference in BOTH pools. Reference-effect contrast = mean with-reference minus common without-reference, WITHOUT quality filtering. Kept separately.',
        'quality_selection':'Strict>.50 with both pools >=2 for all5emotions and>=10high, else>.25 with same coverage; joint language/emotion weighting only if all10cells both pools>=2, else emotion balancing with disclosed language confound. Deterministic top/bottom2 within each joint cell if neither qualifies; exploratory label.',
        'validation':'Eachvoice baseline plus quality layer16 alpha.75 on ten new task/seed blocks. Also reference-effect layer16 .25/.75 and semantic_local .25/.75 for Mira,Selma,Idris on those same fresh blocks. No post-hoc alpha selection for UI.',
        'ui':'Mira keeps exact previously validated layer16 quality contrast. Othercataloguevoices get quality contrasts derived here. Checkbox defaultON for valid vectors; negative quality-proxy screens remain tryable with explicit warnings, not advertised as improvements. Custom/no-ref/LoRA/wronghash bypass.',
        'limitations':['Reference/no-reference trajectories diverge; means also include token-history, prosody, duration and generic reference-presence effects. Not a pure causal speaker vector.',
            'Common unconditioned control reused across all voices; voice effects are not independent controls.',
            'Only ten fresh validation tasks per candidate and shared batch RNG. Learned quality proxies do not guarantee clear sound.',
            'No MLP training in this study; speaker-embedding-to-direction generalization would require separate speaker-disjoint training/validation and multiple references per speaker.']}
    analysis.frozen_json(ROOT/'design.json',plan)
    return plan


def read_results(phase=None,voice=None):
    out=[json.loads(p.read_text()) for p in sorted((ROOT/'results').glob('*.json'))]
    return [r for r in out if (phase is None or r['phase']==phase) and (voice is None or r['voice']==voice)]


def make_rows(voice,phase,condition='baseline',intervention=None):
    return [{**task,'voice':voice,'phase':phase,'condition':condition,
        'reference_present':voice!='none','intervention':intervention or {'points':[],'alpha':0},
        'id':f'{phase}-{voice}-{condition}-{task["block"]}'} for task in tasks(phase)]


def save_audio(id_,wave):
    folder=ROOT/'audio';folder.mkdir(exist_ok=True)
    wav=folder/(id_+'.wav');mp3=wav.with_suffix('.mp3')
    sf.write(wav,np.asarray(wave,dtype=np.float32),48000,subtype='FLOAT')
    subprocess.run(['ffmpeg','-nostdin','-v','error','-y','-i',str(wav),'-map_metadata','-1',
        '-ac','1','-ar','48000','-c:a','libmp3lame','-b:a','96k',str(mp3)],check=True,timeout=30)
    return str(mp3.relative_to(ROOT))


class RemoteQuality:
    def __init__(self,url):self.url=url.rstrip('/')
    def score(self,waves,rows):
        paths=[];(ROOT/'audio').mkdir(exist_ok=True)
        for row,wave in zip(rows,waves):
            path=ROOT/'audio'/(row['id']+'.wav');sf.write(path,wave,48000,subtype='FLOAT');paths.append(str(path))
        req=urllib.request.Request(self.url+'/score',data=json.dumps({'rows':rows,'paths':paths}).encode(),headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(req,timeout=180) as f:return json.load(f)['scores']


@torch.inference_mode()
def run(voices,device,score_url,phase='collection'):
    settings.DEVICE=device;settings.SCORE_DEVICE='cuda:1';settings.COMPILE_GLOBAL=False
    settings.FAST_INFERENCE=True;torch.set_num_threads(4);torch.cuda.set_device(device)
    from engine import SmallEngine
    from dpo_adapters import AdapterSelection
    engine=SmallEngine();engine.configure_adapters(AdapterSelection())
    engine.configure_speaker_steering(None,{'active':False,'reason':'offline_research'})
    quality=RemoteQuality(score_url)
    design=freeze()
    for voice in voices:
        voice_reference=None if voice=='none' else references.fixed_clip(voice)[0]
        groups=[('baseline',{'points':[],'alpha':0})]
        if phase=='validation':
            groups.append(('quality-layer16-a0.75',{'points':['layer_16'],'alpha':.75,'file':str(BANK/(voice+'.npz')),'key':'layer_16__contrast'}))
            if voice in ('emolia_c1699','emolia_c0542','anime_088'):
                groups += [(f'reference-{point}-a{alpha}',{'points':[point],'alpha':alpha,
                    'file':str(ROOT/'reference-vectors'/(voice+'.npz')),'key':point+'__reference_difference'})
                    for point in ('layer_16','semantic_local') for alpha in (.25,.75)]
                extension=json.loads((ROOT/'reference-effect-validation-extension.json').read_text())
                assert voice in extension['voices']
                groups.append(('no-reference-baseline',{'points':[],'alpha':0,'reference_present':False}))
                groups += [(f'reference-without-{point}-a0.75',{'points':[point],'alpha':.75,
                    'file':str(ROOT/'reference-vectors'/(voice+'.npz')),'key':point+'__reference_difference',
                    'reference_present':False}) for point in ('layer_16','semantic_local')]
        for name,intervention in groups:
            rows=make_rows(voice,phase,name,intervention)
            reference=voice_reference if intervention.get('reference_present',True) else None
            for row in rows:row['reference_present']=reference is not None
            vector=None
            if intervention['points']:
                with np.load(intervention['file'],allow_pickle=False) as data:vector=torch.from_numpy(data[intervention['key']].copy())
            for offset in range(0,len(rows),10):
                repeat_rows=rows[offset:offset+10]
                for start in (0,4,8):
                    batch=repeat_rows[start:start+4]
                    paths=[ROOT/'results'/(r['id']+'.json') for r in batch]
                    if all(p.exists() for p in paths):continue
                    if any(p.exists() for p in paths):raise RuntimeError('Partial research batch: stop for explicit recovery')
                    collector=ActivationCollector(len(batch)) if phase=='collection' else None
                    specs=[build(parse_literal(r['prompt'],'caption'),'caption',r['language'],literal=r['prompt'],reference_identity=False) for r in batch]
                    point=intervention['points'][0] if intervention['points'] else None
                    sem=SemanticVectorTransform(vector,point=point,alpha=intervention['alpha']) if point and point.startswith('semantic') else None
                    def stream(branches,p,chunk_frames,guidance=2.,min_frames=0,graph=False):
                        return research_stream_frames(engine,branches,p,chunk_frames,guidance=guidance,
                            collector=collector,collect_layers=[16],layer_vectors={16:vector} if point=='layer_16' else {},
                            factors=intervention['alpha'],semantic_transform=sem,min_frames=min_frames,graph=False)
                    engine.fast.stream_frames=stream
                    info={'device':device,'checkpoint':engine.checkpoint_info,'adapters':[],
                        'cfg':2.,'temperature':1.,'batch_seed':batch[0]['seed'],'batch_size':len(batch),
                        'batch_ids':[r['id'] for r in batch], 'reference_sha256':hashlib.sha256(reference.read_bytes()).hexdigest() if reference else None,
                        'vector_sha256':hashlib.sha256(Path(intervention['file']).read_bytes()).hexdigest() if point else None}
                    tick=time.monotonic()
                    try:
                        waves=engine.generate_many(specs,guidance=2.,seed=batch[0]['seed'],reference_path=reference,
                            temperature=1.,streaming=False,batch_size=len(batch),max_frames=256)
                        values=collector.export() if collector else None
                        scores=quality.score(waves,batch)
                        for i,(row,path,wave,score) in enumerate(zip(batch,paths,waves,scores)):
                            count=len(engine.last_codes[i])
                            if values:
                                assert int(values['counts'][i])==count
                                assert set(values['means'])==set(POINTS)
                                (ROOT/'activations').mkdir(exist_ok=True)
                                np.savez_compressed(ROOT/'activations'/(row['id']+'.npz'),
                                    **{k:t[i].numpy() for k,t in values['means'].items()},frame_count=np.array(count))
                            audio=save_audio(row['id'],wave)
                            write_json(path,{**row,**info,**score,'audio_path':audio,'generated_frames':count,
                                'acoustic_tokens':count*12,'status':'ok','failed':False,'batch_wall_seconds':time.monotonic()-tick})
                    except Exception as e:
                        for row,path in zip(batch,paths):
                            if not path.exists():write_json(path,{**row,**info,'failed':True,'status':'failed','error':str(e),'traceback':traceback.format_exc()})
                        raise
                    print(phase,voice,name,'completed',offset+start+len(batch),'/',len(rows),'seconds',round(time.monotonic()-tick,2),flush=True)
    print('Worker complete',voices,phase,flush=True)


def initialize_scores():
    settings.SCORE_DEVICE='cuda:1';torch.set_num_threads(4);torch.cuda.set_device('cuda:1')
    from engine import Scorers
    from adaptive_reward import RewardPlan
    q=Quality.__new__(Quality);q.device='cuda:1';q.base=Scorers()
    q.base.speakers.prepare('ecapa');q.plan=RewardPlan(terms=[{'dimension':v} for v in EMOTION_KEYS.values()])
    q.base.dimensions.prepare(q.plan);q.base.dimensions.prepare_keys({'vn:ESTH'})
    targets={}
    for voice in voice_catalog():
        fixed=references.fixed_clip(voice['id'])[0]
        anchor=Path(references.profiles()[voice['id']]['anchor'])
        target,_=q.base.speakers.target(fixed,'ecapa');other,_=q.base.speakers.target(anchor,'ecapa')
        targets[voice['id']]=(target,other)
    SCORER.update(quality=q,targets=targets,ready=True)


def derive_vectors():
    """Publish complete quality banks only; paired reference effects stay separate."""
    import catalogue_steering_analysis as stats
    BANK.mkdir(exist_ok=True)
    manifest_path=BANK/'manifest.json'
    manifest=json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'version':1,'checkpoint_sha256':settings.FULL_FT_SHA256,'layer':16,'alpha':.75,
        'voices':{v['id']:{'name':v['name'],'status':'preparing','reason':'collection_in_progress',
            'reference_sha256':v['reference_sha256']} for v in voice_catalog()}}
    # Preserve the exact Mira direction and independent validation from the first study.
    old=settings.HERE/'mira-activation-study'
    mira='emolia_c1699'
    if mira not in manifest['voices'] or manifest['voices'][mira]['status']=='preparing':
        with np.load(old/'vectors/selected.npz',allow_pickle=False) as data:
            np.savez_compressed(BANK/(mira+'.npz'),layer_16__contrast=data['layer_16__contrast'])
        valid=json.loads((old/'validation-summary.json').read_text())['conditions']
        validation=next(c for c in valid if c['condition']=='contrast-layer_16-a0.75')
        item=next(v for v in voice_catalog() if v['id']==mira)
        manifest['voices'][mira]={'name':item['name'],'status':'ready','reference_sha256':item['reference_sha256'],
            'vector_file':mira+'.npz','vector_key':'layer_16__contrast','dimension':1024,
            'sha256':hashlib.sha256((BANK/(mira+'.npz')).read_bytes()).hexdigest(),
            'alpha':.75,'validation':validation,'selection':json.loads((old/'vectors/metadata.json').read_text())['selected'],
            'source':'Exact original Mira high-minus-low direction, not recomputed from pure-caption reference-effect arms.'}
    control=read_results('collection','none')
    all_arrays={}
    for row in read_results('collection'):
        if row.get('failed'):continue
        path=ROOT/'activations'/(row['id']+'.npz')
        if not path.is_file():continue
        with np.load(path,allow_pickle=False) as data:all_arrays[row['id']]={p:data[p].copy() for p in POINTS}
    for voice in voice_catalog():
        id_=voice['id'];rows=read_results('collection',id_)
        if len(rows)!=100 or any(r.get('failed') for r in rows):continue
        if id_!=mira and manifest['voices'][id_]['status']=='preparing':
            vector,metadata=stats.select_vector(rows,all_arrays)
            np.savez_compressed(BANK/(id_+'.npz'),layer_16__contrast=vector)
            manifest['voices'][id_].update(status='unvalidated',vector_file=id_+'.npz',
                vector_key='layer_16__contrast',dimension=1024,
                sha256=hashlib.sha256((BANK/(id_+'.npz')).read_bytes()).hexdigest(),alpha=.75,
                selection=metadata,allow_activation=True,
                warning='Experimenteller Layer-16-Vektor: noch kein unabhängiger Qualitätsnachweis für diese Stimme.')
        if len(control)==100 and not any(r.get('failed') for r in control):
            result=stats.paired_reference_vector(rows,control,all_arrays)
            directory=ROOT/'reference-vectors';directory.mkdir(exist_ok=True)
            archive={p+'__reference_difference':v for p,v in result['vectors'].items()}
            archive.update({p+'__reference_present_mean':v['with'] for p,v in result['arm_means'].items()})
            archive.update({p+'__reference_absent_mean':v['without'] for p,v in result['arm_means'].items()})
            destination=directory/(id_+'.npz')
            if destination.exists():
                with np.load(destination,allow_pickle=False) as saved:
                    if set(saved.files)!=set(archive) or any(not np.array_equal(saved[k],v) for k,v in archive.items()):
                        raise RuntimeError('Frozen reference vector changed: '+id_)
            else:
                np.savez_compressed(destination,**archive)
            write_json(directory/(id_+'.json'),result['metadata'])
        valid=read_results('validation',id_)
        if len([r for r in valid if r['condition'] in ('baseline','quality-layer16-a0.75')])==20:
            audit=stats.assessment(valid,condition='quality-layer16-a0.75')
            if id_==mira and 'prior_validation' not in manifest['voices'][id_]:
                manifest['voices'][id_]['prior_validation']=manifest['voices'][id_]['validation']
            manifest['voices'][id_]['validation']=audit
            failed=audit.get('gate_reasons',[])
            manifest['voices'][id_]['status']='quality_failed' if failed else 'ready' if audit.get('confirmed_gain') else 'experimental'
            manifest['voices'][id_]['warning']=('Qualitäts-/Sprecher-Screen nicht bestanden: '+', '.join(failed)+'. Vektor bleibt zum Ausprobieren aktiv; Checkbox zum Vergleich ausschalten.' if failed else
                None if audit.get('confirmed_gain') else 'Kleiner Pilot: Wirkung dieser Stimme noch nicht eindeutig bestätigt; Qualität kann pro Take schwanken.')
            manifest['voices'][id_]['allow_activation']=True
    # Atomically replace the manifest: ongoing demo requests never see partial JSON.
    tmp=BANK/'manifest.build.json';write_json(tmp,manifest);tmp.replace(manifest_path)
    print('Voice bank:',{k:v['status'] for k,v in manifest['voices'].items()},flush=True)
    return manifest


@asynccontextmanager
async def lifespan(app):
    await asyncio.to_thread(initialize_scores);yield


score_app=FastAPI(lifespan=lifespan)


class ScoreInput(BaseModel):
    rows:list[dict]=Field(min_length=1,max_length=4)
    paths:list[str]=Field(min_length=1,max_length=4)


@score_app.get('/state')
def score_state():return {'ready':SCORER.get('ready',False),'device':'cuda:1'}


@score_app.post('/score')
def score_batch(body:ScoreInput):
    if len(body.rows)!=len(body.paths):raise HTTPException(400,'Mismatched score rows')
    voice=body.rows[0]['voice']
    if any(r['voice']!=voice for r in body.rows):raise HTTPException(400,'Mixed voice score batch')
    key=voice if voice!='none' else 'emolia_c1699'
    if key not in SCORER['targets']:raise HTTPException(400,'Unknown catalogue voice')
    waves=[]
    for name in body.paths:
        path=Path(name).resolve()
        if not path.is_relative_to((ROOT/'audio').resolve()) or path.suffix!='.wav' or not path.is_file():raise HTTPException(400,'Only study WAVs')
        wave,rate=sf.read(path,dtype='float32')
        if rate!=48000 or wave.ndim!=1 or not 24000<=len(wave)<=256*3840+48000:raise HTTPException(400,'Unsupported wave')
        waves.append(wave)
    with LOCK:
        q=SCORER['quality'];q.target,q.anchor=SCORER['targets'][key]
        scores=q.score(waves,body.rows)
        for row in scores:row['ecapa_target']=key;row['conditioning_absent']=voice=='none'
        return {'scores':scores}


def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['design','collection','validation','vectors'])
    p.add_argument('--voices');p.add_argument('--device',default='cuda:0')
    p.add_argument('--score-url',default='http://127.0.0.1:8795')
    args=p.parse_args();plan=freeze()
    if args.action=='vectors':derive_vectors()
    elif args.action!='design':run(args.voices.split(','),args.device,args.score_url,args.action)


if __name__=='__main__':main()
