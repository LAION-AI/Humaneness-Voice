"""Frozen CPU students for live, session-local approximate voice matching.

The 320-ms model is selected by validation only, never by a live user's name or
test performance. Long contexts use completed V5 results when available and
otherwise explicitly fall back to the completed V3 two-second baseline.
"""
from functools import lru_cache
import json,os,threading
from pathlib import Path
import numpy as np
import torch
from speaker_v4.train import MixedStudent

CACHE=Path.home()/'.cache'
LOCK=threading.RLock()

def objective(v):return v['macro_speaker_accuracy']-.5*v['eer']+.1*v['source_utterance_teacher_cosine']

def registry():
    supplied=os.environ.get('ROBO_SPEAKER_MODEL_DIR')
    if supplied:
        root=Path(supplied);index=json.loads((root/'index.json').read_text())
        chosen=next(r for r in index['models'] if r['validation_selected_default'])
        # Published index paths are relative to the repository, not models/.
        fast=root/Path(chosen['file']).name
        models={'fast':{'path':str(fast),'frames':4,'mix_layers':True,'label':'V4 layer mix · 320 ms'},
            'long':{'path':str(root/'v3-l15-2000ms-baseline.pt'),'frames':25,'mix_layers':False,'label':'V3 L15 · 2-s baseline (V5 pending)'}}
        for key,spec in index.get('long_models',{}).items():
            if key in ('long','long3'):models[key]={**spec,'path':str(root/Path(spec['file']).name)}
        return models
    v4=CACHE/'robo-v4';d=json.loads((v4/'results.json').read_text())
    if not d.get('finished'):raise RuntimeError('No completed 320-ms study available')
    chosen=max((r for r in d['factorial'] if r['arm']==d['selected_factorial_arm']),key=lambda r:objective(r['validation']))
    output={'fast':{'path':str(v4/'models'/chosen['checkpoint']),'frames':4,'mix_layers':bool(chosen['mix_layers']),
        'label':f'V4 layer mix · 320 ms · seed {chosen["seed"]}'},
        'long':{'path':str(CACHE/'robo-v3/models/query_gru-best.pt'),'frames':25,'mix_layers':False,
        'label':'V3 L15 · 2-s baseline (V5 optimization pending)'}}
    path=CACHE/'robo-v5/results.json'
    if path.exists():
        v5=json.loads(path.read_text())
        if v5.get('finished'):
            for frames in (25,38):
                recipe=v5['selected_recipe_by_context'][str(frames*80)]
                row=max((r for r in v5['runs'] if r['frames']==frames and r['recipe']==recipe),key=lambda r:objective(r['validation']))
                key='long' if frames==25 else 'long3'
                output[key]={'path':str(CACHE/'robo-v5/models'/row['checkpoint']),'frames':frames,'mix_layers':False,
                    'label':f'V5 optimized · {frames*80} ms · {recipe}'}
    return output

def available():
    try:return {k:{x:v for x,v in entry.items() if x!='path'} for k,entry in registry().items() if Path(entry['path']).is_file()}
    except (OSError,ValueError,KeyError,RuntimeError):return {}

@lru_cache(maxsize=6)
def load_model(path,mix_layers):
    with LOCK:
        saved=torch.load(path,map_location='cpu',weights_only=False)
        model=MixedStudent(mix_layers).eval();model.load_state_dict(saved['model'])
        for p in model.parameters():p.requires_grad_(False)
        return model

class LivePredictor:
    def __init__(self,long_context='long'):
        self.models=registry();self.use_long=long_context is not None
        self.long_context=long_context if long_context in self.models else 'long'
        self.fast=load_model(self.models['fast']['path'],self.models['fast']['mix_layers'])
        spec=self.models[self.long_context];self.long=load_model(spec['path'],False);self.frames=spec['frames']
    def reset(self):pass  # Each observation comes from the current causal encoder window.
    @torch.inference_mode()
    def predict(self,event):
        energy=np.asarray(event['energy_db'],np.float32)
        if not len(energy) or float(energy.max())<-52:
            return {'kind':'silence','embedding':None,'confidence':0.,'long_embedding':None}
        layers=event.get('layer_features')
        if layers is None or np.shape(layers)!=(4,4,1024):
            return {'kind':'warming','embedding':None,'confidence':0.,'long_embedding':None}
        x=torch.from_numpy(np.asarray(layers,np.float32))[None];e=torch.from_numpy(energy[-4:].copy())[None]
        with LOCK:embedding,activity=self.fast(x,e)
        probs=activity.softmax(-1)[0].numpy();kind=('speech','silence','overlap')[int(probs.argmax())]
        if probs[0]<.60 and kind=='speech':kind='uncertain'
        long_vector=None;context=event.get('context_features');db=event.get('context_energy_db')
        if self.use_long and kind=='speech' and context is not None and len(context)>=12 and db is not None:
            n=min(len(context),len(db),self.frames)
            if n>=12:
                h=np.asarray(context[-n:],np.float32);d=np.asarray(db[-n:],np.float32)
                with LOCK:b,_=self.long(torch.from_numpy(h.copy())[None,None],torch.from_numpy(d.copy())[None])
                long_vector=b[0].numpy().copy()
        return {'kind':kind,'embedding':embedding[0].numpy().copy() if kind=='speech' else None,
            'raw_embedding':embedding[0].numpy().copy(),
            'long_embedding':long_vector,'confidence':float(probs[0]),'activity_probabilities':probs.tolist(),
            'context_ms':320,'long_context_ms':self.frames*80,'model':self.models['fast']['label']}
