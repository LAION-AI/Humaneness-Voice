"""Pinned Humaneness Ears quality estimates; no teachers, decoder or audio upload.

Base = Whisper Base; Medium = Whisper Small. Reuse both encoders on scorer GPU.
Scores are decoded with the release's frozen training mean/std. Fixed native
scales bring AudioBox and DNSMOS onto [0,1], not per-batch min/max ranks.
"""
import hashlib
import importlib.util
import json
from pathlib import Path
import threading
import time
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

import settings

REPO='laion/humaneness-ears-base-medium'
REVISION='818506970d93c809a3295a0a02c40dee8ff2bfc3'
WEIGHT_HASHES={'base':'c4d9e05295020cd0fab4e44ee10aae3bdf934b0a7700fea05bb271ffa2e410b4',
    'medium':'00f30de0f7d656d2a1e86bdc629673ff8b576532e1263b02bcebdfc2763e54f1'}
CODE_HASH='7fa88f3c73fbedfd2b805a3b0007edd100da98064ac4dcf0c76659b0840bb7c0'
METRICS={'ce':('audiobox:CE','Content Enjoyment',1.,10.),
    'pq':('audiobox:PQ','Production Quality',1.,10.),
    'cu':('audiobox:CU','Content Usefulness',1.,10.),
    'pc':('audiobox:PC','Production Complexity',1.,10.),
    'mos':('dnsmos:P808_MOS','DNSMOS P.808 MOS',1.,5.),
    'ovrl':('dnsmos:OVRL','DNSMOS Overall',1.,5.)}


class RankingOptions(BaseModel):
    model_config=ConfigDict(extra='forbid')
    mode: Literal['ears_mix','ears_only','legacy']='ears_mix'
    model: Literal['base','medium']='medium'
    metric: Literal['ce','pq','cu','pc','mos','ovrl']='ce'
    weight: float=Field(default=.5,ge=0,le=1,allow_inf_nan=False)


def normalized(raw,metric):
    if not np.isfinite(raw):raise ValueError('Nonfinite Humaneness Ears score')
    _,_,lo,hi=METRICS[metric]
    return float(np.clip((raw-lo)/(hi-lo),0,1))


def combine_ears(item,report,options):
    """Pure sorting means exactly that: no hidden speaker/burst/WER mixture."""
    item['ears']=report
    item['previous_quality']=item['quality']
    value=report.get('normalized')
    if value is None:
        item.update(quality=0.,reward=0.,reward_model=options.mode,ranking_available=False)
        return item
    if options.mode=='ears_only':
        # Sort by raw regression output, preserving order even outside the
        # display scale. Do not silently include WER or clamp away differences.
        item.update(quality=value,reward=float(report['raw']),ranking_available=True,
            reward_model='ears_only',ranking_gate=1.,ears_weight=1.,clap_weight=0.)
    else:
        quality=(1-options.weight)*item['n_clap']+options.weight*value
        item.update(quality=quality,reward=quality*item['gate'],ranking_available=True,
            reward_model='ears_mix',ranking_gate=item['gate'],ears_weight=options.weight,
            clap_weight=1-options.weight)
    return item


def catalog():
    return {'repo':REPO,'revision':REVISION,'default':RankingOptions().model_dump(),
        'models':[{'id':'base','name':'Base · Whisper Base · 26M'},
                  {'id':'medium','name':'Medium · Whisper Small · 97M'}],
        'metrics':[{'id':k,'key':v[0],'name':v[1],'min':v[2],'max':v[3]} for k,v in METRICS.items()],
        'normalization':'clip((raw-min)/(max-min),0,1); VoiceCLAP uses existing fixed normalization',
        'long_audio':'Non-overlapping <=30s windows, duration-weighted raw-score mean; no discarded tail',
        'note':'Distilled regression proxies, not original AudioBox/DNSMOS models or human MOS.'}


class EarsModels:
    def __init__(self,device=None):
        self.device=str(device or settings.SCORE_DEVICE)
        self.lock=threading.RLock();self.models={};self.last_timing={}
        self.root=None

    def load(self,size):
        import torch
        from huggingface_hub import snapshot_download
        from safetensors.torch import load_file
        from transformers import WhisperFeatureExtractor
        with self.lock:
            if size in self.models:return
            if size not in WEIGHT_HASHES:raise ValueError('Unknown Humaneness Ears variant')
            root=Path(snapshot_download(REPO,revision=REVISION,token=False,
                cache_dir=str(settings.HERE/'model-cache/ears/hub'),allow_patterns=[
                    'model.py','classes.json','training_normalization.json','display_normalization.json',
                    'release_manifest.json','LICENSE','NOTICE.md',size+'/config.json',
                    size+'/preprocessor_config.json',size+'/model.safetensors']))
            manifest=json.loads((root/'release_manifest.json').read_text())
            files=['model.py','classes.json','training_normalization.json','display_normalization.json',
                size+'/config.json',size+'/preprocessor_config.json',size+'/model.safetensors']
            for relative in files:
                digest=hashlib.sha256()
                with (root/relative).open('rb') as file:
                    for block in iter(lambda:file.read(8*1024*1024),b''):digest.update(block)
                expected=CODE_HASH if relative=='model.py' else WEIGHT_HASHES[size] if relative.endswith('safetensors') else manifest['files'][relative]['sha256']
                if digest.hexdigest()!=expected:raise RuntimeError('Humaneness Ears integrity check failed: '+relative)
            spec=importlib.util.spec_from_file_location('_small_demo_ears_release',root/'model.py')
            module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
            classes=json.loads((root/'classes.json').read_text())['names']
            norm=json.loads((root/'training_normalization.json').read_text())
            display=json.loads((root/'display_normalization.json').read_text())
            if norm['score_names']!=[r['name'] for r in display['targets']]:
                raise RuntimeError('Humaneness Ears target order mismatch')
            if not all(v[0] in norm['score_names'] for v in METRICS.values()):
                raise RuntimeError('Humaneness Ears quality heads missing')
            # Isolate random constructor initialization from TTS samplers.
            with torch.random.fork_rng(devices=[]):
                model=module.LayeredMultiTaskWhisper(root/size,len(classes),initialize_pretrained=False)
            model.load_state_dict(load_file(root/size/'model.safetensors',device='cpu'),strict=True)
            model.to(self.device).eval()
            extractor=WhisperFeatureExtractor.from_pretrained(root/size,local_files_only=True)
            self.models[size]=(model,extractor,norm)
            self.root=root

    def prepare(self):
        for size in ('base','medium'):self.load(size)

    def predict_conditioning(self,audio16,names):
        """Distilled native 99 axes, normalized exactly as VC preparation.

        This replaces unavailable corpus annotations with measured audio
        predictions of the same named targets, not zeros or prompt guesses.
        """
        import torch
        self.load('medium')
        if len(names)!=99 or len(set(names))!=99:
            raise ValueError('VC needs exactly 99 named score axes')
        mapped=['emo_'+n if i<40 else 'vn_'+n+'_reg' if i<97 else n for i,n in enumerate(names)]
        with self.lock,torch.inference_mode(),torch.autocast('cuda',enabled=self.device.startswith('cuda'),dtype=torch.bfloat16):
            model,extractor,norm=self.models['medium']
            indexes=[norm['score_names'].index(n) for n in mapped]
            mean=np.asarray(norm['score_mean'],np.float32)[indexes]
            std=np.asarray(norm['score_std'],np.float32)[indexes]
            totals=np.zeros((len(audio16),99),np.float64);lengths=np.zeros(len(audio16),np.float64)
            segments=[];owners=[];weights=[]
            for row,wave in enumerate(audio16):
                wave=wave.detach().float().cpu().numpy() if hasattr(wave,'detach') else np.asarray(wave,dtype=np.float32)
                if wave.ndim!=1 or not len(wave) or not np.isfinite(wave).all():raise ValueError('Invalid VC score audio')
                for start in range(0,len(wave),480000):
                    part=wave[start:start+480000];segments.append(np.pad(part,(0,max(0,1600-len(part)))))
                    owners.append(row);weights.append(len(part))
            for start in range(0,len(segments),4):
                features=extractor(segments[start:start+4],sampling_rate=16000,padding='longest',truncation=True,
                    max_length=480000,return_attention_mask=True,return_tensors='np')
                mel=features['input_features'].astype(np.float32);mask=features['attention_mask'].astype(np.int64)
                if mel.shape[-1]%2:mel=np.pad(mel,((0,0),(0,0),(0,1)));mask=np.pad(mask,((0,0),(0,1)))
                out=model(torch.from_numpy(mel).to(self.device),torch.from_numpy(mask).to(self.device),predict_events=False)
                raw=out['scores'][:,indexes].float().cpu().numpy()*std+mean
                if not np.isfinite(raw).all():raise RuntimeError('Nonfinite VC score conditioning')
                for offset,value in enumerate(raw):
                    i=start+offset;totals[owners[i]]+=value*weights[i];lengths[owners[i]]+=weights[i]
            result=(totals/lengths[:,None]).astype(np.float32)
            result[:,:98]/=6.;result[:,98]/=10.
            return result

    def predict(self,audio16,size='medium',metric='ce'):
        """Batched PCM, 16kHz. Every second participates, including >30s tails."""
        import torch
        self.load(size);tick=time.perf_counter()
        segments=[];rows=[];weights=[];durations=[]
        for row,wave in enumerate(audio16):
            wave=wave.detach().float().cpu().numpy() if hasattr(wave,'detach') else np.asarray(wave,dtype=np.float32)
            wave=wave.reshape(-1)
            if not np.isfinite(wave).all():raise ValueError('Nonfinite Ears audio input')
            durations.append(len(wave)/16000)
            for start in range(0,len(wave),480000):
                part=wave[start:start+480000]
                weights.append(len(part));rows.append(row)
                segments.append(np.pad(part,(0,max(0,1600-len(part)))))
        totals=np.zeros((len(audio16),len(METRICS)),np.float64);denom=np.zeros(len(audio16),np.float64)
        with self.lock,torch.inference_mode(),torch.autocast('cuda',enabled=self.device.startswith('cuda'),dtype=torch.bfloat16):
            model,extractor,norm=self.models[size]
            indexes=[norm['score_names'].index(v[0]) for v in METRICS.values()]
            mean=np.asarray(norm['score_mean'])[indexes];std=np.asarray(norm['score_std'])[indexes]
            for start in range(0,len(segments),4):
                features=extractor(segments[start:start+4],sampling_rate=16000,padding='longest',
                    truncation=True,max_length=480000,return_attention_mask=True,return_tensors='np')
                mel=features['input_features'].astype(np.float32)
                mask=features['attention_mask'].astype(np.int64)
                if mel.shape[-1]%2:
                    mel=np.pad(mel,((0,0),(0,0),(0,1)));mask=np.pad(mask,((0,0),(0,1)))
                out=model(torch.from_numpy(mel).to(self.device),torch.from_numpy(mask).to(self.device),predict_events=False)
                raw=out['scores'][:,indexes].float().cpu().numpy()*std+mean
                if not np.isfinite(raw).all():raise RuntimeError('Nonfinite Humaneness Ears predictions')
                for offset,value in enumerate(raw):
                    i=start+offset;totals[rows[i]]+=value*weights[i];denom[rows[i]]+=weights[i]
        results=[]
        for row in range(len(audio16)):
            raw={key:float(v) for key,v in zip(METRICS,totals[row]/max(1,denom[row]))} if denom[row] else {}
            results.append({'available':bool(denom[row]),'model':size,'metric':metric,
                'key':METRICS[metric][0],'raw':raw.get(metric),
                'normalized':normalized(raw[metric],metric) if raw else None,
                'scores_raw':{METRICS[k][0]:v for k,v in raw.items()},'duration_seconds':durations[row],
                'window_count':rows.count(row),'aggregation':'duration-weighted mean over <=30s windows',
                'repo':REPO,'revision':REVISION,'proxy':True})
        self.last_timing={'ears_seconds':time.perf_counter()-tick,'ears_windows':len(segments),'ears_model':size}
        return results
