"""Reference-conditioned identity proxy or real ECAPA speaker embeddings.

VoiceNet attributes are perceptual, not truly invariant identity measurements.
Fixed native-scale bipolar centering avoids positive-vector cosine inflation.
Profile identity anchors stay fixed across changing emotional conditioning clips.
No external speaker service; private uploads never leave the local scorer GPU.
"""
from collections import OrderedDict
import hashlib
from pathlib import Path
import threading
import time
from typing import Literal

from huggingface_hub import snapshot_download
import numpy as np
from pydantic import BaseModel, ConfigDict, Field
import soundfile as sf
import torch
from torch.nn import functional as F
import torchaudio

from adaptive_reward import DIMENSIONS
import settings

IDENTITY_KEYS = tuple('vn:'+key for key in (
    'AGEV','GEND','REGS','R_CHST','R_HEAD','R_ORAL','R_NASL','R_THRT','R_MASK',
    'FULL','BRGT','WARM','ROUG','METL','HARM'))


class SpeakerOptions(BaseModel):
    model_config = ConfigDict(extra='forbid')
    method: Literal['voicenet','ecapa','off'] = 'ecapa'
    weight: float = Field(default=.6, ge=0, le=1, allow_inf_nan=False)


class ReferenceTooShort(ValueError):
    pass


class UnavailableSpeakerReward:
    def __init__(self,options,reason):
        self.options,self.reason=options,reason
        self.last_timing={}
    def score(self,audio16,embeddings,speech):
        return [{'available':False,'method':self.options.method,'reason':self.reason,
                 'cosine':None,'normalized':None} for _ in audio16]


def identity_vector(raw):
    """Uniform feature weighting, per-dimension fixed [-1,1] native scale."""
    low = raw.new_tensor([DIMENSIONS[key]['min'] for key in IDENTITY_KEYS])
    high = raw.new_tensor([DIMENSIONS[key]['max'] for key in IDENTITY_KEYS])
    if not torch.isfinite(raw).all():
        raise ValueError('Non-finite speaker attributes; refusing arbitrary ranking')
    return 2*((raw-low)/(high-low)).clamp(0,1)-1


def combine_identity(item, report, weight, adaptive=False):
    """Exact 40/60 style+speaker default, then the existing bounded WER gate.

With identity on, Genuine/Blend are diagnostic only, not hidden extra weights.
Optional explicit burst alignment is blended by SearchScorer afterwards.
"""
    style = float(item['style_score'] if adaptive else item['n_clap'])
    effective = weight if report.get('available') else 0.
    item.update(speaker=report, speaker_similarity=report.get('cosine'),
                speaker_score=report.get('normalized'), speaker_weight=effective,
                prompt_weight=1-effective, previous_quality=item['quality'])
    item['quality'] = (1-effective)*style + effective*float(report.get('normalized') or 0.)
    item['reward'] = item['quality']*item['gate']
    item['reward_model'] = 'adaptive_plus_speaker' if adaptive else 'clap_plus_speaker'
    return item


def catalog():
    defaults=SpeakerOptions()
    return {'default_method':defaults.method,'default_weight':defaults.weight,
        'ecapa_repo':settings.SPEAKER_REPO,'ecapa_revision':settings.SPEAKER_REVISION,
        'dimensions':[{'id':key,'name':DIMENSIONS[key]['name'],
                       'min':DIMENSIONS[key]['min'],'max':DIMENSIONS[key]['max']} for key in IDENTITY_KEYS],
        'note':'Perceptual timbre proxy, not unique or strictly emotion-invariant speaker identity.'}


def audio_batch(waves, device):
    longest=max(len(w) for w in waves)
    batch=torch.stack([F.pad(w.to(device),(0,longest-len(w))) for w in waves])
    lengths=batch.new_tensor([len(w)/longest for w in waves])
    return batch,lengths


class SpeakerModels:
    def __init__(self, scorers):
        self.scorers=scorers
        self.device=scorers.retriever.device
        self.lock=threading.RLock()
        self.ecapa=None
        self.targets=OrderedDict()

    def prepare(self, method):
        if method=='voicenet':
            self.scorers.dimensions.prepare_keys(set(IDENTITY_KEYS))
        elif method=='ecapa' and self.ecapa is None:
            from speechbrain.inference.speaker import EncoderClassifier
            root=snapshot_download(settings.SPEAKER_REPO,revision=settings.SPEAKER_REVISION,
                allow_patterns=['hyperparams.yaml','*.ckpt','label_encoder.txt'],
                cache_dir=str(Path(settings.HF_CACHE)/'hub'),token=False)
            self.ecapa=EncoderClassifier.from_hparams(source=root,
                overrides={'pretrained_path':root},
                savedir=str(Path(settings.HF_CACHE)/'small-ecapa-pinned'),
                run_opts={'device':self.device})

    @torch.inference_mode()
    def embed(self, audio16, method, embeddings=None):
        if method=='voicenet':
            if embeddings is None:
                batch,_=audio_batch(audio16,self.device)
                embeddings=self.scorers.retriever.model.encode_waveform(batch,sample_rate=16000).float()
            raw=self.scorers.dimensions.predict_voicenet(embeddings,IDENTITY_KEYS)
            return identity_vector(raw),raw
        batch,lengths=audio_batch(audio16,self.device)
        with torch.autocast('cuda',enabled=False):
            vectors=self.ecapa.encode_batch(batch.float(),wav_lens=lengths).reshape(len(audio16),-1).float()
        return vectors,None

    def target(self, path, method):
        path=Path(path)
        # Playback touches mtime; hash immutable content instead so user activity
        # does not repeatedly re-encode an unchanged private reference.
        key=(method,str(path),hashlib.sha256(path.read_bytes()).hexdigest())
        with self.lock:
            self.prepare(method)
            if key in self.targets:
                self.targets.move_to_end(key)
                return self.targets[key]
            wave,sr=sf.read(path,dtype='float32')
            if wave.ndim>1:wave=wave.mean(1)
            wave=wave[:int(sr*settings.REFERENCE_SECONDS)]
            if len(wave)<sr*.5:raise ReferenceTooShort('Reference too short: speaker comparison requires at least 0.5 seconds')
            audio=torchaudio.functional.resample(torch.as_tensor(wave,device=self.device),sr,16000)
            vector,raw=self.embed([audio],method)
            if not torch.isfinite(vector).all() or vector.norm()<1e-8:
                raise ValueError('Reference speaker vector is undefined')
            result=(F.normalize(vector,dim=-1)[0],raw[0].cpu().tolist() if raw is not None else None)
            self.targets[key]=result
            if len(self.targets)>32:self.targets.popitem(last=False)
            return result

    def forget(self, path):
        with self.lock:
            for key in list(self.targets):
                if key[1]==str(path):del self.targets[key]


class SpeakerReward:
    def __init__(self, models, options, reference_path, window_seconds=None, source='reference'):
        self.models,self.options=models,options
        self.window_seconds=window_seconds
        self.source=source
        self.target,self.target_raw=models.target(reference_path,options.method)
        self.last_timing={}

    def describe(self):
        return {'method':self.options.method,'weight':self.options.weight,'prompt_weight':1-self.options.weight,
            'source':self.source,'window_seconds':self.window_seconds,
            'comparison':'latest search chunk' if self.window_seconds else 'whole finished candidate',
            'dimensions':list(IDENTITY_KEYS) if self.options.method=='voicenet' else None,
            'reference_attributes':{DIMENSIONS[key]['name']:value for key,value in zip(IDENTITY_KEYS,self.target_raw)}
                if self.target_raw is not None else None}

    @torch.inference_mode()
    def score(self, audio16, embeddings, speech):
        started=time.perf_counter()
        clips=[w[-int(self.window_seconds*16000):] if self.window_seconds else w for w in audio16]
        # Do not compare silence/requested non-speech or tiny chunks as speakers.
        valid=[i for i,w in enumerate(clips) if len(w)>=8000 and speech[i] and w.square().mean().sqrt().item()>1e-4]
        reports=[{'available':False,'method':self.options.method,
                  'reason':'Too little recognized speech to compare identity','cosine':None,'normalized':None} for _ in clips]
        if valid:
            reuse = embeddings[valid] if not self.window_seconds else None
            with self.models.lock:
                vectors,raw=self.models.embed([clips[i] for i in valid],self.options.method,reuse)
            if not torch.isfinite(vectors).all():raise ValueError('Non-finite speaker embeddings')
            cosine=(F.normalize(vectors,dim=-1) @ self.target).clamp(-1,1).cpu().tolist()
            values=raw.cpu().tolist() if raw is not None else [None]*len(valid)
            for row,(i,sim,value) in enumerate(zip(valid,cosine,values)):
                if vectors[row].norm()<1e-8:continue
                reports[i]={'available':True,'method':self.options.method,'cosine':float(sim),
                    'normalized':float(np.clip((sim+1)/2,0,1)),
                    'attributes':{DIMENSIONS[key]['name']:v for key,v in zip(IDENTITY_KEYS,value)} if value is not None else None}
        self.last_timing={'speaker_seconds':time.perf_counter()-started}
        return reports
