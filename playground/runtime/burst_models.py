"""Pinned locator v2 + commercial five-head ensemble; lazy batched GPU inference.

Architecture/post-processing follow the published implementations. The trained
locator encoder is separate from BUD-E; it must not be substituted with it.
"""
import threading
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download

import settings
from burst_reward import (LOCATOR_REPO,LOCATOR_REVISION,DETECTOR_REPO,DETECTOR_REVISION,WHISPER_REVISION)


class Locator(nn.Module):
    def __init__(self,config):
        super().__init__()
        from transformers import WhisperModel
        self.whisper=WhisperModel(config)
        hidden=max(256,config.d_model//2)
        self.proj=nn.Sequential(nn.Linear(config.d_model,hidden),nn.GELU(),nn.Dropout(.1))
        self.temporal=nn.Sequential(nn.Conv1d(hidden,hidden,7,padding=3),nn.GELU(),nn.Dropout(.1))
        self.out=nn.Linear(hidden,1)
    def forward(self,features):
        hidden=self.proj(self.whisper.encoder(input_features=features).last_hidden_state)
        return self.out(self.temporal(hidden.permute(0,2,1)).permute(0,2,1)).squeeze(-1)


class ClassHead(nn.Module):
    def __init__(self,arch):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(arch['D'],arch['H']),nn.BatchNorm1d(arch['H']),
            nn.GELU(),nn.Dropout(arch['dropout']),nn.Linear(arch['H'],arch['C']))
    def forward(self,x):return self.net(x)


def extract_events(probabilities,threshold=.5,merge_gap=.1,min_duration=.1):
    """20-ms v2 grouping. Physical PCM extent is cropped before this call."""
    active=np.asarray(probabilities) > threshold
    edges=np.diff(np.r_[False,active,False].astype(int))
    intervals=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
    merged=[]
    for start,end in intervals:
        if merged and (start-merged[-1][1])*.02<=merge_gap+1e-9:merged[-1]=(merged[-1][0],end)
        else:merged.append((start,end))
    return [{'start':start*.02,'end':end*.02,'confidence':float(np.mean(probabilities[start:end]))}
            for start,end in merged if (end-start)*.02>=min_duration-1e-9]


class BurstModels:
    def __init__(self,retriever):
        self.retriever=retriever;self.device=retriever.device
        self.locator=None;self.heads=[];self.classes=[];self.families=[]
        self.lock=threading.RLock()

    def prepare(self,typed=True):
        with self.lock:
            if self.locator is None:
                from transformers import WhisperConfig,WhisperFeatureExtractor
                config_path=hf_hub_download('openai/whisper-small','config.json',revision=WHISPER_REVISION,
                                           cache_dir=settings.HF_CACHE+'/hub')
                config=WhisperConfig.from_json_file(config_path)
                config._attn_implementation='sdpa'
                model=Locator(config)
                path=hf_hub_download(LOCATOR_REPO,'model_v2.pt',revision=LOCATOR_REVISION,
                                     cache_dir=settings.HF_CACHE+'/hub')
                state=torch.load(path,map_location='cpu',weights_only=True)
                model.load_state_dict(state,strict=True)
                # Decoder weights are in the public checkpoint but not used
                # by the official segmenter. Verify all keys before dropping it.
                model.whisper.decoder=None
                self.locator=model.to(self.device,dtype=torch.float32).eval()
                feature_path=hf_hub_download('openai/whisper-small','preprocessor_config.json',
                    revision=WHISPER_REVISION,cache_dir=settings.HF_CACHE+'/hub')
                self.extractor=WhisperFeatureExtractor.from_pretrained(str(Path(feature_path).parent))
            if typed and not self.heads:
                heads=[]
                for seed in range(5):
                    path=hf_hub_download(DETECTOR_REPO,f'vocal_burst_mlp_prod_s{seed}.pt',
                        revision=DETECTOR_REVISION,cache_dir=settings.HF_CACHE+'/hub')
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    if state['encoder']!='voiceclap-commercial' or state['arch']['D']!=768:
                        raise ValueError('Unexpected commercial classifier encoder contract')
                    if seed and (state['classes']!=self.classes or state['family']!=self.families):
                        raise ValueError('Classifier ensemble label order differs')
                    head=ClassHead(state['arch']);head.load_state_dict(state['state_dict'],strict=True)
                    heads.append(head.to(self.device).eval())
                    self.classes=state['classes'];self.families=state['family']
                self.heads=heads

    @torch.inference_mode()
    def probabilities(self,audio16):
        """Entire prefix, overlapping 30-s windows; no first-30-s blind spot."""
        jobs=[]
        arrays=[w.detach().float().cpu().numpy() if torch.is_tensor(w) else np.asarray(w,dtype=np.float32) for w in audio16]
        for row,wave in enumerate(arrays):
            for start in range(0,max(1,len(wave)),472000):
                jobs.append((row,start,wave[start:start+480000]))
                if start+480000>=len(wave):break
        sums=[np.zeros(math_frames(len(w)),dtype=np.float32) for w in arrays]
        counts=[np.zeros_like(s) for s in sums]
        for offset in range(0,len(jobs),4):
            batch=jobs[offset:offset+4]
            features=self.extractor([np.pad(w,(0,max(0,480000-len(w)))) for _,_,w in batch],
                                    sampling_rate=16000,return_tensors='pt').input_features.to(self.device)
            probs=torch.sigmoid(self.locator(features)).float().cpu().numpy()
            for (row,start,wave),values in zip(batch,probs):
                frame=start//320;length=min(math_frames(len(wave)),len(sums[row])-frame)
                sums[row][frame:frame+length]+=values[:length];counts[row][frame:frame+length]+=1
        return [s/np.maximum(1,c) for s,c in zip(sums,counts)]

    @torch.inference_mode()
    def detect(self,audio16,typed=True):
        with self.lock,torch.autocast('cuda',enabled=False):
            probabilities=self.probabilities(audio16)
            result=[extract_events(p) for p in probabilities]
            cuts=[]
            for row,(wave,events) in enumerate(zip(audio16,result)):
                duration=len(wave)/16000
                for event in events:
                    event['end']=min(duration,event['end'])
                    event['duration']=event['end']-event['start']
                    if typed:
                        lo=max(0,int((event['start']-.04)*16000));hi=min(len(wave),int((event['end']+.04)*16000))
                        cut=torch.as_tensor(wave[lo:hi],dtype=torch.float32,device=self.device)
                        cuts.append((event,cut))
            for offset in range(0,len(cuts),8) if typed else []:
                batch=cuts[offset:offset+8];longest=max(max(400,len(w)) for _,w in batch)
                inputs=torch.stack([F.pad(w,(0,longest-len(w))) for _,w in batch])
                embeddings=F.normalize(self.retriever.model.encode_waveform(inputs,sample_rate=16000).float(),dim=-1)
                # Follow the card: mean logits of the five initialisations,
                # then one softmax. Not mean of already-softmaxed scores.
                probs=torch.stack([head(embeddings) for head in self.heads]).mean(0).softmax(-1).cpu().tolist()
                for (event,_),values in zip(batch,probs):
                    index=int(np.argmax(values))
                    event.update(probabilities=values,classes=self.classes,families=self.families,
                                 top_label=self.classes[index],top_probability=values[index])
            return result


def math_frames(samples):return int((samples+319)//320)
