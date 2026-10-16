"""Lazy selected-head inference using official GELU VoiceNet and ReLU EIV+.

The VoiceCLAP audio tower is shared with search; only chosen heads are loaded.
EIV+ uses its documented BUD-E encoder, NOT the previous EmoWhisper checker.
Selected large heads are bounded to four resident experts. Full-sequence and
pooled experts consume the exact 30-second-padded encoder sequence.
"""
import threading
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from huggingface_hub import hf_hub_download, snapshot_download

import settings
from adaptive_reward import CATALOG, DIMENSIONS


class VoiceNetHead(nn.Module):
    def __init__(self,architecture):
        super().__init__()
        self.f1=nn.Linear(architecture['D'],architecture['H'])
        self.f2=nn.Linear(architecture['H'],architecture['out'])
    def forward(self,x):return self.f2(F.gelu(self.f1(x)))


class EmpathicHead(nn.Module):
    def __init__(self,input_dim):
        super().__init__()
        self.proj=nn.Linear(input_dim,64)
        layers=[nn.ReLU(),nn.Dropout(0.)]
        size=64
        for hidden in [64,32,16]:
            layers.extend([nn.Linear(size,hidden),nn.ReLU(),nn.Dropout(.1)])
            size=hidden
        layers.append(nn.Linear(size,1))
        self.mlp=nn.Sequential(*layers)
    def forward(self,x):return self.mlp(self.proj(x))


class DimensionPredictors:
    def __init__(self,device):
        self.device=device
        self.lock=threading.RLock()
        self.vn={}; self.eiv={}; self.encoder=None; self.extractor=None
        self.last_timing={}

    def prepare(self,plan):
        """Resolve/download before the measured search starts, never silently skip."""
        self.prepare_keys({t.dimension for t in plan.terms})

    def prepare_keys(self, selected):
        """Internal identity vector may use more than four VoiceNet heads."""
        with self.lock:
            for key in list(self.eiv):
                if any(k.startswith('eiv:') for k in selected) and key not in selected:del self.eiv[key]
            for key in selected:
                meta=DIMENSIONS[key]
                if meta['source']=='voicenet' and key not in self.vn:
                    path=hf_hub_download(CATALOG['voicenet_repo'],f"regression/{key[3:]}.pt",
                        revision=CATALOG['voicenet_revision'],cache_dir=settings.HF_CACHE+'/hub')
                    # Official pinned checkpoints contain NumPy metadata, not
                    # weights alone. Only this trusted pinned repo is accepted.
                    state=torch.load(path,map_location='cpu',weights_only=False)
                    head=VoiceNetHead(state['arch'])
                    head.load_state_dict(state['state_dict'],strict=True)
                    self.vn[key]=(head.to(self.device).eval(),
                        torch.as_tensor(np.asarray(state['mu']),dtype=torch.float32,device=self.device),
                        torch.as_tensor(np.asarray(state['sd']),dtype=torch.float32,device=self.device))
                elif meta['source']=='eiv_plus' and key not in self.eiv:
                    if self.encoder is None:self._load_encoder()
                    path=hf_hub_download(CATALOG['eiv_repo'],meta['filename'],
                        revision=CATALOG['eiv_revision'],cache_dir=settings.HF_CACHE+'/hub')
                    state=torch.load(path,map_location='cpu',weights_only=True)
                    state={k.removeprefix('_orig_mod.'):v for k,v in state.items()}
                    head=EmpathicHead(state['proj.weight'].shape[1])
                    head.load_state_dict(state,strict=True)
                    self.eiv[key]=head.to(self.device,dtype=torch.float32).eval()

    @torch.inference_mode()
    def predict_voicenet(self, embeddings, keys):
        """Shared tower embeddings -> fixed-order native scalar vectors."""
        with self.lock, torch.autocast('cuda', enabled=False):
            unit = F.normalize(embeddings.float(), dim=-1)
            result=[]
            for key in keys:
                head, mu, sd = self.vn[key]
                result.append(head((unit-mu)/sd.clamp_min(1e-6)).reshape(-1))
            return torch.stack(result, dim=-1).float()

    def _load_encoder(self):
        from transformers import WhisperFeatureExtractor, WhisperModel
        root=snapshot_download(CATALOG['encoder_repo'],revision=CATALOG['encoder_revision'],
            allow_patterns=['*.json','*.safetensors'],cache_dir=settings.HF_CACHE+'/hub')
        self.extractor=WhisperFeatureExtractor.from_pretrained(root)
        model=WhisperModel.from_pretrained(root,dtype=torch.float32,attn_implementation='sdpa')
        self.encoder=model.encoder.to(self.device).eval()

    @torch.inference_mode()
    def predict(self,audio16,embeddings,plan):
        started=time.perf_counter()
        output=[{} for _ in audio16]
        with self.lock,torch.autocast('cuda',enabled=False):
            vn=[t.dimension for t in plan.terms if t.dimension.startswith('vn:')]
            eiv=[t.dimension for t in plan.terms if t.dimension.startswith('eiv:')]
            unit=F.normalize(embeddings.float(),dim=-1)
            for key in vn:
                head,mu,sd=self.vn[key]
                values=head((unit-mu)/sd.clamp_min(1e-6)).reshape(-1).cpu().tolist()
                for row,value in zip(output,values):row[key]=value
            # Keep memory bounded for BS 16. This evaluates each waveform with
            # its own 30-s padding. Clips longer than 30 s use the latest 30 s.
            for offset in range(0,len(audio16),4) if eiv else []:
                batch=[w[-480000:].float().cpu().numpy() for w in audio16[offset:offset+4]]
                features=self.extractor(batch,sampling_rate=16000,return_tensors='pt').input_features
                hidden=self.encoder(features.to(self.device,dtype=torch.float32)).last_hidden_state.float()
                flat=hidden.flatten(1)
                pooled=torch.cat([hidden.mean(1),hidden.min(1).values,hidden.max(1).values,hidden.std(1)],dim=-1)
                for key in eiv:
                    head=self.eiv[key]
                    inputs=pooled if head.proj.in_features==3072 else flat
                    values=head(inputs).reshape(-1).cpu().tolist()
                    for row,value in zip(output[offset:offset+4],values):row[key]=value
        self.last_timing={'dimension_seconds':time.perf_counter()-started}
        return output
