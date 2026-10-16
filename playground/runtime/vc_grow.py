"""Pinned 24k speaker_mix adapter, following the published inference schema.

No remote Python execution, unrestricted pickle, optimizer restoration or
silent vanilla fallback. Parent and continuation are loaded in that order.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import settings

TARGET_STATIC = frozenset({'AGEV','GEND','BRGT','HARM','ROUG','SMTH','WARM',
    'R_CHST','R_HEAD','R_MASK','R_MIXD','R_NASL','R_ORAL','R_THRT'})


class LoRALinear(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.lora_down=nn.Linear(base.in_features,128,bias=False)
        self.lora_up=nn.Linear(128,base.out_features,bias=False)
        self.scale=1.

    @property
    def weight(self):return self.base.weight

    @property
    def bias(self):return self.base.bias

    def forward(self,x,scale=1.):
        return self.base(x)+(self.scale*scale)*self.lora_up(self.lora_down(x))


class StyledSpeakerProjection(nn.Module):
    def __init__(self,base):
        super().__init__()
        self.base=base.requires_grad_(False)
        self.style=nn.Sequential(nn.Linear(198,256),nn.SiLU(),nn.Linear(256,80))
        self.condition=None;self.present=None;self.scale=1.

    def forward(self,camp):
        if self.condition is None or self.present is None:
            raise RuntimeError('VC requires complete source/target score conditioning')
        if self.condition.shape!=(len(camp),99):
            raise ValueError('VC conditioning batch must match CAMPPlus batch')
        features=torch.cat((self.condition*self.present,self.present),dim=-1)
        return self.base(camp)+self.scale*self.style(features)


def verified_download(repo,file,revision,expected=None):
    from huggingface_hub import hf_hub_download
    path=Path(hf_hub_download(repo,file,revision=revision,
        cache_dir=str(settings.HERE/'model-cache/chatterbox/hub'),token=False))
    if expected and hashlib.sha256(path.read_bytes()).hexdigest()!=expected:
        raise RuntimeError('VC adapter integrity mismatch: '+file)
    return path


def install(model):
    from safetensors.torch import load_file
    parent=verified_download(settings.VC_GROW_PARENT_REPO,'checkpoints/base_300h_adapter.safetensors',
        settings.VC_GROW_PARENT_REVISION,settings.VC_GROW_PARENT_SHA256)
    checkpoint=verified_download(settings.VC_GROW_REPO,settings.VC_GROW_CHECKPOINT,
        settings.VC_GROW_REVISION,settings.VC_GROW_SHA256)
    order=verified_download(settings.VC_GROW_REPO,'score_order_and_preparation.json',settings.VC_GROW_REVISION)
    names=json.loads(order.read_text())['score_order']
    if len(names)!=99 or len(set(names))!=99 or not TARGET_STATIC.issubset(names):
        raise RuntimeError('Invalid pinned 99-score conditioning order')
    model.requires_grad_(False)
    estimator=model.flow.decoder.estimator
    modules=[(name,module) for name,module in estimator.named_modules()
        if name.endswith(('.attn1.to_q','.attn1.to_k','.attn1.to_v','.attn1.to_out.0'))]
    if len(modules)!=224 or any(not isinstance(m,nn.Linear) for _,m in modules):
        raise RuntimeError('Unexpected S3Gen attention architecture')
    # Constructors must not perturb a simultaneous TTS sampler's RNG.
    with torch.random.fork_rng(devices=[]):
        for name,base in modules:
            parent_name,leaf=name.rsplit('.',1)
            container=estimator.get_submodule(parent_name)
            wrapped=LoRALinear(base).to(device=base.weight.device,dtype=base.weight.dtype)
            if isinstance(container,(nn.ModuleList,nn.Sequential)):container[int(leaf)]=wrapped
            else:setattr(container,leaf,wrapped)
        base=model.flow.spk_embed_affine_layer
        model.flow.spk_embed_affine_layer=StyledSpeakerProjection(base).to(device=base.weight.device,dtype=base.weight.dtype)
    parameters={n:p for n,p in model.named_parameters() if p.requires_grad}
    def assign(tensors):
        if set(parameters)!=set(tensors):raise RuntimeError('VC trainable key mismatch')
        with torch.no_grad():
            for name,param in parameters.items():
                value=tensors[name]
                if value.shape!=param.shape or not torch.isfinite(value).all():
                    raise RuntimeError('Invalid VC adapter tensor: '+name)
                param.copy_(value.to(param))
    assign(load_file(str(parent)))
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if state.get('schema')!='chatterbox-s3gen-grow-v3-ce-emo-speaker' or state.get('arm')!='speaker_mix' or state.get('step')!=24000:
        raise RuntimeError('Unexpected 24k checkpoint schema/arm/step')
    assign(state['trainable'])
    model.requires_grad_(False);model.eval();model.flow.decoder.training_cfg_rate=0.
    return names


def mix_scores(source,target,names):
    source=np.asarray(source,dtype=np.float32);target=np.asarray(target,dtype=np.float32)
    if source.ndim!=2 or source.shape[1]!=99 or target.shape!=(99,) or not np.isfinite(source).all() or not np.isfinite(target).all():
        raise ValueError('Invalid complete normalized VC conditioning scores')
    mixed=source.copy()
    for axis in TARGET_STATIC:mixed[:,names.index(axis)]=target[names.index(axis)]
    return mixed
