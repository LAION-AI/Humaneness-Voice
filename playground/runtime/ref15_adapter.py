"""Pinned Ref15 q/k/v/o rank-128 release; no downloaded code is executed.

Public loader contract: S3 step 32337, rank=alpha=128, 112 semantic targets,
224 A/B tensors. Fusion is BF16 rather than the upstream FP32 residual branch.
"""
import json
import math
from pathlib import Path
import re

from huggingface_hub import hf_hub_download
from safetensors import safe_open
from safetensors.torch import load_file
import torch

import settings
from rank1_adapters import BASE_SHA256, sha256

ID = 'ref15_rank128'
FORMAT='humaneness-voice-small-ref15-qkvo-r128-v1'
TARGETS={f'transformer.layers.{i}.self_attn.{projection}_proj'
         for i in range(28) for projection in ('q','k','v','o')}


def repository_files():
    # Exact artifact and digest from the inspected, immutable release CONTENTS.
    root=settings.HERE/'model-cache/ref15'
    filename='adapter_model.pt'
    path=Path(hf_hub_download(settings.REF15_REPO,filename,revision=settings.REF15_REVISION,
        cache_dir=str(root/'hub'),token=False))
    if sha256(path)!=settings.REF15_SHA256:raise ValueError('Ref15 adapter integrity check failed')
    release={'repo':settings.REF15_REPO,'revision':settings.REF15_REVISION,'file':filename,
        'sha256':settings.REF15_SHA256,'rank':128,'alpha':128,'targets':112,'format':FORMAT}
    return path,{'release_contract':True},release


def parse_payload(path, config, modules):
    """Return validated module A/B pairs and the *declared* alpha/r scale."""
    config=dict(config)
    if Path(path).suffix=='.safetensors':
        with safe_open(path,framework='pt') as handle:
            for key,value in (handle.metadata() or {}).items():
                if key not in config:
                    try:config[key]=json.loads(value)
                    except (ValueError,TypeError):config[key]=value
        tensors=load_file(path,device='cpu')
    else:
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if not isinstance(payload,dict):raise ValueError('Ref15 payload must be a mapping')
        if config.get('release_contract'):
            if (payload.get('format')!=FORMAT or payload.get('base_model')!=settings.REPO
                or payload.get('base_stage')!='S3' or payload.get('base_step')!=32337
                or payload.get('rank')!=128 or payload.get('alpha')!=128):
                raise ValueError('Ref15 release contract mismatch')
            expected={name+f'.{side}.weight' for name in TARGETS for side in ('a','b')}
            if set(payload.get('state_dict',{}))!=expected:
                raise ValueError('Ref15 requires all 224 exact semantic q/k/v/o tensors')
            config['base_model_name_or_path']=payload['base_model']
        for key in ('rank','rank_size','r','alpha','lora_alpha','base_sha256','base_model_name_or_path',
                    'use_rslora','use_dora','rank_pattern','alpha_pattern'):
            if key in payload and key not in config:config[key]=payload[key]
        for field in ('adapter','state_dict','lora_state_dict'):
            if field in payload:
                tensors=payload[field];break
        else:tensors=payload
    if not isinstance(tensors,dict):raise ValueError('Ref15 weights must be a tensor mapping')
    for flag in ('use_dora','use_rslora'):
        if flag in config and not isinstance(config[flag],bool):
            raise ValueError('Ref15 scaling flags must be explicit booleans')
    if config.get('use_dora') or config.get('rank_pattern') or config.get('alpha_pattern'):
        raise ValueError('Ref15 DoRA/per-layer scaling requires a dedicated verified loader')
    rank=config.get('r',config.get('rank_size',config.get('rank')))
    alpha=config.get('lora_alpha',config.get('alpha'))
    if rank is None or alpha is None:
        raise ValueError('Ref15 release must declare rank and LoRA alpha; no scaling is guessed')
    if float(rank)!=128 or not math.isfinite(float(alpha)) or float(alpha)<=0:
        raise ValueError('Ref15 requires rank 128 and a finite positive declared alpha')
    base=config.get('base_sha256')
    if base is not None and base!=BASE_SHA256:raise ValueError('Ref15 base hash mismatch')
    repo=config.get('base_model_name_or_path')
    if repo is not None and str(repo).rstrip('/')!=settings.REPO:
        raise ValueError('Ref15 declared base does not match Humaneness-Voice-Small')
    if base is None and repo is None:
        raise ValueError('Ref15 must declare the compatible base in its payload/config')
    # Strict explicit naming schemes; never infer A/B by tensor shape alone.
    patterns=(r'^(.+)\.lora_([AB])(?:\.default)?\.weight$',r'^(.+)\.([ab])\.weight$')
    pairs={}
    for key,tensor in tensors.items():
        if not isinstance(tensor,torch.Tensor):
            raise ValueError('Ref15 weight mapping contains non-tensor entries')
        match=next((m for pattern in patterns if (m:=re.fullmatch(pattern,key))),None)
        if match is None:raise ValueError('Unknown Ref15 tensor naming: '+key)
        name,side=match[1],match[2].lower()
        if name.startswith('base_model.model.'):name=name[len('base_model.model.'):]
        if name not in modules:raise ValueError('Ref15 target not present in this base: '+name)
        if side in pairs.setdefault(name,{}):raise ValueError('Duplicate Ref15 tensor: '+key)
        pairs[name][side]=tensor
    if not pairs:raise ValueError('Empty Ref15 adapter')
    validated={}
    for name,pair in pairs.items():
        module=modules[name]
        if set(pair)!= {'a','b'}:raise ValueError('Incomplete Ref15 A/B pair: '+name)
        a,b=pair['a'],pair['b']
        if a.shape!=(128,module.in_features) or b.shape!=(module.out_features,128):
            raise ValueError('Ref15 adapter shape mismatch: '+name)
        if not a.is_floating_point() or not b.is_floating_point() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise ValueError('Invalid Ref15 tensors: '+name)
        validated[name]=(a,b)
    scale=float(alpha)/(math.sqrt(128) if config.get('use_rslora') else 128)
    return validated,scale


def load(modules):
    path,config,release=repository_files()
    pairs,scale=parse_payload(path,config,modules)
    root=settings.HERE/'model-cache/ref15'
    root.mkdir(parents=True,exist_ok=True)
    temporary=root/'release.tmp'
    temporary.write_text(json.dumps(release,indent=2));temporary.replace(root/'release.json')
    return pairs,scale,release
