"""Optional DPO + Ref15 adapters fused without replacing CUDA-graph storage.

One DPO plus optional Ref15. BF16 fusion is not identical to an FP32
residual branch. Both semantic and local Talker projections are updated.
"""
from functools import lru_cache
import json
from pathlib import Path
import time

from huggingface_hub import hf_hub_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
import torch

import settings
from rank1_adapters import BASE_SHA256, sha256

IDS = ('subset_pitch_quality_p2_rank016', 'subset_pitch_quality_p2_rank064',
       'rebalanced_rank064', 'rebalanced_rank128', 'original_rank064', 'original_rank128')
REF15_ID = 'ref15_rank128'


class ActiveAdapter(BaseModel):
    id: str
    weight: float


class AdapterSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    # Empty selection always means explicit base (also used by frozen studies
    # and Ref15-only presets). Request defaults are supplied by Turn instead.
    id: str | None = None
    weight: float = Field(default=1., ge=0, le=2, allow_inf_nan=False)
    ref15: bool = False
    ref15_weight: float = Field(default=1.,ge=0,le=2,allow_inf_nan=False)

    @model_validator(mode='after')
    def known(self):
        if self.id is not None and self.id not in IDS:
            raise ValueError('Unknown DPO adapter')
        return self

    def active(self):
        out=[ActiveAdapter(id=self.id,weight=self.weight)] if self.id and self.weight>0 else []
        if self.ref15 and self.ref15_weight>0:
            out.append(ActiveAdapter(id=REF15_ID,weight=self.ref15_weight))
        return out


def download(filename):
    return Path(hf_hub_download(settings.DPO_REPO, filename,
        revision=settings.DPO_REVISION, cache_dir=str(Path(settings.HF_CACHE) / 'hub'), token=False))


@lru_cache(maxsize=6)
def metadata(id_):
    if id_ not in IDS:
        raise ValueError('Unknown DPO adapter')
    return json.loads(download(f'adapters/{id_}.json').read_text())


def catalog():
    # Stable release IDs allow the UI to load without blocking on HF downloads.
    labels = {'subset_pitch_quality_p2': 'Pitch & quality subset · P2',
              'rebalanced': 'Rebalanced DPO', 'original': 'Original DPO'}
    rows = [{'id': id_, 'label': labels[id_.rsplit('_rank', 1)[0]],
             'rank': int(id_.rsplit('rank', 1)[1]), 'default': id_ == settings.DEFAULT_ADAPTER}
            for id_ in IDS]
    return {'repo': settings.DPO_REPO, 'revision': settings.DPO_REVISION,
            'enabled': settings.STAGE in ('S3',settings.FULL_FT_STAGE), 'default': settings.DEFAULT_ADAPTER,
            'base_stage':settings.STAGE,'cross_base':settings.STAGE==settings.FULL_FT_STAGE,
            'preferred_dpo':settings.PREFERRED_DPO_ADAPTER,
            'ref15':{'id':REF15_ID,'label':'Ref15 · längerer Referenz-Kontext','rank':128,
                'repo':settings.REF15_REPO,'revision':settings.REF15_REVISION,
                'verification':'Pinned q/k/v/o release contract; mixtures and 20-second references experimental',
                'reference_required':True,'training_max_frames':187},
            'adapters': rows, 'counts': {'dpo': len(rows),'ref15':1},
            'note': 'Base is default. One DPO and Ref15 can be combined; mixtures are experimental.'}


def validate_selection(selection):
    if settings.STAGE not in ('S3',settings.FULL_FT_STAGE) and selection.active():
        raise ValueError('DPO LoRAs require the verified S3 or Ref15 Full-FT base')


def prefetch():
    for id_ in IDS:
        meta = metadata(id_)
        path = download(f'adapters/{id_}.adapter.pt')
        # Release metadata counts tensor bytes, not the serialized ZIP overhead.
        if sha256(path) != meta['adapter_sha256']:
            raise ValueError('DPO adapter integrity check failed: ' + id_)
    return catalog()


class DPOBank:
    def __init__(self, model, base_checkpoint, *, expected_base_sha256=None):
        self.base_sha256=sha256(base_checkpoint)
        if self.base_sha256 != (expected_base_sha256 or BASE_SHA256):
            raise ValueError('DPO checkpoint base hash mismatch')
        semantic = ('.self_attn.q_proj', '.self_attn.v_proj', '.mlp.gate_proj', '.mlp.up_proj', '.mlp.down_proj')
        talker = ('.attn.c_attn', '.attn.c_proj', '.mlp.fc_in', '.mlp.fc_out')
        self.modules = {name: module for name, module in model.named_modules()
            if isinstance(module, torch.nn.Linear) and (
                name.startswith('transformer.layers.') and name.endswith(semantic) or
                name.startswith('local_transformer.h.') and name.endswith(talker))}
        if len(self.modules) != 144:
            raise ValueError(f'Unexpected DPO architecture: {len(self.modules)} targets, expected 144')
        self.dpo_modules=dict(self.modules)
        self.all_modules={name:m for name,m in model.named_modules() if isinstance(m,torch.nn.Linear)}
        self.original = None
        self.active_key = ()
        self.cache = {}
        self.ref15_release = None

    def tensors(self, id_):
        if id_ in self.cache:
            return self.cache[id_]
        meta = metadata(id_)
        path = download(f'adapters/{id_}.adapter.pt')
        if sha256(path) != meta['adapter_sha256']:
            raise ValueError('DPO adapter integrity check failed')
        payload = torch.load(path, map_location='cpu', weights_only=True)
        rank = int(payload['rank_size'])
        alpha = int(payload['alpha'])
        if (payload.get('format') != 'humaneness-voice-small-dpo-lora-adapter-v1'
                or payload.get('base_sha256') != BASE_SHA256 or rank not in (16, 64, 128)
                or rank != int(id_.rsplit('rank', 1)[1]) or alpha != 2*rank):
            raise ValueError('Unexpected DPO adapter contract')
        tensors = payload['adapter']
        expected = {name + f'.{side}.weight' for name in self.dpo_modules for side in ('a', 'b')}
        if set(tensors) != expected:
            raise ValueError('DPO adapter target names mismatch')
        for name, module in self.dpo_modules.items():
            a, b = tensors[name + '.a.weight'], tensors[name + '.b.weight']
            if a.shape != (rank, module.in_features) or b.shape != (module.out_features, rank):
                raise ValueError('DPO adapter shape mismatch: ' + name)
            if not a.is_floating_point() or not b.is_floating_point() or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Non-finite DPO adapter')
        self.cache[id_] = (tensors, alpha / rank)
        return self.cache[id_]

    def contribution(self,choice):
        if choice.id==REF15_ID:
            if REF15_ID not in self.cache:
                from ref15_adapter import load
                pairs,scale,release=load(self.all_modules)
                self.cache[REF15_ID]=(pairs,scale)
                self.ref15_release=release
            return self.cache[REF15_ID]
        tensors,scale=self.tensors(choice.id)
        return {name:(tensors[name+'.a.weight'],tensors[name+'.b.weight']) for name in self.dpo_modules},scale

    @torch.inference_mode()
    def apply(self, selection):
        validate_selection(selection)
        started = time.perf_counter()
        active = selection.active()
        key = tuple((choice.id,choice.weight) for choice in active)
        if key != self.active_key:
            # Validate every contributor before touching live weights. Mixtures
            # sum deltas from a single immutable base, never the last mixture.
            contributions=[(*self.contribution(choice),choice.weight) for choice in active]
            new_targets=set().union(*(set(pairs) for pairs,_,_ in contributions)) if contributions else set()
            for name in new_targets:
                self.modules.setdefault(name,self.all_modules[name])
            if self.original is None:
                self.original = {}
            for name,module in self.modules.items():
                if name not in self.original:
                    self.original[name]=module.weight.detach().cpu().clone()
            merged = {}
            for name, base in self.original.items():
                weight = base.float().clone()
                for pairs,scale,dose in contributions:
                    if name in pairs:
                        a,b=pairs[name]
                        weight.addmm_(b.float(),a.float(),alpha=scale*dose)
                weight = weight.to(base.dtype)
                if not torch.isfinite(weight).all():
                    raise ValueError('DPO weight update overflows model dtype')
                merged[name] = weight
            device = next(iter(self.modules.values())).weight.device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            backup = {name: m.weight.detach().cpu().clone() for name, m in self.modules.items()}
            try:
                for name, module in self.modules.items():
                    module.weight.copy_(merged[name])
                if device.type == 'cuda':
                    torch.cuda.synchronize(device)
            except Exception:
                for name, module in self.modules.items():
                    module.weight.copy_(backup[name])
                raise
            self.active_key = key
        return {'repo': settings.DPO_REPO, 'revision': settings.DPO_REVISION,
                'selection': selection.model_dump(), 'active': [x.model_dump() for x in active],
                'ref15_release':self.ref15_release if any(x.id==REF15_ID for x in active) else None,
                'base_sha256':getattr(self,'base_sha256',BASE_SHA256),
                'adapter_training_base_sha256':BASE_SHA256,
                'cross_base':bool(active) and getattr(self,'base_sha256',BASE_SHA256)!=BASE_SHA256,
                'experimental_mix':len(active)>1 or (bool(active) and getattr(self,'base_sha256',BASE_SHA256)!=BASE_SHA256),
                'switch_ms': round((time.perf_counter() - started) * 1000),
                'fusion': 'in-place BF16; declared scaling; additive deltas from immutable base'}
