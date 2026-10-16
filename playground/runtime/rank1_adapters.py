"""Pinned packed rank-1 bank, per-request selection and reversible weight fusion.

No wrappers are installed after torch.compile/CUDA graph capture. Updates are
fused into the existing parameter storage under the engine lock, always from
an immutable CPU base copy. This differs numerically from FP32 branch addition
in the reference loader. Multi-family mixtures are experimental, not validated
emotion/identity guarantees. TAR members are read in memory, never extracted.
"""
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import tarfile
import time

from huggingface_hub import hf_hub_download, snapshot_download
from pydantic import BaseModel, ConfigDict, Field, model_validator
from safetensors.torch import load as load_safetensors
import torch

import settings

BASE_SHA256 = '60079723bf797a81107def1e6fc479e5353b3cc51f3780f98ea65a9f5c109f97'
FAMILIES = {'voice': 'profile', 'emotion': 'emotion', 'style': 'vn'}
STYLE_NAMES = {
    'AROU': 'Arousal / activation', 'ARSH': 'Arousal shift', 'EMPH': 'Emphasis',
    'EXPL': 'Content explicitness (not just style)', 'S_ASMR': 'ASMR speaking style',
    'S_DRAM': 'Dramatic speaking style', 'S_RANT': 'Ranting speaking style',
    'TENS': 'Tension', 'VALN': 'Valence (negative to positive)', 'VALS': 'Valence shift',
    'VFLX': 'Tempo shifts', 'VOLT': 'Emotional volatility', 'VULN': 'Vulnerability',
}


class AdapterChoice(BaseModel):
    model_config = ConfigDict(extra='forbid')
    id: str = Field(pattern=r'^(profile|emotion|vn)/[A-Za-z0-9_-]+$', max_length=120)
    weight: float = Field(default=1., ge=0, le=2, allow_inf_nan=False)


class AdapterSelection(BaseModel):
    model_config = ConfigDict(extra='forbid')
    voice: AdapterChoice | None = None
    emotion: AdapterChoice | None = None
    style: AdapterChoice | None = None

    @model_validator(mode='after')
    def correct_families(self):
        for slot, family in FAMILIES.items():
            choice = getattr(self, slot)
            if choice and not choice.id.startswith(family + '/'):
                raise ValueError(f'{slot} requires a {family} adapter')
        return self

    def active(self):
        return [getattr(self, slot) for slot in FAMILIES
                if getattr(self, slot) and getattr(self, slot).weight > 0]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def repository():
    return Path(snapshot_download(settings.LORA_REPO, revision=settings.LORA_REVISION,
        cache_dir=str(Path(settings.HF_CACHE) / 'hub'),
        allow_patterns=['index.json', 'manifest.json', 'README.md'], token=False))


@lru_cache(maxsize=1)
def index_data():
    data = json.loads((repository() / 'index.json').read_text())
    if data.get('base_sha256') != BASE_SHA256 or data.get('base_stage') != 'S3':
        raise ValueError('LoRA bank does not match the pinned S3 checkpoint')
    for id_, meta in data['adapters'].items():
        if (meta['rank'], meta['alpha'], meta['targets']) != (1, 1, 'semantic_qkvo'):
            raise ValueError(f'Unsupported adapter contract: {id_}')
        if meta.get('base_sha256') != BASE_SHA256:
            raise ValueError(f'Incorrect adapter base: {id_}')
    return data


def catalog():
    groups = {slot: [] for slot in FAMILIES}
    for id_, meta in index_data()['adapters'].items():
        slot = next(slot for slot, family in FAMILIES.items() if family == meta['family'])
        label = meta['label']
        if slot == 'style':
            dimension, level = label.rsplit('_', 1)
            label = STYLE_NAMES.get(dimension, dimension) + (' · high' if level == 'high' else ' · low')
        groups[slot].append({'id': id_, 'label': label, 'family': slot,
            'rank': 1, 'training_rows': meta['training_rows']})
    for rows in groups.values():
        rows.sort(key=lambda row: row['label'].casefold())
    return {'repo': settings.LORA_REPO, 'revision': settings.LORA_REVISION,
        'base_stage': 'S3', 'enabled': settings.STAGE == 'S3', 'groups': groups,
        'counts': {slot: len(rows) for slot, rows in groups.items()},
        'note': 'One adapter per family. Multi-family mixtures and dose changes are experimental. '
                'Voice adapters do not require reference audio; a reference may compete with the chosen voice. '
                'Training completion is not proof of emotion/style controllability.'}


def validate_selection(selection):
    if settings.STAGE != 'S3' and selection.active():
        raise ValueError('These rank-1 LoRAs require the S3 checkpoint')
    for slot in FAMILIES:
        choice = getattr(selection, slot)
        if choice and choice.id not in index_data()['adapters']:
            raise ValueError(f'Unknown LoRA: {choice.id}')


def prefetch():
    root = repository()
    index_data()
    manifest = json.loads((root / 'manifest.json').read_text())
    for entry in manifest['archives']:
        path = hf_hub_download(settings.LORA_REPO, entry['file'], revision=settings.LORA_REVISION,
            cache_dir=str(Path(settings.HF_CACHE) / 'hub'), token=False)
        if Path(path).stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
            raise ValueError(f'Archive integrity check failed: {entry["file"]}')
    return catalog()


class Rank1Bank:
    def __init__(self, model, base_checkpoint, *, module_names=None):
        if sha256(base_checkpoint) != BASE_SHA256:
            raise ValueError('Refusing LoRAs: loaded base checkpoint SHA-256 differs from the release')
        self.model = model
        self.index = index_data()
        self.root = repository()
        self.manifest = json.loads((self.root / 'manifest.json').read_text())
        self.names = module_names or [f'transformer.layers.{layer}.self_attn.{proj}'
            for layer in range(28) for proj in ('q_proj', 'k_proj', 'v_proj', 'o_proj')]
        self.modules = {name: model.get_submodule(name) for name in self.names}
        if any(not isinstance(module, torch.nn.Linear) for module in self.modules.values()):
            raise ValueError('Unexpected semantic attention projection architecture')
        self.original = None
        self.active_key = ()
        self.cache = OrderedDict()
        self.verified_archives = set()

    def tensors(self, id_):
        if id_ in self.cache:
            self.cache.move_to_end(id_)
            return self.cache[id_]
        meta = self.index['adapters'][id_]
        entry = next(x for x in self.manifest['archives'] if x['file'] == meta['archive'])
        path = Path(hf_hub_download(settings.LORA_REPO, meta['archive'], revision=settings.LORA_REVISION,
            cache_dir=str(Path(settings.HF_CACHE) / 'hub'), token=False))
        if meta['archive'] not in self.verified_archives:
            if path.stat().st_size != entry['bytes'] or sha256(path) != entry['sha256']:
                raise ValueError('LoRA archive hash mismatch')
            self.verified_archives.add(meta['archive'])
        with tarfile.open(path, 'r:') as archive:
            member = archive.getmember(meta['member'])
            if not member.isfile() or not 0 < member.size <= 4 * 1024 * 1024:
                raise ValueError('Unexpected LoRA TAR member')
            with archive.extractfile(member) as file:
                raw = file.read()
        if hashlib.sha256(raw).hexdigest() != meta['member_sha256']:
            raise ValueError('LoRA member hash mismatch')
        tensors = load_safetensors(raw)
        expected = {name + f'.{side}.weight' for name in self.names for side in ('a', 'b')}
        if set(tensors) != expected:
            raise ValueError('LoRA tensor names do not match semantic q/k/v/o')
        for name, module in self.modules.items():
            a, b = tensors[name + '.a.weight'], tensors[name + '.b.weight']
            if a.shape != (1, module.in_features) or b.shape != (module.out_features, 1):
                raise ValueError('LoRA tensor shape mismatch: ' + name)
            if a.dtype != torch.float32 or b.dtype != torch.float32 or not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError('Non-finite or non-FP32 LoRA tensors')
        self.cache[id_] = tensors
        if len(self.cache) > 32:
            self.cache.popitem(last=False)
        return tensors

    @torch.inference_mode()
    def apply(self, selection):
        """Caller must hold the engine's generation lock; no request-global choice."""
        validate_selection(selection)
        choices = selection.active()
        key = tuple((choice.id, choice.weight) for choice in choices)
        started = time.perf_counter()
        if key != self.active_key:
            # Fully validate every member before changing any live parameter.
            updates = [(self.tensors(choice.id), choice.weight) for choice in choices]
            if self.original is None:
                self.original = {name: module.weight.detach().cpu().clone() for name, module in self.modules.items()}
            merged = {}
            for name, base in self.original.items():
                weight = base.float().clone()
                for tensors, scale in updates:
                    weight.addmm_(tensors[name + '.b.weight'], tensors[name + '.a.weight'], alpha=scale)
                if not torch.isfinite(weight).all():
                    raise ValueError('Non-finite fused LoRA weight')
                merged[name] = weight.to(base.dtype)
                if not torch.isfinite(merged[name]).all():
                    raise ValueError('LoRA update overflows the model dtype')
            device = next(iter(self.modules.values())).weight.device
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            backup = {name: module.weight.detach().cpu().clone() for name, module in self.modules.items()}
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
        return {'repo': settings.LORA_REPO, 'revision': settings.LORA_REVISION,
            'selection': selection.model_dump(), 'active': [choice.model_dump() for choice in choices],
            'experimental_mix': len(choices) > 1, 'switch_ms': round((time.perf_counter() - started) * 1000),
            'backend': 'FP32 update fused into stable BF16 base storage; not bit-identical to reference FP32 branch addition'}


if __name__ == '__main__':
    print(json.dumps({'prefetched': prefetch()['counts']}, indent=2))
