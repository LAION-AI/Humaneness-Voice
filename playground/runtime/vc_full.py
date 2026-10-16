"""Strict, pinned full 40k Humaneness VC load; no parent/LoRA fallback."""
import hashlib
import json

import torch
from safetensors.torch import load_file

import settings
from vc_grow import StyledSpeakerProjection, TARGET_STATIC, verified_download


def install(model):
    checkpoint=verified_download(settings.VC_FULL_REPO,settings.VC_FULL_CHECKPOINT,
        settings.VC_FULL_REVISION,settings.VC_FULL_SHA256)
    order=verified_download(settings.VC_FULL_REPO,'conditioning/score_order.json',settings.VC_FULL_REVISION)
    names=json.loads(order.read_text())['score_order']
    if len(names)!=99 or len(set(names))!=99 or not TARGET_STATIC.issubset(names):
        raise RuntimeError('Invalid full VC conditioning order')
    # The supplied frozen conditioner is byte-identical to our pinned Ears
    # Medium release. Verify all shared assets before reusing that GPU model.
    from pathlib import Path
    from huggingface_hub import snapshot_download
    from ears_reward import REPO, REVISION
    assets=['model.py','classes.json','training_normalization.json',
        'medium/config.json','medium/preprocessor_config.json','medium/model.safetensors']
    shared=Path(snapshot_download(REPO,revision=REVISION,token=False,
        cache_dir=str(settings.HERE/'model-cache/ears/hub'),allow_patterns=assets))
    for relative in assets:
        supplied=verified_download(settings.VC_FULL_REPO,'conditioning/ears-medium/'+relative,settings.VC_FULL_REVISION)
        if hashlib.sha256(supplied.read_bytes()).digest()!=hashlib.sha256((shared/relative).read_bytes()).digest():
            raise RuntimeError('Full VC conditioner differs from pinned shared Ears: '+relative)
    with torch.random.fork_rng(devices=[]):
        model.flow.spk_embed_affine_layer=StyledSpeakerProjection(model.flow.spk_embed_affine_layer)
    # All backbone, tokenizer, CAMP, vocoder and projection keys must match.
    # Never use strict=False or restore optimizer/pickle files for this release.
    model.load_state_dict(load_file(str(checkpoint),device='cpu'),strict=True)
    model.requires_grad_(False).eval()
    model.flow.decoder.training_cfg_rate=0.
    return names
