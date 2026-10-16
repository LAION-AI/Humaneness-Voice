"""The optional converter is a verified complete model, not the old adapter."""
from types import SimpleNamespace
from pathlib import Path
import json

import pytest
import torch
from torch import nn

import settings
import vc_full
from voice_conversion import ConversionManager


def test_new_converter_contract_and_disabled_default():
    report=ConversionManager().describe()
    assert not report['default']
    assert report['model_repo']=='laion/Humaneness-Voice-Converter'
    assert report['step']==40000 and report['arm']=='merged_full'
    assert report['checkpoint']=='model.safetensors'
    assert report['checkpoint_sha256']=='45bbf058ea2a82f1840dc6161f01be929217d28843003f5354336c81b11ea3dd'


@pytest.mark.parametrize('bad_assets,bad_state',[(False,False),(True,False),(False,True)])
def test_full_install_checks_conditioner_and_requires_exact_state(tmp_path,monkeypatch,bad_assets,bad_state):
    names=sorted(vc_full.TARGET_STATIC)+[f'axis{i}' for i in range(85)]
    order=tmp_path/'order.json';order.write_text(json.dumps({'score_order':names}))
    supplied=tmp_path/'supplied';supplied.write_bytes(b'changed' if bad_assets else b'original')
    root=tmp_path/'shared'
    for f in ['model.py','classes.json','training_normalization.json','medium/config.json',
              'medium/preprocessor_config.json','medium/model.safetensors']:
        p=root/f;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(b'original')
    def download(repo,file,revision,expected=None):
        assert repo==settings.VC_FULL_REPO and revision==settings.VC_FULL_REVISION
        if file=='model.safetensors':
            assert expected==settings.VC_FULL_SHA256
            return tmp_path/'weights.safetensors'
        return order if file.endswith('score_order.json') else supplied
    monkeypatch.setattr(vc_full,'verified_download',download)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub,'snapshot_download',lambda *a,**k:str(root))
    model=nn.Module();model.flow=nn.Module()
    model.flow.spk_embed_affine_layer=nn.Linear(192,80)
    model.flow.decoder=SimpleNamespace(training_cfg_rate=.2)
    # Fixture includes the same new style projection as the real full export.
    expected=nn.Module();expected.flow=nn.Module()
    expected.flow.spk_embed_affine_layer=vc_full.StyledSpeakerProjection(nn.Linear(192,80))
    state=expected.state_dict()
    if bad_state:state.pop(next(iter(state)))
    monkeypatch.setattr(vc_full,'load_file',lambda *a,**k:state)
    if bad_assets or bad_state:
        with pytest.raises(RuntimeError):vc_full.install(model)
    else:
        assert vc_full.install(model)==names
        assert not model.training and not any(p.requires_grad for p in model.parameters())
        assert model.flow.decoder.training_cfg_rate==0


def test_english_ui_does_not_restore_old_vc_on_preference():
    html=(Path(__file__).resolve().parents[1]/'index.html').read_text()
    assert '<html lang="en">' in html
    assert "let vcPreference=false" in html
    assert "small.vc.full40k.v1" in html and 'small.vc.grow24k.v1' not in html
    assert 'id="vc" disabled checked' not in html
    from actor_prompt import SYSTEM
    assert 'Gentle, natural and spontaneous delivery.' in SYSTEM
    assert 'explicit anger' in SYSTEM


@pytest.mark.parametrize('explicit',[False,True,None])
def test_fallback_direction_both_surfaces_and_explicit_acting_is_preserved(explicit):
    from director import enforce_default_delivery
    original={'text':'Do not change these words.', 'general':'Intensely angry and shouting.' if explicit else 'Conversation.',
        'caption':'An expressive performance.', 'delivery_explicit':explicit}
    out=enforce_default_delivery(original,True)
    assert out['text']==original['text']
    if explicit is False:
        from prompting import REFERENCE_IDENTITY
        for f in ('caption','general'):
            assert out[f].startswith(REFERENCE_IDENTITY+' Gentle, natural and spontaneous delivery.')
    else:assert out==original
    assert 'Gentle' not in original['caption']
