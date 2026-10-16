import json
from types import SimpleNamespace
from contextlib import nullcontext
import threading

import pytest
import torch
from torch import nn

from rl_adapter import RLLinear, attach_rl
from acting_models import MODELS


def test_residual_is_exact_separate_fp32_and_keeps_base_unchanged():
    base=nn.Linear(3,4).float();original=base.weight.detach().clone()
    layer=RLLinear(base,2)
    with torch.no_grad(): layer.a.fill_(.125);layer.b.fill_(.25)
    x=torch.randn(2,3)
    expected=base(x)+torch.nn.functional.linear(torch.nn.functional.linear(x,layer.a),layer.b)
    assert torch.equal(layer(x),expected)
    assert torch.equal(base.weight,original) and layer.weight is base.weight and layer.bias is base.bias


def miniature():
    model=nn.Module()
    model.transformer=nn.ModuleDict({'block':nn.ModuleDict({key:nn.Linear(3,3) for key in ['q_proj','k_proj','v_proj','o_proj']})})
    model.local_transformer=nn.ModuleDict({'block':nn.ModuleDict({'attn':nn.ModuleDict({'c_attn':nn.Linear(3,9),'c_proj':nn.Linear(3,3)})})})
    return model


def test_adapter_strict_coverage_names_dimensions_and_dtype(tmp_path,monkeypatch):
    import rl_adapter
    root=tmp_path/'base';root.mkdir()
    model=miniature();state={}
    for name,module in model.named_modules():
        if isinstance(module,nn.Linear):
            state[name+'.a']=torch.zeros(128,module.in_features)
            state[name+'.b']=torch.zeros(module.out_features,128)
    release=MODELS['small_rl']
    config={'base_weight_sha256':release['sha256'],'adapter_sha256':release['adapter_sha256'],
        'rank':128,'alpha':128,'residual_scale':1,'adapter_tensors':len(state),
        'adapter_parameters':sum(v.numel() for v in state.values())}
    (tmp_path/'adapter_config.json').write_text(json.dumps(config))
    monkeypatch.setattr(rl_adapter,'load_file',lambda *a,**k:state)
    report=attach_rl(model,root,release)
    assert report['semantic']==4 and report['local']==2 and not report['merged']
    assert all(not p.requires_grad for p in model.parameters())
    state.pop(next(iter(state)))
    with pytest.raises(ValueError,match='tensor names'): attach_rl(miniature(),root,release)


def test_rl_releases_pin_distinct_final_bases_and_adapters():
    from small_app import Turn,candidate_batch_size
    for key,step in [('small_rl',6109),('large_rl',3055)]:
        release=MODELS[key]
        assert release['rl_bundle'] and release['policy_dtype']=='fp32'
        assert f'{step:08d}' in release['checkpoint'] and len(release['adapter_sha256'])==64
        body=Turn(message='Hello',model=key,generation='best',n=8,voice_mode='none')
        assert candidate_batch_size(body)==release['max_batch_size']
    assert Turn(message='Hello',voice_mode='none').temperature==.7


def test_large_memory_mode_is_reversible_without_touching_other_gpu(monkeypatch):
    from model_memory import place_auxiliaries
    calls=[]
    class Module:
        def __init__(self,name):self.name=name
        def to(self,device):calls.append((self.name,device));return self
    state={'scorers':SimpleNamespace(asr=SimpleNamespace(model=Module('asr')),
        ears=SimpleNamespace(models={'base':(Module('ears'),None,None)},lock=threading.RLock())),
        'finisher':SimpleNamespace(available=True,aligner=SimpleNamespace(model=Module('aligner')))}
    monkeypatch.setattr(torch.cuda,'device',lambda *a:nullcontext())
    monkeypatch.setattr(torch.cuda,'empty_cache',lambda:None)
    place_auxiliaries(state,'large_rl')
    assert calls==[('asr','cpu'),('ears','cpu'),('aligner','cpu')]
    place_auxiliaries(state,'small_rl')
    import settings
    assert state['scorers'].asr.device==settings.SCORE_DEVICE and calls[-1]==('aligner',settings.ALIGN_DEVICE)
