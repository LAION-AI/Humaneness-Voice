"""Unit contracts and fusion rollback; real GPU checks are recorded separately."""
import math

import pytest
from safetensors.torch import save_file
import torch

import dpo_adapters as dpo
from ref15_adapter import FORMAT, TARGETS, parse_payload
import settings


@pytest.fixture
def exact_release(tmp_path):
    modules={name:torch.nn.Linear(3,2,bias=False) for name in TARGETS}
    tensors={name+'.a.weight':torch.ones(128,3) for name in TARGETS}
    tensors.update({name+'.b.weight':torch.ones(2,128)/128 for name in TARGETS})
    payload={'format':FORMAT,'base_model':settings.REPO,'base_stage':'S3',
             'base_step':32337,'rank':128,'alpha':128,'state_dict':tensors}
    return tmp_path/'exact.pt',payload,modules


def test_published_ref15_contract_has_scale_one(exact_release):
    path,payload,modules=exact_release
    torch.save(payload,path)
    pairs,scale=parse_payload(path,{'release_contract':True},modules)
    assert len(pairs)==112 and scale==1


@pytest.mark.parametrize('field,value',[('format','unknown'),('base_model','different/base'),
    ('base_stage','S2'),('base_step',32336),('rank',64),('alpha',256)])
def test_published_contract_rejects_other_releases(exact_release,field,value):
    path,payload,modules=exact_release
    torch.save({**payload,field:value},path)
    with pytest.raises(ValueError,match='contract mismatch'):
        parse_payload(path,{'release_contract':True},modules)


def test_published_contract_requires_all_224_tensors(exact_release):
    path,payload,modules=exact_release
    payload['state_dict'].pop(next(iter(payload['state_dict'])))
    torch.save(payload,path)
    with pytest.raises(ValueError,match='224 exact'):
        parse_payload(path,{'release_contract':True},modules)


@pytest.fixture
def payload(tmp_path):
    path=tmp_path/'adapter_model.safetensors'
    tensors={'base_model.model.proj.lora_A.weight':torch.ones(128,3),
             'base_model.model.proj.lora_B.weight':torch.ones(2,128)/128}
    save_file(tensors,path)
    config={'r':128,'lora_alpha':256,'base_sha256':dpo.BASE_SHA256}
    modules={'proj':torch.nn.Linear(3,2,bias=False)}
    return path,tensors,config,modules


def test_explicit_peft_and_payload_scaling(payload,tmp_path):
    path,tensors,config,modules=payload
    pairs,scale=parse_payload(path,config,modules)
    assert set(pairs)=={'proj'} and scale==2
    assert parse_payload(path,{**config,'use_rslora':True},modules)[1]==pytest.approx(256/math.sqrt(128))
    raw=tmp_path/'adapter.pt'
    torch.save({'adapter':{'proj.a.weight':tensors['base_model.model.proj.lora_A.weight'],
        'proj.b.weight':tensors['base_model.model.proj.lora_B.weight']},
        'rank_size':128,'alpha':256,'base_sha256':dpo.BASE_SHA256},raw)
    assert parse_payload(raw,{},modules)[1]==2


@pytest.mark.parametrize('change',[{'r':64},{'lora_alpha':float('nan')},{'base_sha256':'0'*64},
    {'use_dora':True},{'rank_pattern':{'proj':16}},{'alpha_pattern':{'proj':128}},
    {'base_model_name_or_path':'different/base'}])
def test_incompatible_contracts_rejected(payload,change):
    path,_,config,modules=payload
    with pytest.raises(ValueError):parse_payload(path,{**config,**change},modules)


def test_no_guessed_scaling_or_shape_based_name_matching(payload):
    path,tensors,config,modules=payload
    with pytest.raises(ValueError,match='declare rank'):
        parse_payload(path,{'r':128,'base_sha256':dpo.BASE_SHA256},modules)
    with pytest.raises(ValueError,match='compatible base'):
        parse_payload(path,{'r':128,'lora_alpha':256},modules)
    tensors['unrecognized.weight']=torch.zeros(2,3);save_file(tensors,path)
    with pytest.raises(ValueError,match='naming'):parse_payload(path,config,modules)


def toy_bank(tmp_path,monkeypatch,full_ft=False):
    path=tmp_path/'base';path.write_bytes(b'fixture')
    monkeypatch.setattr(dpo,'BASE_SHA256',dpo.sha256(path))
    if full_ft:
        path.write_bytes(b'verified-full-ft-fixture')
    model=torch.nn.Module()
    for prefix in ('transformer.layers','local_transformer.h'):
        parent=model
        for part in prefix.split('.'):
            child=torch.nn.Module();parent.add_module(part,child);parent=child
        for i in range(28 if prefix.startswith('transformer') else 1):
            layer=torch.nn.Module();parent.add_module(str(i),layer)
            categories={'self_attn':('q_proj','v_proj'),'mlp':('gate_proj','up_proj','down_proj')} if prefix.startswith('transformer') else {'attn':('c_attn','c_proj'),'mlp':('fc_in','fc_out')}
            for category,names in categories.items():
                group=torch.nn.Module();layer.add_module(category,group)
                for name in names:
                    module=torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16)
                    with torch.no_grad():module.weight.fill_(1.)
                    group.add_module(name,module)
    model.extra=torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16)
    with torch.no_grad():model.extra.weight.fill_(1.)
    if full_ft:
        with torch.no_grad():
            for param in model.parameters():param.fill_(2.5)
    return dpo.DPOBank(model,path,expected_base_sha256=dpo.sha256(path) if full_ft else None)


def test_full_ft_dpo_opt_in_reports_cross_base_and_restores_actual_loaded_base(tmp_path,monkeypatch):
    bank=toy_bank(tmp_path,monkeypatch,full_ft=True)
    assert bank.base_sha256 != dpo.BASE_SHA256
    pairs={name:(torch.ones(16,3),torch.ones(2,16)/16) for name in bank.dpo_modules}
    monkeypatch.setattr(bank,'contribution',lambda pick:(pairs,2.))
    before={n:m.weight.clone() for n,m in bank.all_modules.items()}
    pointers={n:m.weight.data_ptr() for n,m in bank.all_modules.items()}
    report=bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016'))
    assert report['cross_base'] and report['experimental_mix']
    assert report['base_sha256']==bank.base_sha256
    assert report['adapter_training_base_sha256']==dpo.BASE_SHA256
    assert all(torch.all(m.weight==4.5) for m in bank.dpo_modules.values())
    restored=bank.apply(dpo.AdapterSelection())
    assert not restored['active'] and not restored['cross_base']
    assert all(torch.equal(before[n],m.weight) and m.weight.data_ptr()==pointers[n]
               for n,m in bank.all_modules.items())


def test_full_ft_hash_must_be_verified_explicitly(tmp_path):
    path=tmp_path/'unverified-base';path.write_bytes(b'unknown')
    with pytest.raises(ValueError,match='base'):
        dpo.DPOBank(torch.nn.Module(),path)
    with pytest.raises(ValueError,match='base'):
        dpo.DPOBank(torch.nn.Module(),path,expected_base_sha256=settings.FULL_FT_SHA256)


def test_base_dpo_ref15_combination_switch_and_exact_restore(tmp_path,monkeypatch):
    bank=toy_bank(tmp_path,monkeypatch)
    dpo_pairs={name:(torch.ones(16,3),torch.ones(2,16)/16) for name in bank.dpo_modules}
    target=next(iter(bank.dpo_modules))
    ref_pairs={name:(torch.ones(128,3),torch.ones(2,128)/128) for name in (target,'extra')}
    monkeypatch.setattr(bank,'contribution',lambda pick:(ref_pairs,2.) if pick.id==dpo.REF15_ID else (dpo_pairs,2.))
    pointers={n:m.weight.data_ptr() for n,m in bank.all_modules.items()}
    pick=dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',weight=.5,ref15=True,ref15_weight=.25)
    report=bank.apply(pick)
    assert report['experimental_mix'] and len(report['active'])==2
    assert torch.all(bank.modules[target].weight==2.5) and torch.all(bank.modules['extra'].weight==1.5)
    bank.apply(pick)
    assert torch.all(bank.modules[target].weight==2.5)
    bank.apply(dpo.AdapterSelection(ref15=True))
    assert torch.all(bank.modules[target].weight==3) and torch.all(bank.modules['extra'].weight==3)
    bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',weight=.5))
    assert torch.all(bank.modules[target].weight==2) and torch.all(bank.modules['extra'].weight==1)
    bank.apply(dpo.AdapterSelection())
    assert all(torch.all(m.weight==1) and pointers[n]==m.weight.data_ptr() for n,m in bank.all_modules.items())


def test_failed_second_contributor_leaves_active_weights_unchanged(tmp_path,monkeypatch):
    bank=toy_bank(tmp_path,monkeypatch)
    pairs={name:(torch.ones(16,3),torch.ones(2,16)/16) for name in bank.dpo_modules}
    def contribution(pick):
        if pick.id==dpo.REF15_ID:raise ValueError('Unverified Ref15 format')
        return pairs,2.
    monkeypatch.setattr(bank,'contribution',contribution)
    bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016'))
    before={n:m.weight.clone() for n,m in bank.all_modules.items()}
    with pytest.raises(ValueError,match='Unverified'):
        bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',ref15=True))
    assert all(torch.equal(before[n],m.weight) for n,m in bank.all_modules.items())


def test_zero_strength_and_independent_opt_in():
    assert not dpo.AdapterSelection().active()
    assert not dpo.AdapterSelection(ref15=True,ref15_weight=0).active()
    assert [x.id for x in dpo.AdapterSelection(ref15=True).active()]==[dpo.REF15_ID]
    assert len(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',ref15=True).active())==2
    for invalid in ({'ref15_weight':float('nan')},{'ref15_weight':2.1},{'id':dpo.REF15_ID}):
        with pytest.raises(ValueError):dpo.AdapterSelection(**invalid)
