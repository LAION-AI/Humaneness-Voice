import json
from pathlib import Path
import sys
import threading
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pydantic import ValidationError

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import dpo_adapters as dpo
import dynamic_references as refs
import settings
from storage import AudioStore


def test_defaults_catalog_and_validation():
    from small_app import Turn
    assert Turn(message='Hello').search_seconds == 30
    assert Turn(message='Hello').generation == 'single'
    assert Turn(message='Hello').n == 1 and Turn(message='Hello').streaming
    assert Turn(message='Hello').search_n == 4
    assert Turn(message='Hello').adapters.id is None
    assert not Turn(message='Hello').adapters.ref15
    assert len(dpo.catalog()['adapters']) == 6
    assert sum(x['default'] for x in dpo.catalog()['adapters']) == 0
    assert dpo.catalog()['ref15']['rank']==128 and dpo.catalog()['default'] is None
    for value in ({'id':'../../bad'}, {'voice':{'id':'profile/test'}}, {'weight':float('nan')}):
        with pytest.raises(ValidationError):
            dpo.AdapterSelection.model_validate(value)
    with pytest.raises(ValidationError):
        Turn(message='x',voice_mode='profile',voice_profile='missing')
    with pytest.raises(ValidationError):
        Turn(message='x',voice_mode='none',reference_id='ignored')


def test_dpo_fusion_scaling_targets_and_exact_restore(tmp_path, monkeypatch):
    base = tmp_path/'base';base.write_bytes(b'test')
    monkeypatch.setattr(dpo, 'BASE_SHA256', dpo.sha256(base))
    model = torch.nn.Module()
    for name in ['transformer.layers', 'local_transformer.h']:
        parent = model
        for part in name.split('.'):
            child = torch.nn.Module();parent.add_module(part, child);parent = child
        for layer in range(28 if name.startswith('transformer') else 1):
            block = torch.nn.Module();parent.add_module(str(layer),block)
            categories = {'self_attn':('q_proj','v_proj'), 'mlp':('gate_proj','up_proj','down_proj')} if name.startswith('transformer') else {'attn':('c_attn','c_proj'), 'mlp':('fc_in','fc_out')}
            for category, names in categories.items():
                group=torch.nn.Module();block.add_module(category,group)
                for key in names:
                    linear=torch.nn.Linear(3,2,bias=False,dtype=torch.bfloat16)
                    with torch.no_grad():linear.weight.fill_(1.)
                    group.add_module(key,linear)
    bank=dpo.DPOBank(model,base)
    tensors={}
    for name in bank.modules:
        tensors[name+'.a.weight']=torch.ones(16,3)
        tensors[name+'.b.weight']=torch.full((2,16),.125)
    payload={'format':'humaneness-voice-small-dpo-lora-adapter-v1','base_sha256':dpo.BASE_SHA256,
             'rank_size':16,'alpha':32,'adapter':tensors}
    path=tmp_path/'adapter.pt';torch.save(payload,path)
    monkeypatch.setattr(dpo,'download',lambda *a:path)
    monkeypatch.setattr(dpo,'metadata',lambda *a:{'adapter_sha256':dpo.sha256(path)})
    pointers={name:module.weight.data_ptr() for name,module in bank.modules.items()}
    bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',weight=.5))
    assert all(torch.equal(m.weight,torch.full_like(m.weight,3.)) for m in bank.modules.values())
    bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016',weight=.5))
    bank.apply(dpo.AdapterSelection(id=None))
    assert all(torch.equal(m.weight,torch.ones_like(m.weight)) and m.weight.data_ptr()==pointers[n]
               for n,m in bank.modules.items())
    bank.cache.clear();payload['adapter'].pop(next(iter(tensors)));torch.save(payload,path)
    with pytest.raises(ValueError,match='target names'):
        bank.apply(dpo.AdapterSelection(id='subset_pitch_quality_p2_rank016'))
    assert all(torch.equal(m.weight,torch.ones_like(m.weight)) for m in bank.modules.values())


def test_original_ten_profiles_have_condition_clips():
    data=refs.catalog()
    assert {v['name'] for v in data['voices']} == {'Idris','Selma','Cormac','Mira','Rasmus','Nora','Béla','Anton','Robin','Osku'}
    assert not any(v['dynamic'] for v in data['voices'])
    assert sum(v['recommended'] for v in data['voices']) == 6
    assert data['reference_policy'] == 'fixed'
    for voice in refs.profiles():
        assert all(gid.startswith(voice+'|') for gid in refs.clips(voice))


def test_dynamic_retrieval_filters_speaker_language_and_isolated_burst(tmp_path,monkeypatch):
    clip=tmp_path/'clip';clip.touch()
    monkeypatch.setattr(refs,'profiles',lambda:{'chosen':{'name':'Chosen','anchor':str(clip)}})
    conditions=[{'voice':voice,'lang':lang,'block':block,'gid':str(i),'emotion':emotion}
                for i,(voice,lang,block,emotion) in enumerate([
                    ('other','en','emotion','Anger'), ('chosen','de','emotion','Anger'),
                    ('chosen','en','burst_isolated','Anger'),('chosen','en','emotion','Anger'),
                    ('chosen','en','emotion','Sadness')])]
    monkeypatch.setattr(refs,'clips',lambda voice:{str(i):clip for i in range(5)})
    monkeypatch.setattr(refs,'fixed_clip',lambda voice:(clip,{'key':'fixed','seconds':1,'sha256':'fixture'}))
    retrieval=SimpleNamespace(ok=True,by_voice={'chosen':list(range(5))},cond=conditions,
        C=np.array([[1.,0]]*4+[[0,1.]]),A=np.eye(2),qmu=np.zeros(2),
        emotions=['Anger','Sadness'],embed=lambda text:np.array([1.,0.]) if 'angry' in text else np.array([0.,1.]),lock=threading.Lock())
    spec={'general':'angry delivery','tagged':'','lang':'en','text':'Words never enter query'}
    _, report=refs.choose(retrieval,spec,'chosen',fixed=False)
    assert report['gid']=='3' and report['dynamic'] and 'Words' not in report['query']
    _, report=refs.choose(retrieval,{**spec,'general':'sad delivery'},'chosen',fixed=False)
    assert report['gid']=='4'
    _, report=refs.choose(retrieval,{**spec,'general':''},'chosen',fixed=False)
    assert not report['dynamic']


def test_reference_survives_restart_and_cookie_restores_owner(tmp_path,monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    store=AudioStore(tmp_path,ttl=1)
    owner='a'*32
    item=store.new(owner,'Own voice','reference','A');item.path.touch();item.playback.touch()
    store.persist_references()
    restored=AudioStore(tmp_path,ttl=1)
    assert restored.cleanup(item.touched+100)==[]
    assert restored.get(item.id,owner).label=='Own voice'
    monkeypatch.setattr(small_app,'store',restored)
    monkeypatch.setattr(small_app,'sessions',{})
    client=TestClient(small_app.app);client.cookies.set('small_session',owner)
    assert client.get('/api/state').json()['references'][0]['id']==item.id
    assert TestClient(small_app.app).get('/api/audio/'+item.id).status_code==404
    restored.remove(restored.get(item.id,owner))
    assert not list(tmp_path.iterdir())
