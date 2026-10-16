"""Acting release/API contracts; real GPU/HTTPS tests are separate."""
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import torch

import acting_engine
from acting_models import MODELS
import small_app
from small_app import Turn


def test_acting_default_and_limits():
    turn=Turn(message='Hello')
    assert (turn.model,turn.batch_size,turn.generation,turn.n,turn.streaming,turn.chunk_frames)==('small_rl',1,'single',1,True,10)
    assert turn.voice_profile=='emolia_c1699' and not turn.dynamic_reference and not turn.speaker_steering
    assert turn.format=='structured' and not turn.adapters.active()
    for model,limit in [('large_acting',4),('small_acting',8),('small_alpha',8),('large_alpha',4),('small_rl',8),('large_rl',2)]:
        assert Turn(message='Hello',model=model,batch_size=limit,n=12).batch_size==limit
        with pytest.raises(ValueError):Turn(message='Hello',model=model,batch_size=limit+1)
        with pytest.raises(ValueError):Turn(message='Hello',model=model,generation='search_beam',search_n=limit+1)
    with pytest.raises(ValueError):Turn(message='Hello',adapters={'ref15':True})
    with pytest.raises(ValueError):Turn(message='Hello',speaker_steering=True)


def test_pinned_weight_contracts_and_public_routes():
    client=TestClient(small_app.app)
    catalog=client.get('/api/models/catalog').json()
    assert catalog['default']=='small_rl' and len(catalog['models'])==6
    assert not catalog['steering_available'] and not catalog['adapters_available']
    assert {x['step'] for x in MODELS.values()}=={432,15238,3775,2377,6109,3055}
    assert MODELS['small_acting']['checkpoint']=='checkpoints/HVS-Small-2026-10-12/model_bf16.pt'
    assert MODELS['small_acting']['repo']=='laion/Humaneness-Voice-Small'
    assert all(len(x['sha256'])==64 and len(x['revision'])==40 for x in MODELS.values())
    assert not client.get('/api/adapters/catalog').json()['enabled']
    assert client.get('/acting-ui.js').status_code==200
    assert MODELS['small_alpha']['experimental'] and not MODELS['small_alpha']['training_complete']
    assert MODELS['large_alpha']['experimental'] and not MODELS['large_alpha']['training_complete']


def test_complete_state_prefix_and_tied_head_validation(monkeypatch):
    calls=[]
    model=SimpleNamespace(load_state_dict=lambda state,**kw:calls.append((state,kw)) or (['text_lm_head.weight'],[]),
                          tie_weights=lambda:calls.append('tied'))
    monkeypatch.setattr(acting_engine,'load_file',lambda *a,**kw:{'base.transformer.weight':torch.ones(1)})
    acting_engine.load_acting_state(model,'unused')
    assert list(calls[0][0])==['transformer.weight'] and calls[0][1]['assign']
    assert calls[-1]=='tied'
    monkeypatch.setattr(acting_engine,'load_file',lambda *a,**kw:{'wrong.weight':torch.ones(1)})
    with pytest.raises(ValueError,match='base'):acting_engine.load_acting_state(model,'unused')
    monkeypatch.setattr(acting_engine,'load_file',lambda *a,**kw:{'base.transformer.weight':torch.ones(1)})
    model.load_state_dict=lambda *a,**kw:(['transformer.missing'],[])
    with pytest.raises(ValueError,match='mismatch'):acting_engine.load_acting_state(model,'unused')


def test_bf16_small_load_is_weights_only_mmap_strict_and_retied(monkeypatch):
    calls=[]
    state={'transformer.weight':torch.ones(2,dtype=torch.bfloat16)}
    monkeypatch.setattr(torch,'load',lambda path,**kw:calls.append(kw) or state)
    model=SimpleNamespace(load_state_dict=lambda data,**kw:calls.append((data,kw)),
                          tie_weights=lambda:calls.append('tied'))
    acting_engine.load_acting_state(model,'model_bf16.pt',format='pytorch_state_dict')
    assert calls[0]=={'map_location':'cpu','weights_only':True,'mmap':True}
    assert calls[1]==(state,{'strict':True,'assign':True}) and calls[-1]=='tied'
    monkeypatch.setattr(torch,'load',lambda *a,**k:{'lora_up.weight':torch.ones(2)})
    with pytest.raises(ValueError,match='merged full state'):
        acting_engine.load_acting_state(model,'unused',format='pytorch_state_dict')
    monkeypatch.setattr(torch,'load',lambda *a,**k:{'state':{'weight':torch.ones(2)}})
    with pytest.raises(ValueError,match='tensor state dictionary'):
        acting_engine.load_acting_state(model,'unused',format='pytorch_state_dict')


def test_acting_pack_uses_literal_text_not_tagged_script():
    received=[]
    class Processor:
        def build_user_message(self,**kw):received.append(kw);return {'role':'user','content':'test'}
        def __call__(self,*a,**kw):return {'input_ids':[torch.zeros(4,13,dtype=torch.long)]}
    engine=object.__new__(acting_engine.ActingEngine);engine.proc=Processor();engine.device='cpu'
    instruction='GENERAL: angry\nSCRIPT:\n(intensely angry) [1 seconds duration] Hello.'
    ids,mask=engine.pack(instruction,{'text':'Hello.','frames':20,'lang':'de'},torch.ones(10,12),2)
    assert received[0]['text']=='Hello.' and received[0]['instruction']==instruction
    assert received[0]['reference'] is not None and received[0]['language']=='German'
    assert ids.shape==(2,4,13) and mask.shape==(2,4)


def test_model_switch_endpoint_rejects_busy_server(monkeypatch):
    monkeypatch.setattr(small_app,'STATE',{'busy':True,'phase':'ready'})
    assert TestClient(small_app.app).post('/api/models/select',json={'model':'small_acting'}).status_code==409


def test_model_switch_endpoint_resets_busy_after_failure(monkeypatch):
    monkeypatch.setattr(small_app,'STATE',{'busy':False,'phase':'ready'})
    def fail(model):raise RuntimeError('test load failed')
    monkeypatch.setattr(small_app,'switch_acting_model',fail)
    response=TestClient(small_app.app,raise_server_exceptions=False).post('/api/models/select',json={'model':'small_acting'})
    assert response.status_code==500 and not small_app.STATE['busy']
def test_only_continuation_is_compiled_and_mask_strides_are_stable(monkeypatch):
    import acting_engine
    import torch
    calls=[]
    def eager(**kwargs):calls.append(('eager',kwargs));return 'eager'
    def compiled(**kwargs):calls.append(('compiled',kwargs));return 'compiled'
    monkeypatch.setattr(torch,'compile',lambda f,**k:compiled)
    dispatch=acting_engine.compile_decode_only(eager)
    mask=torch.ones(2,20,dtype=torch.bool)[:,:10]
    assert not mask.is_contiguous()
    assert dispatch(attention_mask=mask,past_key_values=None)=='eager'
    assert dispatch(attention_mask=mask,past_key_values=((1,2),))=='compiled'
    assert all(kw['attention_mask'].is_contiguous() for _,kw in calls)
    assert torch.equal(calls[1][1]['attention_mask'],mask)


@pytest.mark.parametrize('model,limit',[('small_acting',8),('large_acting',4),('small_alpha',8),('large_alpha',4)])
def test_best_of_n_is_parallel_even_with_single_stream_batch_one(model,limit):
    from small_app import candidate_batch_size
    assert candidate_batch_size(Turn(message='Test',model=model,generation='best',n=3,batch_size=1))==3
    assert candidate_batch_size(Turn(message='Test',model=model,generation='all',n=12,batch_size=1))==limit
    assert candidate_batch_size(Turn(message='Test',model=model,generation='single',batch_size=1))==1
