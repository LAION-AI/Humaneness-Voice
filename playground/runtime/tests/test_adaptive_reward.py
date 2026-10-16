import asyncio
import json

import httpx
import pytest

from adaptive_reward import AUTOMATIC_DIMENSIONS, DIMENSIONS, RewardPlan, adaptive_quality, reward_catalog_context


def plan(*terms):return RewardPlan(terms=list(terms))


def test_catalog_native_ranges_and_actual_heads():
    assert len([d for d in DIMENSIONS if d.startswith('vn:')])==57
    assert len([d for d in DIMENSIONS if d.startswith('eiv:')])==55
    assert DIMENSIONS['vn:BKGN']['max']==4
    assert DIMENSIONS['vn:EXPL']['max']==2
    assert DIMENSIONS['eiv:Valence']['min']==-3
    assert DIMENSIONS['eiv:score_speech_quality']['max']==2.4
    assert 'eiv:Jealousy_&_Envy' in DIMENSIONS
    assert 'eiv:Authenticity' not in DIMENSIONS
    assert 'eiv:Warm_vs._Cold' not in AUTOMATIC_DIMENSIONS
    assert 'eiv:Warm_vs._Cold' not in reward_catalog_context(automatic=True)
    for code in ['vn:S_WHIS','vn:S_RANT','vn:S_DRAM','vn:S_CASU','vn:S_FORM','eiv:Anger']:
        assert code in reward_catalog_context()


@pytest.mark.parametrize('terms',[
    [],[{'dimension':'bad'}],[{'dimension':'vn:AROU','weight':0}],
    [{'dimension':'vn:AROU','weight':float('nan')}],
    [{'dimension':'vn:AROU','objective':'target','target':7}],
    [{'dimension':'vn:AROU','objective':'target'}],
    [{'dimension':'vn:AROU','target':3}],
    [{'dimension':'vn:AROU'}]*2,
    [{'dimension':d} for d in list(DIMENSIONS)[:5]],
])
def test_plan_rejects_invalid_or_unbounded_terms(terms):
    with pytest.raises(ValueError):RewardPlan(terms=terms)


def test_fixed_scales_bipolar_weights_and_objectives():
    p=plan({'dimension':'vn:AROU','objective':'target','target':3,'weight':2},
           {'dimension':'eiv:Valence','objective':'minimize','weight':1})
    result=adaptive_quality({'vn:AROU':3,'eiv:Valence':0},p,6,10,.5)
    assert result['style_score']==pytest.approx(5/6)
    assert result['quality']==pytest.approx(11/12)
    assert sum(d['normalized_weight'] for d in result['dimension_scores'])==pytest.approx(1)
    assert result['dimension_scores'][1]['normalized']==.5
    # No min/max fitting to a candidate batch; changing another row cannot
    # change this utility. Extreme out-of-scale outputs are explicitly clipped.
    r=adaptive_quality({'vn:AROU':100},plan({'dimension':'vn:AROU'}),0,0,1)
    assert r['quality']==1
    with pytest.raises(ValueError):adaptive_quality({'vn:AROU':float('inf')},plan({'dimension':'vn:AROU'}),0,0)


def test_turn_modes_and_manual_requirement():
    from small_app import Turn
    assert Turn(message='x').reward_mode=='clap'
    p=plan({'dimension':'eiv:Anger','objective':'target','target':2.5})
    for mode in ['adaptive_auto','adaptive_manual']:
        assert Turn(message='x',reward_mode=mode,reward_plan=p,ranking={'mode':'legacy'}).reward_plan==p
    with pytest.raises(ValueError):Turn(message='x',reward_mode='adaptive_manual')
    with pytest.raises(ValueError):Turn(message='x',reward_style_share=1.5)


def test_preview_catalog_and_validated_luna_plan(monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    async def choose(description):
        assert description=='quiet whisper'
        return plan({'dimension':'vn:S_WHIS','objective':'target','target':5})
    monkeypatch.setattr(small_app.director,'reward_plan',choose)
    monkeypatch.setitem(small_app.STATE,'busy',False)
    c=TestClient(small_app.app)
    assert len(c.get('/api/reward/catalog').json()['dimensions'])==112
    r=c.post('/api/reward/plan',json={'description':'quiet whisper'})
    assert r.status_code==200 and r.json()['terms'][0]['dimension']=='vn:S_WHIS'
    monkeypatch.setitem(small_app.STATE,'busy',True)
    assert c.post('/api/reward/plan',json={'description':'quiet whisper'}).status_code==409


@pytest.mark.parametrize('invalid',['imaginary','eiv:Warm_vs._Cold'])
def test_luna_receives_taxonomy_retries_invalid_dimension(monkeypatch,invalid):
    asyncio.run(check_luna(monkeypatch,invalid))


async def check_luna(monkeypatch,invalid):
    import settings
    from director import Director
    monkeypatch.setattr(settings,'luna_key',lambda:'unit-test-placeholder')
    calls=[]
    def handle(request):
        body=json.loads(request.content); calls.append(body)
        assert 'vn:S_WHIS' in body['messages'][0]['content']
        assert 'eiv:score_overall_quality' in body['messages'][0]['content']
        assert 'not the literal meaning' in body['messages'][0]['content']
        answer={'terms':[{'dimension':invalid}]} if len(calls)==1 else {
            'terms':[{'dimension':'vn:S_WHIS','objective':'target','target':5,'weight':1}],
            'reason':'Whispered delivery.'}
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(answer)}}]})
    director=Director()
    await director.client.aclose()
    director.client=httpx.AsyncClient(transport=httpx.MockTransport(handle))
    result=await director.reward_plan('whisper like a secret')
    assert len(calls)==2 and result.terms[0].dimension=='vn:S_WHIS'
    await director.close()
