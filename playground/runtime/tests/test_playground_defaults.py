"""Live defaults / reference-only identity guidance; no quality claims."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from prompting import REFERENCE_IDENTITY, build, parse_literal


@pytest.mark.parametrize('fmt', ['caption','structured','tags','transcript'])
def test_identity_kept_in_both_cfg_branches_without_changing_words(fmt):
    text='I knew, you would come back.'
    turn={'text':text,'caption':'Anger held back, then soft relief.',
          'general':REFERENCE_IDENTITY+' Anger held back, then soft relief.',
          'script':'(deeply angry, held in) I knew, [0.4 seconds pause] (sigh, 0.3 seconds) you would come back.',
          'tags':['contained_anger']}
    spec=build(turn,fmt,reference_identity=True)
    if fmt=='transcript':
        assert spec['prompt']==spec['unconditional'] and not spec['identity_cfg']
    else:
        assert spec['prompt'].count(REFERENCE_IDENTITY)==1
        assert spec['unconditional'].count(REFERENCE_IDENTITY)==1
        assert spec['identity_cfg']
        assert 'Anger held back' not in spec['unconditional']
    assert REFERENCE_IDENTITY not in spec['general']
    assert parse_literal(spec['prompt'],fmt)['text']==text
    if fmt=='structured':
        for token in ['[0.4 seconds pause]','(sigh, 0.3 seconds)']:
            assert token in spec['prompt'] and token in spec['unconditional']


def test_manual_prompt_and_frozen_study_cfg_remain_unchanged():
    literal='CAPTION: '+REFERENCE_IDENTITY+' Restrained anger.\nTRANSCRIPT: "Stay here."'
    turn=parse_literal(literal,'caption')
    historical=build(turn,'caption',literal=literal)
    live=build(turn,'caption',literal=literal,reference_identity=True)
    assert historical['prompt']==live['prompt']==literal
    assert REFERENCE_IDENTITY not in historical['unconditional']
    assert 'identity_cfg' not in historical
    assert REFERENCE_IDENTITY in live['unconditional']
    plain=literal.replace(REFERENCE_IDENTITY+' ','')
    manual=build(parse_literal(plain,'caption'),'caption',literal=plain,reference_identity=True)
    assert manual['prompt']==plain and not manual['identity_cfg']


def test_identity_boilerplate_inside_spoken_text_is_never_removed():
    turn={'text':REFERENCE_IDENTITY,'caption':'Quiet conversation.'}
    spec=build(turn,'caption',reference_identity=True)
    assert parse_literal(spec['prompt'],'caption')['text']==REFERENCE_IDENTITY
    literal='CAPTION: Quiet conversation.\nTRANSCRIPT: '+json.dumps(REFERENCE_IDENTITY)
    manual=build(parse_literal(literal,'caption'),'caption',literal=literal,reference_identity=True)
    assert not manual['identity_cfg']
    assert manual['prompt']==literal


@pytest.mark.parametrize('reference_enabled',[True,False])
def test_luna_receives_reference_presence_and_identity_policy(monkeypatch,reference_enabled):
    import asyncio
    import httpx
    import settings
    from director import Director
    monkeypatch.setattr(settings,'luna_key',lambda:'test-only')
    response={'text':'Stay here.','script':'(quietly) Stay here.','general':'Quiet.','caption':'Quiet.',
              'tags':['quiet'],'language':'en'}
    def respond(request):
        payload=json.loads(request.content)
        assert payload['model']==settings.LUNA_MODEL and 'tools' not in payload
        assert REFERENCE_IDENTITY in payload['messages'][0]['content']
        assert 'do not\nmake ordinary caption mode longer' in payload['messages'][0]['content']
        assert ('Reference audio enabled: yes' if reference_enabled else 'Reference audio enabled: no') in payload['messages'][-1]['content']
        return httpx.Response(200,json={'choices':[{'message':{'content':json.dumps(response)}}]})
    async def run():
        agent=Director();await agent.client.aclose()
        agent.client=httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            actual,_=await agent.direct('Stay here.','caption','en','perform',[],reference_enabled=reference_enabled)
            assert actual['text']=='Stay here.'
        finally:await agent.close()
    asyncio.run(run())


@pytest.mark.parametrize('enabled', [True, False])
def test_single_live_sse_chunks_before_result_and_private_mp3(tmp_path,monkeypatch,enabled):
    import small_app
    from storage import AudioStore
    from search_scoring import SearchScorer
    wave=np.ones(3*48000,np.float32)*.05
    calls=[]
    def generate(*args,**kwargs):
        calls.append((args,kwargs))
        assert args[1]==1 and kwargs['streaming'] and kwargs['chunk_frames']==10
        for chunk in [wave[:76800],wave[76800:]]:
            kwargs['on_audio'](chunk)
        return [wave]
    monkeypatch.setattr(SearchScorer,'transcribe_batch',lambda self,audio:
        (['Hello world'],[[{'word':'Hello','start':.4,'end':.9},
                          {'word':'world','start':1.,'end':1.8}]]))
    monkeypatch.setattr(small_app,'store',AudioStore(tmp_path))
    engine=SimpleNamespace(generate=generate,last_performance=None,
        configure_adapters=lambda pick:{'selection':pick.model_dump(),'active':[],'switch_ms':0})
    scorers=SimpleNamespace(asr=object(),score=lambda waves,spec:[{'rank':0,'sec':len(waves[0])/48000,
        'reward':1.,'wer':0.,'hyp':'Hello world','clap':.2,'blend':5.,'genuineness':3.}])
    finisher=SimpleNamespace(finish=lambda waves,*args:(waves,[{'enabled':enabled}]))
    monkeypatch.setattr(small_app,'STATE',{'phase':'ready','busy':False,'engine':engine,
        'scorers':scorers,'finisher':finisher})
    client=TestClient(small_app.app)
    response=client.post('/api/turn',json={'message':'CAPTION: Quiet.\nTRANSCRIPT: "Hello world"',
        'input_mode':'manual','format':'caption','voice_mode':'none','generation':'single','n':1,
        'align':enabled,'speaker':{'method':'off'},'ranking':{'mode':'legacy'}})
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    assert not [e for e in events if e['event']=='error']
    chunks=[e for e in events if e['event']=='audio_chunk']
    result=next(e for e in events if e['event']=='result')
    assert chunks and events.index(chunks[0])<events.index(result)
    assert result['generation']=='single' and len(result['candidates'])==1
    assert result['adapters']['selection']['id'] is None
    assert result['live_stream']['enabled']==enabled
    other=TestClient(small_app.app)
    for chunk in chunks:
        audio=client.get(chunk['url'])
        assert audio.status_code==200 and audio.headers['content-type']=='audio/mpeg'
        assert other.get(chunk['url']).status_code==404
    assert not small_app.STATE['busy']
