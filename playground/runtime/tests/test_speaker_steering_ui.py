"""Request-scoped API/UI steering wiring; CPU only, no live mutation."""
import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import small_app
import speaker_steering
from storage import AudioStore


def test_ui_and_api_defaults_off_for_acting():
    from test_ui import Elements
    html=(Path(__file__).resolve().parents[1]/'index.html').read_text()
    attrs=Elements(html).attributes['speaker-steering']
    assert attrs['type']=='checkbox' and 'checked' not in attrs
    assert 'speaker_steering:$(\'speaker-steering\').checked' in html
    assert "$('speaker-steering').checked=false" in html
    assert 'Layer 16' in html and 'experimental' in html
    assert small_app.Turn(message='Hello').speaker_steering is False
    assert small_app.Turn(message='Hello',speaker_steering=False).speaker_steering is False
    with pytest.raises(ValueError,match='incompatible'):
        small_app.Turn(message='Hello',speaker_steering=True)


@pytest.mark.parametrize('mode', ['profile','custom','none'])
def test_only_selected_profile_resolves_vector_no_upload_inheritance(monkeypatch,tmp_path,mode):
    calls=[];configured=[]
    def resolve(engine,voice,path,**options):
        calls.append((voice,path,options))
        active=voice is not None and not options['custom']
        return (np.ones(4) if active else None),{'active':active,'reason':'test'}
    monkeypatch.setattr(speaker_steering,'resolve_speaker_steering',resolve)
    engine=SimpleNamespace(configure_speaker_steering=lambda vector,report:configured.append((vector,report)) or report)
    # Historical manager remains tested without exposing it for Acting.
    body=small_app.Turn.model_construct(message='Hello',voice_mode=mode,speaker_steering=True)
    reference=SimpleNamespace(path=tmp_path/'reference.wav') if mode!='none' else None
    report=asyncio.run(small_app.prepare_speaker_steering(engine,body,reference))
    assert calls[0][0]==(body.voice_profile if mode=='profile' else None)
    assert calls[0][2]['custom']==(mode=='custom')
    assert calls[0][2]['enabled'] is True
    assert report['active']==(mode=='profile')
    if mode!='profile':assert configured[0][0] is None


def test_explicit_lora_disables_vector_even_if_engine_resolver_is_simplified(monkeypatch):
    monkeypatch.setattr(speaker_steering,'resolve_speaker_steering',lambda *a,**kw:
        (np.ones(4),{'active':True,'reason':'test'}))
    configured=[]
    engine=SimpleNamespace(configure_speaker_steering=lambda vector,report:configured.append(vector) or report)
    from dpo_adapters import AdapterSelection
    body=small_app.Turn.model_construct(message='Hello',adapters=AdapterSelection(ref15=True),speaker_steering=True)
    report=asyncio.run(small_app.prepare_speaker_steering(engine,body,SimpleNamespace(path=Path('ref.wav'))))
    assert configured==[None] and not report['active'] and report['reason']=='lora_active'


@pytest.mark.parametrize('failure', [False,True])
def test_profile_sse_reports_applied_vector_and_always_clears_job_state(tmp_path,monkeypatch,failure):
    wave=np.sin(np.arange(48000,dtype=np.float32)*.02)*.05
    fixed=tmp_path/'fixed.wav';sf.write(fixed,wave,48000,subtype='FLOAT')
    configured=[];state={'active':False}
    def configure(vector,report):
        state['active']=vector is not None
        configured.append((state['active'],dict(report)))
        return report
    def generate(*args,**kw):
        assert not state['active']
        if failure:raise RuntimeError('Deliberate generation failure')
        return [wave]
    engine=SimpleNamespace(configure_speaker_steering=configure,generate=generate,last_performance=None,
        configure_adapters=lambda pick:{'selection':pick.model_dump(),'active':[],'switch_ms':0})
    scorers=SimpleNamespace(retriever=None,score=lambda waves,spec:
        [{'rank':0,'sec':1,'reward':1.,'wer':0.,'hyp':'Hello','clap':.2,'blend':5.,'genuineness':3.}])
    finisher=SimpleNamespace(finish=lambda waves,*a:(waves,[{'enabled':False}]))
    monkeypatch.setattr(small_app,'STATE',{'phase':'ready','busy':False,'engine':engine,'scorers':scorers,'finisher':finisher})
    monkeypatch.setattr(small_app,'store',AudioStore(tmp_path/'store'))
    monkeypatch.setattr(small_app.dynamic_references,'choose',lambda *a:
        (fixed,{'voice':small_app.settings.DEFAULT_VOICE,'name':'Selma','mode':'profile','dynamic':False}))
    monkeypatch.setattr(speaker_steering,'resolve_speaker_steering',lambda *a,**kw:
        (None,{'active':False,'enabled':False,'reason':'disabled_by_user','voice':'mira','layer':16,'alpha':.75}))
    response=TestClient(small_app.app).post('/api/turn',json={
        'message':'CAPTION: Quiet.\nTRANSCRIPT: "Hello"','input_mode':'manual','format':'caption',
        'generation':'single','streaming':False,'align':False,'voice_conversion':False,'speaker':{'method':'off'},'ranking':{'mode':'legacy'}})
    events=[json.loads(line[6:]) for line in response.text.splitlines() if line.startswith('data: ')]
    steering=next(e for e in events if e['event']=='speaker_steering')
    assert not steering['active'] and steering['layer']==16
    if failure:assert next(e for e in events if e['event']=='error')
    else:assert not next(e for e in events if e['event']=='result')['speaker_steering']['active']
    assert configured[0][0] is False and configured[-1][0] is False
    assert configured[-1][1]['reason']=='request_complete'
    assert not state['active'] and not small_app.STATE['busy']


def test_report_routes_allow_only_aggregate_json_not_vector_bank(tmp_path,monkeypatch):
    monkeypatch.setattr(small_app.settings,'HERE',tmp_path)
    folder=tmp_path/'catalogue-steering-study';folder.mkdir()
    (folder/'index.html').write_text('<html>Experimental catalog steering</html>')
    (folder/'summary.json').write_text('{"n":10}')
    client=TestClient(small_app.app)
    assert client.get('/catalogue-steering-study/').status_code==200
    assert client.get('/catalogue-steering-study/summary.json').json()=={'n':10}
    for path in ('vectors.npz','manifest.json','private-provenance.json','reference.wav'):
        assert client.get('/catalogue-steering-study/'+path).status_code==404


def test_public_steering_catalog_is_read_only_manager_output(monkeypatch):
    monkeypatch.setattr(speaker_steering,'catalog_speaker_steering',lambda:
        {'default_enabled':True,'voices':[{'voice':'mira','status':'experimental'}]})
    response=TestClient(small_app.app).get('/api/speaker-steering/catalog')
    assert response.status_code==200 and not response.json()['default_enabled']


def test_historical_http_study_payloads_explicitly_opt_out():
    root=Path(__file__).resolve().parents[1]
    for name in ('speaker_study.py','full_ft_speaker_study.py'):
        source=(root/name).read_text()
        assert "'speaker_steering':False" in source
        assert 'Frozen study forbids activation steering' in source
