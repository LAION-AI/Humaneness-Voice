import hashlib

import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import dynamic_references as refs
import settings


def test_defaults_and_fixed_reference_integrity():
    from small_app import Turn
    body = Turn(message='Hello')
    assert body.guidance == 2 and body.temperature == .7
    assert body.seconds is None and body.format == 'structured'
    assert body.adapters.id is None and not body.adapters.ref15
    catalog = refs.catalog()
    assert catalog['default'] == settings.DEFAULT_VOICE == 'emolia_c1699'
    assert [v['id'] for v in catalog['voices'][:6]] == catalog['presets']
    assert len(set(catalog['presets'])) == 6
    assert {v['name'] for v in catalog['voices'][:6]} == {'Selma','Idris','Anton','Robin','Mira','Osku'}
    for voice in catalog['voices']:
        path, item = refs.fixed_clip(voice['id'])
        assert hashlib.sha256(path.read_bytes()).hexdigest() == voice['reference_sha256']
        info = sf.info(path)
        assert info.samplerate == 48000 and info.channels == 1 and info.duration <= 15
        assert item['seconds'] == info.duration
        assert voice['content_enjoyment'] > 5


@pytest.mark.parametrize('voice', refs.fixed_manifest()['selected'])
def test_prompt_language_emotion_seed_do_not_switch_reference(voice):
    angry = {'lang':'en','general':'Intense furious shouting','tagged':'','text':'Go!'}
    neutral = {'lang':'de','general':'Neutral, calm conversation','tagged':'','text':'Hallo.'}
    # No retriever is accessed or required on the default path.
    path, first = refs.choose(None, angry, voice)
    second_path, second = refs.choose(None, neutral, voice)
    assert path == second_path and first['reference_sha256'] == second['reference_sha256']
    assert first['policy'] == second['policy'] == 'fixed'
    assert not first['dynamic'] and not second['dynamic']


def test_only_catalog_reference_audio_is_public():
    from small_app import app
    client = TestClient(app)
    for voice in refs.catalog()['voices'][:6]:
        response = client.get(voice['preview_url'])
        path, _ = refs.fixed_clip(voice['id'], playback=True)
        assert response.status_code == 200 and response.headers['content-type'] == 'audio/mpeg'
        assert response.content == path.read_bytes()
    assert client.get('/api/voices/reference/unknown').status_code == 404
    assert client.get('/fixed-references/selection.json').status_code == 404


def test_actor_fixed_identity_and_emotional_performance_recipe():
    from actor_prompt import SYSTEM
    assert 'each catalog voice uses one fixed reference clip' in SYSTEM
    assert 'GENERAL / SCRIPT is the Acting checkpoint default' in SYSTEM
    assert 'Strong acting is allowed while the' in SYSTEM
    assert 'Do not add\nmale/female voice, age, accent' in SYSTEM
