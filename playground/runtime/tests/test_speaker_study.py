"""Pilot design/statistics/public-report boundaries; no GPU quality claims."""
from collections import Counter
import json
import re

import pytest

import speaker_study as study


def test_frozen_design_is_balanced_and_single_take():
    plan=study.make_design();rows=plan['rows']
    counts=Counter((r['voice'],r['adapter'],r['condition']) for r in rows)
    assert len(rows)==240 and len(counts)==24 and set(counts.values())=={10}
    assert len({r['seed'] for r in rows})==10 and len({r['id'] for r in rows})==240
    for block in range(10):
        matched=[r for r in rows if r['index']==block]
        assert len(matched)==24 and len({r['seed'] for r in matched})==1
    assert plan['ref15_enabled'] is False and plan['alignment'] is False
    assert plan['temperature']==1 and plan['cfg']==1.5


@pytest.mark.parametrize('voice',list(study.VOICES))
def test_affect_only_prompts_do_not_describe_speaker(voice):
    for emotion in study.EMOTIONS:
        for condition in ('caption_affect','script_affect'):
            fmt,prompt=study.prompt_for(condition,voice,emotion,'en')
            assert not re.search(r'\b(male|female|gender|timbre|baritone|bright|dark|young|old|age|register)\b',prompt,re.I)
            assert fmt in ('caption','structured')


@pytest.mark.parametrize('condition',list(study.PROMPTS))
def test_prompt_formats_preserve_target_words(condition):
    from prompting import parse_literal
    for lang,text in study.TEXTS.items():
        fmt,prompt=study.prompt_for(condition,'selma','anger',lang)
        assert parse_literal(prompt,fmt)['text']==text


def test_transcript_control_has_no_effective_cfg_contrast():
    plan=study.make_design()
    for row in plan['rows']:
        assert (row['spec']['prompt']==row['spec']['unconditional'])==(row['condition']=='transcript')


def test_saved_design_cannot_be_silently_rewritten(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    original=study.design();assert study.design()==original
    path=tmp_path/'design.json';changed=json.loads(path.read_text());changed['rows'][0]['seed']=1
    path.write_text(json.dumps(changed))
    with pytest.raises(AssertionError,match='Frozen study'):study.design()


def test_bootstrap_constant_and_bounds():
    assert study.bootstrap([.4]*10)==pytest.approx([.4,.4])
    lo,hi=study.bootstrap([0,1]*5)
    assert 0<=lo<.5<hi<=1


def test_public_report_routes_cannot_expose_sources(tmp_path,monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(small_app.settings,'HERE',tmp_path)
    folder=tmp_path/'speaker-study';folder.mkdir()
    (folder/'index.html').write_text('<html>study</html>')
    (folder/'summary.json').write_text('{"completed":240}')
    client=TestClient(small_app.app)
    assert client.get('/speaker-study/').status_code==200
    assert client.get('/speaker-study.html').status_code==200
    assert client.get('/speaker-study/summary.json').json()['completed']==240
    for private in ('references.json','idris.wav','reference.wav','scores.json','private-ecapa'):
        assert client.get('/speaker-study/'+private).status_code==404
