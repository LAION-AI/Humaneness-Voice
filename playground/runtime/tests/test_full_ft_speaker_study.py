"""Frozen full-FT-only design and public report privacy; no quality claims."""
from collections import Counter

import full_ft_speaker_study as study


def test_full_ft_pilot_balanced_matched_and_adapter_free(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    plan=study.design()
    assert plan==study.design()
    assert plan['checkpoint_sha256']==study.settings.FULL_FT_SHA256
    assert plan['adapters']=='NONE in every request'
    assert len(plan['rows'])==120
    assert set(Counter((r['voice'],r['condition']) for r in plan['rows']).values())=={10}
    for block in range(10):
        matched=[r for r in plan['rows'] if r['block']==block]
        assert len(matched)==12 and len({r['seed'] for r in matched})==1
        assert len({r['text'] for r in matched})==1
    assert len({r['text'] for r in plan['rows']})==10
    assert any('optimality' in limit for limit in plan['limitations'])


def test_validation_has_fresh_texts_seeds_and_fixed_conditions():
    pilot=study.rows_for('pilot',study.CONDITIONS)
    validation=study.rows_for('validation',['baseline','cooler'])
    assert len(validation)==40
    assert not {r['seed'] for r in pilot}&{r['seed'] for r in validation}
    assert not {r['text'] for r in pilot}&{r['text'] for r in validation}
    assert set(Counter((r['voice'],r['condition']) for r in validation).values())=={10}


def test_report_routes_are_read_only_and_hide_reference_sources(tmp_path,monkeypatch):
    import small_app
    from fastapi.testclient import TestClient
    monkeypatch.setattr(small_app.settings,'HERE',tmp_path)
    folder=tmp_path/'full-ft-speaker-study';folder.mkdir()
    (folder/'index.html').write_text('<html>Full FT</html>')
    (folder/'summary-pilot.json').write_text('{"trials":120}')
    client=TestClient(small_app.app)
    assert client.get('/full-ft-speaker-study/').status_code==200
    assert client.get('/full-ft-speaker-study/summary-pilot.json').json()=={'trials':120}
    for private in ('references.json','idris.wav','private-provenance.json','references/idris-15.wav','scores.json'):
        assert client.get('/full-ft-speaker-study/'+private).status_code==404


def test_failed_generation_is_not_discarded_from_reliability():
    row={'failed':True}
    assert study.metric(row,'wer')==1 and study.metric(row,'bad_wer')==1
    assert study.metric(row,'exact')==0 and study.metric(row,'conditioning_cosine')==0


def test_automatic_duration_followup_uses_fresh_matched_texts_and_seeds():
    pilot=study.rows_for('pilot',study.CONDITIONS)
    validation=study.rows_for('validation',['baseline','no_cfg'])
    followup=study.rows_for('followup',study.FOLLOWUP)
    assert len(followup)==60
    assert not {r['text'] for r in followup}&{r['text'] for r in pilot+validation}
    assert not {r['seed'] for r in followup}&{r['seed'] for r in pilot+validation}
    assert set(Counter((r['voice'],r['condition']) for r in followup).values())=={10}
    assert all(c['seconds'] is None and c['temperature']==1 for c in study.FOLLOWUP.values())
