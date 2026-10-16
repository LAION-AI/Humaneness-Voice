import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import mira_activation_analysis as study


def test_disjoint_phases_and_matched_seeds():
    plan = study.design(100, 3)
    assert len(plan['collection_rows']) == 100
    assert len(plan['screen_rows']) == 40
    assert len(plan['semantic_rows']) == 90
    textsets = [set(b['text'] for b in study.blocks(p)) for p in study.TEXTS]
    seedsets = [set(b['seed'] for b in study.blocks(p)) for p in study.TEXTS]
    for i in range(len(textsets)):
        for j in range(i):
            assert not textsets[i] & textsets[j]
            assert not seedsets[i] & seedsets[j]
    rows = study.rows_for('refine', [study.condition(), study.condition([1], .5)])
    for block in range(10):
        pair = [r for r in rows if r['block'] == block]
        assert len(pair) == 2 and pair[0]['seed'] == pair[1]['seed']


def test_balanced_aggregation_and_strict_threshold():
    records, arrays = [], {}
    for eidx, emotion in enumerate(study.EMOTIONS):
        for i in range(2 if eidx else 6):
            key = f'{emotion}-{i}'
            records.append({'id': key, 'emotion': emotion, 'ecapa': .4})
            arrays[key] = {'layer_0': np.array([eidx, 10.])}
        key = f'{emotion}-exact'
        records.append({'id': key, 'emotion': emotion, 'ecapa': .25})
        arrays[key] = {'layer_0': np.array([100., 100.])}
    out = study.aggregate_vectors(records, arrays)
    assert out['metadata']['eligible']
    assert out['metadata']['high_n'] == 14
    np.testing.assert_allclose(out['vectors']['layer_0']['raw_mean'], [2., 10.])
    assert 'contrast' not in out['vectors']['layer_0']  # insufficient low emotion coverage


def scored_rows():
    rows = []
    for i in range(10):
        for c in ('baseline', 'good', 'bad', 'missing'):
            rows.append({'id': f'{c}-{i}', 'phase': 'validation', 'block': i, 'condition': c,
                         'emotion': 'joy', 'ecapa': .2 + (.1 if c == 'good' else 0),
                         'content_enjoyment': 6., 'production_quality': 6., 'mos': 3.8,
                         'wer': .02 + (.1 if c == 'bad' else 0), 'emotion_score': 1.3})
    for row in rows:
        if row['condition'] == 'missing': row.pop('mos')
    return rows


def test_selection_requires_heldout_gain_and_all_quality_metrics():
    summary = study.summarize(scored_rows(), bootstrap_repetitions=100)
    cells = {r['condition']: r for r in summary['conditions']}
    assert cells['good']['eligible'] and cells['good']['heldout_positive_interval']
    assert not cells['baseline']['eligible']
    assert 'wer_regression' in cells['bad']['gate_reasons']
    assert 'incomplete_matched_mos' in cells['missing']['gate_reasons']
    assert study.select_conditions(scored_rows()) == ['good']


def test_failures_are_retained_and_block_gate():
    rows = scored_rows()
    rows[1]['failed'] = True
    rows[1]['error'] = 'out of memory'
    c = next(c for c in study.summarize(rows, bootstrap_repetitions=100)['conditions'] if c['condition'] == 'good')
    assert c['n'] == 10 and c['failures'] == 1 and not c['eligible']


def test_report_embeds_audio_escapes_and_shows_failures(tmp_path):
    audio = tmp_path/'sample.mp3'
    audio.write_bytes(b'ID3unit-test')
    rows = scored_rows()[:2]
    rows[0].update(audio_path=str(audio), text='<script>unsafe</script>', prompt='CAPTION: neutral')
    rows[1].update(failed=True, error='boom')
    out = tmp_path/'index.html'
    result = study.render_report(out, {'limitations': ['pilot only']}, rows)
    page = out.read_text()
    assert result['audio_embedded'] == 1
    assert 'data:audio/mpeg;base64,' in page
    assert '&lt;script&gt;unsafe' in page
    assert 'Generation failure retained' in page
    assert 'No validated improvement' in page


def test_frozen_design_refuses_change(tmp_path):
    p = tmp_path/'design.json'
    study.frozen_json(p, {'a': 1})
    study.frozen_json(p, {'a': 1})
    with pytest.raises(ValueError): study.frozen_json(p, {'a': 2})


def test_refinement_budget_is_declared():
    conditions = study.refinement_conditions(['layer_1', 'layer_2', 'layer_3'])
    assert len(conditions) == 57  # baseline + seven subsets * four alphas * two directions
    assert all(len(c['points']) <= 3 for c in conditions)
    assert 'alpha/number_of_points' in study.design(50)['combination_rule']


def test_point_names_and_exact_cfg_identity():
    from prompting import REFERENCE_IDENTITY
    assert study.condition([0, 27], .5)['points'] == ['layer_0', 'layer_27']
    assert study.condition(['semantic_local'], .25)['points'] == ['semantic_local']
    assert REFERENCE_IDENTITY in study.design()['collection_rows'][0]['prompt']


def test_aesthetics_is_gate_when_available_and_missing_is_not_hidden():
    rows = scored_rows()
    for row in rows: row['aesthetics'] = 4.
    for row in rows:
        if row['condition'] == 'good': row['aesthetics'] = 3.5
    summary = study.summarize(rows, bootstrap_repetitions=20)
    good = next(c for c in summary['conditions'] if c['condition'] == 'good')
    assert not good['eligible'] and 'aesthetics_regression' in good['gate_reasons']


def test_relative_report_audio(tmp_path):
    (tmp_path/'audio').mkdir()
    (tmp_path/'audio/example.mp3').write_bytes(b'ID3relative-audio')
    row = scored_rows()[0]
    row['audio_path'] = 'audio/example.mp3'
    result = study.render_report(tmp_path/'index.html', {}, [row])
    assert result['audio_embedded'] == 1


def test_bounded_refinement_has_baseline_fresh_rows_and_diagnostics():
    rows = []
    for b in study.blocks('screen'):
        for p in (None, 1, 2, 3):
            c = study.condition() if p is None else study.condition([p], .25)
            rows.append({**b, 'id': f'{c["id"]}-{b["block"]}', 'condition': c['id'], 'intervention': c,
                         'ecapa': .20 if p is None else .30-p*.01, 'content_enjoyment': 6.,
                         'production_quality': 6., 'mos': 3.8, 'wer': .02,
                         'emotion_score': None if b['emotion']=='neutral' else 1.3})
    plan = study.build_refinement_plan(rows, ['layer_1__contrast'])
    assert plan['selected_points'] == ['layer_1', 'layer_2', 'layer_3']
    assert len(plan['conditions']) == 24
    assert len(plan['rows']) == 192
    assert not plan['diagnostic_only_if_no_qualified']
    assert plan['conditions'][0]['id'] == 'baseline'
    assert set(r['text'] for r in plan['rows']).isdisjoint(r['text'] for r in rows)


def test_language_and_recorded_batch_bootstrap():
    rows = scored_rows()
    for r in rows:
        r['language'] = 'en' if r['block'] % 2 == 0 else 'de'
        r['anchor_ecapa'] = .4 + (.03 if r['condition']=='good' else 0.)
        r['batch_ids'] = [f'{r["condition"]}-{j}' for j in range((r['block']//4)*4, min(10,(r['block']//4)*4+4))]
    diagnostics = study.language_diagnostics(rows, repetitions=100)
    good_en = next(d for d in diagnostics if d['condition']=='good' and d['language']=='en')
    assert good_en['n'] == 5 and good_en['matched']['ecapa']['mean'] == pytest.approx(.1)
    assert good_en['matched']['anchor_ecapa']['mean'] == pytest.approx(.03)
    pooled = study.language_diagnostics(rows, repetitions=100, by_language=False)
    good = next(d for d in pooled if d['condition']=='good')
    assert good['batch_clustered']['ecapa']['n_pairs'] == 10
    assert good['batch_clustered']['ecapa']['n_clusters'] == 3


def test_report_orders_validation_first_and_documents_capture_and_evidence(tmp_path):
    rows = scored_rows()
    for r in rows: r['language'] = 'en'
    report = tmp_path/'index.html'
    evidence = {'measurement-amendment.json': {'reason':'Batch padding issue', 'change':'Independent waveform encoding'},
                'external_note': {'value':'<script>no</script>'}}
    study.render_report(report, {}, rows, evidence=evidence, narrative=['Do not enable by default.'])
    page = report.read_text()
    assert page.index('Fresh held-out validation') < page.index('Full grid') < page.index('Listen to every take')
    assert 'last conditional prefill position' in page
    assert 'first emitted frame' in page
    assert 'whole-batch bootstrap' in page
    assert 'shared one RNG' not in page  # precise present-tense wording
    assert 'share one RNG' in page
    assert 'Independent waveform encoding' in page
    assert '&lt;script&gt;no' in page
    assert 'Do not enable by default.' in page
    assert 'max-width:100%' in page and 'overflow-wrap:anywhere' in page


def test_high_similarity_language_confound_counts_visible(tmp_path):
    rows = []
    for i in range(23):
        rows.append({'id': f'collection-{i}', 'phase':'collection', 'condition':'baseline', 'block':i,
                     'language':'de' if i else 'en', 'emotion':'neutral'})
    meta = {'selected': {'threshold':.5, 'high_ids':[r['id'] for r in rows]}}
    study.render_report(tmp_path/'index.html', {}, rows, vectors_metadata=meta)
    page = (tmp_path/'index.html').read_text()
    assert '22 DE, 1 EN' in page
    assert 'strict cosine &gt;0.5' in page


def test_reference_players_do_not_change_take_counts_and_downloads_are_safe(tmp_path):
    audio = tmp_path/'ref.mp3'
    audio.write_bytes(b'ID3reference')
    out = tmp_path/'index.html'
    result = study.render_report(out, {}, scored_rows()[:2],
        reference_audio={'Conditioning Mira':audio, 'Independent-file anchor':{'path':audio,'caption':'Same-label, unverified.'}},
        downloads={'Frozen protocol':'design.json','Scores':'results.json','Vector metadata':'vectors.json'})
    page = out.read_text()
    assert result['takes'] == 2 and result['reference_audio_embedded'] == 2
    assert page.count('<article ') == 2
    assert page.count('data:audio/mpeg;base64,') == 2
    assert 'not independently verified' in page
    assert 'href="design.json" download' in page
    with pytest.raises(ValueError):
        study.render_report(out, {}, [], downloads={'bad':'javascript:alert(1)'})


def test_independent_audit_checks_frozen_specs_and_fresh_inputs():
    import copy
    planned = study.rows_for('validation', [study.condition(),study.condition([16],.75,'contrast')], count=2)
    rows = copy.deepcopy(planned)
    for r in rows:
        r.update(adapters=[],cfg=2.,temperature=1.,batch_seed=9,batch_size=2,
            checkpoint={'stage':'S3-Ref15-FullFT','step':26774},reference_sha256='same-fixed-reference',
            acoustic_tokens=120,generated_frames=10,ecapa=.3,anchor_ecapa=.2,content_enjoyment=6.,
            production_quality=6.,mos=4.,aesthetics=3.,wer=0.,emotion_score=None)
    audit = study.audit_validation(rows, {'rows':planned}, [], repetitions=20)
    assert audit['audit']=='PASS' and audit['validation_rows']==4 and audit['prior_rows']==0
    rows[0]['intervention']['alpha']=.9
    with pytest.raises(ValueError, match='Frozen field mismatch'):
        study.audit_validation(rows, {'rows':planned}, [], repetitions=20)
