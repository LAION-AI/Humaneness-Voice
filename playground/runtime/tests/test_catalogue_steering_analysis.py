from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import catalogue_steering_analysis as study


def samples(n=10, high=.6, low=.1):
    rows,arrays=[],{}
    for emotion in study.mira.EMOTIONS:
        for language in ('en','de'):
            for i in range(n):
                id_=f'{emotion}-{language}-{i}'
                ecapa=high if i>=n-2 else low
                rows.append({'id':id_,'emotion':emotion,'language':language,'ecapa':ecapa})
                arrays[id_]=np.full(1024,i,dtype=np.float32)
    return rows,arrays


def manifest():
    return {'candidates':[{'voice':f'voice{i}','name':f'Voice {i}','sha256':'a'*64,
        'seconds':10.,'wav':f'voice{i}.wav'} for i in range(10)]}


def test_threshold_half_and_joint_balance():
    rows,arrays=samples()
    vector,meta=study.select_vector(rows,arrays)
    assert meta['threshold']==.5 and meta['joint_language_balanced']
    assert len(meta['high_ids'])==20 and len(meta['low_ids'])==80
    np.testing.assert_allclose(vector,5.)  # high8.5 minuslow3.5
    assert meta['vector_sha256']


def test_lower_threshold_is_explicit_and_both_pools_must_cover_emotions():
    rows,arrays=samples(high=.4)
    _,meta=study.select_vector(rows,arrays)
    assert meta['threshold']==.25 and not meta['threshold_pools'][0]['eligible']


def test_rank_fallback_is_joint_balanced_and_exploratory():
    rows,arrays=samples(high=.2,low=.1)
    vector,meta=study.select_vector(rows,arrays)
    assert meta['threshold'] is None and meta['exploratory_rank_fallback']
    assert set(meta['high_ids']).isdisjoint(meta['low_ids'])
    assert meta['joint_language_balanced']
    np.testing.assert_allclose(vector,8.)


def test_ties_are_deterministic_and_missing_cell_does_not_invent_vector():
    rows,arrays=samples(n=5,high=.2,low=.2)
    a,meta=study.select_vector(rows,arrays)
    b,other=study.select_vector(list(reversed(rows)),arrays)
    np.testing.assert_allclose(a,b)
    assert meta['high_ids']==other['high_ids']
    rows=[r for r in rows if not(r['emotion']=='fear' and r['language']=='en')]
    with pytest.raises(ValueError,match='fewer than four'):
        study.select_vector(rows,arrays)


def test_design_pins_all_ten_voices_and_fresh_validation():
    plan=study.design(manifest())
    assert len(plan['collection_rows'])==1000
    assert len(plan['validation_rows'])==200
    assert {r['text'] for r in plan['collection_rows']}.isdisjoint(r['text'] for r in plan['validation_rows'])
    assert all(v['reference_sha256']=='a'*64 for v in plan['voices'])


def test_paired_schedule_ten_actual_seeds_per_identical_condition():
    plan=study.paired_design(manifest())
    assert len(plan['with_reference_rows'])==1000 and len(plan['without_reference_rows'])==100
    for condition in {r['condition_id'] for r in plan['without_reference_rows']}:
        rows=[r for r in plan['without_reference_rows'] if r['condition_id']==condition]
        assert len({r['actual_batch_seed'] for r in rows})==10
        assert len({r['text'] for r in rows})==1
        assert all(r['reference_identity'] is False for r in rows)
        assert all('reference speaker' not in r['prompt'] and study.mira.IDENTITY not in r['prompt'] for r in rows)


def paired_inputs():
    plan=study.paired_design(manifest())
    with_rows=[r for r in plan['with_reference_rows'] if r['voice']=='voice0']
    without_rows=plan['without_reference_rows']
    arrays={}
    for rows,delta in ((with_rows,2.),(without_rows,0.)):
        for r in rows:
            r['generated_frames']=50
            r['ecapa']=0. # ensures no high-sim filtering enters the estimate
            arrays[r['id']]={study.POINT:np.full(1024,delta),
                'semantic_global':np.full(1024,3*delta),'semantic_local':np.full(2560,4*delta)}
    return with_rows,without_rows,arrays


def test_paired_effect_ignores_quality_and_balances_conditions():
    with_rows,without_rows,arrays=paired_inputs()
    result=study.paired_reference_vector(with_rows,without_rows,arrays)
    np.testing.assert_allclose(result['vectors']['layer_16'],2.)
    np.testing.assert_allclose(result['vectors']['semantic_global'],6.)
    np.testing.assert_allclose(result['vectors']['semantic_local'],8.)
    assert result['metadata']['paired_n']==100 and not result['metadata']['quality_selection']
    assert all(n==10 for n in result['metadata']['repeats_per_condition'].values())


def test_paired_effect_rejects_mismatched_prompt_or_silent_pair_deletion():
    with_rows,without_rows,arrays=paired_inputs()
    without_rows[0]['prompt']='CAPTION: male voice'
    with pytest.raises(ValueError,match='differ in prompt'):
        study.paired_reference_vector(with_rows,without_rows,arrays)
    with_rows,without_rows,arrays=paired_inputs()
    with pytest.raises(ValueError,match='unmatched pairs'):
        study.paired_reference_vector(with_rows,without_rows[:-1],arrays)


def audit_rows(diffs=None):
    rows=[]
    for i in range(10):
        for condition in ('baseline','steered'):
            rows.append({'id':f'{condition}-{i}','voice':'voice0','phase':'catalogue_validation',
                'block':i,'condition':condition,'language':'en' if i%2==0 else 'de','emotion':'joy',
                'ecapa':.3+(0 if condition=='baseline' else (diffs[i] if diffs is not None else .05)),
                'content_enjoyment':6.,'production_quality':7.,'mos':4.,'wer':.02})
    return rows


def test_lightweight_audit_does_not_require_unmeasured_emotion_head():
    result=study.assessment(audit_rows())
    assert result['default_enabled'] and result['confirmed_gain']
    assert result['n']==10 and not result['gate_reasons']


def test_positive_unconfirmed_pilot_allowed_but_warned():
    result=study.assessment(audit_rows([.12,-.08]*5))
    assert result['default_enabled'] and result['experimental']
    assert not result['confirmed_gain']


def test_quality_regression_bypasses_default():
    rows=audit_rows()
    for row in rows:
        if row['condition']=='steered': row['content_enjoyment']=5.5
    result=study.assessment(rows)
    assert not result['default_enabled'] and 'content_enjoyment_regression' in result['gate_reasons']


def test_mlp_not_trained_and_holdout_requires_speakers():
    notes=study.mlp_recommendations()
    assert notes['train_now'] is False and 'whole speakers' in notes['splits']


def test_report_keeps_per_voice_audits_and_embeds_only_evaluation(tmp_path):
    rows=audit_rows()
    audio=tmp_path/'sample.mp3';audio.write_bytes(b'ID3unit-test')
    for row in rows:row['audio_path']=str(audio)
    rows.append({'id':'collection-example','phase':'collection','voice':'voice0','audio_path':str(audio)})
    report=study.assessment(rows[:-1])
    result=study.render_catalogue_report(tmp_path/'index.html',rows,{'voice0':report},
        {'voices':{'voice0':{'status':'ready','selection':{'method':'threshold_high_minus_low','threshold':.5}}}},
        downloads={'Results':'results.json'})
    page=(tmp_path/'index.html').read_text()
    assert result['collection_takes_not_embedded']==1
    assert result['evaluation_takes']==20 and result['audio_embedded']==20
    assert page.count('<article ')==20
    assert 'quality screens remain tryable' in page
    assert 'not trained' in page
    assert '10' in page and 'ten distinct actual seeded batches' in page
    assert 'data-filter="voice"' in page
    assert 'href="results.json" download' in page


def test_report_does_not_label_pending_voice_as_validated_mira(tmp_path):
    study.render_catalogue_report(tmp_path/'index.html',[],[],{'voices':{'voice0':{'status':'preparing'}}})
    page=(tmp_path/'index.html').read_text()
    assert 'not yet derived' in page
    assert 'No improvement claim is supported' in page


def test_custom_no_reference_baseline_propagates_to_language_diagnostics(tmp_path):
    rows=audit_rows()
    for row in rows:
        row['condition']='no-reference-baseline' if row['condition']=='baseline' else 'reference-without-layer_16-a0.75'
    report=study.assessment(rows,baseline='no-reference-baseline',condition='reference-without-layer_16-a0.75')
    assert report['baseline']=='no-reference-baseline'
    assert report['per_language']['en']['matched']['ecapa']['mean']==pytest.approx(.05)
    study.render_catalogue_report(tmp_path/'index.html',rows,[report],{})
    page=(tmp_path/'index.html').read_text()
    assert 'Without-reference replacement' in page
    assert 'versus no-reference-baseline' in page


def test_generic_reference_effect_and_centered_vectors_are_diagnostic_only():
    result=study.reference_effect_diagnostics({'a':{'layer_16':np.array([1.,0.])},
                                              'b':{'layer_16':np.array([1.,2.])}})
    np.testing.assert_allclose(result['generic_vectors']['layer_16'],[1.,1.])
    np.testing.assert_allclose(result['centered_vectors']['a']['layer_16'],[0.,-1.])
    assert result['metadata']['voice_count']==2
    assert 'not been generation-tested' in result['metadata']['interpretation']
