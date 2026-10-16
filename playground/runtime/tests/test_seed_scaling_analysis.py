import math

import numpy as np
import pytest

from seed_scaling_analysis import tail_vector, assessment, compare_conditions, direction_stability, insert_report_extension


def collection(n=50):
    rows=[];arrays={}
    for cell in range(10):
        for repeat in range(n):
            id_=f'{cell}-{repeat}'
            rows.append({'id':id_,'repeat':repeat,'condition_id':str(cell),
                         'ecapa':repeat/n,'failed':False})
            arrays[id_]=np.full(1024,repeat,dtype=np.float32)
    return rows,arrays


def test_nested_selection_and_rounding():
    rows,arrays=collection()
    vector,meta=tail_vector(rows,arrays,50,5)
    assert len(meta['cells'])==10
    assert all(c['high_n']==3 and c['low_n']==13 for c in meta['cells'])
    assert np.allclose(vector,48-6)
    assert meta['cells'][0]['effective_top_percent']==6


def test_small_pool_alias_and_fixed_low_tail():
    rows,arrays=collection()
    a,am=tail_vector(rows,arrays,10,5)
    b,bm=tail_vector(rows,arrays,10,10)
    assert np.array_equal(a,b)
    c,cm=tail_vector(rows,arrays,10,25)
    assert np.allclose(a,9-1)
    assert np.allclose(c,8-1)
    assert am['cells'][0]['low_ids']==cm['cells'][0]['low_ids']
    assert all(int(id_.split('-')[1])<10 for v in am['cells'] for id_ in v['high_ids'])


def test_missing_or_failed_seeds_are_not_dropped():
    rows,arrays=collection()
    with pytest.raises(ValueError,match='Incomplete'):
        tail_vector(rows[1:],arrays,10,5)
    rows[0]['failed']=True
    with pytest.raises(ValueError,match='failed'):
        tail_vector(rows,arrays,10,5)


def test_assessment_and_paired_comparison():
    rows=[]
    for block in range(20):
        for c,delta in [('baseline',0),('steered',.04)]:
            row={'block':block,'condition':c,'batch_seed':block//4,'ecapa':.3+delta,
                 'anchor_ecapa':.3+delta,'content_enjoyment':6.,'production_quality':7.,
                 'mos':3.,'aesthetics':6.,'genuineness':6.,'wer':0.,'emotion_score':None}
            rows.append(row)
    a=assessment(rows,'steered')
    assert a['confirmed_gain'] and a['quality_safe']
    assert a['matched']['ecapa']['batch_clusters']==5
    assert math.isclose(compare_conditions(rows,'steered','baseline')['ecapa']['mean'],.04)
    rows.pop()
    with pytest.raises(ValueError,match='Unmatched'):
        assessment(rows,'steered')


def test_extended_sizes_and_one_percent():
    rows,arrays=collection(200)
    extra=[{**r,'id':'second-'+r['id'],'condition_id':'second-'+r['condition_id']} for r in rows]
    extended={**arrays,**{'second-'+k:v for k,v in arrays.items()}}
    v,m=tail_vector(rows+extra,extended,200,1,expected_cells=20,
                   allowed_sizes=(50,100,200),allowed_percentages=(10,5,2,1))
    assert all(c['high_n']==2 and c['low_n']==50 for c in m['cells'])
    assert np.allclose(v,198.5-24.5)


def test_stability_is_deterministic_collection_only():
    rows,arrays=collection()
    a=direction_stability(rows,arrays,10,5,repetitions=4)
    assert a==direction_stability(rows,arrays,10,5,repetitions=4)
    assert a['collection_only'] is True and a['resamples']==4
    assert np.allclose(a['cosine_interval'],[1,1],atol=1e-5)


def test_report_updates_preserve_both_studies_and_original_content():
    html='<html>Original content</html>'
    html=insert_report_extension(html,'seed-scaling','<section>First</section>')
    html=insert_report_extension(html,'strict-tail','<section>Second</section>')
    html=insert_report_extension(html,'seed-scaling','<section>First updated</section>')
    assert 'Original content' in html and '<section>Second</section>' in html
    assert '<section>First updated</section>' in html
    assert html.count('<!-- seed-scaling-extension -->')==1
    legacy='<html>Original<!-- seed-scaling-extension --><section>Legacy first</section><section>Later</section></html>'
    upgraded=insert_report_extension(legacy,'seed-scaling','<section>Rebuilt</section>')
    assert '<section>Later</section>' in upgraded and 'Legacy first' not in upgraded
