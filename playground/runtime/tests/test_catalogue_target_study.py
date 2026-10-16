from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import pytest

import catalogue_target_study as study
from catalogue_target_analysis import LAYERS, aggregate, speaker_split


def data():
    rows = []; means = {}
    for _, batch in study.batches('mira'):
        for row in batch:
            rows.append({**row, 'ecapa':row['repeat']/100, 'generated_frames':30, 'failed':False})
            means[row['id']] = np.stack([np.full(1024,row['repeat']+layer,dtype=np.float32) for layer in LAYERS])
    return rows, means


def test_fixed_cells_and_100_actual_seeds():
    groups = defaultdict(list)
    all_rows = []
    for name, batch in study.batches('mira'):
        assert len({r['seed'] for r in batch}) == 1
        assert len({r['condition_id'] for r in batch}) == len(batch)
        assert len(batch) in (2,4)
        for row in batch:
            groups[row['condition_id']].append(row); all_rows.append(row)
            assert not row['reference_identity']
            assert all(word not in row['prompt'].lower() for word in ('male voice','female voice','timbre','speaker identity'))
    assert len(all_rows) == 1000 and len({r['id'] for r in all_rows}) == 1000
    assert len(groups) == 10
    for group in groups.values():
        assert len(group) == len({r['seed'] for r in group}) == 100
        assert len({r['prompt'] for r in group}) == 1
    assert not {r['text'] for r in study.cells()} & {r['text'] for r in study.cells('validation')}


def test_top_two_of_100_for_all_five_layers():
    rows, means = data(); arrays, meta = aggregate(rows,means)
    assert arrays['delta'].shape == (5,1024)
    assert arrays['cell_deltas'].shape == (10,5,1024)
    assert np.allclose(arrays['delta'],98.5-12)
    assert meta['selected_high_takes'] == 20
    assert meta['selected_low_takes'] == 250
    assert all(c['high_n']==2 and c['low_n']==25 for c in meta['cells'])
    for i,layer in enumerate(LAYERS):
        assert np.array_equal(arrays[f'layer_{layer}__contrast'],arrays['delta'][i])
    assert all({int(id_.split('-')[-1])//10 for id_ in c['high_ids']}=={98,99} for c in meta['cells'])


def test_50_seed_secondary_is_fixed_prefix():
    rows, means = data(); arrays, meta = aggregate(rows,means,50)
    assert np.allclose(arrays['delta'],49-6)
    assert meta['selected_high_takes'] == 10 and meta['selected_low_takes'] == 130
    assert all(int(id_.split('-')[-1])//10 < 50 for c in meta['cells'] for id_ in c['high_ids'])


def test_missing_failures_nonfinite_and_wrong_layers_rejected():
    rows,means=data()
    with pytest.raises(ValueError,match='Incomplete'):aggregate(rows[1:],means)
    rows[0]['failed']=True
    with pytest.raises(ValueError,match='Failed'):aggregate(rows,means)
    rows[0]['failed']=False; rows[0]['ecapa']=float('nan')
    with pytest.raises(ValueError,match='Failed'):aggregate(rows,means)
    rows[0]['ecapa']=0.; means[rows[0]['id']]=np.zeros((4,1024))
    with pytest.raises(ValueError):aggregate(rows,means)
    means[rows[0]['id']]=np.full((5,1024),float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):aggregate(rows,means)


def test_split_is_stable_and_reference_grouped():
    a='ab'*32
    assert speaker_split(a)==speaker_split(a)
    assert set(speaker_split(str(i)) for i in range(1000))=={'train','validation','test'}


def test_atomic_json_and_frozen_design(tmp_path):
    path=tmp_path/'artifact.json'
    study.freeze(path,{'a':1})
    assert study.freeze(path,{'a':1})=={'a':1}
    with pytest.raises(ValueError,match='Immutable'):study.freeze(path,{'a':2})
    assert json.loads(path.read_text())=={'a':1}
    assert list(tmp_path.glob('*.tmp'))==[]


def test_atomic_batch_commit_and_readback(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    monkeypatch.setattr(study,'save_mp3',lambda p,w:p.write_bytes(b'MP3'))
    folder=tmp_path/'voices/mira'
    key,batch=next(study.batches('mira'))
    rows=[{**r,'generated_frames':20,'ecapa':.5,'failed':False} for r in batch]
    values=np.zeros((4,5,1024),dtype=np.float32)
    study.batch_commit(folder,key,rows,values,np.full(4,20),np.zeros((4,192)),[np.ones(100)]*4)
    loaded,means=study.read_collection('mira')
    assert loaded==rows and len(means)==4
    assert not list((folder/'batches').glob('.partial*'))
    with pytest.raises(ValueError,match='overwrite'):
        study.batch_commit(folder,key,rows,values,np.full(4,20),np.zeros((4,192)),[np.ones(100)]*4)


def test_partial_batches_never_consumed(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    orphan=tmp_path/'voices/mira/batches/.partial-r000-g0'
    orphan.mkdir(parents=True);(orphan/'results.json').write_text('[]')
    assert study.read_collection('mira')==([], {})


def test_complete_target_export_has_training_input_all_layers_and_provenance(tmp_path,monkeypatch):
    monkeypatch.setattr(study,'ROOT',tmp_path)
    rows,means=data()
    monkeypatch.setattr(study,'read_collection',lambda voice:(rows,means))
    folder=tmp_path/'voices/mira';folder.mkdir(parents=True)
    embedding=np.ones(192,dtype=np.float32)/np.sqrt(np.float32(192))
    np.savez(folder/'speaker-input.npz',speaker_embedding=embedding)
    (tmp_path/'design.json').write_text(json.dumps({'voices':[{'id':'mira','name':'Mira','split':'train'}]}))
    meta=study.derive('mira')
    assert meta['status']=='training_target_only_not_validated'
    assert len(meta['cell_order'])==10
    with np.load(tmp_path/'targets/mira/n100-top2.npz',allow_pickle=False) as result:
        assert result['delta'].shape==(5,1024)
        assert np.array_equal(result['speaker_embedding'],embedding)
        assert all(f'layer_{layer}__contrast' in result for layer in LAYERS)
    study.derive('mira')  # Exact immutable target can be resumed.
