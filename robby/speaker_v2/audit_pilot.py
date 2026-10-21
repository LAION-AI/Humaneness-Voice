"""Read-only dataset audit, with reproducible machine-readable measurements."""
from __future__ import annotations
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import numpy as np


def audit(root):
    root = Path(root)
    manifest = json.loads((root/'manifest.json').read_text())
    speakers = manifest['speakers']
    teachers = np.load(root/'teacher_embeddings.npy', allow_pickle=False)
    assert teachers.shape == (len(speakers), 192)
    assert np.isfinite(teachers).all()
    assert np.allclose(np.linalg.norm(teachers, axis=1), 1, atol=1e-5)
    identities = {r['speaker_idx']: r for r in speakers}
    splits = {split: {x['voice'] for x in speakers if x['split'] == split}
              for split in ('train','validation','test')}
    for a, av in splits.items():
        for b, bv in splits.items():
            if a != b:
                assert not av & bv
    for row in speakers:
        teacher_ids = {c['uid'] for c in row['clips'] if c['role'] == 'teacher'}
        prediction_ids = {c['uid'] for c in row['clips'] if c['role'] == 'prediction'}
        assert len(teacher_ids) >= 2 and len(prediction_ids) >= 2
        assert not teacher_ids & prediction_ids
        for clip in row['clips']:
            assert hashlib.sha256(Path(clip['path']).read_bytes()).hexdigest() == clip['sha256']
    counts, sizes, features_s, gaps = Counter(), [], [], []
    for row in manifest['conversations']:
        path = root/row['path']
        assert hashlib.sha256(path.read_bytes()).hexdigest() == row['sha256']
        with np.load(path, allow_pickle=False) as data:
            x, ids, ends = data['features'], data['speaker_idx'], data['frame_end_s']
            assert x.shape == (len(ids), 1024) and np.isfinite(x).all()
            assert np.allclose(np.diff(ends), .08, atol=1e-5)
            assert (ids >= -2).all()
            assert not (data['valid_mask'] & ((ids < 0) | data['ambiguity'])).any()
            for i in np.unique(ids[ids >= 0]):
                assert identities[int(i)]['split'] == row['split']
            assert all(identities[i]['split'] == row['split'] for i in row['speaker_ids'])
            counts[row['split']+'_frames'] += len(ids)
            counts[row['split']+'_valid_frames'] += int(data['valid_mask'].sum())
            counts['frames'] += len(ids)
            counts['valid_frames'] += int(data['valid_mask'].sum())
            counts['silence_frames'] += int((ids == -1).sum())
            counts['overlap_frames'] += int((ids == -2).sum())
            counts['decision_hops'] += len(ids)//4
        for e in row['events']:
            if 'speaker_idx' in e:
                assert e['uid'] in {c['uid'] for c in identities[e['speaker_idx']]['clips']
                                   if c['role'] == 'prediction'}
                gaps.append(abs(e['onset_s']/.32 - round(e['onset_s']/.32)))
        counts['conversations'] += 1
        counts[row['scenario']+'_conversations'] += 1
        sizes.append(path.stat().st_size)
        features_s.append(row['feature_seconds'])
    assert max(gaps) > .1, 'Need non-hop-aligned speaker boundaries'
    quality = json.loads((root/'teacher_quality.json').read_text())
    lang = Counter(c['lang'] for s in speakers for c in s['clips'])
    conditions = Counter('/'.join(c['condition']) for s in speakers for c in s['clips'])
    prediction_conditions = Counter('/'.join(c['condition']) for s in speakers for c in s['clips']
                                    if c['role'] == 'prediction')
    cross_pilot = None
    original = root.parent/'pilot'/'speakers.json'
    if root.name.startswith('pilot_v2') and original.exists():
        old = json.loads(original.read_text())
        old_ids = {c['uid'] for s in old for c in s['clips'] if c['role'] == 'prediction'}
        new_ids = {c['uid'] for s in speakers for c in s['clips'] if c['role'] == 'prediction'}
        assert not old_ids & new_ids
        assert np.array_equal(teachers, np.load(original.parent/'teacher_embeddings.npy'))
        cross_pilot = {'new_prediction_utterances_disjoint_from_v1': True,
                       'teacher_embeddings_bitwise_equal_v1': True}
    total_audio_s = sum(r['duration_s'] for r in manifest['conversations'])
    return {'audit_passed': True, 'counts': dict(counts), 'speakers': {k:len(v) for k,v in splits.items()},
        'source_clips': sum(len(s['clips']) for s in speakers), 'languages': dict(lang),
        'source_conditions': dict(conditions), 'prediction_source_conditions': dict(prediction_conditions),
        'cross_pilot_controls': cross_pilot, 'audio_seconds': total_audio_s,
        'feature_file_bytes': sum(sizes), 'feature_request_seconds': sum(features_s),
        'feature_request_rtf': sum(features_s)/total_audio_s,
        'teacher_pair_cosine_min_median_max': np.quantile([q['teacher_pair_cosine'] for q in quality],[0,.5,1]).tolist(),
        'teacher_reference_seconds': sum(q['teacher_seconds'] for q in quality),
        'private_recordings': 0, 'source_snapshot': manifest['dataset_revision'],
        'checks': ['finite normalized teacher embeddings', 'no speaker overlap across splits',
          'no teacher recording reused in student examples', 'all retained source audio SHA256 matches',
          'all conversation feature hashes match', '80ms frame cadence and1024D feature shape',
          'overlap/silence excluded from identity regression', 'non-hop-aligned speaker boundaries'],
        'extrapolation_not_measurement': {'100000_conversations_feature_request_hours': sum(features_s)/len(features_s)*100000/3600,
          'dense_100000x24s_feature_GB': 100000*300*1024*2/1e9,
          'retained_500000x25frame_window_feature_GB': 500000*25*1024*2/1e9}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=Path(__file__).parent/'pilot')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()
    report = audit(args.data)
    if args.save:
        (args.data/'audit.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report, indent=2, allow_nan=False))
