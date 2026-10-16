"""CPU-only, matched-cell Top2% high-minus-low training targets."""
from collections import defaultdict
import hashlib
import math

import numpy as np

LAYERS = (14, 15, 16, 17, 18)  # zero-based Qwen block outputs


def aggregate(rows, means, n=100):
    """Equal take weighting within each tail, then equal cell weighting.

    Both tails have the SAME reference. This is not reference-minus-no-reference.
    No per-layer re-selection: every layer uses the same ECAPA-ranked takes.
    """
    if n not in (50, 100):
        raise ValueError('Only prospectively declared N50/N100')
    cells = defaultdict(list)
    for row in rows:
        if row['repeat'] < n:
            cells[row['condition_id']].append(row)
    if len(cells) != 10:
        raise ValueError('Need all ten emotion/language cells')
    highs, lows, selections = [], [], []
    for cell, candidates in sorted(cells.items()):
        if len(candidates) != n or {r['repeat'] for r in candidates} != set(range(n)):
            raise ValueError('Incomplete seed prefix: ' + cell)
        if any(r.get('failed') or not np.isfinite(r['ecapa']) or r['generated_frames'] <= 0
               for r in candidates):
            raise ValueError('Failed/nonfinite/empty samples cannot be silently dropped')
        ranked = sorted(candidates, key=lambda r: (r['ecapa'], r['id']))
        high = ranked[-math.ceil(n * .02):]
        low = ranked[:math.ceil(n * .25)]
        if {r['id'] for r in high} & {r['id'] for r in low}:
            raise ValueError('Overlapping pools')
        a = np.stack([means[r['id']] for r in high]).astype(np.float64)
        b = np.stack([means[r['id']] for r in low]).astype(np.float64)
        if a.shape != (len(high), 5, 1024) or b.shape != (len(low), 5, 1024):
            raise ValueError('Need all five 1024D layers')
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError('Nonfinite activation')
        highs.append(a.mean(0)); lows.append(b.mean(0))
        selections.append({'cell': cell, 'high_n': len(high), 'low_n': len(low),
            'high_ids': [r['id'] for r in high], 'low_ids': [r['id'] for r in low],
            'high_ecapa': float(np.mean([r['ecapa'] for r in high])),
            'low_ecapa': float(np.mean([r['ecapa'] for r in low])),
            'all_ecapa': float(np.mean([r['ecapa'] for r in candidates]))})
    high_cells, low_cells = np.stack(highs), np.stack(lows)
    cell_deltas = (high_cells-low_cells).astype(np.float32)
    high_mean = high_cells.mean(0).astype(np.float32)
    low_mean = low_cells.mean(0).astype(np.float32)
    delta = (high_cells-low_cells).mean(0).astype(np.float32)
    arrays = {'layers': np.asarray(LAYERS, dtype=np.int64), 'delta': delta,
              'high_mean': high_mean, 'low_mean': low_mean, 'cell_deltas': cell_deltas}
    for i, layer in enumerate(LAYERS):
        arrays[f'layer_{layer}__contrast'] = delta[i]
    metadata = {'n_per_cell': n, 'total_takes': n*10, 'top_percent': 2,
        'bottom_percent': 25, 'selected_high_takes': 10*math.ceil(n*.02),
        'selected_low_takes': 10*math.ceil(n*.25), 'cells': selections,
        'layers': list(LAYERS), 'l2_by_layer': np.linalg.norm(delta, axis=1).tolist(),
        'delta_sha256': hashlib.sha256(delta.tobytes()).hexdigest(),
        'mean_high_ecapa': float(np.mean([c['high_ecapa'] for c in selections])),
        'mean_low_ecapa': float(np.mean([c['low_ecapa'] for c in selections])),
        'mean_collection_ecapa': float(np.mean([c['all_ecapa'] for c in selections])),
        'status': 'training_target_only_not_validated',
        'selection_metric': 'pinned ECAPA cosine against unchanged conditioning reference',
        'quality_warning': 'Top ECAPA is not Top Content Enjoyment. Quality/emotion/WER are not screened in this large-scale collection.'}
    return arrays, metadata


def speaker_split(reference_sha256):
    """Exact duplicate references cannot leak across train/validation/test."""
    bucket = int(hashlib.sha256(('speaker-target-v1:'+reference_sha256).encode()).hexdigest()[:8], 16) % 100
    return 'train' if bucket < 80 else 'validation' if bucket < 90 else 'test'
