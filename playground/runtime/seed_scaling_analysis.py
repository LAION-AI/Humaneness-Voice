"""CPU-only, fixed-cell tail selection and held-out seed-scaling analysis."""
from __future__ import annotations

from collections import defaultdict
import hashlib
import math

import numpy as np

SIZES = (10, 20, 50)
PERCENTAGES = (25, 10, 5)
METRICS = ('ecapa', 'anchor_ecapa', 'content_enjoyment', 'production_quality',
           'mos', 'wer', 'aesthetics', 'genuineness', 'emotion_score')


def insert_report_extension(existing, name, section):
    """Replace only the named extension; retain earlier/later report content."""
    start=f'<!-- {name}-extension -->'
    end=f'<!-- /{name}-extension -->'
    replacement=start+section+end
    if start in existing:
        pos=existing.index(start)
        if end in existing[pos:]:
            stop=existing.index(end,pos)+len(end)
        else:
            # Upgrade the initial extension format, which had no end marker.
            # Each extension is one outer section with no nested sections.
            close=existing.find('</section>',pos)
            if close<0:raise ValueError('Unterminated existing study section')
            stop=close+len('</section>')
        return existing[:pos]+replacement+existing[stop:]
    close='</body>' if '</body>' in existing else '</html>'
    if close not in existing:raise ValueError('Report has no closing HTML boundary')
    return existing.replace(close,replacement+close,1)


def condition_name(n, p):
    return f'n{n}-top{p}'


def tail_vector(rows, arrays, n, percentage, *, expected_cells=10,
                allowed_sizes=SIZES, allowed_percentages=PERCENTAGES):
    """Top ceil(n*p/100) minus FIXED bottom ceil(n*.25), within each cell.

    Nested prefixes are fixed by repeat number, never by observed similarity.
    Equal weight per sample, then equal weight per emotion/language cell.
    """
    if n not in allowed_sizes or percentage not in allowed_percentages:
        raise ValueError('Unsupported prospective condition')
    groups = defaultdict(list)
    for row in rows:
        if row['repeat'] < n:
            groups[row['condition_id']].append(row)
    if len(groups) != expected_cells:
        raise ValueError('Expected all collection cells')
    vectors, metadata = [], []
    for cell, values in sorted(groups.items()):
        if len(values) != n or {r['repeat'] for r in values} != set(range(n)):
            raise ValueError('Incomplete nested seed prefix: ' + cell)
        if any(r.get('failed') or not np.isfinite(r['ecapa']) for r in values):
            raise ValueError('Do not silently replace failed collection samples')
        ranked = sorted(values, key=lambda r: (r['ecapa'], r['id']))
        high_n, low_n = math.ceil(n*percentage/100), math.ceil(n*.25)
        high, low = ranked[-high_n:], ranked[:low_n]
        if set(r['id'] for r in high) & set(r['id'] for r in low):
            raise ValueError('Overlapping contrast pools')
        a = np.stack([arrays[r['id']] for r in high]).astype(np.float64)
        b = np.stack([arrays[r['id']] for r in low]).astype(np.float64)
        if a.shape != (high_n, 1024) or b.shape != (low_n, 1024):
            raise ValueError('Layer 16 must be 1024D')
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError('Nonfinite activation')
        vectors.append(a.mean(0)-b.mean(0))
        metadata.append({'cell':cell, 'high_n':high_n, 'low_n':low_n,
            'effective_top_percent':100*high_n/n,
            'high_ids':[r['id'] for r in high], 'low_ids':[r['id'] for r in low],
            'high_ecapa':float(np.mean([r['ecapa'] for r in high])),
            'low_ecapa':float(np.mean([r['ecapa'] for r in low]))})
    vector = np.mean(vectors, axis=0).astype(np.float32)
    return vector, {'n_per_cell':n, 'total_collection':10*n, 'top_percent':percentage,
        'low_percent':25, 'cells':metadata, 'l2':float(np.linalg.norm(vector)),
        'rms':float(np.sqrt(np.mean(vector**2))),
        'sha256':hashlib.sha256(vector.tobytes()).hexdigest(),
        'mean_selected_ecapa':float(np.mean([c['high_ecapa'] for c in metadata])),
        'mean_rejected_ecapa':float(np.mean([c['low_ecapa'] for c in metadata]))}


def interval(values, clusters, seed=83419):
    """Matched mean, cluster bootstrap by actual generation batch seed."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return None
    groups = [[i for i, key in enumerate(clusters) if key == c] for c in sorted(set(clusters))]
    rng = np.random.default_rng(seed)
    means = np.array([np.mean(values[np.concatenate([groups[i] for i in draw])])
        for draw in rng.integers(0, len(groups), size=(4000, len(groups)))])
    lo, hi = np.quantile(means, (.025, .975))
    return {'mean':float(values.mean()), 'low':float(lo), 'high':float(hi),
        'pairs':len(values), 'batch_clusters':len(groups),
        'positive_pairs':int(np.sum(values > 0))}


def assessment(rows, condition, baseline='baseline'):
    controls = {r['block']:r for r in rows if r['condition'] == baseline}
    takes = sorted([r for r in rows if r['condition'] == condition], key=lambda r:r['block'])
    if not takes or set(controls) != {r['block'] for r in takes}:
        raise ValueError('Unmatched validation rows')
    if any(r.get('failed') for r in takes + list(controls.values())):
        raise ValueError('Failed take in evaluation; no silent dropping')
    out = {'condition':condition, 'n':len(takes), 'means':{}, 'baseline_means':{}, 'matched':{}}
    for key in METRICS:
        valid = [r for r in takes if r.get(key) is not None and controls[r['block']].get(key) is not None]
        out['means'][key] = float(np.mean([r[key] for r in valid])) if valid else None
        out['baseline_means'][key] = float(np.mean([controls[r['block']][key] for r in valid])) if valid else None
        out['matched'][key] = interval([r[key]-controls[r['block']][key] for r in valid],
                                       [r['batch_seed'] for r in valid])
    delta = lambda key: out['matched'][key]['mean']
    gates = []
    for key, limit in (('content_enjoyment', -.30), ('production_quality', -.30), ('mos', -.20),
                       ('aesthetics', -.30), ('emotion_score', -.20)):
        if out['matched'][key] and delta(key) < limit:
            gates.append(key+'_regression')
    if delta('wer') > .03: gates.append('wer_regression')
    if out['means']['wer'] > .15: gates.append('absolute_wer_limit')
    out['bad_wer_rate_delta'] = float(np.mean([r['wer']>.25 for r in takes]) -
                                       np.mean([controls[r['block']]['wer']>.25 for r in takes]))
    if out['bad_wer_rate_delta'] > .10: gates.append('catastrophic_wer_regression')
    out['quality_gate_reasons'] = gates
    out['quality_safe'] = not gates
    out['confirmed_gain'] = not gates and delta('ecapa') >= .01 and out['matched']['ecapa']['low'] > 0
    out['strata']={}
    for axis in ('language','emotion','textset'):
        labels=sorted({str(r[axis]) for r in takes if axis in r})
        if not labels:continue
        out['strata'][axis]={}
        for label in labels:
            subset=[r for r in takes if str(r.get(axis))==label]
            details={'n':len(subset),'means':{},'matched_means':{}}
            for metric in METRICS:
                valid=[r for r in subset if r.get(metric) is not None and controls[r['block']].get(metric) is not None]
                details['means'][metric]=float(np.mean([r[metric] for r in valid])) if valid else None
                details['matched_means'][metric]=float(np.mean([r[metric]-controls[r['block']][metric] for r in valid])) if valid else None
            out['strata'][axis][label]=details
    return out


def compare_conditions(rows, a, b):
    """Paired A-minus-B, used for prospective plateau diagnostics."""
    aa = {r['block']:r for r in rows if r['condition']==a}
    bb = {r['block']:r for r in rows if r['condition']==b}
    if not aa or set(aa)!=set(bb): raise ValueError('Incomplete condition comparison')
    return {key:interval([aa[i][key]-bb[i][key] for i in sorted(aa)
                         if aa[i].get(key) is not None and bb[i].get(key) is not None],
                        [aa[i]['batch_seed'] for i in sorted(aa)
                         if aa[i].get(key) is not None and bb[i].get(key) is not None])
            for key in ('ecapa','content_enjoyment','production_quality','mos','wer')}


def direction_stability(rows, arrays, n, percentage, repetitions=60, **selection_options):
    """Collection-only bootstrap uncertainty, NOT extra generated validation.

    Resample the n-prefix independently within ten cells. Report directional
    cosine and norm relative to the original estimator; no score gain implied.
    """
    center,_=tail_vector(rows,arrays,n,percentage,**selection_options)
    cells=defaultdict(list)
    for row in rows:
        if row['repeat']<n: cells[row['condition_id']].append(row)
    rng=np.random.default_rng(59172+n*100+percentage)
    cosines=[];norms=[]
    for _ in range(repetitions):
        draw=[];activations={}
        for cell,values in sorted(cells.items()):
            values=sorted(values,key=lambda r:r['repeat'])
            for repeat,index in enumerate(rng.integers(0,n,size=n)):
                original=values[index];id_=f'{cell}-bootstrap-{repeat}'
                draw.append({**original,'id':id_,'repeat':repeat})
                activations[id_]=arrays[original['id']]
        vector,_=tail_vector(draw,activations,n,percentage,**selection_options)
        norm=float(np.linalg.norm(vector));norms.append(norm)
        cosines.append(float(vector@center/max(norm*float(np.linalg.norm(center)),1e-12)))
    return {'collection_only':True,'resamples':repetitions,
        'cosine_to_estimate_median':float(np.median(cosines)),
        'cosine_interval':[float(v) for v in np.quantile(cosines,(.025,.975))],
        'l2_interval':[float(v) for v in np.quantile(norms,(.025,.975))],
        'interpretation':'Within-cell prefix bootstrap; no new audio or held-out quality evidence.'}
