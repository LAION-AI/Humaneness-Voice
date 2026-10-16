"""Rebuild the English, self-contained report from retained study artifacts."""
from collections import Counter
import hashlib
import json

import mira_activation_analysis as analysis
from mira_activation_study import ROOT, REFERENCE, read_results, write_json


def load(name):
    path = ROOT / name
    return json.loads(path.read_text()) if path.is_file() else None


def build():
    rows = read_results()
    summary = analysis.summarize(rows)
    validation = [c for c in summary['conditions'] if c['phase'] == 'validation']
    winners = [c['condition'] for c in validation if c['heldout_positive_interval']]
    counts = dict(Counter(r['phase'] for r in rows))
    metadata = load('vectors/metadata.json')
    primary = ROOT / 'vectors/selected.npz'
    language_counts = Counter((r['language'], r['emotion']) for r in rows
                             if r['id'] in set(metadata['selected']['high_ids']))
    evidence = {
        'phase_counts': counts,
        'generation_failures': sum(r['failed'] for r in rows),
        'primary_vector_sha256': hashlib.sha256(primary.read_bytes()).hexdigest(),
        'primary_high_language_emotion_counts': {f'{k[0]} / {k[1]}':v for k,v in language_counts.items()},
        'instrumentation': load('instrumentation-check.json'),
        'measurement_amendment': load('measurement-amendment.json'),
        'reference_controls': load('reference-controls.json'),
        'semantic_bridge': load('semantic-bridge-diagnostic.json'),
        'balanced_diagnostic': load('balanced-diagnostic-summary.json'),
        'balanced_vector_selection': load('vector-diagnostics/balanced025-metadata.json'),
        'balanced_startup_note': load('balanced-execution-note.json'),
        'vector_forensics': load('vector-diagnostics-cosine050/report.json'),
        'validation_selection': load('validation-plan.json'),
        'independent_validation_audit': load('independent-validation-audit.json'),
        'source_manifest': load('source-manifest.json'),
        'reliability_overview': load('reliability-overview.json'),
        'https_browser_check': load('browser-check.json'),
    }
    conclusions = {
        'takes': len(rows), 'phase_counts': counts,
        'generation_failures': evidence['generation_failures'],
        'validation_takes': counts.get('validation', 0),
        'gate_and_positive_text_bootstrap_candidates': winners,
        'live_demo_steering_enabled': False,
        'interpretation': 'All gains are specific to one Mira conditioning clip and automated metrics; validation and per-language/anchor checks take precedence over discovery means. ECAPA cosine is not a match percentage. Cluster intervals and multiple-search limitations must be considered.',
        'primary_vectors': evidence['primary_vector_sha256'],
    }
    conclusions['validation_by_language'] = analysis.language_diagnostics(
        [r for r in rows if r['phase'] == 'validation'])
    conclusions['validation_whole_batch_intervals'] = analysis.language_diagnostics(
        [r for r in rows if r['phase'] == 'validation'], by_language=False)
    conclusions['best_tested_candidate'] = next((c for c in validation
        if c['condition'] == 'contrast-layer_16-a0.75'), None)
    write_json(ROOT / 'summary.json', summary)
    write_json(ROOT / 'conclusions.json', conclusions)
    # No user uploads or private scoring embeddings are included in public JSON.
    public_rows = [{k:v for k,v in r.items() if k not in {'speaker_embedding', 'traceback'}} for r in rows]
    write_json(ROOT / 'results.json', public_rows)
    narrative = [
        f'{len(rows)} retained generations; {evidence["generation_failures"]} technical generation failures. The study is offline, one take per condition/block, without best-of-N selection, conversion, trimming or quality retries.',
        'Raw means did not give a dependable speaker direction: all 28 first-screen layer conditions failed at least one safeguard, and every raw semantic condition lost reference cosine. Some refinements improved discovery scores, but discovery and validation are separate.',
        'The >0.50 primary pool has 23 examples: 22 German and one English. Equal-emotion averaging does not remove this language/phonetic/session confound. Secondary >0.25 directions balance language × emotion jointly, but also change threshold, contrast definition and perturbation norm; they cannot isolate only the language effect.',
        'The strongest pilot routes use high-minus-low or high-minus-all differences rather than uncentered means. The latter is a secondary diagnostic, not the primary vector. Generic-mean controls use different norms, so they do not by themselves prove a speaker-specific causal mechanism.',
        'Total alpha is divided across the selected sites in two/three-layer conditions. Alpha 0.5 at two layers means 0.25 per layer. Raw and RMS-relative doses are not directly comparable; all intervention definitions are retained.',
        'The live playground keeps its existing defaults. No activation steering is installed or enabled merely because a pilot score rose. Listen to matched takes, especially English and emotional cases, before interpreting a reference-score gain as improved identity.',
    ]
    if 'contrast-layer_16-a0.75' in winners:
        narrative.insert(1,
            'Best tested candidate: add 0.75 × the emotion-balanced high-minus-low Mira direction to block 16 output (zero-based; the 17th global transformer block, 1024 dimensions), in both CFG branches, using raw scaling, not RMS-relative scaling. On 20 fresh matched cases, reference ECAPA rises 0.376 → 0.451, Content Enjoyment 6.097 → 6.252, WER falls 6.75% → 2.00%, and alternate-file ECAPA rises 0.155 → 0.204. Speaker gains occur in both English and German. These are measured means, not proof that every take improves.')
        narrative.insert(2,
            'The speaker gain +0.0745 has a paired 95% interval [+0.0332,+0.1174]; whole-batch resampling also stays positive [+0.0393,+0.1056]. Production quality and predicted MOS improve in the mean. Aesthetics, genuineness and emotion means do not collapse, but their intervals are wide: there is no established emotion/aesthetics improvement. Layer 16 at 0.75 is the best validated setting tested, not a continuous optimum or a universal speaker direction.')
        narrative.insert(3,
            'The literal raw-layer mean at block 17 / alpha 0.25 does not establish a speaker gain (+0.0079, interval crosses zero). The primary projected semantic contrast at 0.75 remains suggestive (+0.0409, interval crosses zero). The language-balanced secondary semantic route loses its pilot advantage (−0.0158). This is why pilot rankings alone should not set production defaults.')
    result = analysis.render_report(ROOT / 'index.build.html', load('design.json'), rows,
        summary=summary, vectors_metadata=metadata, audio_root=ROOT,
        evidence=evidence, narrative=narrative,
        reference_audio={'Mira conditioning reference — 10.26 seconds':REFERENCE.with_suffix('.mp3'),
            'Same-corpus-label independent-file anchor — 6.72 seconds':ROOT/'audio/mira-independent-anchor.mp3'},
        downloads={'Frozen protocol':'design.json','All results':'results.json',
            'Summary':'summary.json','All primary vectors':'vectors.npz',
            'Validation selection and amendment':'validation-plan.json'})
    (ROOT / 'index.build.html').replace(ROOT / 'index.html')
    write_json(ROOT / 'report-build.json', result)
    print(json.dumps({**conclusions, **result}, indent=2))


if __name__ == '__main__':
    build()
