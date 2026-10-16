"""Rebuild the English, self-contained audit from immutable experiment rows."""
import json
from datetime import datetime, timezone

from catalogue_steering_study import ROOT, BANK, read_results, derive_vectors, voice_catalog
from catalogue_steering_analysis import assessment, render_catalogue_report
import dynamic_references


def main():
    bank=derive_vectors()
    rows=read_results()
    reports=[]
    for voice in voice_catalog():
        evaluation=[r for r in rows if r['phase']=='validation' and r['voice']==voice['id']]
        conditions=sorted({r['condition'] for r in evaluation}-{'baseline','no-reference-baseline'})
        for condition in conditions:
            baseline='no-reference-baseline' if condition.startswith('reference-without-') else 'baseline'
            reports.append(assessment(evaluation,baseline=baseline,condition=condition))
    public=[{k:v for k,v in row.items() if k not in ('speaker_embedding','traceback')} for row in rows]
    summary={'generated_at':datetime.now(timezone.utc).isoformat(),'takes':len(rows),
        'collection':sum(r['phase']=='collection' for r in rows),
        'validation':sum(r['phase']=='validation' for r in rows),
        'technical_failures':sum(bool(r.get('failed')) for r in rows),'reports':reports,
        'voice_bank':bank,'mlp_trained':False,'multi_layer_grid_run':False}
    for filename,data in [('results.json',public),('summary.json',summary)]:
        temp=ROOT/(filename+'.build');temp.write_text(json.dumps(data,ensure_ascii=False,indent=2));temp.replace(ROOT/filename)
    plan=json.loads((ROOT/'design.json').read_text())
    evidence={'prospective_no_reference_extension':json.loads((ROOT/'reference-effect-validation-extension.json').read_text()),
        'production_gpu_canary':json.loads((ROOT.parent/'test-results/speaker-steering-gpu/report.json').read_text())}
    references={v['name']:dynamic_references.fixed_clip(v['id'],playback=True)[0] for v in voice_catalog()}
    path=ROOT/'index.build.html'
    result=render_catalogue_report(path,rows,reports,bank,plan=plan,audio_root=ROOT,
        evidence=evidence,reference_audio=references,
        narrative=['Quality contrasts show promising held-out speaker gains for Mira, Nora and Osku. Effects are not uniform across voices; failed screens stay visible and experimentally selectable at the user’s request.',
            'Unfiltered reference-minus-no-reference averages did not establish a reliable, quality-preserving general speaker-conditioning mechanism. Semantic interventions often fail safeguards. These vectors are not the UI default.',
            'Ten fresh evaluation tasks per condition are a small proxy pilot. Confidence intervals are exploratory, not corrected for the many comparisons. Emotion-specific improvements and universal speaker generalization are not established. No MLP or three-layer coefficient grid was trained or run.'],
        downloads={'Results ledger':'results.json','Summary and vector audit':'summary.json','Frozen protocol':'design.json'})
    path.replace(ROOT/'index.html')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
