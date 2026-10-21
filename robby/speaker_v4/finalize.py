"""Read-only training watcher + final English report synthesis; no GPU or tuning.

Waits for the actual study worker to finish. Never promotes a model, changes
training, reads microphone data, deletes caches, or treats partial runs as final.
"""
import argparse,json,os,shutil,time
from pathlib import Path
import numpy as np
from .common import ROOT,WORK,FEATURES,atomic,digest,report,STATE

def summarize():
    from .train import aggregates,objective
    results=json.loads((WORK/'results.json').read_text())
    if not results.get('finished'):raise RuntimeError('Training not complete; refusing final claims')
    for stage in ('stage1','factorial'):results[stage+'_aggregates']=aggregates(results[stage])
    conclusions=[];baseline=results['original_frozen_320ms_control']['test']
    control=results['original_frozen_320ms_control']
    candidates=[(objective(control['validation']),'original_frozen','query_gru_legacy',[control])]
    for stage in ('stage1','factorial'):
        for arm in results[stage+'_aggregates']:
            rows=[r for r in results[stage] if r['arm']==arm]
            candidates.append((float(np.mean([objective(r['validation']) for r in rows])),stage,arm,rows))
    _,stage,best,rows=max(candidates,key=lambda x:x[0])
    results['best_overall_by_validation']={'stage':stage,'arm':best,'note':'Operational selection, not a causal full-data/subset comparison'}
    test_mean=float(np.mean([r['test']['macro_speaker_accuracy'] for r in rows]))
    conclusions.append(f"Original frozen 320-ms control: {100*baseline['macro_speaker_accuracy']:.2f}% test macro identification. Best overall option by validation, including that original control: {best} ({stage}), {100*test_mean:.2f}% test macro identification, a descriptive change of {100*(test_mean-baseline['macro_speaker_accuracy']):+.2f} percentage points. New arms average three seeds; the original control is one frozen checkpoint. Test is among ten previously benchmarked synthetic identities.")
    first=results['stage1_aggregates'];special=first['specialize_only']['validation']
    for arm in sorted(a for a in first if a.startswith('stable')):
        other=first[arm]['validation']
        conclusions.append(f"{arm} versus specialization alone (validation means): macro accuracy {100*(other['macro_speaker_accuracy']['mean']-special['macro_speaker_accuracy']['mean']):+.2f} pp; genuine enrollment cosine {other['positive_cosine']['mean']-special['positive_cosine']['mean']:+.4f}; impostor cosine {other['impostor_cosine']['mean']-special['impostor_cosine']['mean']:+.4f}; EER {100*(other['eer']['mean']-special['eer']['mean']):+.2f} pp. A higher genuine cosine is not automatically better discrimination.")
    base=[r for r in results['factorial'] if r['arm']=='base12'];paired={}
    for arm in sorted({r['arm'] for r in results['factorial']} - {'base12'}):
        treatment=[r for r in results['factorial'] if r['arm']==arm];paired[arm]={}
        for split in ('validation','test'):
            speakers=sorted(base[0][split]['per_speaker']);differences=[]
            for speaker in speakers:
                a=np.mean([r[split]['per_speaker'][speaker]['accuracy'] for r in treatment])
                b=np.mean([r[split]['per_speaker'][speaker]['accuracy'] for r in base]);differences.append(a-b)
            draws=np.random.default_rng(943).choice(differences,(5000,len(differences))).mean(1)
            paired[arm][split]={'macro_accuracy_delta_pp':float(np.mean(differences)*100),
                'paired_speaker_bootstrap_95ci_pp':[float(v*100) for v in np.quantile(draws,[.025,.975])],
                'identities':len(speakers),'seeds_averaged_before_bootstrap':len(base)}
        point=paired[arm]['test'];lo,hi=point['paired_speaker_bootstrap_95ci_pp']
        conclusions.append(f"Matched factorial {arm} versus base12: test macro accuracy {point['macro_accuracy_delta_pp']:+.2f} pp; paired identity bootstrap interval [{lo:+.2f}, {hi:+.2f}] pp. This interval uses only ten synthetic identities; it is not evidence of real-human reliability.")
    results['paired_factorial_comparisons']=paired
    cleaned=json.loads((WORK/'cleanup_manifest.json').read_text())
    resources={'surplus_downloads_removed':len(cleaned['files']),'bytes_removed':sum(r['bytes'] for r in cleaned['files']),
        'cleanup_manifest_sha256':digest(WORK/'cleanup_manifest.json'),'feature_cache_files':{},'checkpoints':{},
        'working_volume_free_bytes':shutil.disk_usage(WORK).free,'nvme_volume_free_bytes':shutil.disk_usage(ROOT.parent).free,
        'feature_cache_volume_free_bytes':shutil.disk_usage(FEATURES).free,'feature_cache_location':str(FEATURES.resolve()),
        'feature_cache_is_ephemeral':FEATURES.resolve().is_relative_to(Path('/dev/shm'))}
    if (WORK/'cache_migration.json').exists():resources['cache_migration']=json.loads((WORK/'cache_migration.json').read_text())
    for path in FEATURES.glob('*.npy'):resources['feature_cache_files'][path.name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    for path in (WORK/'models').glob('*-best.pt'):resources['checkpoints'][path.name]={'bytes':path.stat().st_size,'sha256':digest(path)}
    results['resources_and_integrity']=resources
    conclusions.append('No model was automatically enabled as the live user filter. Pooling spans of 320 ms already include past encoder context. Cold-start real-user verification and secure authentication remain untested.')
    results['conclusions']=conclusions;STATE.update(json.loads((ROOT/'progress.json').read_text()))
    report(results,True);atomic(WORK/'finalized.json',{'completed':time.time(),'best_group':best,'stage':stage})

def main(wait=False):
    if wait:
        atomic(ROOT/'report_watcher.json',{'pid':os.getpid(),'started':time.time()})
        dead_checks=0
        while True:
            path=WORK/'results.json'
            if path.exists() and json.loads(path.read_text()).get('finished'):break
            status=json.loads((ROOT/'progress.json').read_text()) if (ROOT/'progress.json').exists() else {}
            if status.get('error'):
                STATE.update(status);report();return
            record=json.loads((ROOT/'runner.json').read_text()) if (ROOT/'runner.json').exists() else {}
            worker=Path('/proc',str(record.get('pid',0)),'cmdline')
            if not worker.exists():
                dead_checks+=1
                if dead_checks>=6:return
                time.sleep(5);continue
            dead_checks=0
            time.sleep(20)
    summarize()

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');main(parser.parse_args().wait)
