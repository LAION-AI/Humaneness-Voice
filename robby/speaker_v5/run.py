"""Queue behind V4, then run and report all frozen context ablations."""
import argparse,json,os,time
from pathlib import Path
import numpy as np
from .common import ROOT,WORK,OLD,SEEDS,CONTEXTS,ARMS,atomic,digest,guard,update,report

def wait_previous():
    update(phase='Queued behind the 320-ms layer/hard-negative study')
    report()  # initial HTML must not retain a safety-stop heading after resume
    missing=0
    while True:
        guard();previous=WORK.parent/'robo-v4';path=previous/'results.json';final=previous/'finalized.json'
        if path.exists() and json.loads(path.read_text()).get('finished') and final.exists():return
        progress=ROOT.parent/'speaker_v4/progress.json';state=json.loads(progress.read_text()) if progress.exists() else {}
        if state.get('error'):raise RuntimeError('Earlier study stopped: '+state['error'])
        record=ROOT.parent/'speaker_v4/runner.json';pid=json.loads(record.read_text()).get('pid',0) if record.exists() else 0
        try:alive=b'speaker_v4.run' in Path('/proc',str(pid),'cmdline').read_bytes()
        except FileNotFoundError:alive=False
        # Worker may finish while its CPU-only finalizer is still writing hashes.
        finished=path.exists() and json.loads(path.read_text()).get('finished')
        missing=0 if alive or finished else missing+1
        if missing>=6:raise RuntimeError('Earlier worker no longer running; refusing to assume completion')
        update(phase='Queued behind the 320-ms layer/hard-negative study',previous_phase=state.get('phase'),previous_windows=state.get('layer_windows'))
        time.sleep(20)

def main(wait=True):
    guard();record=ROOT/'runner.json'
    if record.exists():
        pid=json.loads(record.read_text()).get('pid',0)
        try:
            if pid!=os.getpid() and b'speaker_v5.run' in Path('/proc',str(pid),'cmdline').read_bytes():raise RuntimeError('V5 already active')
        except FileNotFoundError:pass
    atomic(record,{'pid':os.getpid(),'started':time.time()})
    protocol={'declared_at':'2026-10-19','contexts_frames':list(CONTEXTS),'contexts_ms':[f*80 for f in CONTEXTS],
        'recipes':{a:{'centroid_weight':v[0],'pair_mode':v[1]} for a,v in ARMS.items()},'seeds':list(SEEDS),
        'epochs':8,'batch_size':256,'learning_rate':.0002,'early_stop':'at least 5 epochs, patience 3',
        'training_windows':96000,'validation_windows':10000,'test_windows':10000,'split':'original 480/10/10 identity-disjoint',
        'warm_start_sha256':digest(OLD/'models/query_gru-best.pt'),'layer':15,'encoder_history_seconds':8,
        'selection':'per-context validation macro accuracy - .5 EER + .1 source-utterance cosine; average 3 seeds',
        'test_policy':'all configurations and context winners frozen first; ten test speakers already benchmarked in earlier studies',
        'speaker_target':'speaker in newest 320 ms, not majority of all context',
        'source_revision':'88ea857d6a6f20839674dfde30936d8fd6b23c62','gpu':'0, sequential after V4; TTS GPU1 untouched'}
    path=WORK/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol changed; refusing silent experiment mutation')
    atomic(path,protocol)
    results=json.loads((WORK/'results.json').read_text()) if (WORK/'results.json').exists() else {'protocol':protocol,'runs':[]}
    report(results)
    if results.get('finished'):return
    # A deliberate resume must not keep presenting yesterday's safety stop as
    # a current failure. The historical error remains in events.jsonl.
    update(phase='Resuming saved context study',error=None)
    if wait:wait_previous()
    import torch
    from . import data
    from .train import train_one,summarize_groups,objective,evaluate,prototypes,paired
    from speaker_v4.train import MixedStudent
    torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.25,0)
    update(phase='Preparing current-speaker onset slices');data.onset_ages()
    update(phase='Preparing bounded 3.04-s prefix feature cache');cache=data.extract();guard(cache)
    for frames in CONTEXTS:
        pending=[(recipe,seed) for recipe in ARMS for seed in SEEDS if not any(r['frames']==frames and r['recipe']==recipe and r['seed']==seed for r in results['runs'])]
        if not pending:continue
        update(phase='Loading bounded L15 context cache',context_ms=frames*80)
        resident=data.Resident(frames)
        for recipe,seed in pending:
            results['runs'].append(train_one(resident,frames,recipe,seed));results['aggregates']=summarize_groups(results['runs']);report(results)
        del resident;torch.cuda.empty_cache()
    results['selected_recipe_by_context']={str(f*80):max(ARMS,key=lambda a:np.mean([objective(r['validation']) for r in results['runs'] if r['frames']==f and r['recipe']==a])) for f in CONTEXTS}
    results.setdefault('selection_frozen_at',time.time());report(results)
    proto=prototypes()
    for frames in CONTEXTS:
        if all(r.get('test') for r in results['runs'] if r['frames']==frames):continue
        resident=data.Resident(frames)
        for row in results['runs']:
            if row['frames']!=frames or row.get('test'):continue
            update(phase='Frozen held-out context evaluation',context_ms=frames*80,recipe=row['recipe'],seed=row['seed'])
            saved=torch.load(WORK/'models'/row['checkpoint'],map_location=resident.device,weights_only=False)
            model=MixedStudent(False).to(resident.device);model.load_state_dict(saved['model']);model.eval()
            row['test']=evaluate(model,resident,'test',proto,row['validation']['far05_threshold'])
            del model,saved;report(results)
        del resident;torch.cuda.empty_cache()
    results['aggregates']=summarize_groups(results['runs']);comparisons={};conclusions=[]
    for recipe in ARMS:
        baseline=[r for r in results['runs'] if r['frames']==4 and r['recipe']==recipe]
        for frames in (25,38):
            rows=[r for r in results['runs'] if r['frames']==frames and r['recipe']==recipe]
            label=f'{frames*80}ms/{recipe} versus 320ms/{recipe}';comparisons[label]=paired(rows,baseline,'test')
    for frames in CONTEXTS:
        selected=results['selected_recipe_by_context'][str(frames*80)];arm=f'{frames*80}ms/{selected}';group=results['aggregates'][arm]
        t=group['test'];conclusions.append(f"Validation-selected {frames*80}-ms recipe: {selected}. Test macro identification {100*t['macro_speaker_accuracy']['mean']:.2f}%, EER {100*t['eer']['mean']:.2f}%, source-utterance cosine {t['source_utterance_teacher_cosine']['mean']:.4f}, genuine enrollment cosine {t['positive_cosine']['mean']:.4f}, impostor cosine {t['impostor_cosine']['mean']:.4f}; student-only GPU batch-1 {group['student_gpu_batch1_ms']:.3f} ms.")
    # Paired recipe-selected comparisons are also chosen without test feedback.
    winners={f:[r for r in results['runs'] if r['frames']==f and r['recipe']==results['selected_recipe_by_context'][str(f*80)]] for f in CONTEXTS}
    for a,b in ((25,4),(38,4),(38,25)):
        label=f'selected {a*80}ms versus selected {b*80}ms';comparison=paired(winners[a],winners[b],'test');comparisons[label]=comparison
        lo,hi=comparison['paired_identity_bootstrap_95ci_pp'];conclusions.append(f"{label}: {comparison['delta_macro_accuracy_pp']:+.2f} pp macro accuracy, paired identity bootstrap interval [{lo:+.2f}, {hi:+.2f}] pp. Only ten synthetic test identities; examine onset slices and impostor similarity before preferring longer context.")
    old=json.loads((OLD/'models/results.json').read_text());winner=next(r for r in old['results'] if r.get('selected'))
    results['original_frozen_controls']={s:winner[s+'_contexts'] for s in ('validation','test')}
    results['paired_comparisons']=comparisons;results['conclusions']=conclusions
    results['resources']={'derived_cache':str(cache),'ephemeral':True,'derived_bytes':sum(p.stat().st_size for p in cache.glob('*.npy')),
        'persistent_checkpoints':{p.name:{'bytes':p.stat().st_size,'sha256':digest(p)} for p in (WORK/'models').glob('*-best.pt')},
        'no_new_source_downloads':True,'no_original_data_removed':True}
    results['finished']=True;results['completed']=time.time();update(phase='Context comparison complete; no live model promotion',completed=results['completed']);report(results)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--no-wait',action='store_true');args=parser.parse_args()
    try:main(not args.no_wait)
    except BaseException as exc:
        update(phase='Stopped safely; context study is resumable',error=type(exc).__name__+': '+str(exc)[:500]);report();raise
