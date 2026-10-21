"""Resumable sequential study. CUDA_VISIBLE_DEVICES=0 python -m speaker_v4.run"""
import json,os,time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,OLD,WORK,FEATURES,SEEDS,atomic,guard,update,report,digest,cleanup_unused
from . import data
from .train import Resident,Pairs,train_one,objective,aggregates,final_tests

def main():
    guard();record=ROOT/'runner.json'
    if record.exists():
        old=json.loads(record.read_text());pid=old.get('pid',0)
        try:
            if pid!=os.getpid() and b'speaker_v4.run' in Path('/proc',str(pid),'cmdline').read_bytes():raise RuntimeError('Another V4 study is active')
        except FileNotFoundError:pass
    atomic(record,{'pid':os.getpid(),'started':time.time()})
    if not (WORK/'cleanup_manifest.json').exists():cleanup_unused()
    guard();torch.set_num_threads(4);torch.cuda.set_per_process_memory_fraction(.20,0)
    protocol={'seeds':list(SEEDS),'stage1_arms':{'specialize_only':0.,'stable50':.5,'stable80':.8},
        'factorial_arms':['base12','layers3','hard4','layers3_hard4','random_pairs_control'],
        'frames':4,'frame_ms':80,'layer_ids_one_based':[4,8,12,15],'original_encoder_window_seconds':8,
        'epochs':8,'batch_size':512,'learning_rate':.0002,'layer_weight_learning_rate':.003,
        'selection':'320-ms validation macro accuracy - .5 EER + .1 utterance cosine; select alpha by mean of three seeds',
        'split':'original 480/10/10 disjoint identities; original ten test identities reused for frozen comparisons',
        'baseline_checkpoint_sha256':digest(OLD/'models/query_gru-best.pt'),
        'source_dataset_revision':'88ea857d6a6f20839674dfde30936d8fd6b23c62',
        'feature_bytes_planned':116000*3*4*1024+116000*3*4*2,'disk_floor_GiB':3}
    path=WORK/'protocol.json'
    if path.exists() and json.loads(path.read_text())!=protocol:raise RuntimeError('Protocol changed since study start')
    atomic(path,protocol)
    results=json.loads((WORK/'results.json').read_text()) if (WORK/'results.json').exists() else {'protocol':protocol,'stage1':[],'factorial':[]}
    baseline=json.loads((OLD/'models/results.json').read_text())
    winner=next(r for r in baseline['results'] if r.get('selected'))
    results['original_frozen_320ms_control']={'validation':winner['validation_contexts']['320ms'],'test':winner['test_contexts']['320ms']}
    update(phase='Preparing stable training-speaker targets');data.stable_targets();report(results)
    update(phase='Loading bounded L15 four-frame GPU cache')
    resident=None if results.get('selected_stage1_stable_arm') else Resident(np.arange(500000))
    for arm,alpha in protocol['stage1_arms'].items():
        for seed in SEEDS:
            if any(r['arm']==arm and r['seed']==seed for r in results['stage1']):continue
            row=train_one(resident,arm,seed,alpha,OLD/'models/query_gru-best.pt')
            results['stage1'].append(row);results['stage1_aggregates']=aggregates(results['stage1']);report(results)
    stable_arms=['stable50','stable80'];weights={**protocol['stage1_arms'],'stable20':.2}
    means={arm:np.mean([objective(r['validation']) for r in results['stage1'] if r['arm']==arm]) for arm in stable_arms}
    specialize=np.mean([objective(r['validation']) for r in results['stage1'] if r['arm']=='specialize_only'])
    if max(means.values())<specialize or (WORK/'validation_extension.json').exists():
        extension=WORK/'validation_extension.json'
        if not extension.exists():atomic(extension,{'declared_at':time.time(),'test_results_seen':False,'additional_alpha':.2,
            'reason':'Both stronger centroid mixtures underperform specialization alone on validation; test a gentler mixture',
            'validation_objectives_before_extension':{**means,'specialize_only':float(specialize)}})
        results['validation_only_extension']=json.loads(extension.read_text())
        for seed in SEEDS:
            if any(r['arm']=='stable20' and r['seed']==seed for r in results['stage1']):continue
            results['stage1'].append(train_one(resident,'stable20',seed,.2,OLD/'models/query_gru-best.pt'))
            results['stage1_aggregates']=aggregates(results['stage1']);report(results)
        stable_arms.append('stable20');means['stable20']=np.mean([objective(r['validation']) for r in results['stage1'] if r['arm']=='stable20'])
    selected=max(means,key=means.get);results['selected_stage1_stable_arm']=selected
    alpha=weights[selected];results['selected_centroid_weight']=alpha
    # Do not evaluate any newly trained model on test until every experiment is frozen.
    del resident;torch.cuda.empty_cache()
    update(phase='Preparing exact paired lower-layer features',selected_centroid_weight=alpha);report(results)
    data.metadata();data.extract_layers();guard()
    resident=Resident(data.subset(),with_layers=True);pairs=Pairs(resident,data.stable_targets())
    configs=[('base12',False,None),('layers3',True,None),('hard4',False,'hard'),('layers3_hard4',True,'hard'),('random_pairs_control',False,'random')]
    for arm,mix,mode in configs:
        for seed in SEEDS:
            if any(r['arm']==arm and r['seed']==seed for r in results['factorial']):continue
            warm=next(r for r in results['stage1'] if r['arm']==selected and r['seed']==seed)
            row=train_one(resident,arm,seed,alpha,WORK/'models'/warm['checkpoint'],mix_layers=mix,pair_mode=mode,pairs=pairs)
            results['factorial'].append(row);results['factorial_aggregates']=aggregates(results['factorial']);report(results)
    groups=results['factorial_aggregates']
    picked=max(groups,key=lambda arm:np.mean([objective(r['validation']) for r in results['factorial'] if r['arm']==arm]))
    results['selected_factorial_arm']=picked;results.setdefault('selection_frozen_at',time.time());report(results)
    update(phase='All arms frozen; evaluating original held-out test benchmark',selected_arm=picked)
    final_tests(results,resident,'factorial');del pairs,resident;torch.cuda.empty_cache()
    masks=data.splits();resident=Resident(np.r_[masks['validation'],masks['test']]);final_tests(results,resident,'stage1')
    results['stage1_aggregates']=aggregates(results['stage1']);results['factorial_aggregates']=aggregates(results['factorial'])
    results['finished']=True;results['completed']=time.time()
    update(phase='320-ms specialization and factorial study complete',selected_arm=picked,completed=time.time());report(results,True)

if __name__=='__main__':
    try:main()
    except BaseException as exc:
        update(phase='Stopped safely; retained work is resumable',error=type(exc).__name__+': '+str(exc)[:500]);report()
        raise
