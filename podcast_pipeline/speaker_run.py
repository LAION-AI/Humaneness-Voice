#!/usr/bin/env python3
"""Generate fresh, offline Best-of-N podcasts with a 50/50 identity reward."""
import argparse
import json
from pathlib import Path
import time
from recipe import HERE, WORKSPACE, GENERAL, MODEL_REPO, MODEL_REVISION, dump, sha, specification
from run import setup

def main(a):
    setup(a.device)
    import numpy as np
    import torch
    import soundfile as sf
    from safetensors.torch import save_file
    from generation import Generator
    from offline_batch import OfflineBatch
    from speaker_scoring import SpeakerScorer, FORMULA, ECAPA_REPO, ECAPA_REVISION
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch._dynamo.config.cache_size_limit=64
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True)
    refs=json.loads((HERE/'references.json').read_text())
    scripts=[HERE/(f'script-{l}.json' if a.prompt_style=='legacy' else f'script-speaker-{l}.json') for l in ('en','de')]
    jobs=[(json.loads(p.read_text())['language'],t) for p in scripts for t in json.loads(p.read_text())['turns'][:a.limit or None]]
    config=dict(model=MODEL_REPO,revision=MODEL_REVISION,general=GENERAL,references=refs,
        reference_codes_sha256={s:sha(HERE/r['codes']) for s,r in refs.items()},
        scripts={str(p):sha(p) for p in scripts},n=a.n,mode='offline-turn',temperature=1.,cfg=2.5,
        reward=FORMULA,speaker_model=ECAPA_REPO,speaker_revision=ECAPA_REVISION,
        wer_role='final audit only; no multiplier or eligibility filter',voice_conversion=False,
        prompt_style=a.prompt_style,
        sentence_prompt=('Original single turn-level direction; no repeated sentence tags' if a.prompt_style=='legacy'
            else 'Very genuine, very natural, very authentic; context-appropriate emotion'),device=a.device)
    if (out/'configuration.json').exists():assert json.loads((out/'configuration.json').read_text())==config
    else:dump(out/'configuration.json',config)
    dump(out/'source-files.json',{p.name:sha(p) for p in HERE.glob('*.py')})
    tick=time.perf_counter();scorer=SpeakerScorer(a.device,refs)
    g=Generator(a.model,refs,a.device);offline=OfflineBatch(g)
    g.engine.fast.semantic.limit=2;g.engine.fast.graph_limit=2
    load_s=time.perf_counter()-tick
    targets=torch.stack(list(scorer.targets.values()))
    dump(out/'reference-controls.json',dict(speakers=list(scorer.targets),cosine=(targets@targets.T).cpu().tolist(),
        seconds=scorer.reference_seconds,dimensions=int(targets.shape[-1])))
    save_file({s:v.cpu() for s,v in scorer.targets.items()},str(out/'reference-embeddings.safetensors'))
    # Real-model regression: the generic batching path reproduces the existing
    # one-segment runner before enabling prefill reuse or larger groups.
    if a.n==4:
        language,turn=jobs[0];seed=81108
        expected,code,_,_=g.run(turn,language,n=4,mode='turn',chunk_frames=13,max_frames=39,seed=seed,force_winner=0)
        offline.shared=False
        waves,codes,_=offline.synthesize(jobs[:1],n=4,chunk_frames=13,max_frames=39,seed=seed)
        check=dict(same_codes=bool(torch.equal(code,codes[0])),same_pcm=bool(np.array_equal(expected,waves[0])))
        dump(out/'offline-regression.json',check);assert all(check.values()),check
        offline.shared=True
    offline.check_prefill(jobs[:1],a.n)
    if len(jobs)>1:offline.check_prefill(jobs[:2],a.n)
    dump(out/'prefill-checks.json',dict(enabled=offline.shared,checks=offline.prefill_checks))
    # Fixed 104-frame timing probes; truncated probe audio is NEVER an episode.
    probes=[]
    for count in (1,2):
        if count*a.n>16 or count>len(jobs):continue
        for repetition in range(2):
            _,_,perf=offline.synthesize(jobs[:count],n=a.n,max_frames=104,chunk_frames=26,seed=82108)
            probes.append(dict(**perf,repetition=repetition,seconds_per_segment=perf['wall_seconds']/count))
            print('SPEED PROBE',count,repetition,round(perf['wall_seconds'],3),flush=True)
    warm=[p for p in probes if p['repetition']==1]
    groups=min(warm,key=lambda p:p['seconds_per_segment'])['groups']
    dump(out/'speed-pilot.json',dict(probes=probes,selected_groups=groups,n=a.n,
        note='One-segment probe uses the first English turn; two-segment probe uses the first two. Fixed 104-frame cap, warm second repeat. Normalized per-segment comparison is approximate. Probe audio is incomplete and discarded. Different batch shapes change random draws; not an audio-quality comparison.'))
    print('PRODUCTION groups',groups,'N',a.n,'total candidates',groups*a.n,flush=True)
    production_start=time.perf_counter();written=0;batches=[]
    for language in ('en','de'):
        pending=[job for job in jobs if job[0]==language and not (out/f'bestof{a.n}-speaker5050'/f"{language}-{job[1]['index']:02d}"/'result.json').exists()]
        for offset in range(0,len(pending),groups):
            batch=pending[offset:offset+groups];seed=202611080+10000*(language=='de')+batch[0][1]['index']*31
            tick=time.perf_counter();waves,codes,perf=offline.synthesize(batch,n=a.n,chunk_frames=26,seed=seed)
            scores,timing=scorer.score_speakers(waves,[t['speaker'] for _,t in batch for _ in range(a.n)])
            total_s=time.perf_counter()-tick
            perf.update(timing,batch_wall_seconds=total_s,wall_seconds=total_s/len(batch))
            batches.append(dict(language=language,turns=[t['index'] for _,t in batch],**perf))
            dump(out/'production-batches.json',batches)
            for group,(lang,turn) in enumerate(batch):
                local=scores[group*a.n:(group+1)*a.n]
                winner=max(range(a.n),key=lambda i:(local[i]['reward'],-i));idx=group*a.n+winner
                folder=out/f'bestof{a.n}-speaker5050'/f"{lang}-{turn['index']:02d}";folder.mkdir(parents=True,exist_ok=True)
                for i in range(a.n):sf.write(folder/f'candidate-{i}.flac',waves[group*a.n+i],48000,subtype='PCM_24')
                sf.write(folder/'raw.flac',waves[idx],48000,subtype='PCM_24')
                save_file({'codes':codes[idx]},str(folder/'selected-codes.safetensors'))
                per=dict(**perf,mode='turn',first_commit_seconds=total_s,raw_seconds=len(waves[idx])/48000,
                    final_scores=local,winner=winner,group_index=group,group_count=len(batch),
                    timing_allocation='batch wall time divided equally among independent segments; first_commit is full batch latency')
                dump(folder/'result.json',dict(language=lang,turn=turn,variant=f'bestof{a.n}-speaker5050',
                    reference=refs[turn['speaker']],performance=per,prompt=specification(turn,lang)['prompt'],
                    generated_unix=time.time()))
                written+=1
                print('TURN',lang,turn['index'],'winner',winner,'G',round(local[winner]['ears']['genuineness_0_6'],3),
                    'ECAPA',round(local[winner]['ecapa_cosine'],3),'batch_s',round(total_s,2),flush=True)
    g.close()
    dump(out/'generation-complete.json',dict(load_seconds=load_s,production_seconds=time.perf_counter()-production_start,
        turns=written,n=a.n,groups=groups,finished_unix=time.time()))
    print('GENERATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',default=str(WORKSPACE/'podcast-episodes-speaker5050-legacy/generation'))
    p.add_argument('--prompt-style',choices=['legacy','sentence'],default='legacy',
        help='Original restrained turn-level directions by default; sentence retains the archived experiment')
    p.add_argument('--device',default='cuda:0');p.add_argument('--n',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
