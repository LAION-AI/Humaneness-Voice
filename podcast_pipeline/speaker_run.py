#!/usr/bin/env python3
"""Generate fresh, offline Best-of-N podcasts with a 50/50 identity reward."""
import argparse
import json
import os
from pathlib import Path
import time
from concurrent.futures import ThreadPoolExecutor
from archive import archive_batch
from recipe import HERE, WORKSPACE, GENERAL, MODEL_REPO, MODEL_REVISION, dump, sha, specification
from run import setup

def main(a):
    if not 1 <= a.n <= 16 or a.groups < 0 or a.groups*a.n > 16 or a.limit < 0:
        raise ValueError("N must be 1..16; groups*N <=16; limit must be nonnegative")
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
    scripts=[HERE/(f'script-{l}.json' if a.prompt_style=='legacy' else f'script-{a.prompt_style if a.prompt_style=="spontaneous" else "speaker"}-{l}.json') for l in ('en','de')]
    jobs=[(json.loads(p.read_text())['language'],t) for p in scripts for t in json.loads(p.read_text())['turns'][:a.limit or None]]
    config=dict(model=MODEL_REPO,revision=MODEL_REVISION,general=GENERAL,references=refs,
        reference_codes_sha256={s:sha(HERE/r['codes']) for s,r in refs.items()},
        scripts={str(p):sha(p) for p in scripts},n=a.n,mode='offline-turn',temperature=1.,cfg=2.5,
        reward=FORMULA,speaker_model=ECAPA_REPO,speaker_revision=ECAPA_REVISION,
        wer_role='final audit only; no multiplier or eligibility filter',voice_conversion=False,
        optimization_flags={k:v for k,v in os.environ.items() if k.startswith('HVS_') and not k.endswith('CACHE')},
        prompt_style=a.prompt_style,requested_groups=a.groups,archive_mode='bounded-cpu-thread',
        sentence_prompt=('Original single turn-level direction; no repeated sentence tags' if a.prompt_style=='legacy'
            else 'Sparse spontaneous, relaxed, gentle inline cues' if a.prompt_style=='spontaneous' else 'Very genuine, very natural, very authentic; context-appropriate emotion'),device=a.device)
    if (out/'configuration.json').exists():assert json.loads((out/'configuration.json').read_text())==config
    else:dump(out/'configuration.json',config)
    source_hashes={str(p.relative_to(HERE)):sha(p) for p in list(HERE.glob('*.py'))+list((HERE/'runtime').glob('*.py'))}
    stamp=out/'source-files.json'
    if stamp.exists():
        if json.loads(stamp.read_text()) != source_hashes:raise ValueError('Runtime changed; use a new output directory')
    else:dump(stamp,source_hashes)
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
    if not 0<=a.groups<=16//a.n:raise ValueError("groups * N must be <=16")
    if a.groups:
        groups=a.groups
        dump(out/"speed-pilot.json",dict(probes=[],selected_groups=groups,n=a.n,note="Explicit measured batch configuration; timing sweep omitted, correctness checks retained"))
    else:
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
    production_start=time.perf_counter();written=0
    batch_file=out/"production-batches.json"
    batches=json.loads(batch_file.read_text()) if batch_file.exists() else []
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix="podcast-archive")
    archive_job=None;archive_stats=[];archive_wait_s=0.
    for language in ('en','de'):
        pending=[job for job in jobs if job[0]==language]
        for offset in range(0,len(pending),groups):
            batch=pending[offset:offset+groups]
            if all((out/f'bestof{a.n}-speaker5050'/f"{language}-{t['index']:02d}"/'result.json').exists() for _,t in batch):continue
            seed=202611080+10000*(language=='de')+batch[0][1]['index']*31
            tick=time.perf_counter();waves,codes,perf=offline.synthesize(batch,n=a.n,chunk_frames=26,seed=seed)
            scores,timing=scorer.score_speakers(waves,[t['speaker'] for _,t in batch for _ in range(a.n)])
            total_s=time.perf_counter()-tick
            perf.update(timing,batch_wall_seconds=total_s,wall_seconds=total_s/len(batch))
            indices=[t['index'] for _,t in batch]
            batches=[b for b in batches if not (b['language']==language and b['turns']==indices)]
            batches.append(dict(language=language,turns=indices,**perf))
            dump(out/'production-batches.json',batches)
            # At most one previous batch is writing while this GPU batch runs.
            if archive_job is not None:
                wait=time.perf_counter();archive_stats.append(archive_job.result());archive_wait_s+=time.perf_counter()-wait
            archive_job=pool.submit(archive_batch,out,batch,waves,codes,scores,perf,refs,a.n,total_s)
            written+=len(batch)
    if archive_job is not None:
        wait=time.perf_counter();archive_stats.append(archive_job.result());archive_wait_s+=time.perf_counter()-wait
    pool.shutdown(wait=True)
    dump(out/'archive-timing.json',dict(batches=archive_stats,blocked_seconds=archive_wait_s,
        cpu_seconds=sum(x['seconds'] for x in archive_stats),bounded_pending_batches=1))
    g.close()
    dump(out/'generation-complete.json',dict(load_seconds=load_s,production_seconds=time.perf_counter()-production_start,
        turns=len(jobs),new_or_replayed_turns=written,n=a.n,groups=groups,finished_unix=time.time()))
    print('GENERATION COMPLETE',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True)
    p.add_argument('--output',default=str(WORKSPACE/'podcast-episodes-speaker5050-legacy/generation'))
    p.add_argument('--prompt-style',choices=['legacy','sentence','spontaneous'],default='legacy',
        help='Original restrained turn-level directions by default; sentence retains the archived experiment')
    p.add_argument('--groups',type=int,default=0,help='0 measures automatic choice; explicit group count skips only timing sweep')
    p.add_argument('--device',default='cuda:0');p.add_argument('--n',type=int,default=4);p.add_argument('--limit',type=int,default=0)
    main(p.parse_args())
