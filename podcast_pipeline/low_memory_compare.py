"""Phased configurable Best-of-N comparison: policy -> release -> codec + judges -> final audit.

The exact same sampling loop can defer waveform decoding. Stored acoustic tokens
allow scoring/reassembly without simultaneously keeping policy and codec in VRAM.
No cross-stage speedup claim: record stage costs and concurrent GPU workloads.
"""
import argparse,gc,json,os,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from recipe import HERE,GENERAL,MODEL_REPO,MODEL_REVISION,dump,sha
from run import setup
from inputs import load_inputs
from speaker_run import arguments
from languages import LANGUAGE_SEEDS

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--model',required=True);p.add_argument('--output',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--script',action='append');p.add_argument('--n-values',type=int,nargs='+',default=[2,1]);p.add_argument('--candidates',type=int,default=8);a=p.parse_args()
    if not a.n_values or len(set(a.n_values))!=len(a.n_values) or any(n<1 or n>a.candidates or a.candidates%n for n in a.n_values) or not 1<=a.candidates<=16:raise ValueError('N values must divide the 1..16 simultaneous candidate budget')
    setup(a.device)
    import numpy as np
    import torch
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    from safetensors.torch import save_file,load_file
    from phased_generate import TokenEngine
    from offline_batch import OfflineBatch
    from generation import DecoderOnly,DecoderGraph
    from speaker_scoring import SpeakerScorer,FORMULA,ECAPA_REPO,ECAPA_REVISION
    from archive import archive_batch
    torch.set_num_threads(4);torch.cuda.set_device(a.device);torch._dynamo.config.cache_size_limit=64
    out=Path(a.output).resolve();out.mkdir(parents=True,exist_ok=True);start=time.perf_counter()
    defaults=arguments(['--model',a.model,'--device',a.device]+[x for path in (a.script or []) for x in ('--script',path)]);refs,scripts,jobs=load_inputs(defaults)
    modes=[(n,a.candidates//n) for n in a.n_values]
    source={str(f.relative_to(HERE)):sha(f) for f in list(HERE.glob('*.py'))+list((HERE/'runtime').glob('*.py'))}
    Processor=get_class_from_dynamic_module('processing_moss_tts.MossTTSLocalProcessor',a.model)
    proc=Processor.from_pretrained(a.model,codec_path=os.environ.get('PODCAST_CODEC','OpenMOSS-Team/MOSS-Audio-Tokenizer-v2'),codec_weight_dtype='bf16',codec_compute_dtype='bf16')
    anchors={}
    for name,r in refs.items():
        if not r.get('codes'):raise ValueError('This memory-constrained comparison requires prepared reference codes')
        with np.load(HERE/r['codes']) as data:anchors[name]=torch.from_numpy(data[r['code_key']].astype(np.int64))
    from types import SimpleNamespace
    old=torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16);engine=TokenEngine(Path(a.model),proc,anchors)
    finally:torch.set_default_dtype(old)
    engine.fast.semantic.limit=1;engine.fast.graph_limit=1
    g=SimpleNamespace(device=a.device,engine=engine,proc=proc,codec_streams={});offline=OfflineBatch(g)
    policy_load=time.perf_counter()-start;all_batches=[]
    for n,groups in modes:
        folder=out/f'n{n}'/'generation'
        if (folder/'configuration.json').exists():raise ValueError('Phased runs require a fresh output directory; saved candidates must not be mixed with a new run')
        folder.mkdir(parents=True,exist_ok=True);(folder/'codes').mkdir(exist_ok=True)
        dump(folder/'process-start.json',dict(created_unix=time.time()))
        config=dict(model=MODEL_REPO,revision=MODEL_REVISION,general=GENERAL,references=refs,
            reference_codes_sha256={s:sha(HERE/r['codes']) for s,r in refs.items()},scripts={str(f):sha(f) for f in scripts},
            effective_jobs=[[lang,t] for lang,t in jobs],n=n,mode='offline-turn',temperature=1.,cfg=2.5,
            reward=FORMULA,speaker_model=ECAPA_REPO,speaker_revision=ECAPA_REVISION,wer_role='final audit only; no multiplier or eligibility filter',
            voice_conversion=False,prompt_style='spontaneous',device=a.device,requested_groups=groups,
            archive_mode='bounded-cpu-thread',execution='phased policy then codec/judges',single_take_online_scoring=False,
            optimization_flags={k:v for k,v in os.environ.items() if k.startswith('HVS_') and not k.endswith('CACHE')})
        dump(folder/'configuration.json',config);dump(folder/'source-files.json',source)
        # CPU snapshots avoid doubling the GPU KV cache during correctness checks.
        offline.check_prefill(jobs[:groups],n)
        shared_ok=offline.shared
        offline.shared=False
        _,expected,_=offline.synthesize(jobs[:1],n=n,max_frames=39,chunk_frames=13,seed=81108,decode_audio=False)
        offline.shared=shared_ok
        _,actual,_=offline.synthesize(jobs[:1],n=n,max_frames=39,chunk_frames=13,seed=81108,decode_audio=False)
        same=len(expected)==len(actual) and all(torch.equal(x,y) for x,y in zip(expected,actual));assert same
        dump(folder/'token-regression.json',dict(shared_and_full_prefill_codes_equal=same,n=n,frames=39))
        dump(folder/'prefill-checks.json',dict(enabled=offline.shared,checks=offline.prefill_checks))
        dump(folder/'speed-pilot.json',dict(selected_groups=groups,n=n,probes=[],note=f'Explicit {a.candidates}-candidate batches; phased memory fallback on a shared GPU.'))
        for lang in dict.fromkeys(lang for lang,_ in jobs):
            pending=[job for job in jobs if job[0]==lang]
            for offset in range(0,len(pending),groups):
                batch=pending[offset:offset+groups];seed=202611080+LANGUAGE_SEEDS[lang]+batch[0][1]['index']*31
                _,codes,perf=offline.synthesize(batch,n=n,chunk_frames=26,seed=seed,decode_audio=False)
                key=f'{lang}-{batch[0][1]["index"]:02d}'
                path=folder/'codes'/f'{key}.safetensors';save_file({str(i):c for i,c in enumerate(codes)},str(path))
                row=dict(n=n,groups=groups,language=lang,jobs=batch,codes=str(path),token_performance=perf)
                dump(folder/'codes'/f'{key}.json',row);all_batches.append(row)
                print('TOKENS',n,key,round(perf['wall_seconds'],2),'s',flush=True)
    # Preserve CPU codec; release all policy/graph references before loading judges.
    pad=engine.model.config.audio_pad_token_id
    engine.close();del offline,g,engine;gc.collect();torch.cuda.empty_cache()
    token_done=time.perf_counter()
    proc.audio_tokenizer.decoder.to(a.device);proc.audio_tokenizer.quantizer.to(a.device)
    scorer=SpeakerScorer(a.device,refs)
    decoder_load=time.perf_counter()-token_done;codec=proc.audio_tokenizer
    # Real candidate tokens: unfused DecoderOnly versus its optional fused path.
    example=load_file(all_batches[0]['codes']);test=torch.full((4,26,12),pad,device=a.device,dtype=torch.long)
    for i in range(4):
        c=example[str(i)][:26];test[i,:len(c)].copy_(c)
    # StreamingCodec probes the first parameter's device, so use DecoderOnly as
    # the unfused native arithmetic and compare the optional fused path exactly.
    def decode_test():
        with torch.inference_mode(),torch.cuda.device(a.device),DecoderOnly(codec,4,pad) as d:return d.decode(test)[0].clone()
    native=decode_test();import codec_fused;codec_fused.install(codec);fused=decode_test()
    exact=torch.equal(native,fused)
    if not exact:codec_fused.uninstall(codec)
    del native,fused,test
    decoders={};batches={n:[] for n,_ in modes};archive_times={n:[] for n,_ in modes};waits={n:0. for n,_ in modes};future=None;previous_n=None
    pool=ThreadPoolExecutor(max_workers=1)
    try:
        for row in all_batches:
            n=row['n'];folder=out/f'n{n}'/'generation';batch=row['jobs'];raw=load_file(row['codes']);codes=[raw[str(i)] for i in range(len(raw))];count=len(codes)
            tick=time.perf_counter();pieces=[[] for _ in codes]
            if count not in decoders:decoders[count]=DecoderGraph(codec,count,pad,26)
            length=max(map(len,codes));packed=torch.full((count,length,12),pad,device=a.device,dtype=torch.long)
            for i,c in enumerate(codes):packed[i,:len(c)].copy_(c)
            with torch.inference_mode(),torch.cuda.device(a.device),decoders[count] as d:
                for part in packed.split(26,dim=1):
                    waves,sizes=d.decode(part);waves=waves.cpu().numpy()
                    for i,size in enumerate(sizes.cpu().tolist()):
                        if size:pieces[i].append(waves[i,:size].copy())
            waves=[np.concatenate(x) for x in pieces];decode_s=time.perf_counter()-tick
            if n==1:
                scores=[dict(reward=None,ranking_skipped=True,reference_speaker=t['speaker']) for _,t in batch]
                timing=dict(scoring_seconds=0.,ears_seconds=0.,speaker_seconds=0.)
            else:scores,timing=scorer.score_speakers(waves,[t['speaker'] for _,t in batch for _ in range(n)])
            phase2=time.perf_counter()-tick;total=row['token_performance']['wall_seconds']+phase2
            perf=dict(row['token_performance']);perf.update(timing,codec_seconds=decode_s,batch_wall_seconds=total,wall_seconds=total/len(batch),
                deferred_decode_seconds=phase2,execution='phased',codec_fused_bit_identical=exact)
            batches[n].append(dict(language=row['language'],turns=[t['index'] for _,t in batch],**perf))
            dump(folder/'production-batches.json',batches[n])
            if future is not None:
                wait=time.perf_counter();archive_times[previous_n].append(future.result());waits[previous_n]+=time.perf_counter()-wait
            future=pool.submit(archive_batch,folder,batch,waves,codes,scores,perf,refs,n,total);previous_n=n
        if future is not None:
            wait=time.perf_counter();archive_times[previous_n].append(future.result());waits[previous_n]+=time.perf_counter()-wait
    finally:pool.shutdown(wait=True)
    for n,groups in modes:
        folder=out/f'n{n}'/'generation';target=torch.stack(list(scorer.targets.values()))
        dump(folder/'reference-controls.json',dict(speakers=list(scorer.targets),cosine=(target@target.T).cpu().tolist(),dimensions=int(target.shape[-1]),seconds=scorer.reference_seconds))
        dump(folder/'codec-regression.json',dict(fused_bit_identical=exact,frames=26,batch=4,source='actual generated candidates'))
        dump(folder/'archive-timing.json',dict(batches=archive_times[n],blocked_seconds=waits[n],cpu_seconds=sum(x['seconds'] for x in archive_times[n])))
        dump(folder/'generation-complete.json',dict(turns=len(jobs),n=n,groups=groups,
            production_seconds=sum(x['batch_wall_seconds'] for x in batches[n]),finished_unix=time.time(),load_seconds=None,
            note='Sum of token + decoding + ranking batch times; shared loads/correctness tests/wait between phases are in parent receipt, not per-mode end-to-end latency.'))
    decoders.clear();del scorer,proc;gc.collect();torch.cuda.empty_cache()
    dump(out/'generation-complete.json',dict(n=a.n_values,policy_load_seconds=policy_load,decoder_judge_load_seconds=decoder_load,
        generation_job_seconds=time.perf_counter()-start,phased=True,concurrent_unrelated_training=True,
        note='Policy and codec/judges do not occupy VRAM together. Policy batch size is recorded per mode. Not hardware-load matched to prior N4.'))
    print('PHASED GENERATION COMPLETE',flush=True)
if __name__=='__main__':main()
