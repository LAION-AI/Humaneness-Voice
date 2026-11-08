"""Complete-turn batching across independent podcast segments, without streaming.

Layout is [all conditional groups, all neutral groups], each group has N takes.
Prompt prefill computes each identical group once, then copies its KV state.
No voice, generated history, or reward is shared between different segments.
"""
import time
from contextlib import nullcontext
import numpy as np
import torch
from recipe import specification, SR
from performance_core import combine_branches
from generation import prefill_bounded, DecoderGraph

def prefill_repeated(bank,ids,mask,n,microbatch=2):
    if len(ids)%n:raise ValueError('Incomplete candidate group')
    length=ids.shape[1]
    # Never reuse KV for distinct inputs or padding patterns.
    for a in range(0,len(ids),n):
        if not torch.equal(ids[a:a+n],ids[a:a+1].expand_as(ids[a:a+n])) or not torch.equal(mask[a:a+n],mask[a:a+1].expand_as(mask[a:a+n])):
            raise ValueError('Shared prefill requires identical candidate prompts')
    bank.allowed.zero_();bank.allowed[:,:length].copy_(mask);bank.position.fill_(length)
    local=[]
    for a in range(0,len(ids)//n,microbatch):
        unique=torch.arange(a,min(a+microbatch,len(ids)//n),device=ids.device)*n
        out=bank.forward(inputs_embeds=bank.model._build_inputs_embeds(ids[unique]),
            input_ids=None,attention_mask=mask[unique],past_key_values=None,use_cache=True,return_dict=True)
        if not bank.cache:
            for k,v in out.past_key_values:
                shape=(bank.nb,bank.capacity,*k.shape[2:]);bank.cache.append((k.new_zeros(shape),v.new_zeros(shape)))
        for (k,v),(stored_k,stored_v) in zip(out.past_key_values,bank.cache):
            for j,idx in enumerate(unique.tolist()):
                stored_k[idx:idx+n,:length].copy_(k[j:j+1].expand(n,*k.shape[1:]))
                stored_v[idx:idx+n,:length].copy_(v[j:j+1].expand(n,*v.shape[1:]))
        local.append(bank.model._global_hidden_to_local(out.last_hidden_state[:,-1]).to(
            bank.model.local_transformer.ln_f.weight.dtype).repeat_interleave(n,dim=0))
    return torch.cat(local)

class OfflineBatch:
    def __init__(self,generator):self.g=generator;self.shared=True;self.prefill_checks=[]

    def pack(self,jobs,n,guidance):
        e=self.g.engine;branches=[]
        for kind in (['prompt','unconditional'] if guidance!=1 else ['prompt']):
            for language,turn in jobs:
                spec=specification(turn,language)
                branches.append(e.pack(spec[kind],spec,e.refs[turn['speaker']],n))
        return combine_branches(branches,e.model.config.pad_token_id,e.model.config.audio_pad_token_id)

    @torch.inference_mode()
    def check_prefill(self,jobs,n=4):
        e=self.g.engine
        with torch.cuda.device(self.g.device),torch.autocast('cuda',dtype=torch.bfloat16):
            ids,mask=self.pack(jobs,n,2.5);bank=e.fast.semantic.prepare(ids,320)
            torch.cuda.synchronize();tick=time.perf_counter()
            expected=prefill_bounded(bank,ids,mask).detach().cpu();torch.cuda.synchronize()
            baseline_s=time.perf_counter()-tick
            cache=[tuple(t[:,:len(mask[0])].detach().cpu() for t in pair) for pair in bank.cache]
            tick=time.perf_counter();actual=prefill_repeated(bank,ids,mask,n).detach().cpu();torch.cuda.synchronize()
            optimized_s=time.perf_counter()-tick
            same_cache=all(torch.equal(t[:,:ids.shape[1]].cpu(),wanted) for pair,old in zip(bank.cache,cache) for t,wanted in zip(pair,old))
            same_local=torch.equal(expected,actual)
            result=dict(groups=len(jobs),n=n,baseline_seconds=baseline_s,shared_seconds=optimized_s,
                same_cache=same_cache,same_local=same_local,max_abs_local_error=float((expected-actual).abs().max()))
            self.shared=self.shared and same_cache and same_local
            self.prefill_checks.append(result)
            del cache,expected,actual
            print('SHARED PREFILL CHECK',result,'enabled',self.shared,flush=True)
            return result

    @torch.inference_mode()
    def synthesize(self,jobs,n=4,chunk_frames=26,max_frames=320,seed=71108,temperature=1.,guidance=2.5,decode_audio=True):
        count=len(jobs);total=count*n
        if not 1<=total<=16:raise ValueError('Total simultaneous candidates must be 1..16')
        g=self.g;e=g.engine;model=e.model;cfg=model.config
        start=time.perf_counter();token_s=codec_s=0.
        p=dict(max_new_tokens=max_frames,audio_temperature=temperature,audio_top_p=.95,
            audio_top_k=50,audio_repetition_penalty=1.,text_temperature=1.,text_top_p=1.,text_top_k=50)
        with torch.cuda.device(g.device),torch.autocast('cuda',dtype=torch.bfloat16):
            ids,initial=self.pack(jobs,n,guidance);length=ids.shape[1];nb=2 if guidance!=1 else 1
            masks=torch.zeros(total*nb,length+max_frames,dtype=torch.bool,device=g.device)
            masks[:,:length]=initial;bank=e.fast.semantic.prepare(ids,max_frames)
            model_input=ids;finished=torch.zeros(total,dtype=torch.bool,device=g.device)
            code_parts=[];parts=[[] for _ in range(total)];steps=0
            key=(total,chunk_frames)
            if decode_audio and key not in g.codec_streams:g.codec_streams[key]=DecoderGraph(g.proc.audio_tokenizer,total,cfg.audio_pad_token_id,chunk_frames)
            capture_before=e.fast.capture_seconds
            torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
            torch.cuda.reset_peak_memory_stats(g.device)
            with (g.codec_streams[key] if decode_audio else nullcontext()) as decoder:
                while steps<max_frames:
                    tick=time.perf_counter();new=[]
                    for _ in range(min(chunk_frames,max_frames-steps)):
                        mask=masks[:,:length+steps]
                        if steps==0:
                            local=prefill_repeated(bank,model_input,mask,n) if self.shared else prefill_bounded(bank,model_input,mask)
                        else:local=bank.decode(model_input,mask)
                        token,sampled=e.fast.frame(local,total,nb,guidance,p,graph=True)
                        active=token.eq(cfg.audio_assistant_slot_token_id)&~finished
                        finished|=token.eq(cfg.audio_end_token_id)
                        if not bool(active.any()):break
                        frame=sampled.masked_fill(~active[:,None],cfg.audio_pad_token_id)
                        new.append(frame.clone())
                        row=model._build_generation_row(total,ids.device,frame)
                        row[:,0,0]=torch.where(active,row[:,0,0],cfg.pad_token_id)
                        model_input=row.repeat(nb,1,1);masks[:,length+steps]=active.repeat(nb);steps+=1
                    torch.cuda.synchronize(g.device);token_s+=time.perf_counter()-tick;tick=time.perf_counter()
                    if new:
                        codes=torch.stack(new,dim=1);code_parts.append(codes)
                        if decode_audio:
                            wave,sizes=decoder.decode(codes)
                            sizes=sizes.cpu().tolist();wave=wave.float().cpu().numpy()
                            for i,size in enumerate(sizes):
                                if size:parts[i].append(wave[i,:size].copy())
                    codec_s+=time.perf_counter()-tick
                    if not new or bool(finished.all()):break
            code_tensor=torch.cat(code_parts,dim=1)
            waves=[np.concatenate(p) if p else np.empty(0,np.float32) for p in parts]
            assert not decode_audio or all(len(w)>0 and np.isfinite(w).all() for w in waves)
            codes=[c[c[:,0]!=cfg.audio_pad_token_id].cpu().contiguous() for c in code_tensor]
            perf=dict(n=n,groups=count,total_candidates=total,wall_seconds=time.perf_counter()-start,
                token_seconds=token_s,codec_seconds=codec_s,seed=seed,steps=steps,hit_frame_cap=steps>=max_frames,
                deferred_decoding=not decode_audio,shared_prefill=self.shared,chunk_frames=chunk_frames,temperature=temperature,cfg=guidance,
                graph_capture_seconds=e.fast.capture_seconds-capture_before,
                max_memory_allocated_bytes=torch.cuda.max_memory_allocated(g.device),
                max_memory_reserved_bytes=torch.cuda.max_memory_reserved(g.device))
            return waves,codes,perf
