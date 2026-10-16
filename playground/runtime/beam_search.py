"""Causal reward-guided chunk search with persistent global/codec KV caches.

Winner-of-N: keep one prefix and fork it N ways at each chunk.
Top-2: retain up to two global winners, split N children (extra to the best).
Only common acoustic
token prefixes are irreversible streaming output; unrelated beams are not spliced.
"""
import math
import time

import numpy as np
import torch

from performance_core import combine_branches, StreamingCodec


def expansion_indices(winners, n):
    if not winners or n < len(winners):
        raise ValueError('Candidate budget must cover retained beam count')
    quotient, remainder = divmod(n, len(winners))
    return [index for position,index in enumerate(winners)
            for _ in range(quotient + (position < remainder))]


def common_prefix(codes, rows):
    if len(rows) == 1:
        return int(codes[rows[0], :, 0].ne(-1).sum())
    same = codes[rows].eq(codes[rows[0]].unsqueeze(0)).all(-1).all(0)
    same &= codes[rows[0], :, 0].ne(-1)
    differing = (~same).nonzero()
    return int(differing[0]) if len(differing) else codes.shape[1]


def reorder_codec(codec, indexes, n):
    """Pinned codec state has per-row tensors, all with batch axis zero.

    Copy via index_select, including duplicates; never share mutable child KV.
    Fail closed if a new codec revision adds an unsupported state shape.
    """
    for module in codec.modules():
        state = getattr(module, '_streaming_state', None)
        if state is None:
            continue
        for name, value in vars(state).items():
            if torch.is_tensor(value):
                if value.ndim == 0 or value.shape[0] != n:
                    raise ValueError(f'Unsupported codec state tensor {name}: {tuple(value.shape)}')
                setattr(state, name, value.index_select(0, indexes))


class ChunkSearch:
    def __init__(self, engine, scorer):
        self.engine, self.scorer = engine, scorer

    @torch.inference_mode()
    def generate(self, spec, n=8, beam_width=1, chunk_frames=10, max_seconds=6.6,
                 guidance=1.5, temperature=1., seed=777, reference_path=None, on_commit=None, conversion=None):
        if not 1 <= n <= 16 or beam_width not in [1,2]:
            raise ValueError('Search supports batch sizes 1 through 16 and beam width 1/2')
        requested_beam_width = beam_width
        beam_width = min(n, beam_width)
        if chunk_frames not in [10,20]:
            raise ValueError('Search chunk must be 800 or 1600 ms')
        engine, mdl = self.engine, self.engine.model
        cfg = mdl.config
        budget = min(750, math.ceil(max_seconds/.08))
        p = {'max_new_tokens':budget, 'audio_temperature':temperature, 'audio_top_p':.95,
             'audio_top_k':50, 'audio_repetition_penalty':1., 'text_temperature':0. if temperature==0 else 1.,
             'text_top_p':1., 'text_top_k':50}
        timeline, committed, first_commit = [], 0, None
        stopped_by_alignment=False
        committed_samples=0
        vc_states=[conversion.new_stream(i) for i in range(n)] if conversion else None
        started = time.perf_counter()
        with engine.lock, torch.cuda.device(engine.device), torch.autocast('cuda',dtype=torch.bfloat16):
            reference = engine.reference(reference_path) if reference_path else None
            branches = [engine.pack(spec['prompt'],spec,reference,n)]
            if guidance != 1 and spec['prompt'] != spec['unconditional']:
                branches.append(engine.pack(spec['unconditional'],spec,reference,n))
            ids, initial = combine_branches(branches,cfg.pad_token_id,cfg.audio_pad_token_id)
            nb, length = len(branches), ids.shape[1]
            masks = torch.zeros(n*nb,length+budget,dtype=torch.bool,device=engine.device)
            masks[:,:length] = initial
            past, model_input, steps = None, ids, 0
            codes = torch.empty(n,0,12,dtype=torch.long,device=engine.device)
            finished = torch.zeros(n,dtype=torch.bool,device=engine.device)
            pcm = [np.empty(0,dtype=np.float32) for _ in range(n)]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            capture_start = engine.fast.capture_seconds
            decoder = engine.proc.audio_tokenizer
            with StreamingCodec(decoder,n,cfg.audio_pad_token_id) as streaming:
                while steps < budget:
                    phase_start = time.perf_counter()
                    event_start,event_end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
                    event_start.record()
                    new = []
                    new_pcm=[np.zeros(0,np.float32) for _ in range(n)]
                    for _ in range(min(chunk_frames,budget-steps)):
                        out = mdl.transformer(inputs_embeds=mdl._build_inputs_embeds(model_input),input_ids=None,
                            attention_mask=masks[:,:length+steps],past_key_values=past,use_cache=True,
                            position_ids=None,output_attentions=False,output_hidden_states=False,
                            return_dict=True,cu_seqlens=None,num_sequences=None)
                        past = out.past_key_values
                        local = mdl._global_hidden_to_local(out.last_hidden_state[:,-1]).to(mdl.local_transformer.ln_f.weight.dtype)
                        stop,sampled = engine.fast.frame(local,n,nb,guidance,p,graph=True)
                        continues = stop.eq(cfg.audio_assistant_slot_token_id) & ~finished
                        finished |= stop.eq(cfg.audio_end_token_id)
                        if not bool(continues.any()):
                            break
                        frame = sampled.masked_fill(~continues.unsqueeze(-1),cfg.audio_pad_token_id)
                        new.append(frame)
                        row = mdl._build_generation_row(n,ids.device,frame)
                        row[:,0,0] = torch.where(continues,row[:,0,0],cfg.pad_token_id)
                        model_input = row.repeat(nb,1,1)
                        masks[:,length+steps] = continues.repeat(nb)
                        steps += 1
                    if new:
                        chunk = torch.stack(new,dim=1)
                        codes = torch.cat((codes,chunk),dim=1)
                        wave,lengths = streaming.decode(chunk)
                        for i in range(n):
                            new_pcm[i]=wave[i,:int(lengths[i])].float().cpu().numpy().copy()
                            if conversion is None:pcm[i] = np.concatenate((pcm[i],new_pcm[i]))
                    event_end.record()
                    event_end.synchronize()
                    generation_seconds = time.perf_counter()-phase_start
                    endpoint = steps>=budget or bool(finished.all()) or not new
                    conversion_start=time.perf_counter()
                    if conversion:
                        finals=[endpoint or bool(finished[i]) for i in range(n)]
                        converted_batch=conversion.chunk_many(new_pcm,vc_states,finals) if hasattr(conversion,'chunk_many') else [
                            conversion.chunk(new_pcm[i],vc_states[i],final=finals[i]) for i in range(n)]
                        for i,converted in enumerate(converted_batch):
                            pcm[i]=np.concatenate((pcm[i],converted))
                    conversion_seconds=time.perf_counter()-conversion_start if conversion else 0.
                    # Scoring exits cuda:0 autocast so cuda:1's FP32 quality
                    # tower/head precision remains unchanged.
                    with torch.autocast('cuda', enabled=False):
                        scores = self.scorer.score(pcm,spec['text'],[endpoint or bool(x) for x in finished.cpu()])
                    order = sorted(range(n),key=lambda i:(not scores[i].get('ranking_available',True),-scores[i]['reward'],scores[i]['prefix_wer'],i))
                    winners = order[:beam_width] if not endpoint else order[:1]
                    safe = common_prefix(codes,winners)
                    if safe < committed:
                        raise RuntimeError('Search attempted to revise committed acoustic prefix')
                    safe_samples=safe*3840
                    if conversion:
                        from voice_conversion import common_pcm_prefix
                        safe_samples=common_pcm_prefix(pcm,winners,safe_samples)
                    if safe_samples<committed_samples:
                        raise RuntimeError('Search attempted to revise committed converted audio')
                    committed_seconds = (safe_samples-committed_samples)/48000
                    entry = {'step':len(timeline)+1,'frame':steps,'retained_rows':winners,
                        'committed_audio_seconds':committed_seconds,'committed_until_seconds':safe_samples/48000,
                        'voice_conversion_seconds':conversion_seconds,
                        'generation_wall_seconds':generation_seconds,'tts_codec_device_seconds':event_start.elapsed_time(event_end)/1000,
                        'scores':scores, **self.scorer.last_timing}
                    if safe_samples > committed_samples:
                        if first_commit is None:
                            first_commit = time.perf_counter()-started
                        segment = pcm[winners[0]][committed_samples:safe_samples].copy()
                        if on_commit:
                            stopped_by_alignment=bool(on_commit(segment,entry))
                        committed_samples=safe_samples
                    committed = safe
                    timeline.append(entry)
                    if endpoint or stopped_by_alignment:
                        winner = winners[0]
                        break
                    indexes = torch.tensor(expansion_indices(winners,n),device=engine.device)
                    branch_indexes = torch.cat([indexes+branch*n for branch in range(nb)])
                    past = tuple(tuple(value.index_select(0,branch_indexes) for value in layer) for layer in past)
                    masks = masks.index_select(0,branch_indexes)
                    model_input = model_input.index_select(0,branch_indexes)
                    codes = codes.index_select(0,indexes)
                    finished = finished.index_select(0,indexes)
                    pcm = [pcm[i].copy() for i in indexes.cpu().tolist()]
                    if conversion:
                        parents=indexes.cpu().tolist()
                        vc_states=[vc_states[parent].fork(seed+104729*(child+1)+10007*steps)
                                   for child,parent in enumerate(parents)]
                    reorder_codec(decoder,indexes,n)
                else:
                    winner = 0
            output = pcm[winner]
            if conversion and not vc_states[winner].done:
                output=np.concatenate((output,conversion.chunk(np.zeros(0,np.float32),vc_states[winner],final=True)))
            engine.last_codes = [codes[winner][codes[winner,:,0].ne(cfg.audio_pad_token_id)].cpu()]
            capture_seconds = engine.fast.capture_seconds-capture_start
        total = time.perf_counter()-started
        seconds = len(output)/48000
        if not seconds:
            raise RuntimeError('Search generated no audio; try another seed/reference')
        performance = {'n':n,'beam_width':beam_width,'requested_beam_width':requested_beam_width,
            'chunk_ms':chunk_frames*80,'max_seconds':max_seconds,
            'actual_seconds':seconds,'search_wall_seconds':total,'search_rtf':total/seconds,
            'first_commit_seconds':first_commit,'tts_codec_seconds':sum(x['tts_codec_device_seconds'] for x in timeline),
            'scoring_seconds':sum(x['total_seconds'] for x in timeline),'graph_capture_seconds':capture_seconds,
            'voice_conversion_seconds':sum(x['voice_conversion_seconds'] for x in timeline),
            'streaming_rule':'Winner commits immediately; top-2 emits only common acoustic-token prefixes',
            'includes':'reference encoding, packing, global/local TTS, codec, GPU transfers, batched ASR/timestamps, shared VoiceCLAP, heads, selected Humaneness Ears/VoiceNet/EIV+, burst locator/classifier if configured, CPU selection and KV cloning',
            'excludes':'model loading, one-time style text-query embedding, Luna, MP3 encoding, final optional alignment/evaluation, network/playback',
            'device':str(engine.device),'scorer_device':str(self.scorer.device),
            'timeline':timeline,'final_search_score':scores[winner],
            'stopped_by_alignment':stopped_by_alignment,
            'hit_frame_limit':steps>=budget and not bool(finished[winner])}
        if conversion:
            performance['streaming_rule']='Only identical converted PCM shared by retained beams is committed; VC states fork with TTS caches'
            performance['includes']+=', Chatterbox conversion before all candidate rewards'
        return output,performance
