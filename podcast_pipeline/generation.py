"""Large Podcast model: batched turns and genuine causal winner-of-N search.

All branches share weights, never mutable KV state. The selected semantic KV
is copied in place to preserve CUDA graph addresses; codec state forks too.
"""
import os
import gc
import json
import time
from types import MethodType
from pathlib import Path
import numpy as np
import torch
from recipe import SR, specification, chunk_choice
from performance_core import combine_branches, StreamingCodec, _stable_cache_update

class DecoderOnly:
    """Streaming codec without allocating unused encoder states on the GPU.

    The decoder is identical to upstream _decode_frame. Avoid its device probe:
    after encoder offload, next(codec.parameters()) points to CPU.
    """
    def __init__(self, codec, n, pad=-1):
        self.codec, self.n, self.pad = codec, n, pad
        self.modules = [m for m in codec.decoder.modules() if hasattr(m, '_init_streaming_state')]
    def __enter__(self):
        for module in self.modules:
            assert module._streaming_state is None
            module._streaming_state = module._init_streaming_state(self.n)
        return self
    def __exit__(self, *args):
        for module in self.modules:
            module._streaming_state = None
    def decode(self, codes):
        valid = codes[:,:,0].ne(self.pad)
        lengths = valid.sum(-1)
        codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
        for module in self.modules:
            module._streaming_state.set_exec_mask(lengths > 0)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            audio = self.codec.quantizer.decode_codes(codes.permute(2,0,1).contiguous()).float()
            for module in self.codec.decoder:
                audio, lengths = module(audio, lengths)
            audio, lengths = self.codec._restore_channels_from_codec(audio, lengths)
        return audio.float().mean(1), lengths

class DecoderGraph(DecoderOnly):
    """One persistent graph for the common chunk shape, including N>1.

    Short terminal blocks use the same state eagerly. Unlike the demo's batch-1
    bank, this captures only one useful shape instead of all twenty shapes.
    """
    def __init__(self, codec, n, pad=-1, frames=13):
        super().__init__(codec,n,pad)
        self.frames=frames
        self.states=[m._init_streaming_state(n) for m in self.modules]
        self.graph=None
        self.capture_seconds=0.
    def reset(self):
        for state in self.states:state.reset(torch.ones_like(state.exec_mask))
    def __enter__(self):
        for module,state in zip(self.modules,self.states):
            assert module._streaming_state is None
            module._streaming_state=state
            if hasattr(type(module),'_update_streaming_cache'):
                module._update_streaming_cache=MethodType(_stable_cache_update,module)
        if self.graph is None:
            tick=time.perf_counter()
            device=next(self.codec.decoder.parameters()).device
            self.static=torch.zeros(self.n,self.frames,12,dtype=torch.long,device=device)
            stream=torch.cuda.Stream(device=device)
            stream.wait_stream(torch.cuda.current_stream(device))
            with torch.cuda.stream(stream):
                for _ in range(2):super().decode(self.static)
            torch.cuda.current_stream(device).wait_stream(stream)
            self.reset()
            self.graph=torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph,stream=stream):self.output=super().decode(self.static)
            torch.cuda.current_stream(device).wait_stream(stream)
            torch.cuda.synchronize(device)
            self.capture_seconds=time.perf_counter()-tick
        self.reset()
        return self
    def __exit__(self,*args):
        for module in self.modules:
            module._streaming_state=None
            module.__dict__.pop('_update_streaming_cache',None)
    def decode(self,codes):
        if codes.shape[1]!=self.frames:return super().decode(codes)
        self.static.copy_(codes)
        self.graph.replay()
        return self.output[0].clone(),self.output[1].clone()

def fork_codec(codec,indexes,n):
    """Preserve tensor addresses used by the decoder graph while cloning rows."""
    for module in codec.decoder.modules():
        state=getattr(module,'_streaming_state',None)
        if state is None:continue
        for name,value in vars(state).items():
            if torch.is_tensor(value):
                if value.ndim==0 or value.shape[0]!=n:
                    raise ValueError(f'Unknown codec state shape: {name} {tuple(value.shape)}')
                value.copy_(value.index_select(0,indexes))

def prefill_bounded(bank, ids, mask, microbatch=2):
    """Bound transient MLP activations; the same batched KV layout is retained."""
    length = ids.shape[1]
    bank.allowed.zero_()
    bank.allowed[:, :length].copy_(mask)
    bank.position.fill_(length)
    local = []
    for start in range(0, len(ids), microbatch):
        stop = min(len(ids), start+microbatch)
        out = bank.forward(inputs_embeds=bank.model._build_inputs_embeds(ids[start:stop]),
            input_ids=None, attention_mask=mask[start:stop], past_key_values=None,
            use_cache=True, return_dict=True)
        if not bank.cache:
            for key, value in out.past_key_values:
                shape = (bank.nb, bank.capacity, *key.shape[2:])
                bank.cache.append((key.new_zeros(shape), value.new_zeros(shape)))
        for (key,value),(stored_key,stored_value) in zip(out.past_key_values,bank.cache):
            stored_key[start:stop,:length].copy_(key)
            stored_value[start:stop,:length].copy_(value)
        local.append(bank.model._global_hidden_to_local(out.last_hidden_state[:,-1]).to(
            bank.model.local_transformer.ln_f.weight.dtype))
        del out
    return torch.cat(local)

def fork_semantic(bank, winner, n, branches, used):
    indexes = torch.tensor([winner+b*n for b in range(branches) for _ in range(n)],
                           device=bank.device)
    # Only past positions have meaning. Retain allocations/addresses held by graphs.
    for pair in bank.cache:
        for value in pair:
            value[:, :used].copy_(value[:, :used].index_select(0, indexes))
    for value in (bank.allowed, bank.ids, bank.query_valid):
        value.copy_(value.index_select(0, indexes))
    return indexes

class Generator:
    def __init__(self, model_root, reference_config, device, scorer=None):
        from transformers.dynamic_module_utils import get_class_from_dynamic_module
        from phased_generate import TokenEngine
        from recipe import HERE, references
        import soundfile as sf
        self.device, self.scorer = device, scorer
        root = Path(model_root)
        Processor = get_class_from_dynamic_module('processing_moss_tts.MossTTSLocalProcessor', str(root))
        self.proc = Processor.from_pretrained(str(root), codec_path=os.environ.get('PODCAST_CODEC','OpenMOSS-Team/MOSS-Audio-Tokenizer-v2'),
                    codec_weight_dtype='bf16', codec_compute_dtype='bf16')
        self.proc.audio_tokenizer.to(device).eval()
        refs = {}
        paths = references(reference_config)
        for name, item in reference_config.items():
            if item.get('codes'):
                with np.load(HERE/item['codes']) as source:
                    refs[name] = torch.from_numpy(source[item['code_key']].astype(np.int64))
            else:
                wave, sr = sf.read(paths[name], dtype='float32', always_2d=True)
                with torch.cuda.device(device), torch.autocast('cuda', dtype=torch.bfloat16):
                    refs[name] = self.proc.encode_audios_from_wav(torch.from_numpy(wave.mean(1)).unsqueeze(0), sr, n_vq=12)[0].cpu()
            assert refs[name].ndim == 2 and refs[name].shape[1] == 12 and 63 <= len(refs[name]) <= 250
        # Prove decoder-only residency against the upstream streaming primitive
        # before enabling it; this also checks heterogeneous padded batch rows.
        with torch.inference_mode(), torch.cuda.device(device):
            audio_pad=int(json.loads((root/'config.json').read_text())['audio_pad_token_id'])
            test_codes=next(iter(refs.values()))[:26].to(device).unsqueeze(0).repeat(4,1,1)
            test_codes[2,13:]=audio_pad
            test_codes[3,21:]=audio_pad
            def decode_test(factory):
                chunks=[];lengths=[]
                with factory(self.proc.audio_tokenizer,4,audio_pad) as stream:
                    for part in test_codes.split(13,dim=1):
                        audio,size=stream.decode(part)
                        chunks.append(audio.clone());lengths.append(size.clone())
                torch.cuda.synchronize(device)
                return chunks,lengths
            native,native_sizes=decode_test(StreamingCodec)
        # Reference encoding has finished. The large encoder is never called
        # by synthesis, so keep it in CPU RAM rather than occupying VRAM.
        self.proc.audio_tokenizer.encoder.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
        with torch.inference_mode(),torch.cuda.device(device):
            direct,direct_sizes=decode_test(DecoderOnly)
            assert all(torch.equal(a,b) for a,b in zip(native,direct))
            assert all(torch.equal(a,b) for a,b in zip(native_sizes,direct_sizes))
            import codec_fused
            codec_fused.install(self.proc.audio_tokenizer)
            fused,fused_sizes=decode_test(DecoderOnly)
            exact=all(torch.equal(a,b) for a,b in zip(native,fused)) and all(torch.equal(a,b) for a,b in zip(native_sizes,fused_sizes))
            if not exact:
                codec_fused.uninstall(self.proc.audio_tokenizer)
            self.codec_validation=dict(decoder_only_bit_identical=True,fused_bit_identical=exact,
                fused_enabled=exact,batch_size=4,frames=26,heterogeneous_lengths=True)
            graph_stream=DecoderGraph(self.proc.audio_tokenizer,4,audio_pad,13)
            graph_audio=[];graph_sizes=[]
            with graph_stream as stream:
                for part in test_codes.split(13,dim=1):
                    audio,size=stream.decode(part);graph_audio.append(audio);graph_sizes.append(size)
            self.codec_validation['graph_bit_identical']=all(torch.equal(a,b) for a,b in zip(native,graph_audio)) and all(torch.equal(a,b) for a,b in zip(native_sizes,graph_sizes))
            assert self.codec_validation['graph_bit_identical'], 'Batched codec graph changed PCM'
            self.codec_streams={(4,13):graph_stream}
            del native,native_sizes,direct,direct_sizes,fused,fused_sizes,test_codes
            print('CODEC CHECK',self.codec_validation,flush=True)
        # Avoid redundant FP32 allocations in the model constructor.
        previous = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            self.engine = TokenEngine(root, self.proc, refs)
        finally:
            torch.set_default_dtype(previous)
        assert self.engine.device == device
        self.engine.fast.graph_limit = 2
        gc.collect()
        torch.cuda.empty_cache()
        print('RESIDENT GPU GB',round(torch.cuda.memory_allocated(device)/1e9,3),
              'policy GB',round(sum(p.numel()*p.element_size() for p in self.engine.model.parameters())/1e9,3),
              'codec GPU GB',round(sum(p.numel()*p.element_size() for p in self.proc.audio_tokenizer.parameters() if p.is_cuda)/1e9,3),flush=True)

    @torch.inference_mode()
    def run(self, turn, language, n=4, mode='turn', chunk_frames=13, max_frames=320,
            temperature=1., guidance=2.5, seed=8117, on_commit=None, force_winner=None):
        if not 1 <= n <= 16 or mode not in ('turn', 'chunk') or not 1 <= chunk_frames <= 40:
            raise ValueError('N must be 1..16; mode turn/chunk; chunk 1..40 codec frames')
        if self.scorer is None and n > 1 and force_winner is None:
            raise ValueError('N>1 needs a scorer')
        eng = self.engine
        model, cfg = eng.model, eng.model.config
        spec = specification(turn, language)
        p = dict(max_new_tokens=max_frames, audio_temperature=temperature, audio_top_p=.95,
                 audio_top_k=50, audio_repetition_penalty=1., text_temperature=1., text_top_p=1., text_top_k=50)
        started = time.perf_counter()
        with torch.cuda.device(self.device), torch.autocast('cuda', dtype=torch.bfloat16):
            branches = [eng.pack(spec['prompt'], spec, eng.refs[turn['speaker']], n)]
            if guidance != 1:
                branches.append(eng.pack(spec['unconditional'], spec, eng.refs[turn['speaker']], n))
            ids, initial = combine_branches(branches, cfg.pad_token_id, cfg.audio_pad_token_id)
            nb, length = len(branches), ids.shape[1]
            masks = torch.zeros(n*nb, length+max_frames, dtype=torch.bool, device=self.device)
            masks[:, :length] = initial
            bank = eng.fast.semantic.prepare(ids, max_frames)
            model_input = ids
            finished = torch.zeros(n, dtype=torch.bool, device=self.device)
            codes = torch.empty(n, 0, 12, dtype=torch.long, device=self.device)
            pcm = [np.empty(0, np.float32) for _ in range(n)]
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            steps, committed, first_commit, winner = 0, 0, None, 0
            previous_progress, stagnant = 0, 0
            timeline, final_scores = [], None
            generation_s, codec_s, fork_s = 0., 0., 0.
            capture_before = eng.fast.capture_seconds
            torch.cuda.reset_peak_memory_stats(self.device)
            decoder = self.proc.audio_tokenizer
            stream_key=(n,chunk_frames)
            if stream_key not in self.codec_streams:
                self.codec_streams[stream_key]=DecoderGraph(decoder,n,cfg.audio_pad_token_id,chunk_frames)
            assert self.codec_streams[stream_key].pad==cfg.audio_pad_token_id
            with self.codec_streams[stream_key] as streaming:
                while steps < max_frames:
                    tick = time.perf_counter()
                    new = []
                    for _ in range(min(chunk_frames, max_frames-steps)):
                        mask = masks[:, :length+steps]
                        local = prefill_bounded(bank, model_input, mask) if steps == 0 else bank.decode(model_input, mask)
                        token, sampled = eng.fast.frame(local, n, nb, guidance, p, graph=True)
                        continues = token.eq(cfg.audio_assistant_slot_token_id) & ~finished
                        finished |= token.eq(cfg.audio_end_token_id)
                        if not bool(continues.any()):
                            break
                        frame = sampled.masked_fill(~continues.unsqueeze(-1), cfg.audio_pad_token_id)
                        new.append(frame.clone())
                        row = model._build_generation_row(n, ids.device, frame)
                        row[:,0,0] = torch.where(continues, row[:,0,0], cfg.pad_token_id)
                        model_input = row.repeat(nb,1,1)
                        masks[:,length+steps] = continues.repeat(nb)
                        steps += 1
                    torch.cuda.synchronize(self.device)
                    elapsed = time.perf_counter()-tick
                    generation_s += elapsed
                    tick = time.perf_counter()
                    if new:
                        chunk = torch.stack(new, dim=1)
                        codes = torch.cat((codes, chunk), dim=1)
                        wave, sizes = streaming.decode(chunk)
                        for i in range(n):
                            part = wave[i,:int(sizes[i])].float().cpu().numpy().copy()
                            pcm[i] = np.concatenate((pcm[i], part))
                    codec_elapsed = time.perf_counter()-tick
                    codec_s += codec_elapsed
                    endpoint = steps >= max_frames or bool(finished.all()) or not new
                    entry = dict(frame=steps, generation_seconds=elapsed, codec_seconds=codec_elapsed)
                    choose = mode == 'chunk' and n > 1
                    if choose:
                        if force_winner is not None:
                            winner = force_winner
                            scores = []
                        else:
                            # Ears sees the cumulative prefix: 1s alone is too noisy.
                            with torch.autocast('cuda', enabled=False):
                                scores, timing = self.scorer.score(pcm, turn['text'],
                                    [endpoint or bool(f) for f in finished.cpu()])
                            entry.update(timing)
                            winner,complete,guarded=chunk_choice(scores,turn['text'],finished.cpu().tolist(),previous_progress,stagnant)
                            if guarded:
                                entry['progress_guard'] = True
                            progress = scores[winner]['matched_words']
                            stagnant = stagnant+1 if progress <= previous_progress else 0
                            previous_progress = max(previous_progress, progress)
                            if complete:
                                endpoint = True
                                entry['script_complete_stop'] = True
                        entry.update(winner=winner, scores=scores)
                        final_scores = scores
                    if n == 1 or choose:
                        if not len(pcm[winner]):
                            raise RuntimeError('Winner has no audio')
                        part = pcm[winner][committed:].copy()
                        if len(part):
                            if first_commit is None:
                                first_commit = time.perf_counter()-started
                            entry.update(commit_start_s=committed/SR,
                                commit_end_s=len(pcm[winner])/SR,
                                commit_wall_seconds=time.perf_counter()-started)
                            if on_commit:
                                on_commit(part, entry)
                            committed = len(pcm[winner])
                    timeline.append(entry)
                    if endpoint:
                        break
                    if choose:
                        tick = time.perf_counter()
                        branch_index = fork_semantic(bank, winner, n, nb, length+steps)
                        masks = masks.index_select(0, branch_index)
                        model_input = model_input.index_select(0, branch_index)
                        index = torch.full((n,), winner, dtype=torch.long, device=self.device)
                        codes = codes.index_select(0, index)
                        finished = finished.index_select(0, index)
                        pcm = [pcm[winner].copy() for _ in range(n)]
                        fork_codec(decoder,index,n)
                        torch.cuda.synchronize(self.device)
                        fork_s += time.perf_counter()-tick
                # N=1 has no ranking cost. Whole-turn selection waits for all N.
            if mode == 'turn' and n > 1:
                if force_winner is None:
                    with torch.autocast('cuda', enabled=False):
                        final_scores, timing = self.scorer.score(pcm, turn['text'])
                    timeline.append({'final_scoring': True, **timing, 'scores': final_scores})
                    eligible = [i for i,s in enumerate(final_scores) if s['wer'] <= .15] or list(range(n))
                    winner = max(eligible, key=lambda i: (final_scores[i]['reward'], -i))
                else:
                    winner = force_winner
                first_commit = time.perf_counter()-started
                if on_commit:
                    on_commit(pcm[winner], {'winner': winner, 'commit_start_s': 0.,
                        'commit_end_s': len(pcm[winner])/SR, 'commit_wall_seconds': first_commit})
            raw = pcm[winner]
            if not len(raw):
                raise RuntimeError('Empty output')
            selected_codes = codes[winner][codes[winner,:,0] != cfg.audio_pad_token_id].cpu()
            wall = time.perf_counter()-started
            perf = dict(n=n, mode=mode, chunk_frames=chunk_frames, chunk_seconds=chunk_frames*.08,
                seed=seed, temperature=temperature, cfg=guidance, device=self.device,
                wall_seconds=wall, raw_seconds=len(raw)/SR, raw_rtf=wall/(len(raw)/SR),
                first_commit_seconds=first_commit, token_seconds=generation_s, codec_seconds=codec_s,
                fork_seconds=fork_s, scoring_seconds=sum(t.get('scoring_seconds',0) for t in timeline),
                graph_capture_seconds=eng.fast.capture_seconds-capture_before,
                max_memory_allocated_bytes=torch.cuda.max_memory_allocated(self.device),
                max_memory_reserved_bytes=torch.cuda.max_memory_reserved(self.device),
                hit_frame_cap=steps >= max_frames, winner=winner, timeline=timeline,
                final_scores=final_scores, prompt_tokens=length,codec_validation=self.codec_validation,
                codec_graph_capture_seconds=self.codec_streams[stream_key].capture_seconds,
                completion_rule='suffix_confirmed_reward_winner_v2')
            arrivals=[e['commit_wall_seconds']-e['commit_start_s'] for e in timeline if 'commit_wall_seconds' in e]
            perf['minimum_gap_free_start_seconds']=max(arrivals) if arrivals else first_commit
            return raw, selected_codes, perf, pcm if mode == 'turn' else None

    def close(self):
        self.engine.close()
        self.codec_streams.clear()
        self.proc.audio_tokenizer.to('cpu')
        gc.collect()
        torch.cuda.empty_cache()
