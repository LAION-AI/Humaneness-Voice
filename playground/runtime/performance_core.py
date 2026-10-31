"""Measured Small inference paths: fused CFG and reusable local CUDA graphs.

No weights, precision of the TTS model, codebooks, CFG equation, or sampling
distribution are changed. The legacy path remains available for comparison.
"""
from collections import OrderedDict
import time
import os
from types import MethodType

import torch
import torch.nn.functional as F


def combine_branches(branches, pad_token, audio_pad):
    length = max(ids.shape[1] for ids, _ in branches)
    ids_list, mask_list = [], []
    for ids, mask in branches:
        amount = length - ids.shape[1]
        if amount:
            padding = torch.full((ids.shape[0], amount, ids.shape[-1]), audio_pad,
                                 dtype=ids.dtype, device=ids.device)
            padding[:, :, 0] = pad_token
            ids = torch.cat((padding, ids), dim=1)
            mask = F.pad(mask, (amount, 0), value=False)
        ids_list.append(ids)
        mask_list.append(mask)
    return torch.cat(ids_list), torch.cat(mask_list).bool()


def fast_sample(mdl, logits, temperature, top_k, top_p):
    """Same distribution as the release's top-k mask + full-sort top-p, sampled among the k survivors."""
    if not top_k or int(top_k) >= logits.shape[-1]:
        return mdl._sample_next_token(logits=logits, do_sample=True, temperature=temperature,
            top_k=top_k, top_p=top_p, previous_token_ids=None, repetition_penalty=1.0)
    values, indices = torch.topk(logits.float() / float(temperature), int(top_k), dim=-1)
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        drop = torch.softmax(values, dim=-1).cumsum(dim=-1) > float(top_p)
        drop = torch.cat((torch.zeros_like(drop[..., :1]), drop[..., :-1]), dim=-1)
        values = values.masked_fill(drop, -torch.inf)
    return indices.gather(-1, torch.multinomial(torch.softmax(values, dim=-1), 1)).squeeze(-1)


RAMP_START=int(os.environ.get('HVS_RAMP_START','5'))
# Optional explicit ramp, e.g. '3,5,8,12': these chunk sizes first, then chunk_frames.
RAMP_SCHEDULE=[int(x) for x in os.environ.get('HVS_RAMP_SCHEDULE','').split(',') if x.strip()]


class FastGenerator:
    def __init__(self, engine, graph_limit=4):
        self.engine = engine
        self.graphs = OrderedDict()
        self.graph_limit = graph_limit
        self.capture_seconds = 0.0
        self.candidate_ids = engine.model._local_text_candidate_ids(torch.device(engine.device))
        self.semantic=None
        if os.environ.get('HVS_SEMANTIC_CUDA_GRAPH','0')=='1':
            from semantic_graph import SemanticGraphs
            self.semantic=SemanticGraphs(engine.model)
        self.fast_sample=os.environ.get('HVS_FAST_SAMPLE','0')=='1'
        self.chunk_ramp=os.environ.get('HVS_CHUNK_RAMP','0')=='1'
        if os.environ.get('HVS_LOCAL_COMPILE','0')=='1':
            self.local_frame=torch.compile(self.local_frame,dynamic=False,fullgraph=True)
        if os.environ.get('HVS_LOCAL_TRITON','0')=='1':
            # Bandwidth-tuned Triton kernels for the 12 codebook steps; the
            # compiled frame above stays as fallback (see local_fused.py).
            import local_fused
            local_fused.install(self)

    def clear_graphs(self):
        self.graphs.clear()
        if self.semantic:self.semantic.banks.clear()

    def local_frame(self, local_input, n, nb, guidance, p, cache):
        mdl = self.engine.model
        for layer in cache:
            layer["length"] = 0
        hidden, past = mdl._decode_local_hidden_states_with_cache(local_input.unsqueeze(1), cache)
        hidden = hidden[:, -1]
        if not mdl._use_binary_local_text_head():
            raise ValueError("Small fast path requires its released binary stop head")
        stop_logits = mdl.local_text_lm_head(hidden[:n])
        stop_index = mdl._sample_next_token(logits=stop_logits,
            do_sample=p["text_temperature"] > 0, temperature=p["text_temperature"],
            top_k=p["text_top_k"], top_p=p["text_top_p"])
        token = self.candidate_ids[stop_index]
        codes = []
        for channel in range(mdl.config.n_vq):
            logits = mdl.audio_lm_heads[channel](hidden).float()
            if nb == 2:
                conditional, neutral = logits[:n], logits[n:]
                logits = neutral + guidance * (conditional - neutral)
            if self.fast_sample:
                code = fast_sample(mdl, logits, p["audio_temperature"], p["audio_top_k"], p["audio_top_p"])
            else:
                code = mdl._sample_next_token(logits=logits, do_sample=True,
                    temperature=p["audio_temperature"], top_k=p["audio_top_k"],
                    top_p=p["audio_top_p"], previous_token_ids=None,
                    repetition_penalty=1.0)
            codes.append(code)
            if channel + 1 < mdl.config.n_vq:
                shared_code = code.repeat(nb)
                embed = mdl.audio_embeddings[channel](shared_code)
                hidden, past = mdl._decode_local_hidden_states_with_cache(embed.unsqueeze(1), past)
                hidden = hidden[:, -1]
        return token, torch.stack(codes, dim=-1)

    def frame(self, local_input, n, nb, guidance, p, graph=False):
        mdl = self.engine.model
        if not graph:
            cache = mdl._new_static_local_past_key_values(n * nb, 13,
                        local_input.device, local_input.dtype)
            return self.local_frame(local_input, n, nb, guidance, p, cache)
        key = (n, nb, float(guidance), p["audio_temperature"], p["text_temperature"],
               p["audio_top_k"], p["audio_top_p"])
        if key not in self.graphs:
            started = time.monotonic()
            rng = torch.cuda.get_rng_state(local_input.device)
            static = torch.empty_like(local_input)
            static.copy_(local_input)
            cache = mdl._new_static_local_past_key_values(n * nb, 13,
                        local_input.device, local_input.dtype)
            # Warm on a side stream before capturing, as recommended by PyTorch.
            stream = torch.cuda.Stream(device=local_input.device)
            stream.wait_stream(torch.cuda.current_stream(local_input.device))
            with torch.cuda.stream(stream):
                for _ in range(3):
                    self.local_frame(static, n, nb, guidance, p, cache)
            torch.cuda.current_stream(local_input.device).wait_stream(stream)
            cuda_graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(cuda_graph):
                output = self.local_frame(static, n, nb, guidance, p, cache)
            self.graphs[key] = (cuda_graph, static, output, cache)
            torch.cuda.set_rng_state(rng, local_input.device)
            self.capture_seconds += time.monotonic() - started
            while len(self.graphs) > self.graph_limit:
                self.graphs.popitem(last=False)
        self.graphs.move_to_end(key)
        cuda_graph, static, output, _ = self.graphs[key]
        static.copy_(local_input)
        cuda_graph.replay()
        return output

    @torch.inference_mode()
    def stream_frames(self, branches, p, chunk_frames, guidance=1.5,
                      min_frames=0, graph=False):
        mdl, cfg = self.engine.model, self.engine.model.config
        if p["audio_repetition_penalty"] != 1.0:
            raise ValueError("Fast path supports the demo's unchanged repetition penalty 1.0 only")
        ids, initial_mask = combine_branches(branches, cfg.pad_token_id, cfg.audio_pad_token_id)
        nb, n = len(branches), branches[0][0].shape[0]
        budget, length = int(p["max_new_tokens"]), ids.shape[1]
        mask_buffer = torch.zeros(n * nb, length + budget, dtype=torch.bool, device=ids.device)
        mask_buffer[:, :length] = initial_mask
        frames, past = [], None
        finished = torch.zeros(n, dtype=torch.bool, device=ids.device)
        last = 0
        yield_sizes = []
        model_input = ids
        semantic=self.semantic.prepare(ids,budget) if self.semantic else None
        for step in range(budget):
            mask = mask_buffer[:, :length + step]
            if semantic:
                local_input=semantic.prefill(model_input,mask) if step==0 else semantic.decode(model_input,mask)
            else:
                embeddings = mdl._build_inputs_embeds(model_input)
                out = mdl.transformer(inputs_embeds=embeddings, input_ids=None,
                    attention_mask=mask, past_key_values=past, use_cache=True,
                    position_ids=None, output_attentions=False, output_hidden_states=False,
                    return_dict=True, cu_seqlens=None, num_sequences=None)
                past = out.past_key_values
                local_input = mdl._global_hidden_to_local(out.last_hidden_state[:, -1]).to(mdl.local_transformer.ln_f.weight.dtype)
            token, sampled = self.frame(local_input, n, nb, guidance, p, graph)
            continues = token.eq(cfg.audio_assistant_slot_token_id) & ~finished
            finished = finished | token.eq(cfg.audio_end_token_id)
            if step < min_frames:
                continues = torch.ones_like(continues)
                finished = torch.zeros_like(finished)
            if not bool(continues.any().item()):
                break
            frame = sampled.masked_fill(~continues.unsqueeze(-1), cfg.audio_pad_token_id)
            frames.append(frame)
            row = mdl._build_generation_row(n, ids.device, frame)
            row[:, 0, 0] = torch.where(continues, row[:, 0, 0], cfg.pad_token_id)
            model_input = row.repeat(nb, 1, 1)
            mask_buffer[:, length + step] = continues.repeat(nb)
            # Optional short initial block, then the normal codec batch size.
            # This changes only yield boundaries, not token sampling or caches.
            block=p.get('first_chunk_frames',chunk_frames) if last==0 else chunk_frames
            if self.chunk_ramp:
                # Small first chunk, then double up to chunk_frames: each block
                # finishes before the previous one has played (RTF < 0.5).
                if RAMP_SCHEDULE and not p.get('first_chunk_frames'):
                    block=min(chunk_frames,RAMP_SCHEDULE[len(yield_sizes)]) if len(yield_sizes)<len(RAMP_SCHEDULE) else chunk_frames
                else:
                    block=min(chunk_frames,max(1,p.get('first_chunk_frames') or RAMP_START)*2**len(yield_sizes))
            if len(frames) - last >= block:
                yield_sizes.append(len(frames) - last)
                last = len(frames)
                yield frames
        if len(frames) != last:
            yield frames


CODEC_GRAPH = os.environ.get('HVS_CODEC_GRAPH', '0') == '1'
CODEC_GRAPH_MAX = 20
_CODEC_GRAPHS = {}


def _stable_cache_update(module, state, keys, values, positions, *current):
    # Idea from kadirnar/fast-moss-tokenizer: run the upstream update, then keep
    # the cache at its previous addresses so captured CUDA graphs stay valid.
    type(module)._update_streaming_cache(module, state, keys, values, positions, *current)
    keys.copy_(state.cached_keys)
    values.copy_(state.cached_values)
    positions.copy_(state.cached_positions)
    state.cached_keys, state.cached_values, state.cached_positions = keys, values, positions


class CodecGraphs:
    """Persistent batch-1 decoder streaming state with one CUDA graph per chunk size.

    The decoder's bounded KV caches are allocated at full context length once;
    each request only resets them, so replayed graphs read fixed addresses.
    The encoder is never put into streaming mode (reference encoding unaffected).
    """
    def __init__(self, codec, audio_pad):
        self.codec = codec
        self.audio_pad = audio_pad
        self.device = next(codec.parameters()).device
        self.modules = [m for m in codec.decoder.modules() if hasattr(m, '_init_streaming_state')]
        self.states = [m._init_streaming_state(1) for m in self.modules]
        if os.environ.get('HVS_CODEC_FUSED', '0') == '1':
            # Fused Triton decode kernels (bit-identical), see codec_fused.py.
            import codec_fused
            codec_fused.install(codec, self.states)
        self.graphs = {}
        self.capture_seconds = 0.0

    def attach(self):
        if any(m._streaming_state is not None for m in self.modules):
            raise RuntimeError('Codec decoder is already streaming')
        for module, state in zip(self.modules, self.states):
            module._streaming_state = state
            if hasattr(type(module), '_update_streaming_cache'):
                module._update_streaming_cache = MethodType(_stable_cache_update, module)
        if not self.graphs:
            self.capture()
        self.reset()

    def detach(self):
        for module in self.modules:
            module._streaming_state = None
            module.__dict__.pop('_update_streaming_cache', None)

    def reset(self):
        for state in self.states:
            state.reset(torch.ones_like(state.exec_mask))

    def eager(self, codes):
        valid = codes[:, :, 0].ne(self.audio_pad)
        lengths = valid.sum(-1)
        codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
        self.codec._set_streaming_exec_mask(lengths > 0)
        with torch.autocast('cuda', dtype=torch.bfloat16, enabled=codes.is_cuda):
            result = self.codec._decode_frame(codes.permute(2, 0, 1).contiguous(), lengths)
        return result.audio.float().mean(1), result.audio_lengths

    @torch.inference_mode()
    def capture(self):
        started = time.monotonic()
        pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        stream.wait_stream(torch.cuda.current_stream(self.device))
        for frames in range(1, CODEC_GRAPH_MAX + 1):
            static = torch.zeros(1, frames, 12, dtype=torch.long, device=self.device)
            with torch.cuda.stream(stream):
                for _ in range(2):
                    self.eager(static)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool, stream=stream):
                output = self.eager(static)
            self.graphs[frames] = (graph, static, output)
        torch.cuda.current_stream(self.device).wait_stream(stream)
        torch.cuda.synchronize(self.device)
        self.capture_seconds = time.monotonic() - started

    def decode(self, codes):
        entry = self.graphs.get(codes.shape[1])
        if entry is None or codes.shape[0] != 1:
            return self.eager(codes)
        graph, static, (audio, lengths) = entry
        static.copy_(codes)
        graph.replay()
        # The next replay overwrites the graph's outputs; callers keep chunks.
        return audio.clone(), lengths.clone()


class StreamingCodec:
    """True incremental PCM decoding: one codec KV cache per batch row.

    Never re-decode a growing prefix. Uses the pinned codec's stateful frame
    primitive under its streaming context, the same primitive as batch_decode.
    With HVS_CODEC_GRAPH=1, batch-1 streams replay per-chunk-size CUDA graphs
    over a persistent decoder state (bit-identical output, fewer kernel launches).
    """
    def __init__(self, codec, batch_size, audio_pad=-1):
        self.codec = codec
        self.audio_pad = audio_pad
        self.graphs = None
        if CODEC_GRAPH and batch_size == 1 and next(codec.parameters()).is_cuda:
            key = id(codec)
            if key not in _CODEC_GRAPHS or _CODEC_GRAPHS[key].codec is not codec:
                _CODEC_GRAPHS[key] = CodecGraphs(codec, audio_pad)
            self.graphs = _CODEC_GRAPHS[key]
            self.context = None
        else:
            self.context = codec.streaming(batch_size=batch_size)

    def __enter__(self):
        if self.graphs is not None:
            self.graphs.attach()
        else:
            self.context.__enter__()
        return self

    def __exit__(self, *args):
        if self.graphs is not None:
            self.graphs.detach()
            return False
        return self.context.__exit__(*args)

    def decode(self, codes):
        if self.graphs is not None:
            return self.graphs.decode(codes)
        # Input B,T,12; codec expects 12,B,T.
        valid = codes[:, :, 0].ne(self.audio_pad)
        lengths = valid.sum(-1)
        codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
        self.codec._set_streaming_exec_mask(lengths > 0)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=codes.is_cuda):
            result = self.codec._decode_frame(codes.permute(2, 0, 1).contiguous(), lengths)
        return result.audio.float().mean(1), result.audio_lengths
