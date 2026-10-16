"""Measured Small inference paths: fused CFG and reusable local CUDA graphs.

No weights, precision of the TTS model, codebooks, CFG equation, or sampling
distribution are changed. The legacy path remains available for comparison.
"""
from collections import OrderedDict
import time

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


class FastGenerator:
    def __init__(self, engine, graph_limit=4):
        self.engine = engine
        self.graphs = OrderedDict()
        self.graph_limit = graph_limit
        self.capture_seconds = 0.0
        self.candidate_ids = engine.model._local_text_candidate_ids(torch.device(engine.device))

    def clear_graphs(self):
        self.graphs.clear()

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
        model_input = ids
        for step in range(budget):
            mask = mask_buffer[:, :length + step]
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
            if len(frames) - last >= chunk_frames:
                last = len(frames)
                yield frames
        if len(frames) != last:
            yield frames


class StreamingCodec:
    """True incremental PCM decoding: one codec KV cache per batch row.

    Never re-decode a growing prefix. Uses the pinned codec's stateful frame
    primitive under its streaming context, the same primitive as batch_decode.
    """
    def __init__(self, codec, batch_size, audio_pad=-1):
        self.codec = codec
        self.audio_pad = audio_pad
        self.context = codec.streaming(batch_size=batch_size)

    def __enter__(self):
        self.context.__enter__()
        return self

    def __exit__(self, *args):
        return self.context.__exit__(*args)

    def decode(self, codes):
        # Input B,T,12; codec expects 12,B,T.
        valid = codes[:, :, 0].ne(self.audio_pad)
        lengths = valid.sum(-1)
        codes = codes.masked_fill(~valid.unsqueeze(-1), 0)
        self.codec._set_streaming_exec_mask(lengths > 0)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=codes.is_cuda):
            result = self.codec._decode_frame(codes.permute(2, 0, 1).contiguous(), lengths)
        return result.audio.float().mean(1), result.audio_lengths
