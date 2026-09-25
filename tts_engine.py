"""Frame-by-frame streaming for MOSS-TTS-Local 4.55B voice-acting v2.

The model emits 12-codebook RVQ frames at 12.5 Hz.  Rather than waiting for the
whole sequence, we decode a sliding window every `chunk_frames` frames and emit
PCM as it settles, so playback can start ~1 s in.

The windowing (hold-back + overlap-add crossfade + left context) follows LAION's
own reference playground server, which is the implementation the model card
points at for click-free streaming; decoding only the trailing CTX+new frames
keeps it O(n) instead of O(n^2) while preserving timbre across seams.
"""
import threading, time

import numpy as np
import torch

import config
import steer_engine
import timed_script


class TTSEngine:
    def __init__(self, repo=None, codec=None, device="cuda:0", dtype=None):
        from transformers import AutoModel, AutoProcessor
        repo = repo or config.TTS_REPO
        codec = codec or config.CODEC_REPO
        self.device = device
        self.lora = None            # set by the app once the bank is built
        self.vectors = None         # steer_engine.VectorPack, likewise; None = no steering
        self.last_prompt_unc = None  # the neutralised branch, when guidance ran
        dtype = dtype or (torch.float32 if str(device) == "cpu" else torch.bfloat16)
        self.lock = threading.Lock()      # one GPU, serialise generations

        t0 = time.time()
        print(f"[tts] loading processor + codec on {device} ...", flush=True)
        self.proc = AutoProcessor.from_pretrained(repo, trust_remote_code=True,
                                                  codec_path=codec)
        self.proc.audio_tokenizer = self.proc.audio_tokenizer.to(device).eval()
        print(f"[tts] loading {repo} ...", flush=True)
        # flash-attn 2 is incompatible with this remote-code attention; sdpa is
        # what the model card recommends.
        self.model = AutoModel.from_pretrained(
            repo, trust_remote_code=True, dtype=dtype,
            attn_implementation="sdpa").to(device).eval()
        self.sr = int(self.proc.model_config.sampling_rate)
        # 12.5 Hz RVQ frames -> an exact, integral number of samples per frame
        self.samples_per_frame = int(round(self.sr / config.FRAME_RATE))
        print(f"[tts] ready in {time.time()-t0:.1f}s (sr={self.sr})", flush=True)

    # ------------------------------------------------------------ token loop
    @staticmethod
    def _sample_continue_or_end(mdl, lh, p, stop_bias=None):
        """Decide "another frame" vs "stop", with a thumb on the scale.

        Each step the model picks between exactly two tokens: keep going, or end
        the take.  It ends a few words early often enough to be annoying, so a
        constant bias is subtracted from the end token's logit before sampling.
        In nats: 1.0 makes stopping about e times less likely at any given step,
        which shifts a marginal decision without ever making stopping impossible
        — the model still ends when it is actually confident.

        Falls back to the model's own sampler when the bias is off, so the
        default path is bit-for-bit what it was.
        """
        bias = float(stop_bias if stop_bias is not None else config.STOP_BIAS)
        if abs(bias) < 1e-6:
            return mdl._sample_next_assistant_text_token(
                local_hidden_states=lh, do_sample=True,
                temperature=p["text_temperature"], top_k=p["text_top_k"],
                top_p=p["text_top_p"])

        cand = mdl._local_text_candidate_ids(lh.device)      # [continue, end]
        if mdl._use_binary_local_text_head() and mdl.local_text_lm_head is not None:
            logits = mdl.local_text_lm_head(lh)
        else:
            logits = mdl.text_lm_head(lh).index_select(dim=-1, index=cand)
        logits = logits.clone()
        logits[..., 1] -= bias                               # 1 = the end token
        idx = mdl._sample_next_token(
            logits=logits, do_sample=True, temperature=p["text_temperature"],
            top_k=p["text_top_k"], top_p=p["text_top_p"])
        return cand[idx]

    @torch.inference_mode()
    def _stream_frames(self, input_ids, attention_mask, p, chunk_frames,
                       min_frames=0, inj=steer_engine.NULL):
        """The single-branch loop.  `inj` is the steering injector, or the no-op one.

        With `steer_engine.NULL` every `inj.*` call returns its argument unchanged, so this
        path is bit-for-bit what it was before steering existed.  That is not a nicety: a
        zero-strength run through the identical code path is the control the research
        harness relies on, and it held exactly there (510 cells, maximum absolute
        difference 0.0).
        """
        mdl = self.model
        mdl._resolve_fixed_nq(n_vq_for_inference=None, nq=None)
        if input_ids.ndim == 2:
            input_ids = input_ids.unsqueeze(0)
        if attention_mask is None:
            attention_mask = torch.ones(input_ids.shape[:2], dtype=torch.bool,
                                        device=input_ids.device)
        elif attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        attention_mask = attention_mask.to(device=input_ids.device, dtype=torch.bool)

        cfg = mdl.config
        budget = int(p["max_new_tokens"] or 4096)
        cur, cur_mask, cur_in = input_ids, attention_mask, input_ids
        frames = []
        finished = torch.zeros(input_ids.shape[0], dtype=torch.bool,
                               device=input_ids.device)
        pkv = None
        ldt = mdl.local_transformer.ln_f.weight.dtype
        last = 0

        # History for the repetition penalty, kept in one preallocated buffer.
        # The obvious `torch.stack(frames)` rebuilds the entire history on every
        # frame, which is O(n^2) copies and n extra kernel launches over a take —
        # and this loop is launch-bound at batch 1, so that is not free.
        B = input_ids.shape[0]
        hist_buf = torch.zeros(B, budget, int(cfg.n_vq), dtype=torch.long,
                               device=input_ids.device)
        n_hist = 0

        for _ in range(budget):
            hist = hist_buf[:, :n_hist] if n_hist else None
            emb = inj.apply_emb(mdl._build_inputs_embeds(cur_in))
            go = mdl.transformer(input_ids=None, past_key_values=pkv,
                                 attention_mask=cur_mask, position_ids=None,
                                 inputs_embeds=emb, use_cache=True,
                                 output_attentions=False, output_hidden_states=False,
                                 return_dict=True, cu_seqlens=None, num_sequences=None)
            gh = inj.apply_final(go.last_hidden_state[:, -1, :])
            lg = mdl._global_hidden_to_local(gh).to(dtype=ldt)
            lph, lpp = mdl._decode_local_hidden_states_with_cache(lg.unsqueeze(1))
            lh = inj.apply_loc(lph[:, -1, :])

            nt = self._sample_continue_or_end(
                mdl, lh, p, stop_bias=p.get("stop_bias"))
            cont = nt.eq(int(cfg.audio_assistant_slot_token_id)) & ~finished
            finished = finished | nt.eq(int(cfg.audio_end_token_id))
            # About one take in six ends at a sentence boundary with half the line
            # still unspoken (measured word recall 0.49 on those).  Rather than
            # cutting the reply into separate generations — which produces an
            # audible seam in voice and emotion between sentences — simply refuse
            # the end token until the duration we asked for is plausibly used.
            # Healthy takes reach 85-100% of the budget, truncated ones ~35%, so a
            # floor well below the normal range only catches the failures.
            if min_frames is not None and not isinstance(min_frames, int):
                # one floor per row: a batch holds lines of different lengths
                below = min_frames > len(frames)
                cont = cont | below
                finished = finished & ~below
            elif min_frames and len(frames) < min_frames:
                cont = torch.ones_like(cont)
                finished = torch.zeros_like(finished)
            if not bool(cont.any().item()):
                break

            ftoks = []
            for ci in range(int(cfg.n_vq)):
                logit = mdl.audio_lm_heads[ci](lh)
                tok = mdl._sample_next_token(
                    logits=logit, do_sample=True,
                    temperature=p["audio_temperature"], top_k=p["audio_top_k"],
                    top_p=p["audio_top_p"],
                    previous_token_ids=None if hist is None else hist[:, :, ci],
                    repetition_penalty=p["audio_repetition_penalty"])
                ftoks.append(tok)
                if ci + 1 < int(cfg.n_vq):
                    cl = mdl.audio_embeddings[ci](tok).to(dtype=ldt)
                    lth, lpp = mdl._decode_local_hidden_states_with_cache(
                        cl.unsqueeze(1), past_key_values=lpp)
                    lh = lth[:, -1, :]

            nf = torch.stack(ftoks, dim=-1).masked_fill(
                ~cont.unsqueeze(-1), int(cfg.audio_pad_token_id))
            frames.append(nf)
            hist_buf[:, n_hist] = nf
            n_hist += 1
            row = mdl._build_generation_row(batch_size=input_ids.shape[0],
                                            device=input_ids.device, audio_token_ids=nf)
            if bool((~cont).any().item()):
                row[~cont, 0, 0] = int(cfg.pad_token_id)
                row[~cont, 0, 1:] = int(cfg.audio_pad_token_id)
            cur = torch.cat([cur, row], dim=1)
            cur_mask = torch.cat([cur_mask, cont.unsqueeze(1)], dim=1)
            cur_in = row
            pkv = go.past_key_values
            if len(frames) - last >= chunk_frames:
                last = len(frames)
                yield frames
        yield frames

    # ------------------------------------------------------- guided token loop
    @torch.inference_mode()
    def _stream_frames_cfg(self, branches, p, chunk_frames, min_frames=0,
                           inj=steer_engine.NULL, guidance=1.0, steer_both=True):
        """Classifier-free guidance: the same loop with two conditioned branches.

            logits = logits_uncond + g * (logits_cond - logits_uncond)

        `branches` is [(ids, mask), (ids, mask)] and branch 0 is CONDITIONAL.  The two differ
        only in the instruction: branch 1 has the performance direction removed from GENERAL
        and every round-bracket cue removed from the script, while the words, the durations,
        the pauses, the burst tags and the `Tokens` budget are byte-identical (see
        `timed_script.neutralise`).  So the difference between the two logit sets is about
        affect and nothing else.

        **g = 1 cancels the unconditional term exactly** and reproduces ordinary sampling
        from the conditional branch, through this identical code path -- both branches still
        run, both caches still advance, the same draws happen in the same order.  That is
        what makes it a control rather than a second condition.

        WHY THE BRANCHES INTERLEAVE AT CHANNEL LEVEL, not frame level.  The twelve audio
        channels of a frame are sampled autoregressively: channel c+1's logits depend on the
        token sampled for channel c.  Running branch A for a whole frame and then branch B
        would leave the two conditioned on DIFFERENT channel prefixes, and their difference
        would then be mostly about that divergence rather than about the condition.  Both
        branches are therefore advanced with the SAME sampled token at every channel step.

        WHAT IS NOT GUIDED.  The continue/end decision is taken on the conditional branch
        alone.  It is a structural token: guiding it would change how long the clip is, and
        the duration error would then be measuring guidance's effect on the stopping rule
        rather than on the performance.

        WHY THIS DOES NOT STREAM.  Measured cost is 1.93x at batch 1.  docs/ADAPTERS.md §1
        records realtime factor 1.0 as the streaming budget and 0.764 as the live merged
        baseline, which puts a guided take at ~1.47 -- the player would starve.  The caller
        therefore passes a chunk size larger than any take, so exactly one decode happens,
        at the end.  See docs/LEVERS.md.
        """
        mdl = self.model
        mdl._resolve_fixed_nq(n_vq_for_inference=None, nq=None)
        st = []
        for ids, am in branches:
            if ids.ndim == 2:
                ids = ids.unsqueeze(0)
            if am is None:
                am = torch.ones(ids.shape[:2], dtype=torch.bool, device=ids.device)
            elif am.ndim == 1:
                am = am.unsqueeze(0)
            am = am.to(device=ids.device, dtype=torch.bool)
            st.append({"ids": ids, "am": am, "mi": ids, "pkv": None,
                       "loc": None, "lpkv": None})
        nb = len(st)
        cfg = mdl.config
        dev = st[0]["ids"].device
        B = st[0]["ids"].shape[0]
        budget = int(p["max_new_tokens"] or 4096)
        ldt = mdl.local_transformer.ln_f.weight.dtype
        frames = []
        finished = torch.zeros(B, dtype=torch.bool, device=dev)
        hist_buf = torch.zeros(B, budget, int(cfg.n_vq), dtype=torch.long, device=dev)
        n_hist = 0
        last = 0
        g = float(guidance)

        try:
            for _ in range(budget):
                hist = hist_buf[:, :n_hist] if n_hist else None
                for i, sb in enumerate(st):
                    # One set of hooks lives on the shared model, so branch selection is a
                    # flag rather than a second Injector: with steer_both False only the
                    # conditional branch is steered.
                    inj.enabled = bool(steer_both or i == 0)
                    emb = inj.apply_emb(mdl._build_inputs_embeds(sb["mi"]))
                    go = mdl.transformer(
                        input_ids=None, past_key_values=sb["pkv"],
                        attention_mask=sb["am"], position_ids=None, inputs_embeds=emb,
                        use_cache=True, output_attentions=False,
                        output_hidden_states=False, return_dict=True,
                        cu_seqlens=None, num_sequences=None)
                    sb["pkv"] = go.past_key_values
                    gh = inj.apply_final(go.last_hidden_state[:, -1, :])
                    lg = mdl._global_hidden_to_local(gh).to(dtype=ldt)
                    lph, sb["lpkv"] = mdl._decode_local_hidden_states_with_cache(
                        lg.unsqueeze(1))
                    sb["loc"] = inj.apply_loc(lph[:, -1, :])
                inj.enabled = True

                nt = self._sample_continue_or_end(
                    mdl, st[0]["loc"], p, stop_bias=p.get("stop_bias"))
                cont = nt.eq(int(cfg.audio_assistant_slot_token_id)) & ~finished
                finished = finished | nt.eq(int(cfg.audio_end_token_id))
                if min_frames is not None and not isinstance(min_frames, int):
                    # one floor per row, as in the single-branch loop: a batch
                    # holds lines of different lengths, and best-of-N sends a
                    # whole batch through here
                    below = min_frames > len(frames)
                    cont = cont | below
                    finished = finished & ~below
                elif min_frames and len(frames) < min_frames:
                    cont = torch.ones_like(cont)
                    finished = torch.zeros_like(finished)
                if not bool(cont.any().item()):
                    break

                ftoks = []
                for ci in range(int(cfg.n_vq)):
                    logit = mdl.audio_lm_heads[ci](st[0]["loc"]).float()
                    if nb > 1 and abs(g - 1.0) > 1e-6:
                        unc = mdl.audio_lm_heads[ci](st[1]["loc"]).float()
                        logit = unc + g * (logit - unc)
                    tok = mdl._sample_next_token(
                        logits=logit, do_sample=True,
                        temperature=p["audio_temperature"], top_k=p["audio_top_k"],
                        top_p=p["audio_top_p"],
                        previous_token_ids=None if hist is None else hist[:, :, ci],
                        repetition_penalty=p["audio_repetition_penalty"])
                    ftoks.append(tok)
                    if ci + 1 < int(cfg.n_vq):
                        cl = mdl.audio_embeddings[ci](tok).to(dtype=ldt)
                        for sb in st:
                            lth, sb["lpkv"] = mdl._decode_local_hidden_states_with_cache(
                                cl.unsqueeze(1), past_key_values=sb["lpkv"])
                            sb["loc"] = lth[:, -1, :]

                nf = torch.stack(ftoks, dim=-1).masked_fill(
                    ~cont.unsqueeze(-1), int(cfg.audio_pad_token_id))
                frames.append(nf)
                hist_buf[:, n_hist] = nf
                n_hist += 1
                row = mdl._build_generation_row(batch_size=B, device=dev,
                                                audio_token_ids=nf)
                if bool((~cont).any().item()):
                    row = row.clone()
                    row[~cont, 0, 0] = int(cfg.pad_token_id)
                    row[~cont, 0, 1:] = int(cfg.audio_pad_token_id)
                for sb in st:
                    sb["ids"] = torch.cat([sb["ids"], row], dim=1)
                    sb["am"] = torch.cat([sb["am"], cont.unsqueeze(1)], dim=1)
                    sb["mi"] = row
                if len(frames) - last >= chunk_frames:
                    last = len(frames)
                    yield frames
        finally:
            inj.enabled = True
        yield frames

    # ------------------------------------------------------------- public API
    @staticmethod
    def frames_for(text, per_word=None):
        """Duration budget in 12.5 Hz codec frames.

        The model card is explicit: pass `tokens ~= words * 6`, otherwise the
        model picks its own length and routinely stops before the line is done.
        """
        n = len([w for w in str(text).split() if w.strip()])
        return max(24, int(n * (per_word or config.TOKENS_PER_WORD)))

    def build_inputs(self, text, instruction, language="English", ref_codes=None,
                     seed=0, tokens=None):
        kw = dict(text=text, instruction=instruction, language=language)
        if tokens != -1:          # -1 = omit entirely, for A/B against the default
            kw["tokens"] = int(tokens) if tokens else self.frames_for(text)
        if ref_codes is not None:
            # a list: the processor embeds each entry as its own reference segment,
            # so an identity anchor can be stacked in front of the delivery clip
            refs = ref_codes if isinstance(ref_codes, (list, tuple)) else [ref_codes]
            kw["reference"] = [r.to(torch.long) for r in refs if r is not None]
        torch.manual_seed(int(seed))
        msg = self.proc.build_user_message(**kw)
        batch = self.proc([[msg]], mode="generation")
        # the processor's own rendering of the <user_inst> block — what the model
        # is actually handed, kept so the demo can show it verbatim instead of
        # showing the script as it looked before the timing was worked out
        self.last_prompt = msg.get("content") if isinstance(msg, dict) else None
        return batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device)

    @torch.inference_mode()
    def stream_pcm(self, text, instruction, language="English", ref_codes=None,
                   lora_specs=None, lever_plan=None, instruction_unc=None,
                   text_unc=None, **over):
        """Yield ("start"|"pcm"|"end", payload) as the audio is generated.

        `lever_plan` is a `levers.Plan`: which of the three levers runs this turn.  With
        none, or with a plan whose mode is `adapter`, this is exactly the path it always
        was.  `instruction_unc` / `text_unc` are the neutralised prompt that forms the
        second branch when the plan asks for guidance.
        """
        p = dict(config.DEFAULTS)
        p.update({k: v for k, v in over.items() if v is not None})
        want_steer = bool(lever_plan is not None and lever_plan.steer)
        want_cfg = bool(lever_plan is not None and lever_plan.wants_cfg
                        and instruction_unc)

        with self.lock:
            # adapters are attached and detached inside the generation lock, so a
            # turn can never run under another turn's adapters
            applied = []
            if self.lora is not None:
                try:
                    applied = self.lora.apply(lora_specs)
                except Exception as e:
                    print(f"[tts] lora apply failed: {e}", flush=True)
                    self.lora.clear()
            t0 = time.time()
            n_tok = p.get("tokens") if p.get("tokens") else self.frames_for(text)
            iid, am = self.build_inputs(text, instruction, language, ref_codes,
                                        p["seed"], tokens=n_tok)
            branch1 = None
            if want_cfg:
                # Same reference clips, same Tokens, same words -- only the direction is
                # gone.  Built second so `last_prompt` (what the UI shows) stays the
                # conditional one; the neutralised text is exposed separately.
                cond_prompt = getattr(self, "last_prompt", None)
                branch1 = self.build_inputs(text_unc or text, instruction_unc, language,
                                            ref_codes, p["seed"], tokens=n_tok)
                self.last_prompt_unc = getattr(self, "last_prompt", None)
                self.last_prompt = cond_prompt
            # The steering hooks live on the model, so they MUST be removed again or the
            # next take inherits them -- a leaked hook is invisible and would make every
            # later generation a superposition of the ones before it.
            inj = steer_engine.NULL
            realised = {}
            if want_steer and not (self.vectors is not None and self.vectors.available):
                lever_plan.drop_steer("the steering vectors are not loaded on this server")
                want_steer = False
            if want_steer:
                try:
                    specs, missing = steer_engine.build_specs(
                        lever_plan.steer, self.vectors, self.device)
                    realised = steer_engine.realised_magnitude(specs)
                    over_cap = {k: v for k, v in realised.items()
                                if v > config.STEER_REALISED_CEILING + 1e-6}
                    if over_cap:
                        # Components sharing a layer sum there, so the realised magnitude
                        # is not the nominal alpha.  Refuse rather than trim: a
                        # composition that lands past the ceiling was not the one measured.
                        print(f"[steer] refusing: realised magnitude {over_cap} exceeds "
                              f"the ceiling {config.STEER_REALISED_CEILING}", flush=True)
                        lever_plan.drop_steer(
                            f"refused: realised magnitude {over_cap} past the ceiling "
                            f"{config.STEER_REALISED_CEILING}")
                        want_steer = False
                    else:
                        if missing:
                            print(f"[steer] not in the pack: {missing}", flush=True)
                        inj = steer_engine.Injector(self.model, specs)
                except Exception as e:
                    print(f"[steer] injector failed: {e}", flush=True)
                    lever_plan.drop_steer(f"steering failed to attach: {e}")
                    inj = steer_engine.NULL
                    want_steer = False
                    realised = {}
            if lever_plan is not None and want_steer:
                lever_plan.realised_alpha = realised
            if lever_plan is not None and lever_plan.wants_cfg and not want_cfg:
                lever_plan.drop_cfg("no neutralised prompt was built for this take")
            prep_ms = (time.time() - t0) * 1000
            # never let the frame budget cut the line off before the duration we
            # just asked for; leave generous headroom for pauses and bursts
            p["max_new_tokens"] = max(int(p["max_new_tokens"] or 0),
                                      int(abs(n_tok) * config.TOKEN_HEADROOM) + 64)

            SR = self.sr
            HB = int(config.HOLDBACK_S * SR)
            XF = int(config.CROSSFADE_S * SR)
            CTX = config.CTX_FRAMES

            started = False
            last_frames = None   # this take's own codes, reused as context later
            emitted = 0          # absolute samples handed to the client
            emitted_f = 0
            pending = np.zeros(0, np.float32)
            win = None
            base = 0
            n_sent = 0

            min_frames = int(abs(n_tok) * config.MIN_FRAME_FRACTION)
            if want_cfg:
                # A guided take costs 1.93x and would starve the player, so it is rendered
                # whole and then handed over: one chunk larger than any possible take means
                # exactly one decode, at the end.  The client sees a long time-to-first-audio
                # and then the complete reply, which is honest about what it is doing.
                # The plan already decided this: `both` returns 0.209 of word error and
                # 0.75 of genuineness for 18 % of the effect.
                steer_both = (lever_plan.steer_branch == "both")
                gen = self._stream_frames_cfg(
                    [(iid, am), branch1], p, chunk_frames=10 ** 9,
                    min_frames=min_frames, inj=inj,
                    guidance=lever_plan.guidance, steer_both=steer_both)
            else:
                gen = self._stream_frames(iid, am, p, int(p["chunk_frames"]),
                                          min_frames=min_frames, inj=inj)
            try:
                for frames in gen:
                    last_frames = frames
                    if not frames:
                        continue
                    ws = max(0, emitted_f - CTX)
                    codes = torch.stack(frames[ws:], dim=0)[:, 0, :]
                    w = self.proc.decode_audio_codes([codes], return_stereo=False)[0]
                    w = w.detach().cpu().float()
                    win = (w.mean(0) if w.dim() > 1 else w).numpy().astype(np.float32)
                    # The codec is exactly SR/12.5 samples per frame (3840 at 48 kHz).
                    # Deriving it from the decoded length instead lets rounding drift
                    # accumulate once the window starts sliding, and the splice point
                    # walks off the frame grid — which is audible as a stutter well
                    # into a long reply, not at the start.
                    spf = self.samples_per_frame
                    base = ws * spf

                    if not started:
                        started = True
                        yield "start", {"sr": SR, "ttfa_gpu_ms": round((time.time()-t0)*1000, 1),
                                        "prep_ms": round(prep_ms, 1)}

                    end = base + len(win) - HB          # settled boundary in absolute samples
                    if end <= emitted + pending.size:
                        continue
                    lo = max(0, emitted - base)
                    seg = win[lo:end - base].copy()
                    if pending.size and seg.size > pending.size:
                        f = np.linspace(0, 1, pending.size, dtype=np.float32)
                        seg[:pending.size] = pending * (1 - f) + seg[:pending.size] * f
                    if seg.size > XF:
                        pending = seg[-XF:].copy()
                        out = seg[:-XF]
                    else:
                        pending = seg.copy()
                        out = np.zeros(0, np.float32)
                    emitted = end - pending.size
                    emitted_f = emitted // spf
                    if out.size:
                        n_sent += out.size
                        yield "pcm", out

                if win is not None and (base + len(win)) > emitted:   # flush held-back tail
                    seg = win[max(0, emitted - base):].copy()
                    if pending.size and seg.size > pending.size:
                        f = np.linspace(0, 1, pending.size, dtype=np.float32)
                        seg[:pending.size] = pending * (1 - f) + seg[:pending.size] * f
                    n_sent += seg.size
                    yield "pcm", seg

                gpu_ms = (time.time() - t0) * 1000
                dur = n_sent / SR
                if self.lora is not None:
                    self.lora.clear()
                # The generated frames are already 12-codebook MOSS codes — the exact
                # format `reference=` wants — so the tail of this take can seed the
                # next turn without any re-encoding.
                tail = None
                if last_frames:
                    n = min(len(last_frames), config.TAIL_FRAMES)
                    tail = torch.stack(last_frames[-n:], dim=0)[:, 0, :].cpu().clone()
                yield "end", {"gpu_total_ms": round(gpu_ms, 1), "tail_codes": tail,
                              "audio_sec": round(dur, 3),
                              "frames": emitted_f, "tokens": n_tok,
                              "rtf": round((gpu_ms / 1000) / dur, 3) if dur > 0 else None,
                              "loras": [{"name": n, "lam": l} for n, l in applied],
                              # what actually ran, so a bad take can be traced to its
                              # configuration rather than guessed at
                              "levers": (lever_plan.payload() if lever_plan is not None
                                         else None),
                              "prompt_unc": self.last_prompt_unc if want_cfg else None}
            finally:
                # A leaked forward hook is invisible: the model keeps generating,
                # slightly wrong, and nothing in the logs says so.  This runs even if
                # the client abandons the stream mid-reply.
                inj.close()

    # ------------------------------------------------------ multi-part takes
    @staticmethod
    def split_script(script):
        """Split a SCRIPT into sentence-sized parts, cues kept with their sentence.

        The model ends a take by emitting an end token, and it turns out to do
        that at a sentence boundary maybe one time in six — the second half of
        the reply is then simply never spoken (measured recall 0.49 on such a
        take).  Generating one sentence at a time makes an end token at a
        sentence boundary the correct outcome rather than a truncation.
        """
        import re
        # Splitting costs a fresh prefill of the whole instruction per part, which
        # pushed the realtime factor over 1.0 and starved the player.  Short
        # replies were not where takes were being truncated anyway, so they stay
        # in one piece and keep the fast path.
        bare_all = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", script or "")
        if len(bare_all.split()) <= config.SPLIT_MIN_WORDS:
            return [script] if script else []
        parts, buf = [], ""
        # split after . ! ? … that are not inside brackets
        for tok in re.split(r"(?<=[.!?…])\s+", (script or "").strip()):
            if not tok:
                continue
            buf = (buf + " " + tok).strip() if buf else tok
            # keep going while the piece is only a cue, or still very short
            bare = re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", buf).strip()
            if len(bare.split()) >= 6:
                parts.append(buf)
                buf = ""
        if buf:
            if parts and len(re.sub(r"\([^)]*\)|\[[^\]]*\]", " ", buf).split()) < 4:
                parts[-1] += " " + buf          # a stray fragment joins the previous part
            else:
                parts.append(buf)
        return parts or ([script] if script else [])

    @torch.inference_mode()
    def stream_utterance(self, script, general, strip, language="English",
                         ref_codes=None, lora_specs=None, speed=1.0,
                         reads_as=None, lever_plan=None, general_unc=None, **over):
        """Stream a whole reply as a sequence of sentence-sized takes.

        Yields the same ("start"|"pcm"|"end") protocol as stream_pcm, with one
        start at the first audio and one end carrying the totals.
        """
        parts = self.split_script(script)
        t0 = time.time()
        tail = None
        prompt_unc = None
        started = False
        total_samples = 0
        n_tok_total = 0
        applied = []
        levers_payload = None
        for idx, part in enumerate(parts):
            part_over = dict(over)
            instruction_unc = text_unc = None
            if config.TIMED_SCRIPT:
                # SFT3's format: the script carries its own timing, and the Text
                # field repeats it byte for byte rather than being stripped bare.
                tagged, frames, plain = timed_script.render(part, speed=speed)
                if not plain:
                    continue
                text = tagged
                lc = "DE" if str(language).lower().startswith(("ger", "de")) else "EN"
                gl = timed_script.general_line(
                    general, frames / config.FRAME_RATE, lc, reads_as)
                instruction = f"GENERAL: {gl}\nSCRIPT:\n{tagged}"
                if general_unc:
                    # The unconditional branch: the standing half of GENERAL with no
                    # performance direction and no "reads as" clause, and the same script
                    # with its round-bracket directions removed.  Durations, pauses, burst
                    # tags, words and the Tokens budget are byte-identical, so the two
                    # branches differ in affect and in nothing else.
                    plain_tagged = timed_script.neutralise(tagged)
                    gl_unc = timed_script.general_line(
                        general_unc, frames / config.FRAME_RATE, lc, None)
                    instruction_unc = f"GENERAL: {gl_unc}\nSCRIPT:\n{plain_tagged}"
                    text_unc = plain_tagged
                if frames:
                    part_over["tokens"] = frames
            else:
                text = strip(part)
                if not text.strip():
                    continue
                instruction = f"GENERAL: {general}\nSCRIPT:\n{part}"
                if general_unc:
                    instruction_unc = (f"GENERAL: {general_unc}\nSCRIPT:\n"
                                       + timed_script.neutralise(part))
                    text_unc = text
            for kind, payload in self.stream_pcm(
                    text=text, instruction=instruction, language=language,
                    ref_codes=ref_codes, lora_specs=lora_specs,
                    lever_plan=lever_plan, instruction_unc=instruction_unc,
                    text_unc=text_unc, **part_over):
                if kind == "pcm":
                    total_samples += len(payload)
                    yield "pcm", payload
                elif kind == "start":
                    if not started:
                        started = True
                        p = dict(payload)
                        p["parts"] = len(parts)
                        p["prompt"] = getattr(self, "last_prompt", None)
                        yield "start", p
                elif kind == "end":
                    n_tok_total += payload.get("tokens") or 0
                    applied = payload.get("loras", applied)
                    if payload.get("prompt_unc"):
                        prompt_unc = payload["prompt_unc"]
                    levers_payload = payload.get("levers", levers_payload)
                    # keep the last part's tail; it is what continues the voice
                    tail = payload.get("tail_codes", tail)
        gpu_ms = (time.time() - t0) * 1000
        dur = total_samples / self.sr
        yield "end", {"gpu_total_ms": round(gpu_ms, 1), "audio_sec": round(dur, 3),
                      "tokens": n_tok_total, "parts": len(parts), "loras": applied,
                      "tail_codes": tail, "levers": levers_payload,
                      # the neutralised branch, when guidance ran.  stream_pcm puts it
                      # in its own end payload; without forwarding it here the turn
                      # path never sees it and the documented field is unreachable.
                      "prompt_unc": prompt_unc,
                      "rtf": round((gpu_ms / 1000) / dur, 3) if dur > 0 else None}

    @torch.inference_mode()
    def generate_batch(self, items, lora_specs=None, **over):
        """Generate several takes in one forward pass — no streaming.

        The streaming path is batch 1 by necessity: audio has to start playing
        before the line is finished.  A sweep does not need that, and running
        ten utterances together turns ten launch-bound decodes into one.  The
        generation loop was already written with a batch dimension, so this only
        has to build a padded batch and cut the results apart again.

        `items`: dicts with text, instruction, language, tokens, ref_codes, and
        optionally instruction_unc / text_unc.  When `guidance` > 1 and every item
        carries a neutralised prompt, the batch runs the two-branch guided loop —
        both branches padded to the same width, which is why the neutralised half
        is built as its own batch rather than per item.

        Returns a list of float32 arrays at self.sr.
        """
        p = dict(config.DEFAULTS)
        p.update({k: v for k, v in over.items() if v is not None})
        # Best-of-N passes the same item N times.  Sampling is seeded once for
        # the whole batch, so without a per-row offset every candidate would be
        # the same take and the ranking would be choosing between copies.  The
        # loop draws one multinomial per row from a single generator state, so
        # the rows differ as long as the batch is sampled at all -- but the seed
        # is what makes a *rerun* differ, and identical items make that visible.
        p["seed_per_item"] = bool(over.get("seed_per_item"))
        with self.lock:
            applied = []
            if self.lora is not None:
                try:
                    applied = self.lora.apply(lora_specs)
                except Exception as e:
                    print(f"[tts] lora apply failed: {e}", flush=True)
                    self.lora.clear()
            try:
                msgs = []
                for it in items:
                    kw = dict(text=it["text"], instruction=it.get("instruction", ""),
                              language=it.get("language", "English"),
                              tokens=int(it["tokens"]))
                    refs = it.get("ref_codes")
                    if refs is not None and not isinstance(refs, (list, tuple)):
                        refs = [refs]
                    if refs is not None and len(refs):
                        kw["reference"] = [r.to(torch.long) for r in refs if r is not None]
                    msgs.append([self.proc.build_user_message(**kw)])
                torch.manual_seed(int(p["seed"] or 0))
                batch = self.proc(msgs, mode="generation")   # left-padded
                iid = batch["input_ids"].to(self.device)
                am = batch["attention_mask"].to(self.device)
                g = float(over.get("guidance") or 1.0)
                branch1 = None
                if g > 1.0001 and all(it.get("instruction_unc") for it in items):
                    umsgs = []
                    for it in items:
                        kw = dict(text=it.get("text_unc") or it["text"],
                                  instruction=it["instruction_unc"],
                                  language=it.get("language", "English"),
                                  tokens=int(it["tokens"]))
                        refs = it.get("ref_codes")
                        if refs is not None and not isinstance(refs, (list, tuple)):
                            refs = [refs]
                        if refs is not None and len(refs):
                            kw["reference"] = [r.to(torch.long) for r in refs
                                               if r is not None]
                        umsgs.append([self.proc.build_user_message(**kw)])
                    torch.manual_seed(int(p["seed"] or 0))
                    ub = self.proc(umsgs, mode="generation")
                    branch1 = (ub["input_ids"].to(self.device),
                               ub["attention_mask"].to(self.device))
                toks = [int(it["tokens"]) for it in items]
                p["max_new_tokens"] = int(max(toks) * config.TOKEN_HEADROOM) + 64
                minf = torch.tensor(
                    [int(t * config.MIN_FRAME_FRACTION) for t in toks],
                    device=self.device)
                frames = None
                if branch1 is not None:
                    for fr in self._stream_frames_cfg(
                            [(iid, am), branch1], p, chunk_frames=10 ** 9,
                            min_frames=minf, guidance=g):
                        frames = fr
                else:
                    for fr in self._stream_frames(iid, am, p, chunk_frames=10 ** 9,
                                                  min_frames=minf):
                        frames = fr
                if not frames:
                    return [np.zeros(0, np.float32) for _ in items]
                allf = torch.stack(frames, dim=1)      # (B, T, n_vq)
                pad = int(self.model.config.audio_pad_token_id)
                out = []
                for bi in range(allf.shape[0]):
                    c = allf[bi]
                    live = (c != pad).any(dim=-1)
                    n = int(live.nonzero()[-1].item()) + 1 if bool(live.any()) else 0
                    if n <= 0:
                        out.append(np.zeros(0, np.float32))
                        continue
                    w = self.proc.decode_audio_codes([c[:n]], return_stereo=False)[0]
                    out.append(w.reshape(-1).float().cpu().numpy())
                return out
            finally:
                if self.lora is not None:
                    self.lora.unapply()

    def warmup(self):
        try:
            t0 = time.time()
            for kind, _ in self.stream_pcm(
                    "Hello there, this is a warm up.",
                    "GENERAL: A calm friendly voice. Pristine studio recording.",
                    max_new_tokens=60, chunk_frames=24):
                pass
            print(f"[tts] warmup done in {time.time()-t0:.1f}s", flush=True)
        except Exception as e:
            print(f"[tts] warmup skipped: {e}", flush=True)
