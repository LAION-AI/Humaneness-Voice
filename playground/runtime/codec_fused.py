"""Fused Triton kernels for the MOSS Audio Tokenizer v2 streaming *decoder* (opt-in, HVS_CODEC_FUSED=1).

Portions adapted from kadirnar/fast-moss-tokenizer (https://github.com/kadirnar/fast-moss-tokenizer,
commit 92c0c1e), Copyright kadirnar, licensed under the Apache License, Version 2.0
(http://www.apache.org/licenses/LICENSE-2.0). Adapted ideas: cached immutable weight casts
(here: fp32 LayerNorm affine and weight-norm conv weights), a fused rotary kernel that keeps the
native FP32 roundings (enable_fp_fusion=False), and a fused layer-scale residual add. The hardware
and version guards of the original were removed; everything here is plain Triton (no NVRTC), so it
runs on Jetson Thor (sm_110, CUDA 13, torch 2.14, Triton 3.8). Changes: decode-only, bf16 codec
weights, streaming attention KV-cache/mask/RoPE fused, in-place stable caches for CUDA graphs.

Every kernel reproduces the arithmetic of the pinned modeling code (revision f6e20e5) operation by
operation (same FP32 roundings, same bf16 casts, same SDPA call with the same key order), so the
waveforms are expected to be bit-identical to the unfused (graphed) decode. Unsupported cases fall
back to the original forwards.

Use:
    import codec_fused
    codec_fused.install(codec, states)   # once, before CUDA-graph capture (idempotent)
    codec_fused.uninstall(codec)          # restores the original forwards
"""
import math
import os
from types import MethodType

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from triton.language.extra import libdevice

ENABLED = os.environ.get('HVS_CODEC_FUSED', '0') == '1'


# ----------------------------------------------------------------------------------------------
# Kernels
# ----------------------------------------------------------------------------------------------
@triton.jit
def _prep_kernel(P, CK, CV, Freq, Off, Q, KA, VA, n_rows, T, CTX,
                 H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr):
    """q = rope(q_cur); k_all = cat(cache_k, rope(k_cur)); v_all = cat(cache_v, v_cur).

    P is the in_proj output (B, T, 3, H, D) bf16. RoPE follows apply_rope(): fp32 phase
    freqs*(offset+t), libdevice cos/sin (== torch's cosf/sinf), separately rounded fp32 products
    (no FMA), one bf16 cast. Even lane: x*c - x'*s; odd lane: x'*s + x*c (x' = pair partner).
    """
    L = CTX + T
    r = tl.program_id(0) * ROWS + tl.arange(0, ROWS)[:, None]
    d = tl.arange(0, D)[None, :]
    valid = r < n_rows
    j = r % L
    bh = r // L
    h = bh % H
    b = bh // H
    cur = valid & (j >= CTX)
    old = valid & (j < CTX)
    t = j - CTX
    cbase = (bh * CTX + j) * D
    ck = tl.load(CK + cbase + d, old, 0.)
    cv = tl.load(CV + cbase + d, old, 0.)
    E = H * D
    pbase = (b * T + t) * 3 * E + h * D
    partner = d ^ 1
    q = tl.load(P + pbase + d, cur, 0.).to(tl.float32)
    qp = tl.load(P + pbase + partner, cur, 0.).to(tl.float32)
    k = tl.load(P + pbase + E + d, cur, 0.).to(tl.float32)
    kp = tl.load(P + pbase + E + partner, cur, 0.).to(tl.float32)
    v = tl.load(P + pbase + 2 * E + d, cur, 0.)
    off = tl.load(Off + b, valid, 0).to(tl.float32)
    phase = tl.load(Freq + d // 2) * (off + t.to(tl.float32))
    c = libdevice.cos(phase)
    s = libdevice.sin(phase)
    even = (d % 2) == 0
    qo = tl.where(even, q * c - qp * s, qp * s + q * c)
    ko = tl.where(even, k * c - kp * s, kp * s + k * c)
    ty = KA.dtype.element_ty
    abase = (bh * L + j) * D + d
    tl.store(KA + abase, tl.where(cur, ko.to(ty), ck), valid)
    tl.store(VA + abase, tl.where(cur, v, cv), valid)
    tl.store(Q + (bh * T + t) * D + d, qo.to(ty), cur)


@triton.jit
def _bias_kernel(CP, Off, POSK, BIAS, T, CTX, LD, BLOCK: tl.constexpr):
    """Additive SDPA bias == torch's conversion of _build_streaming_sdpa_bias's bool mask.

    Grid (B*T, cdiv(L, BLOCK)). Row tq==0 also materializes pos_k = cat(cached_pos, pos_q).
    """
    row = tl.program_id(0)
    b = row // T
    tq = row % T
    L = CTX + T
    i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    m = i < L
    off = tl.load(Off + b)
    old = tl.load(CP + b * CTX + i, m & (i < CTX), -1)
    posk = tl.where(i < CTX, old, off + (i - CTX))
    posq = off + tq
    delta = posq - posk
    ok = (posk >= 0) & (delta >= 0) & (delta < CTX)
    val = tl.where(ok, 0.0, float('-inf'))
    tl.store(BIAS + row * LD + i, val.to(BIAS.dtype.element_ty), m)
    tl.store(POSK + b * L + i, posk, m & (tq == 0))


@triton.jit
def _update_kernel(KA, VA, POSK, CK, CV, CP, Off, Exec, B, T, CTX, n_kv,
                   H: tl.constexpr, D: tl.constexpr, ROWS: tl.constexpr, PBLOCK: tl.constexpr):
    """caches = where(exec, (k_all, v_all, pos_k)[..., -CTX:], caches) in place; offset += T."""
    pid = tl.program_id(0)
    L = CTX + T
    n_kv_progs = tl.cdiv(n_kv, ROWS)
    if pid < n_kv_progs:
        r = pid * ROWS + tl.arange(0, ROWS)[:, None]
        d = tl.arange(0, D)[None, :]
        valid = r < n_kv
        j = r % CTX
        bh = r // CTX
        b = bh // H
        ex = tl.load(Exec + b, valid, 0) != 0
        w = valid & ex
        src = (bh * L + T + j) * D + d
        dst = (bh * CTX + j) * D + d
        tl.store(CK + dst, tl.load(KA + src, w, 0.), w)
        tl.store(CV + dst, tl.load(VA + src, w, 0.), w)
    else:
        q = pid - n_kv_progs
        i = q * PBLOCK + tl.arange(0, PBLOCK)
        pvalid = i < B * CTX
        pb = i // CTX
        pj = i % CTX
        pex = tl.load(Exec + pb, pvalid, 0) != 0
        pw = pvalid & pex
        tl.store(CP + pb * CTX + pj, tl.load(POSK + pb * L + T + pj, pw, 0), pw)
        if q == 0:
            bb = tl.arange(0, PBLOCK)
            mb = bb < B
            exb = tl.load(Exec + bb, mb, 0) != 0
            o = tl.load(Off + bb, mb, 0)
            tl.store(Off + bb, tl.where(exb, o + T, o), mb)


@triton.jit
def _scale_add_kernel(X, U, S, Y, N, C, ROUND_BF16: tl.constexpr, BLOCK: tl.constexpr):
    """y = residual.float() + (scale * update) with the bf16 rounding of torch's bf16 mul."""
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < N
    x = tl.load(X + i, m, 0.).to(tl.float32)
    u = tl.load(U + i, m, 0.).to(tl.float32)
    s = tl.load(S + i % C, m, 0.).to(tl.float32)
    prod = s * u
    if ROUND_BF16:
        prod = prod.to(tl.bfloat16).to(tl.float32)
    tl.store(Y + i, x + prod, m)


def scale_add(x, update, scale):
    out = torch.empty(x.shape, device=x.device, dtype=torch.float32)
    n = x.numel()
    rb = torch.promote_types(scale.dtype, update.dtype) == torch.bfloat16
    _scale_add_kernel[(triton.cdiv(n, 1024),)](x, update, scale, out, n, x.shape[-1], rb, 1024,
                                              num_warps=4, enable_fp_fusion=False)
    return out


# ----------------------------------------------------------------------------------------------
# Patched forwards
# ----------------------------------------------------------------------------------------------
def _attn_forward(original):
    def forward(self, query, cu_seqlens=None, max_seqlen=None, position_ids=None, input_lengths=None):
        state = self._streaming_state
        if (state is None or query.dim() != 3 or self.context is None or self.rope is None
                or self.resolve_attention_implementation(query, True) != 'sdpa'
                or torch.is_grad_enabled() or not query.is_cuda):
            return original(query, cu_seqlens, max_seqlen, position_ids, input_lengths)
        B, T, _ = query.shape
        H = self.num_heads
        D = self.embed_dim // H
        if D & (D - 1) or D % 2:
            return original(query, cu_seqlens, max_seqlen, position_ids, input_lengths)
        projected = self.in_proj(query)
        if projected.dtype not in (torch.bfloat16, torch.float16) or not projected.is_contiguous():
            return original(query, cu_seqlens, max_seqlen, position_ids, input_lengths)
        ck, cv, cp = self._ensure_streaming_cache(state, B, projected.device, projected.dtype)
        CTX = self.context
        L = CTX + T
        freq = self._fused_freqs.get(D)
        if freq is None:
            # identical to apply_rope(): exp(arange(D//2) * (-log(max_period) * 2 / D)) in fp32
            ds = torch.arange(D // 2, device=query.device, dtype=torch.float32)
            freq = self._fused_freqs[D] = torch.exp(ds * (-math.log(self.rope.max_period) * 2 / D))
        q = torch.empty((B, H, T, D), device=query.device, dtype=projected.dtype)
        k_all = torch.empty((B, H, L, D), device=query.device, dtype=projected.dtype)
        v_all = torch.empty_like(k_all)
        n_rows = B * H * L
        ROWS = 16
        _prep_kernel[(triton.cdiv(n_rows, ROWS),)](
            projected, ck, cv, freq, state.offset, q, k_all, v_all, n_rows, T, CTX,
            H, D, ROWS, num_warps=4, enable_fp_fusion=False)
        # Additive bias laid out exactly like torch's mem-efficient preprocessing (16-aligned rows).
        LD = (L + 15) // 16 * 16
        bias = torch.empty((B, 1, T, LD), device=query.device, dtype=projected.dtype)[..., :L]
        pos_k = torch.empty((B, L), device=query.device, dtype=torch.long)
        _bias_kernel[(B * T, triton.cdiv(L, 1024))](cp, state.offset, pos_k, bias, T, CTX, LD, 1024,
                                                    num_warps=4)
        out = F.scaled_dot_product_attention(q, k_all, v_all, bias, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, T, self.embed_dim)
        n_kv = B * H * CTX
        PBLOCK = 1024
        grid = triton.cdiv(n_kv, 8) + triton.cdiv(B * CTX, PBLOCK)
        _update_kernel[(grid,)](k_all, v_all, pos_k, ck, cv, cp, state.offset, state.exec_mask,
                                B, T, CTX, n_kv, H, D, 8, PBLOCK, num_warps=4)
        return self.out_proj(out)
    return forward


def _norm_forward(original, weight, bias):
    """LayerNorm with the fp32 affine cast cached (autocast would cast the bf16 params every call)."""
    params = (original.__self__.weight, original.__self__.bias)
    versions = tuple(p._version if p is not None else None for p in params)

    def forward(self, x):
        if (self.weight is not params[0] or self.bias is not params[1]
                or tuple(p._version if p is not None else None for p in (self.weight, self.bias)) != versions):
            return original(x)
        return F.layer_norm(x, self.normalized_shape, weight, bias, self.eps)
    return forward


def _layer_forward(original):
    def forward(self, x, **kwargs):
        s1, s2 = self.layer_scale_1, self.layer_scale_2
        if (torch.is_grad_enabled() or not x.is_cuda or not hasattr(s1, 'scale') or not hasattr(s2, 'scale')
                or not s1.channel_last or not s2.channel_last):
            return original(x, **kwargs)
        for norm, block, scale in ((self.norm1, self.self_attn, s1.scale), (self.norm2, self.ffn, s2.scale)):
            normed = norm(x)
            update = block(normed, **kwargs) if block is self.self_attn else block(normed)
            if (normed.dtype == torch.float32 and update.shape == x.shape and x.is_contiguous()
                    and update.is_contiguous() and scale.dtype in (torch.bfloat16, torch.float32)
                    and update.dtype in (torch.bfloat16, torch.float32)):
                x = scale_add(x, update, scale)
            else:
                x = x.to(normed) + scale * update
        return x
    return forward


def _wn_conv_forward(original, weight):
    """Weight-norm conv with the parametrized weight (g * v / ||v||) computed once."""
    module = original.__self__
    parameters = tuple(module.parametrizations.weight.parameters())
    versions = tuple(p._version for p in parameters)

    def forward(self, x):
        current = tuple(self.parametrizations.weight.parameters())
        if (len(current) != len(parameters) or any(a is not b for a, b in zip(current, parameters))
                or tuple(p._version for p in current) != versions):
            return original(x)
        return self._conv_forward(x, weight, self.bias)
    return forward


def _shared_exec_mask(original, codec):
    def set_mask(self, exec_mask):
        shared = getattr(codec, '_fused_exec_mask', None)
        states = [m._streaming_state for m in codec.modules()
                  if getattr(m, '_streaming_state', None) is not None]
        if shared is not None and states and all(s.exec_mask is shared for s in states):
            shared[:] = exec_mask.to(torch.bool).to(shared.device)
        else:
            original(exec_mask)
    return set_mask


# ----------------------------------------------------------------------------------------------
# Install / uninstall
# ----------------------------------------------------------------------------------------------
def install(codec, states=None):
    """Patch the decoder of `codec` in place (idempotent). `states`: CodecGraphs.states (optional).

    With `states` (all batch-1, persistent), every decoder streaming state shares one exec_mask
    tensor, so _set_streaming_exec_mask is a single copy instead of ~190.
    Call before any CUDA-graph capture; graphs captured before install keep the old kernels.
    """
    saved = getattr(codec, '_fused_saved', None)
    if saved is None:
        saved = []
        _patch(codec, saved)
        codec._fused_saved = saved
    if states:
        shared = states[0].exec_mask
        if all(s.exec_mask.shape == shared.shape and s.exec_mask.device == shared.device for s in states):
            for s in states:
                s.exec_mask = shared
            codec._fused_exec_mask = shared
            if '_set_streaming_exec_mask' not in codec.__dict__:
                saved.append((codec, '_set_streaming_exec_mask', False, None))
                codec._set_streaming_exec_mask = MethodType(
                    _shared_exec_mask(codec._set_streaming_exec_mask, codec), codec)
    return saved


def _patch(codec, saved):
    def replace(module, name, value):
        saved.append((module, name, name in module.__dict__, module.__dict__.get(name)))
        setattr(module, name, value)

    with torch.no_grad():
        for module in codec.decoder.modules():
            kind = type(module).__name__
            if kind == 'MossAudioTokenizerMultiheadAttention':
                module._fused_freqs = {}
                replace(module, 'forward', MethodType(_attn_forward(module.forward), module))
            elif kind == 'MossAudioTokenizerTransformerLayer':
                replace(module, 'forward', MethodType(_layer_forward(module.forward), module))
            elif type(module) is torch.nn.LayerNorm and module.elementwise_affine:
                w = module.weight.detach().float().clone()
                b = module.bias.detach().float().clone() if module.bias is not None else None
                replace(module, 'forward', MethodType(_norm_forward(module.forward, w, b), module))
        # Weight-norm convs of the quantizer decode path (fp32, autocast disabled there).
        for name, module in codec.quantizer.named_modules():
            if (name.rsplit('.', 1)[-1] in ('out_proj', 'output_proj') and isinstance(module, torch.nn.Conv1d)
                    and torch.nn.utils.parametrize.is_parametrized(module, 'weight')):
                with torch.autocast('cuda', enabled=False):
                    w = module.weight.detach().clone()
                replace(module, 'forward', MethodType(_wn_conv_forward(module.forward, w), module))


def uninstall(codec):
    saved = getattr(codec, '_fused_saved', None)
    if saved is None:
        return
    for module, name, existed, old in reversed(saved):
        if existed:
            setattr(module, name, old)
        else:
            delattr(module, name)
    codec._fused_saved = None
    codec.__dict__.pop('_fused_exec_mask', None)
