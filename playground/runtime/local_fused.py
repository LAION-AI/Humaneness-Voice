"""Opt-in Triton path for the 12 codebook steps of one audio frame (HVS_LOCAL_TRITON=1).

Replaces FastGenerator.local_frame (1-layer local transformer with a 13-slot
static KV cache, binary stop head, 12 audio heads with CFG, top-k/top-p
sampling).  Per codebook step t:

  1. LayerNorm 1 + c_attn GEMV (+bias)                       -- step 0 only, see table below
  2. RoPE + KV-cache write + attention over <=13 cached positions (one program per row/head)
  3. c_proj GEMV + bias + residual
  4. LayerNorm 2 + fc_in GEMV + bias + SiLU
  5. fc_out GEMV + bias + residual
  6. final LayerNorm + audio head GEMV + CFG mix (fp32 logits)
  7. sampling: temperature, top-k, top-p, inverse-CDF draw from one torch.rand
     uniform (+ the binary stop head on step 0)
=> 7 + 11*6 = 73 kernels per frame (+1 torch.rand), versus hundreds before.

Why it is faster: the frame is weight-bandwidth bound (hidden 2560, MLP 9728:
~157 MB of bf16 weights per step, x12 = 1.9 GB per frame).
  * Hand-written batch-2 GEMVs stream weights at ~230 GB/s on Thor (cuBLAS
    picks kernels reaching ~75-130 GB/s for these M=2 shapes).
  * qkv table: steps t >= 1 feed emb_{t-1}[code] into the transformer, and
    LN1 + c_attn of an embedding row does not depend on anything else, so
    LN1(emb_c) @ Wqkv^T + b is precomputed once for all 11 x 1024 codes
    (173 MB, built in <1 s at init).  This removes 39 MB of weight reads from
    11 of 12 steps (-25 % bytes).  HVS_LOCAL_TRITON_TABLE=0 disables it.
  * c_proj is loaded with evict_last and the other weights with evict_first,
    so c_proj tends to stay in the 32 MB L2 between steps (~4 %).

Numerics: same bf16 weights; every point where the reference rounds to bf16
(LayerNorm outputs, linear outputs, RoPE output, attention output, residual
sums, logits) is rounded to bf16 here too; accumulation is fp32.  Sampling
keeps the distribution of performance_core.fast_sample (top-k first, then
top-p over the k survivors, softmax of logits/temperature; ties broken
towards the smaller index like torch.topk); only the random stream differs.
Static shapes, no host syncs: CUDA-graph capturable.  Supports n == 1 with
nb in (1, 2) (the live path); other shapes/parameters fall back to the
original local_frame.  The table assumes the local weights are not modified
after FastGenerator init (true for the engine's merged bf16 weights).
"""
import math
import os

import torch
import triton
import triton.language as tl


# ----------------------------------------------------------------------------- kernels

@triton.jit
def _stats(x_ptr, eps, K: tl.constexpr, KP: tl.constexpr):
    """Mean and rstd of one row of length K (two-pass, fp32)."""
    rk = tl.arange(0, KP)
    km = rk < K
    x = tl.load(x_ptr + rk, mask=km, other=0.0).to(tl.float32)
    mean = tl.sum(x, 0) / K
    d = tl.where(km, x - mean, 0.0)
    return mean, 1.0 / tl.sqrt(tl.sum(d * d, 0) / K + eps)


@triton.jit
def _ln_chunk(x_ptr, rk, mean, rstd, g, b):
    x = tl.load(x_ptr + rk).to(tl.float32)
    return ((x - mean) * rstd * g + b).to(tl.bfloat16).to(tl.float32)


@triton.jit
def _ln_gemv_kernel(x_ptr, emb_ptr, code_ptr, code_col,
                    lnw_ptr, lnb_ptr, w_ptr, b_ptr, y_ptr, res_ptr, N, eps,
                    K: tl.constexpr, KP: tl.constexpr, TWO: tl.constexpr, BN: tl.constexpr,
                    BK: tl.constexpr, GATHER: tl.constexpr, WRITE_RES: tl.constexpr, EV: tl.constexpr,
                    ACT: tl.constexpr):
    """y[rows,N] = act(LN(x) @ W^T + b), bf16 out, 1 or 2 rows (TWO).
    GATHER: both rows are emb[code[0, code_col]] (CFG branches share the code)."""
    pid = tl.program_id(0)
    if GATHER:
        code = tl.load(code_ptr + code_col)
        x0p = emb_ptr + code * K
        x1p = x0p
    else:
        x0p = x_ptr
        x1p = x_ptr + K
    m0, r0 = _stats(x0p, eps, K, KP)
    if TWO:
        m1, r1 = _stats(x1p, eps, K, KP)
    if WRITE_RES:
        if pid == 0:  # residual stream starts as the raw input row(s)
            rk = tl.arange(0, KP)
            km = rk < K
            tl.store(res_ptr + rk, tl.load(x0p + rk, mask=km), mask=km)
            if TWO:
                tl.store(res_ptr + K + rk, tl.load(x1p + rk, mask=km), mask=km)
    rn = pid * BN + tl.arange(0, BN)
    acc0 = tl.zeros([BN], tl.float32)
    acc1 = tl.zeros([BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], eviction_policy=EV).to(tl.float32)
        g = tl.load(lnw_ptr + rk).to(tl.float32)
        bb = tl.load(lnb_ptr + rk).to(tl.float32)
        acc0 += tl.sum(w * _ln_chunk(x0p, rk, m0, r0, g, bb)[None, :], 1)
        if TWO:
            acc1 += tl.sum(w * _ln_chunk(x1p, rk, m1, r1, g, bb)[None, :], 1)
    bias = tl.load(b_ptr + rn).to(tl.float32)
    y0 = (acc0 + bias).to(tl.bfloat16)
    if ACT == 1:  # SiLU on the bf16 linear output, like nn.SiLU on a bf16 tensor
        f = y0.to(tl.float32)
        y0 = (f / (1.0 + tl.exp(-f))).to(tl.bfloat16)
    tl.store(y_ptr + rn, y0)
    if TWO:
        y1 = (acc1 + bias).to(tl.bfloat16)
        if ACT == 1:
            f = y1.to(tl.float32)
            y1 = (f / (1.0 + tl.exp(-f))).to(tl.bfloat16)
        tl.store(y_ptr + N + rn, y1)


@triton.jit
def _gemv_res_kernel(a_ptr, w_ptr, b_ptr, h_ptr, N,
                     K: tl.constexpr, TWO: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
                     EV: tl.constexpr):
    """h[rows,N] = h + bf16(a @ W^T + b)   (in place, bf16)."""
    pid = tl.program_id(0)
    rn = pid * BN + tl.arange(0, BN)
    acc0 = tl.zeros([BN], tl.float32)
    acc1 = tl.zeros([BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], eviction_policy=EV).to(tl.float32)
        acc0 += tl.sum(w * tl.load(a_ptr + rk).to(tl.float32)[None, :], 1)
        if TWO:
            acc1 += tl.sum(w * tl.load(a_ptr + K + rk).to(tl.float32)[None, :], 1)
    bias = tl.load(b_ptr + rn).to(tl.float32)
    o0 = (acc0 + bias).to(tl.bfloat16).to(tl.float32)
    tl.store(h_ptr + rn, (tl.load(h_ptr + rn).to(tl.float32) + o0).to(tl.bfloat16))
    if TWO:
        o1 = (acc1 + bias).to(tl.bfloat16).to(tl.float32)
        tl.store(h_ptr + N + rn, (tl.load(h_ptr + N + rn).to(tl.float32) + o1).to(tl.bfloat16))


@triton.jit
def _attn_kernel(qkv_ptr, kc_ptr, vc_ptr, cos_ptr, sin_ptr, o_ptr, t, scale,
                 table_ptr, emb_ptr, code_ptr, h_ptr,
                 H: tl.constexpr, D: tl.constexpr, L: tl.constexpr,
                 HALF_P: tl.constexpr, DP: tl.constexpr, LP: tl.constexpr, GATHER: tl.constexpr):
    """One (row, head): RoPE on q/k at position t, write k/v to the static cache slot t,
    softmax attention over slots 0..t.  Cache layout [rows, L, H, D] (the release's).
    GATHER (steps t>0): q/k/v come from the precomputed table row of the sampled code
    (= LN1(emb) @ Wqkv^T + b), and this program also initialises its slice of the
    residual stream h with the embedding row."""
    r = tl.program_id(0)
    hh = tl.program_id(1)
    E: tl.constexpr = H * D
    half = tl.arange(0, HALF_P)
    hm = half < D // 2
    if GATHER:
        code = tl.load(code_ptr + t - 1)
        base = table_ptr + code * 3 * E + hh * D
        rd0 = tl.arange(0, DP)
        tl.store(h_ptr + r * E + hh * D + rd0, tl.load(emb_ptr + code * E + hh * D + rd0, mask=rd0 < D),
                 mask=rd0 < D)
    else:
        base = qkv_ptr + r * 3 * E + hh * D
    qe = tl.load(base + 2 * half, mask=hm, other=0.0).to(tl.float32)
    qo = tl.load(base + 2 * half + 1, mask=hm, other=0.0).to(tl.float32)
    ke = tl.load(base + E + 2 * half, mask=hm, other=0.0).to(tl.float32)
    ko = tl.load(base + E + 2 * half + 1, mask=hm, other=0.0).to(tl.float32)
    c = tl.load(cos_ptr + t * (D // 2) + half, mask=hm, other=0.0).to(tl.float32)
    s = tl.load(sin_ptr + t * (D // 2) + half, mask=hm, other=0.0).to(tl.float32)
    qe2 = (qe * c - qo * s).to(tl.bfloat16).to(tl.float32)
    qo2 = (qo * c + qe * s).to(tl.bfloat16).to(tl.float32)
    ke2 = (ke * c - ko * s).to(tl.bfloat16)
    ko2 = (ko * c + ke * s).to(tl.bfloat16)
    rd = tl.arange(0, DP)
    dm = rd < D
    v_now = tl.load(base + 2 * E + rd, mask=dm, other=0.0)
    slot = (r * L + t) * E + hh * D
    tl.store(kc_ptr + slot + 2 * half, ke2, mask=hm)
    tl.store(kc_ptr + slot + 2 * half + 1, ko2, mask=hm)
    tl.store(vc_ptr + slot + rd, v_now, mask=dm)
    # earlier slots from the cache, current slot from registers
    rl = tl.arange(0, LP)
    lm = rl < t
    kb = kc_ptr + (r * L + rl[:, None]) * E + hh * D
    kce = tl.load(kb + 2 * half[None, :], mask=lm[:, None] & hm[None, :], other=0.0).to(tl.float32)
    kco = tl.load(kb + 2 * half[None, :] + 1, mask=lm[:, None] & hm[None, :], other=0.0).to(tl.float32)
    sc = tl.sum(kce * qe2[None, :] + kco * qo2[None, :], 1)
    s_now = tl.sum(ke2.to(tl.float32) * qe2 + ko2.to(tl.float32) * qo2, 0)
    sc = tl.where(rl == t, s_now, sc) * scale
    sc = tl.where(rl <= t, sc, -float("inf"))
    p = tl.exp(sc - tl.max(sc, 0))
    p = p / tl.sum(p, 0)
    vb = vc_ptr + (r * L + rl[:, None]) * E + hh * D
    vv = tl.load(vb + rd[None, :], mask=lm[:, None] & dm[None, :], other=0.0).to(tl.float32)
    vv = tl.where((rl == t)[:, None], v_now.to(tl.float32)[None, :], vv)
    o = tl.sum(p[:, None] * vv, 0)
    tl.store(o_ptr + r * E + hh * D + rd, o.to(tl.bfloat16), mask=dm)


@triton.jit
def _head_kernel(h_ptr, lnw_ptr, lnb_ptr, w_ptr, out_ptr, guidance, eps,
                 K: tl.constexpr, KP: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, EV: tl.constexpr,
                 TWO: tl.constexpr):
    """logits[V] (fp32) = CFG(bf16(LN_f(h) @ W^T)); TWO (CFG): neutral + g*(cond-neutral)."""
    pid = tl.program_id(0)
    m0, r0 = _stats(h_ptr, eps, K, KP)
    if TWO:
        m1, r1 = _stats(h_ptr + K, eps, K, KP)
    rn = pid * BN + tl.arange(0, BN)
    acc0 = tl.zeros([BN], tl.float32)
    acc1 = tl.zeros([BN], tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        w = tl.load(w_ptr + rn[:, None] * K + rk[None, :], eviction_policy=EV).to(tl.float32)
        g = tl.load(lnw_ptr + rk).to(tl.float32)
        bb = tl.load(lnb_ptr + rk).to(tl.float32)
        acc0 += tl.sum(w * _ln_chunk(h_ptr, rk, m0, r0, g, bb)[None, :], 1)
        if TWO:
            acc1 += tl.sum(w * _ln_chunk(h_ptr + K, rk, m1, r1, g, bb)[None, :], 1)
    cond = acc0.to(tl.bfloat16).to(tl.float32)
    if TWO:
        neu = acc1.to(tl.bfloat16).to(tl.float32)
        cond = neu + guidance * (cond - neu)
    tl.store(out_ptr + rn, cond)


@triton.jit
def _sortable(x):
    b = x.to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _sample_kernel(logits_ptr, u_ptr, codes_ptr, col, temp, top_k, top_p,
                   h_ptr, lnw_ptr, lnb_ptr, stop_w_ptr, cand_ptr, tok_ptr, eps, text_temp, text_top_p,
                   V: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                   USE_TOP_P: tl.constexpr, STOP: tl.constexpr, STOP_MODE: tl.constexpr,
                   STOP_TOP_P: tl.constexpr):
    """Row i: sample one code from softmax(logits/T) restricted to top-k, then top-p
    (token kept iff the probability mass ranked strictly before it is <= top_p,
    i.e. performance_core.fast_sample).  Inverse CDF with one uniform."""
    i = tl.program_id(0)
    rv = tl.arange(0, V)
    x = tl.load(logits_ptr + i * V + rv) / temp
    # descending sort of (value, -index): ties -> smaller index first, like torch.topk
    key = (_sortable(x).to(tl.int64) << 32) | (V - 1 - rv).to(tl.int64)
    key = tl.sort(key, 0, descending=True)
    idx = V - 1 - (key & 0xFFFFFFFF).to(tl.int32)
    vb = (key >> 32).to(tl.int32)
    vb = vb ^ ((vb >> 31) & 0x7FFFFFFF)
    v = vb.to(tl.float32, bitcast=True)
    keep = rv < top_k
    m = tl.max(tl.where(keep, v, -float("inf")), 0)
    e = tl.where(keep, tl.exp(v - m), 0.0)
    if USE_TOP_P:
        pr = e / tl.sum(e, 0)
        before = tl.cumsum(pr, 0) - pr
        keep = keep & ((before <= top_p) | (rv == 0))
        e = tl.where(keep, e, 0.0)
    cw = tl.cumsum(e, 0)
    u = tl.load(u_ptr + i * 13 + col)
    target = u * tl.sum(e, 0)
    nkeep = tl.sum(keep.to(tl.int32), 0)
    sel = tl.minimum(tl.sum((keep & (cw <= target)).to(tl.int32), 0), nkeep - 1)
    code = tl.sum(tl.where(rv == sel, idx, 0), 0)
    tl.store(codes_ptr + i * 12 + col, code.to(tl.int64))
    if STOP:
        # binary stop head on the frame's first hidden state (rows < n)
        rk = tl.arange(0, KP)
        km = rk < K
        hx = tl.load(h_ptr + i * K + rk, mask=km, other=0.0).to(tl.float32)
        mean = tl.sum(hx, 0) / K
        d = tl.where(km, hx - mean, 0.0)
        rstd = 1.0 / tl.sqrt(tl.sum(d * d, 0) / K + eps)
        g = tl.load(lnw_ptr + rk, mask=km, other=0.0).to(tl.float32)
        bb = tl.load(lnb_ptr + rk, mask=km, other=0.0).to(tl.float32)
        hn = (d * rstd * g + bb).to(tl.bfloat16).to(tl.float32)
        w0 = tl.load(stop_w_ptr + rk, mask=km, other=0.0).to(tl.float32)
        w1 = tl.load(stop_w_ptr + K + rk, mask=km, other=0.0).to(tl.float32)
        l0 = tl.sum(hn * w0, 0).to(tl.bfloat16).to(tl.float32)
        l1 = tl.sum(hn * w1, 0).to(tl.bfloat16).to(tl.float32)
        if STOP_MODE == 0:  # argmax (do_sample False or top_k == 1); first index on ties
            pick = (l1 > l0).to(tl.int32)
        else:
            a0 = l0 / text_temp
            a1 = l1 / text_temp
            mx = tl.maximum(a0, a1)
            e0 = tl.exp(a0 - mx)
            e1 = tl.exp(a1 - mx)
            p1 = e1 / (e0 + e1)
            if STOP_TOP_P:
                # sorted desc: the second token is dropped iff P(first) > top_p
                big = tl.maximum(p1, 1.0 - p1)
                if big > text_top_p:
                    p1 = (l1 > l0).to(tl.float32)
            us = tl.load(u_ptr + i * 13 + 12)
            pick = (us < p1).to(tl.int32)
        tok = tl.load(cand_ptr + pick)
        tl.store(tok_ptr + i, tok)


# ----------------------------------------------------------------------------- module

def _np2(x):
    return 1 << (int(x) - 1).bit_length()


class FusedLocalFrame:
    """Drop-in for FastGenerator.local_frame (same signature and outputs)."""

    def __init__(self, mdl, candidate_ids, fallback=None, use_table=True):
        self.mdl = mdl
        self.fallback = fallback
        self.cand = candidate_ids
        lt = mdl.local_transformer
        self.ok = (len(lt.h) == 1 and mdl._use_binary_local_text_head()
                   and str(getattr(lt.h[0].attn, 'position_embedding_type', '')) == 'rope'
                   and not lt.h[0].attn.scale_attn_by_inverse_layer_idx
                   and lt.h[0].attn.scale_attn_weights
                   and type(lt.h[0].mlp.act).__name__ in ('SiLUActivation', 'SiLU')
                   and int(mdl.config.n_vq) == 12
                   and all(int(s) == 1024 for s in mdl.config.audio_codebook_sizes))
        blk = lt.h[0]
        a = blk.attn
        self.H, self.D, self.E = a.num_heads, a.head_dim, a.embed_dim
        self.inner = blk.mlp.fc_in.out_features
        self.eps = float(blk.ln_1.eps)
        c = lambda t: t.detach().contiguous()
        self.ln1 = (c(blk.ln_1.weight), c(blk.ln_1.bias))
        self.ln2 = (c(blk.ln_2.weight), c(blk.ln_2.bias))
        self.lnf = (c(lt.ln_f.weight), c(lt.ln_f.bias))
        self.qkv = (c(a.c_attn.weight), c(a.c_attn.bias))
        self.proj = (c(a.c_proj.weight), c(a.c_proj.bias))
        self.fin = (c(blk.mlp.fc_in.weight), c(blk.mlp.fc_in.bias))
        self.fout = (c(blk.mlp.fc_out.weight), c(blk.mlp.fc_out.bias))
        self.heads = [c(h.weight) for h in mdl.audio_lm_heads]
        self.embs = [c(e.weight) for e in mdl.audio_embeddings]
        self.stop_w = c(mdl.local_text_lm_head.weight)
        self.ok = self.ok and all(t.dtype == torch.bfloat16 for t in
                                  (self.qkv[0], self.fin[0], self.heads[0], self.embs[0], self.stop_w))
        # RoPE table from the release's own module (cos/sin cast to bf16 like the reference)
        dev = self.qkv[0].device
        pos = torch.arange(13, device=dev).unsqueeze(0)
        cos, sin = a.rotary_emb(pos, device=dev, dtype=torch.bfloat16)  # [1,13,1,D] interleaved pairs
        self.cos = cos[0, :, 0, 0::2].contiguous()
        self.sin = sin[0, :, 0, 0::2].contiguous()
        self.scale = 1.0 / math.sqrt(self.D)
        # GEMV tiling per weight (BN, BK, num_warps, num_stages); tuned on Jetson Thor (20 SMs)
        self.cfg = dict(qkv=(16, 512, 4, 2), proj=(4, 256, 2, 3), fin=(8, 512, 2, 2),
                        fout=(8, 256, 4, 3), head=(8, 512, 2, 2))
        # L2 policy: stream the big weights with evict_first so c_proj (13 MB) tends to
        # stay in the 32 MB L2 across the 12 steps (~4 % faster on Thor)
        self.evict = dict(qkv='evict_first', proj='evict_last', fin='evict_first',
                          fout='evict_first', head='evict_first')
        self.record = None
        self.table = None
        if self.ok and use_table:
            self.build_table()

    @torch.no_grad()
    def build_table(self):
        """qkv rows for every (codebook c < 11, code): LN1(emb_c[code]) @ Wqkv^T + b, bf16.
        Steps t >= 1 feed emb_{t-1}[code] into the local transformer, so their c_attn GEMV
        (39 MB of weights per step) becomes a 15 KB gather.  Same bf16 ops as the
        reference (F.layer_norm + F.linear on bf16).  11 x 1024 x 7680 bf16 = 173 MB.
        Assumes frozen local weights (true after the engine merged its RL adapter)."""
        E = self.E
        tab = torch.empty(11, 1024, 3 * E, device=self.qkv[0].device, dtype=torch.bfloat16)
        for c in range(11):
            a = torch.nn.functional.layer_norm(self.embs[c], (E,), self.ln1[0], self.ln1[1], self.eps)
            tab[c] = torch.nn.functional.linear(a, self.qkv[0], self.qkv[1])
        self.table = tab

    def supported(self, local_input, n, nb, p, cache):
        M = n * nb
        return (self.ok and n == 1 and nb in (1, 2) and local_input.dtype == torch.bfloat16
                and local_input.is_cuda and tuple(local_input.shape) == (M, self.E)
                and local_input.is_contiguous()
                and len(cache) == 1 and tuple(cache[0]['key'].shape) == (M, 13, self.H, self.D)
                and cache[0]['key'].dtype == torch.bfloat16
                and float(p['audio_temperature']) > 0
                and p.get('audio_repetition_penalty', 1.0) == 1.0)

    def __call__(self, local_input, n, nb, guidance, p, cache):
        if not self.supported(local_input, n, nb, p, cache):
            if self.fallback is None:
                raise ValueError('HVS_LOCAL_TRITON: unsupported local frame configuration')
            return self.fallback(local_input, n, nb, guidance, p, cache)
        M, E, K, I = nb, self.E, self.E, self.inner
        two = nb == 2
        dev = local_input.device
        x0 = local_input
        h = torch.empty(M, E, device=dev, dtype=torch.bfloat16)      # residual stream
        qkv = torch.empty(M, 3 * E, device=dev, dtype=torch.bfloat16)
        att = torch.empty(M, E, device=dev, dtype=torch.bfloat16)
        f = torch.empty(M, I, device=dev, dtype=torch.bfloat16)
        logits = torch.empty(1, 1024, device=dev, dtype=torch.float32)
        codes = torch.empty(1, 12, device=dev, dtype=torch.long)
        token = torch.empty(1, device=dev, dtype=torch.long)
        u = torch.rand(1, 13, device=dev, dtype=torch.float32)  # 12 codes + stop
        kc, vc = cache[0]['key'], cache[0]['value']
        cache[0]['length'] = 13
        top_k = p['audio_top_k']
        top_k = 1024 if not top_k or int(top_k) >= 1024 else int(top_k)
        top_p = p['audio_top_p']
        use_top_p = top_p is not None and 0.0 < float(top_p) < 1.0
        tt, tk, tp = float(p['text_temperature']), p['text_top_k'], p['text_top_p']
        stop_mode = 0 if (tt <= 0 or (tk is not None and int(tk) == 1)) else 1
        stop_top_p = tp is not None and 0.0 < float(tp) < 1.0
        c = self.cfg
        KP = _np2(K)
        for t in range(12):
            gather = t > 0 and self.table is not None
            if not gather:
                emb = self.embs[t - 1] if t > 0 else x0
                BN, BK, nw, ns = c['qkv']
                _ln_gemv_kernel[(3 * E // BN,)](x0, emb, codes, t - 1, self.ln1[0], self.ln1[1],
                    self.qkv[0], self.qkv[1], qkv, h, 3 * E, self.eps,
                    K=K, KP=KP, TWO=two, BN=BN, BK=BK, GATHER=t > 0, WRITE_RES=True, ACT=0, EV=self.evict['qkv'],
                    num_warps=nw, num_stages=ns)
            _attn_kernel[(M, self.H)](qkv, kc, vc, self.cos, self.sin, att, t, self.scale,
                self.table[t - 1] if gather else qkv, self.embs[t - 1] if gather else x0, codes, h,
                H=self.H, D=self.D, L=13, HALF_P=_np2(self.D // 2), DP=_np2(self.D), LP=16,
                GATHER=gather, num_warps=1)
            BN, BK, nw, ns = c['proj']
            _gemv_res_kernel[(E // BN,)](att, self.proj[0], self.proj[1], h, E,
                K=E, TWO=two, BN=BN, BK=BK, EV=self.evict['proj'], num_warps=nw, num_stages=ns)
            BN, BK, nw, ns = c['fin']
            _ln_gemv_kernel[(I // BN,)](h, h, codes, 0, self.ln2[0], self.ln2[1],
                self.fin[0], self.fin[1], f, h, I, self.eps,
                K=K, KP=KP, TWO=two, BN=BN, BK=BK, GATHER=False, WRITE_RES=False, ACT=1, EV=self.evict['fin'],
                num_warps=nw, num_stages=ns)
            BN, BK, nw, ns = c['fout']
            _gemv_res_kernel[(E // BN,)](f, self.fout[0], self.fout[1], h, E,
                K=I, TWO=two, BN=BN, BK=BK, EV=self.evict['fout'], num_warps=nw, num_stages=ns)
            BN, BK, nw, ns = c['head']
            _head_kernel[(1024 // BN,)](h, self.lnf[0], self.lnf[1], self.heads[t], logits,
                float(guidance), self.eps, K=K, KP=KP, BN=BN, BK=BK, TWO=two, EV=self.evict['head'],
                num_warps=nw, num_stages=ns)
            if self.record is not None:  # debugging/validation only (eager)
                self.record.append((h.clone(), logits.clone()))
            _sample_kernel[(1,)](logits, u, codes, t, float(p['audio_temperature']),
                top_k, float(top_p) if use_top_p else 1.0,
                h, self.lnf[0], self.lnf[1], self.stop_w, self.cand, token, self.eps,
                tt if tt > 0 else 1.0, float(tp) if stop_top_p else 1.0,
                V=1024, K=K, KP=KP, USE_TOP_P=use_top_p, STOP=(t == 0), STOP_MODE=stop_mode,
                STOP_TOP_P=stop_top_p, num_warps=8)
        return token, codes


def install(fast_generator):
    """Route fast_generator.local_frame through the Triton path (falls back when unsupported)."""
    fused = FusedLocalFrame(fast_generator.engine.model, fast_generator.candidate_ids,
                            fallback=fast_generator.local_frame,
                            use_table=os.environ.get('HVS_LOCAL_TRITON_TABLE', '1') == '1')
    fast_generator.local_frame = fused
    return fused
