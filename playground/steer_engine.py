"""Steering vectors: a direction added to the model's own hidden state while it generates.

    h  <-  h + alpha * (v / ||v||) * ||h||

This is the second of the three levers the training side measured (the first is the LoRA
merge weight this server already uses, the third is guidance — see `cfg` in `tts_engine.py`).
The mechanism, the normalisation and the injection points are ported from the research
harness `actf_steer.py` (`Injector`, `build_specs`) rather than re-derived, because the
convention is load-bearing in two places that are easy to get wrong:

**alpha is dimensionless.**  It is the fraction of the CURRENT hidden state's own magnitude
added along `v`, not an absolute step.  Raw difference-vector norms span three orders of
magnitude across the 36 layers (||h|| runs 6.4 at the embeddings to 2237 at layer 35), so a
fixed absolute alpha would be a no-op in one place and a catastrophe in another.

**Only the last position of each forward pass is steered.**  During the prefill that is the
position that emits frame 0; during decoding it is the single new frame slot.  So exactly the
positions whose hidden state produces a generated frame are steered, and the prompt's own
representation is left alone.

WHERE THE VECTORS COME FROM.  A 99-dimension library extracted from this checkpoint's own
activations, one difference-of-means direction per (attribute, layer).  The full research
file is 112 MB and is NOT in this repository; `setup/build_steering_pack.py` distils it to
the ~5 MB the server actually needs — each dimension at its own top-5 layers — and
`config.STEER_PACK` points at the result.  With no pack on disk every steering mode
degrades to `adapter` and says so in the response payload.  See docs/LEVERS.md.

WHICH LAYERS.  Per attribute, not global: Anger is most decodable at h21 h20 h19, genuineness
at h12 h13 h21, blend at h25 h22 h20.  `taps: "top1"` means "that attribute's own best layer".
k is small and per family — emotion is free only at k = 1, the quality axes break at k >= 2,
the delivery axes want 3-5.

COST.  These are forward hooks, and `docs/ADAPTERS.md` §1 records that forward hooks for the
whole adapter stack took the realtime factor from 0.737 to 1.29 and were rejected.  That was
**536 extra kernel launches per token**.  This is one to five, each a norm and a fused
multiply-add on a [1, 1, 2560] slice, which is the regime §2 already accepted for the twelve
tied-module hooks ("twelve hooks is not 536").  It has not been measured on this hardware —
see the smoke test asked for in the pull request.
"""
import json
import os
import threading

import numpy as np
import torch

import config


# --------------------------------------------------------------------------- the arithmetic
def _add(h, v, al):
    """h + alpha * unit(v) * ||h||, computed in fp32 and cast back.

    fp32 deliberately: a steering direction is a *difference* of means, one to two orders of
    magnitude smaller than the means themselves, and doing the accumulation in bf16 quantises
    it to a few significant bits.
    """
    f = h.float()
    n = f.norm(dim=-1, keepdim=True)
    return (f + float(al) * n * v.float()).to(h.dtype)


class Injector:
    """The hooks for one steering configuration.

    `close()` MUST be called or the next generation inherits them — a leaked hook is
    invisible and would make every later take a superposition of the ones before it.  The
    engine calls it in a `finally`.

    Ported from `actf_steer.Injector`.  `specs` is a list of (tap, unit direction, alpha),
    which is what `build_specs` returns.
    """

    def __init__(self, model, specs, n_layers=None):
        self.enabled = True
        self.handles = []
        self.emb = self.final = self.loc = None
        self.taps = sorted({int(t) for t, _v, _a in specs})
        layers = getattr(getattr(model, "transformer", None), "layers", None)
        n_layers = int(n_layers or (len(layers) if layers is not None else 0))
        self.n_layers = n_layers
        for tap, v, al in specs:
            u = v / (v.norm() + 1e-9)
            if tap == 0:
                self.emb = (u, al)
            elif 1 <= tap <= n_layers - 1:
                self.handles.append(
                    layers[tap - 1].register_forward_hook(self._mk(u, al)))
            elif tap == n_layers:
                # the vector that goes into the acoustic decoder
                self.final = (u, al)
            elif tap == 37:
                # the local (talker) transformer's hidden state AT THE FRAME SLOT only, not
                # at the eleven within-frame channel steps: the frame slot is exactly the
                # position tap 37 was measured at, and steering the channel steps as well
                # would be a different intervention from the one the vector was estimated for
                self.loc = (u, al)
            else:
                raise ValueError(f"bad tap {tap} for a {n_layers}-layer stack")

    def _mk(self, u, al):
        def hook(mod, inp, out):
            if not self.enabled or al == 0.0:
                return None
            h = out[0]
            h = h.clone()
            h[:, -1, :] = _add(h[:, -1, :], u, al)
            return (h,) + tuple(out[1:])
        return hook

    def apply_emb(self, e):
        if self.emb is None or not self.enabled:
            return e
        u, al = self.emb
        e = e.clone()
        e[:, -1, :] = _add(e[:, -1, :], u, al)
        return e

    def apply_final(self, g):
        if self.final is None or not self.enabled:
            return g
        u, al = self.final
        return _add(g, u, al)

    def apply_loc(self, l):
        if self.loc is None or not self.enabled:
            return l
        u, al = self.loc
        return _add(l, u, al)

    def close(self):
        for h in self.handles:
            h.remove()
        self.handles = []


class _NullInjector:
    """Does nothing, so the generation loop has no branch in it.

    A zero-strength run through the identical code path is the control the research harness
    relies on, and `comb_recommendations.json` records it holding exactly: 510 cells, maximum
    absolute difference 0.0.  Keeping the loop shape identical is what makes that true here
    too.
    """
    taps = ()

    # `enabled` is a settable no-op so the guided loop can toggle branch selection without
    # caring whether steering is on: assigning to a plain class attribute here would mutate
    # the shared singleton for every other generation in the process.
    @property
    def enabled(self):
        return False

    @enabled.setter
    def enabled(self, _value):
        pass

    def apply_emb(self, e):
        return e

    def apply_final(self, g):
        return g

    def apply_loc(self, l):
        return l

    def close(self):
        pass


NULL = _NullInjector()


# --------------------------------------------------------------------------- the vector pack
class VectorPack:
    """The distilled steering vectors plus the per-dimension layer ranking.

    Two file shapes are accepted:

      * the **server pack** written by `setup/build_steering_pack.py`: `names` (99 strings),
        `taps` (99 x K int), `V` (99 x K x 2560 float32), `rank` (a JSON string).  ~5 MB.
      * the **research file** `p3_vectors_ext.npz` (112 MB) together with `tap_rank.json`,
        for a box that happens to have them.

    Missing files are not an error: `available` is False and every steering mode degrades to
    `adapter` with a reason in the payload.  A dial that reads a value while the thing it
    names is switched off is worse than no dial (docs/LEARNINGS.md), so the degrade is
    reported rather than silently applied.
    """

    def __init__(self, path=None, rank_path=None, device="cpu"):
        self.path = path or config.STEER_PACK
        self.rank_path = rank_path or config.STEER_TAP_RANK
        self.device = device
        self.available = False
        self.error = None
        self.names = []
        self._vec = {}          # (name, tap) -> np.ndarray[2560]
        self._taps = {}         # name -> [tap, ...] in rank order
        self.rank = {}
        self._cache = {}
        self._lock = threading.Lock()
        try:
            self._load()
            self.available = bool(self._vec)
        except Exception as e:                      # noqa: BLE001 - never fail start-up
            self.error = f"{type(e).__name__}: {e}"
            print(f"[steer] vectors unavailable ({self.error}); steering modes will fall "
                  "back to adapter. Guidance does NOT need this file and is unaffected.",
                  flush=True)

    # -- loading ------------------------------------------------------------
    def _load(self):
        if not self.path or not os.path.exists(self.path):
            raise FileNotFoundError(self.path or "config.STEER_PACK is unset")
        z = np.load(self.path, allow_pickle=True)
        keys = set(z.files)
        if {"names", "taps", "V"} <= keys:
            self._load_pack(z)
        elif {"dim_kind", "dim_name", "taps", "d_hi_lo"} <= keys:
            self._load_research(z)
        else:
            raise ValueError(f"unrecognised vector file: keys {sorted(keys)}")
        print(f"[steer] {len(self.names)} dimensions from {self.path} "
              f"({sum(len(v) for v in self._taps.values())} (dim, layer) vectors, "
              f"{sum(a.nbytes for a in self._vec.values()) / 1e6:.1f} MB)", flush=True)

    def _load_pack(self, z):
        names = [str(x) for x in z["names"]]
        taps = np.asarray(z["taps"])
        V = np.asarray(z["V"], dtype=np.float32)
        for i, nm in enumerate(names):
            row = []
            for j, t in enumerate(taps[i]):
                t = int(t)
                if t < 0:
                    continue
                row.append(t)
                self._vec[(nm, t)] = V[i, j]
            self._taps[nm] = row
        self.names = names
        if "rank" in z.files:
            self.rank = json.loads(str(z["rank"]))
        elif self.rank_path and os.path.exists(self.rank_path):
            self.rank = json.load(open(self.rank_path))["rank"]

    def _load_research(self, z):
        if not (self.rank_path and os.path.exists(self.rank_path)):
            raise FileNotFoundError(
                "the research vector file needs tap_rank.json alongside it "
                f"(config.STEER_TAP_RANK = {self.rank_path!r})")
        self.rank = json.load(open(self.rank_path))["rank"]
        taps = [int(t) for t in z["taps"]]
        tap_ix = {t: i for i, t in enumerate(taps)}
        names = [f"{k}:{d}" for k, d in zip(z["dim_kind"], z["dim_name"])]
        A = z["d_hi_lo"]
        for i, nm in enumerate(names):
            want = self._ranked_taps(nm, config.STEER_PACK_K)
            row = []
            for t in want:
                if t not in tap_ix:
                    continue
                row.append(t)
                self._vec[(nm, t)] = np.asarray(A[i, tap_ix[t]], dtype=np.float32)
            self._taps[nm] = row
        self.names = names

    def _ranked_taps(self, name, k):
        """The k layers where THIS attribute is most decodable, from tap_rank.json."""
        dim = name.split(":", 1)[1] if ":" in name else name
        r = self.rank.get(dim)
        if not r:
            return []
        out = []
        for x in r["ranked"][:k]:
            out.append(37 if x == "loc" else int(x[1:]))
        return out

    # -- lookup -------------------------------------------------------------
    def has(self, name):
        return bool(self._taps.get(name))

    def taps_for(self, name, k):
        """The first k of this attribute's own ranked layers that the pack actually holds."""
        return list(self._taps.get(name, ()))[:max(0, int(k))]

    def direction(self, name, tap):
        v = self._vec.get((name, int(tap)))
        return None if v is None else v


# --------------------------------------------------------------------------- spec building
def build_specs(components, pack, device, dtype=torch.float32):
    """[{key, alpha, taps}] -> [(tap, unit direction tensor, summed magnitude)].

    Ported from `actf_steer.build_specs`.  Each component is normalised to unit length
    **before** it is weighted, so `alpha` means the same fraction of the hidden state's own
    magnitude for every component regardless of how large that attribute's raw
    difference-of-means happens to be — otherwise a combination is silently dominated by
    whichever attribute separates most.  Components that land on the same layer are summed
    there and the sum is re-normalised, with the magnitude carried separately, so the
    injector still receives a unit direction and one alpha.

    A zero-magnitude layer is KEPT rather than dropped: alpha = 0 has to run the identical
    code path to be the control it claims to be, and a combination whose components cancel
    at one layer is a real result, not an absent hook.
    """
    acc = {}
    missing = []
    for c in components:
        key = c["key"]
        al = float(c.get("alpha", 0.0))
        taps = c.get("layers")
        if not taps:
            missing.append(key)
            continue
        for tap in taps:
            d = pack.direction(key, tap)
            if d is None:
                missing.append(f"{key}@h{tap}")
                continue
            n = float(np.linalg.norm(d))
            d = (d / n) if n > 0 else d
            acc[tap] = acc.get(tap, 0.0) + al * d
    out = []
    for tap, v in sorted(acc.items()):
        n = float(np.linalg.norm(v))
        u = v / n if n > 0 else np.zeros_like(v)
        out.append((int(tap),
                    torch.from_numpy(np.ascontiguousarray(u)).to(device=device, dtype=dtype),
                    n))
    return out, missing


def realised_magnitude(specs):
    """The alpha actually applied at each layer, which is not the nominal alpha.

    Two components sharing a layer sum there: Anger at 0.10 with genuineness at 0.10 realises
    0.1604 at their shared h21.  The whitepaper requires this to be computed and logged, and
    it is what `config.STEER_ALPHA_CEILING` is checked against.
    """
    return {int(t): round(float(a), 4) for t, _v, a in specs}
