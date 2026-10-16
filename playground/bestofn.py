#!/usr/bin/env python3
"""Generate a turn N times and keep the best one.

The case for this is in the burst recipes themselves: they quote an `N` per
class — "at a hit rate of 0.27 it takes 8 candidates for 90 % confidence" — and
without best-of-N that column is advice nobody can act on.  The same holds for
the emotion adapters, whose own card says to generate several candidates and
rank them because the base model drifts under strong emotion.

All N candidates are one batched forward pass.  The streaming path is batch 1
because audio must start before the line ends; choosing between candidates is
the opposite situation, so it uses `generate_batch` and the whole set costs
little more than one take.

## The reward

    R = (norm(genuineness) + norm(blend) + 2 * norm(clap)
         + 2 * norm(burst)) * gate(WER)

`clap` is the cosine between the take's VoiceCLAP-commercial *audio* embedding
and the *text* embedding of what the director asked for — GENERAL plus every
round bracket in the script.  It carries double weight because it is the only
term that measures whether the take is the performance that was requested; the
other two measure whether it is a good take of anything.

`burst` is new (2026-09-07, study `vb_opt`, protocol section 71).  The four
terms above ask whether a take is good and whether it is the performance that
was requested.  None of them asks whether the SOUND the script names is
actually in the audio, and measured over 1,940 candidate sets only `clap`
tracks burst presence at all (r +0.148); `blend` points the wrong way (-0.084).
Restricted to the configuration the server is always in — a carrier voice with
its own `sft3_voice` adapter — the old reward's burst lift is +0.0104 (t 1.51),
indistinguishable from zero.  With this term the strict hit rate of the
DELIVERED take rises +0.0454 (t 7.60, +37 % relative), scored by a detector
other than the one ranking.  It is a trade: CLAP falls 0.0055 (t -6.82), which
is 3.6 % of the range a re-ranker can move.  See `config.BON_BURST_WEIGHT` for
why 2.0 and for the unguided caveat.

The term is `s_strict + 0.5*(s_fam - s_strict) + 0.25*s_pres + 0.5*agree`,
averaged over the bursts the turn actually requested.  A turn that requests none
carries **no `burst` key**, and `rank` then forces the term to exactly 0 so the
ranking is unchanged.  Note that leaving it to `_norm` is not enough: `_norm` of
a constant vector returns 0.5, and a constant inside the sum still re-weights
every candidate once the per-candidate WER gate multiplies it.

Normalisation is within the candidate set, not against an absolute scale.  Only
the ranking matters, the three scorers have unrelated ranges (0-6, 0-10, a
cosine), and their absolute values are not calibrated against human judgement
anyway.

`gate(WER)` is `1.0` when the inverse word error rate is above 0.85 and the
inverse rate itself below it.  Intelligibility is a threshold, not a
preference: a take everyone can understand should not beat another take everyone
can understand for being marginally more understandable, but a take that garbles
the line has to lose regardless of how good it sounds.
"""
import re

import os
import numpy as np

import config


def direction_text(general, script):
    """What the director asked for: GENERAL plus every round bracket."""
    bits = []
    if general:
        bits.append(" ".join(str(general).split()))
    bits += [re.sub(r",?\s*[0-9]*\.?[0-9]+\s*(?:s|sec|seconds?)\b", "", c).strip()
             for c in re.findall(r"\(([^)]{2,160})\)", str(script or ""))]
    return " ".join(b for b in bits if b)[:600]


def _norm(v):
    """Min-max inside the candidate set; all-equal collapses to a constant."""
    a = np.asarray(v, dtype=np.float64)
    if not len(a):
        return a
    lo, hi = float(np.nanmin(a)), float(np.nanmax(a))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-9:
        return np.ones_like(a) * 0.5
    return (a - lo) / (hi - lo)


# Our burst labels are prose; the detector knows 17 names.  An exact match
# gives the strict term, anything else is scored at the family level only —
# the honest thing for a distinction the detector was never trained to make.
_BURST_ALIAS = {
    "scream": "Scream", "shriek": "Scream", "chuckle": "Chuckle",
    "laugh": "Chuckle", "giggle": "Breathy Giggle", "snicker": "Chuckle",
    "sharp inhale": "Sharp Inhale", "gasp": "Sharp Inhale",
    "fearful gasp": "Sharp Inhale", "surprised gasp": "Sharp Inhale",
    "deep breath": "Deep Breath", "heavy breathing": "Heavy Breathing",
    "panting": "Panting", "yawn": "Yawn", "humming": "Humming",
    "soft hum": "Soft Hum", "hum": "Soft Hum", "relief sigh": "Relief Sigh",
    "wistful sigh": "Wistful Sigh", "contented sigh": "Wistful Sigh",
    "exasperated sigh": "Exasperated Sigh", "sigh": "Exasperated Sigh",
    "frustrated groan": "Frustrated Groan", "groan": "Frustrated Groan",
    "exhausted groan": "Exhausted Groan", "grunt": "Affirmative Grunt",
}
_BURST_FAMILY = {
    "scream": "scream", "shriek": "scream", "wail": "scream",
    "laugh": "laugh", "giggle": "laugh", "chuckle": "laugh",
    "sigh": "sigh", "breath": "breath", "inhale": "breath",
    "exhale": "breath", "gasp": "breath", "pant": "breath", "hum": "hum",
    "groan": "groan", "grunt": "groan", "moan": "groan", "yawn": "yawn",
}


def _resolve_burst(label, classes, families):
    """(class index or None, family name) for one of our burst labels."""
    s = str(label or "").strip().lower().replace("_", " ")
    name = _BURST_ALIAS.get(s)
    if name is None:
        best = ""
        for k, v in _BURST_ALIAS.items():
            if k in s and len(k) > len(best):
                best, name = k, v
    idx = classes.index(name) if name in (classes or []) else None
    fam = families[idx] if idx is not None else None
    if fam is None:
        for w, f in _BURST_FAMILY.items():
            if w in s:
                fam = f
                break
    return idx, fam


def gate(wer, knee=None):
    """The intelligibility factor: the inverse word error rate, straight.

    This was flattened to 1.0 above an inverse rate of 0.85, on the argument that
    intelligibility is a threshold rather than a preference.  Removed: the flat
    region covered most of a candidate set -- six of eight in one run -- so the
    factor stopped separating exactly where the candidates were closest, and the
    ranking there fell to the three perceptual terms alone.  Multiplying by the
    raw inverse rate keeps a small intelligibility difference as a small reward
    difference all the way up.

    `knee` is still honoured when passed, so the old behaviour is one argument
    away, but nothing sets it now.
    """
    inv = 1.0 - float(max(0.0, min(1.0, wer)))
    k = config.BON_WER_KNEE if knee is None else knee
    if k is not None and k > 0 and inv > k:
        return 1.0
    return inv


def rank(cands):
    """`cands`: dicts with genuineness, blend, clap, wer.  Adds reward + rank."""
    if not cands:
        return cands
    g = _norm([c.get("genuineness", 0.0) for c in cands])
    b = _norm([c.get("blend", 0.0) for c in cands])
    p = _norm([c.get("clap", 0.0) for c in cands])
    # A turn that asked for no sound carries no `burst` key at all, and then the
    # term must be EXACTLY zero.  `_norm` of a constant vector returns 0.5, not
    # 0 -- and a constant 0.5 inside the sum is NOT harmless, because the sum is
    # multiplied by a per-candidate gate afterwards, so it quietly re-weights
    # every candidate by its own word error rate.  Measured on a 3-candidate
    # fixture: it changed the winner.  Hence the explicit guard.
    _has_burst = any(c.get("burst") is not None for c in cands)
    u = (_norm([c.get("burst", 0.0) for c in cands]) if _has_burst
         else [0.0] * len(cands))
    for i, c in enumerate(cands):
        c["n_genuineness"] = round(float(g[i]), 4)
        c["n_blend"] = round(float(b[i]), 4)
        c["n_clap"] = round(float(p[i]), 4)
        c["n_burst"] = round(float(u[i]), 4)
        c["gate"] = round(gate(c.get("wer", 0.0)), 4)
        c["reward"] = round(
            float((g[i] + b[i] + config.BON_CLAP_WEIGHT * p[i]
                   + config.BON_BURST_WEIGHT * u[i]) * c["gate"]), 4)
    order = sorted(range(len(cands)), key=lambda i: -cands[i]["reward"])
    for pos, i in enumerate(order):
        cands[i]["rank"] = pos
    return cands


class Judge:
    """The three scorers plus transcription, loaded once."""

    def __init__(self, device=None, asr=None):
        self.ok = False
        self.asr = asr
        dev = device or config.BON_DEVICE
        try:
            import sys
            # Publication portability only: same heads and numerical path.
            scorer_root=os.environ.get('HUMANENESS_SCORER_DIR','/mnt/nvme/moss-15-v2-assets')
            genuine_dir=os.path.join(scorer_root,'vc_genuineness')
            blend_dir=os.path.join(scorer_root,'vc_blend')
            sys.path.insert(0, genuine_dir)
            sys.path.insert(0, blend_dir)
            from blend_model import CommercialBlendScorer
            from genuineness_scorer import GenuinenessScorer
            self.gen = GenuinenessScorer(
                pkg_dir=genuine_dir,
                model="full", device=dev)
            self.bl = CommercialBlendScorer(
                pkg_dir=blend_dir, device=dev)
            self.ok = True
        except Exception as e:
            print(f"[bestofn] scorers unavailable ({type(e).__name__}: {e})",
                  flush=True)
            return
        # the CLAP tower is already loaded for retrieval; reuse it rather than
        # holding a second copy of the same weights on the same card
        self.clap = None
        print(f"[bestofn] judge ready on {dev}", flush=True)

    def attach_clap(self, retriever):
        self.clap = retriever

    def _clap_sim(self, wav48, want_vec):
        if self.clap is None or want_vec is None:
            return 0.0
        try:
            import torch
            x = np.asarray(wav48, np.float32).reshape(-1)
            n = (len(x) // 3) * 3
            w16 = x[:n].reshape(-1, 3).mean(1)
            self.clap._load()
            with torch.no_grad():
                v = self.clap.model.encode_waveform(
                    torch.tensor(w16, dtype=torch.float32,
                                 device=self.clap.device)[None],
                    sample_rate=16000)[0].float().cpu().numpy()
            v = v - self.clap.qmu
            v /= max(np.linalg.norm(v), 1e-8)
            return float(np.dot(v, want_vec))
        except Exception as e:
            print(f"[bestofn] clap failed: {e}", flush=True)
            return 0.0

    # ---------------------------------------------------------- bursts ----
    # The head is 768 -> 256 -> 17 on `encode_waveform` output, which is the
    # tower already loaded for retrieval — `FastScorer.emb.encode_waveform` and
    # `laion/voiceclap-commercial` were measured identical to cosine 1.000000.
    # So this runs in-process on five MLPs of about a megabyte each; no second
    # encoder, no service, and the local language model keeps its card.
    _heads = None
    _hcls = None
    _hfam = None

    def _load_heads(self):
        if Judge._heads is not None:
            return bool(Judge._heads)
        import glob
        import torch
        import torch.nn as nn
        Judge._heads = []
        try:
            # Pick the snapshot that actually HAS the heads: a partial
            # download leaves a second snapshot directory, and taking [0]
            # silently found the older one without `commercial/`.
            files = []
            for snap in sorted(glob.glob(config.BURST_DETECTOR_PATH)):
                got = sorted(glob.glob(os.path.join(
                    snap, "commercial", "vocal_burst_mlp_prod_s*.pt")))
                if len(got) > len(files):
                    files = got
            if not files:
                raise FileNotFoundError(
                    "no commercial/ heads under "
                    f"{config.BURST_DETECTOR_PATH} — run "
                    "python setup/fetch_all.py")

            class Head(nn.Module):
                def __init__(self, D, H, C, dropout):
                    super().__init__()
                    # BatchNorm, not LayerNorm — the checkpoint carries
                    # running_mean, and guessing would normalise wrongly.
                    self.net = nn.Sequential(
                        nn.Linear(D, H), nn.BatchNorm1d(H), nn.GELU(),
                        nn.Dropout(dropout), nn.Linear(H, C))

                def forward(self, x):
                    return self.net(x)

            for f in files:
                ck = torch.load(f, map_location="cpu", weights_only=False)
                a = ck["arch"]
                assert str(ck.get("encoder", "")).startswith("voiceclap-comm"), \
                    f"{f} expects {ck.get('encoder')}, not the retrieval tower"
                h = Head(a["D"], a["H"], a["C"], a["dropout"])
                h.load_state_dict(ck["state_dict"])
                Judge._heads.append(h.eval().to(self.clap.device))
                Judge._hcls, Judge._hfam = ck["classes"], ck["family"]
            print(f"[burst] {len(Judge._heads)} commercial heads on "
                  f"{self.clap.device}", flush=True)
        except Exception as e:
            print(f"[burst] heads unavailable: {type(e).__name__}: {e}",
                  flush=True)
            Judge._heads = []
        return bool(Judge._heads)

    def burst_scores(self, waves, sr, bursts):
        """One burst score per candidate, or None each.

        Localised to `[t-1, t+2]` around each onset in windows of
        BON_BURST_WIN_S at BON_BURST_HOP_S — localised measured 0.2082 against
        0.1907 whole-clip.
        """
        n = len(waves)
        if not bursts or not config.BON_BURST_READY or self.clap is None:
            return [None] * n
        try:
            import torch
            self.clap._load()
            if not self._load_heads():
                return [None] * n
            cls, fam = Judge._hcls, Judge._hfam
            nb = cls.index("no_burst")
            win = int(config.BON_BURST_WIN_S * 16000)
            hop = max(1, int(config.BON_BURST_HOP_S * 16000))
            out = []
            for w in waves:
                x = np.asarray(w, np.float32).reshape(-1)
                m = (len(x) // 3) * 3
                x16 = x[:m].reshape(-1, 3).mean(1)
                vals = []
                for lab, t in bursts:
                    lo = max(0, int((t - 1.0) * 16000))
                    hi = min(len(x16), int((t + 2.0) * 16000) + win)
                    segs = [x16[a:a + win]
                            for a in range(lo, max(lo + 1, hi - win + 1), hop)
                            if a + win <= len(x16)]
                    if not segs:
                        continue
                    with torch.no_grad():
                        e = torch.stack([
                            self.clap.model.encode_waveform(
                                torch.tensor(g, dtype=torch.float32,
                                             device=self.clap.device)[None],
                                sample_rate=16000)[0].float() for g in segs])
                        if e.shape[0] == 1:      # BatchNorm refuses one row
                            e = torch.cat([e, e], 0)
                            p = torch.stack([torch.softmax(h(e), -1)
                                             for h in Judge._heads]).mean(0)[:1]
                        else:
                            p = torch.stack([torch.softmax(h(e), -1)
                                             for h in Judge._heads]).mean(0)
                    p = p.cpu().numpy()
                    idx, family = _resolve_burst(lab, cls, fam)
                    fi = [i for i, f in enumerate(fam) if f == family] \
                        if family else []
                    s_strict = float(p[:, idx].max()) if idx is not None else 0.0
                    s_fam = float(p[:, fi].sum(1).max()) if fi else s_strict
                    s_pres = float((1.0 - p[:, nb]).max())
                    agree = 1.0 if any(int(k) in fi for k in p.argmax(1)) else 0.0
                    vals.append(s_strict + 0.5 * (s_fam - s_strict)
                                + 0.25 * s_pres + 0.5 * agree)
                out.append(float(np.mean(vals)) if vals else None)
            return out
        except Exception as e:
            print(f"[burst] scoring skipped: {type(e).__name__}: {e}",
                  flush=True)
            return [None] * n

    def want_vector(self, general, script):
        """The centred text embedding of what was asked for."""
        if self.clap is None:
            return None
        txt = direction_text(general, script)
        if not txt:
            return None
        try:
            v = self.clap.embed(txt) - self.clap.qmu
            return v / max(np.linalg.norm(v), 1e-8)
        except Exception:
            return None

    def score(self, waves, sr, plain, general=None, script=None, want=None,
              tagged=None):
        """One dict per candidate, ready for `rank`.

        `script` is the director's own text, used for the wanted-attribute
        vector.  `tagged` is the RENDERED script — it carries the durations, so
        it is the only one from which a burst's onset can be computed.
        """
        import os
        import tempfile

        import soundfile as sf
        from eval_tail import trailing_words, wer as _wer
        if want is None:
            want = self.want_vector(general, script)
        out = []
        for w in waves:
            c = {"sec": round(len(w) / float(sr), 3), "genuineness": 0.0,
                 "blend": 0.0, "clap": 0.0, "wer": 1.0, "extra_w": 0, "hyp": ""}
            if w is None or len(w) < sr // 8:
                out.append(c)
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, np.asarray(w, np.float32), sr)
                path = f.name
            try:
                c["genuineness"] = float(self.gen.score(path))
                c["blend"] = float(self.bl.score(path))
            except Exception as e:
                print(f"[bestofn] scorer: {e}", flush=True)
            finally:
                os.unlink(path)
            c["clap"] = self._clap_sim(w, want)
            if self.asr is not None and plain:
                try:
                    x = np.asarray(w, np.float32)
                    n = (len(x) // 3) * 3
                    hyp = self.asr.transcribe(x[:n].reshape(-1, 3).mean(1))
                    c["hyp"] = hyp
                    c["wer"] = float(_wer(plain, hyp))
                    c["extra_w"] = int(trailing_words(plain, hyp)[0])
                except Exception as e:
                    print(f"[bestofn] asr: {e}", flush=True)
            out.append(c)
        # Is the sound the script names actually in the audio?  None of the
        # four terms above asks that -- only `clap` tracks burst presence at
        # all (r +0.148) and `blend` points the wrong way.  Scored by the
        # detector service, localised to each burst's own onset, and soft in
        # every direction: no service means None for every candidate, and
        # `rank` then forces the term to exactly 0.
        try:
            import timed_script as _ts
            _bursts = _ts.burst_onsets(tagged or "")
            if _bursts:
                for c, v in zip(out, self.burst_scores(waves, sr, _bursts)):
                    if v is not None:
                        c["burst"] = v
        except Exception as e:
            print(f"[burst] skipped: {type(e).__name__}: {e}", flush=True)

        return out
