"""What the user's voice sounds like: emotions and VoiceNet dimensions.

Two independent scorers, both running on whatever the microphone captured:

  emotions   laion/Empathic-Insight-Voice-Small — 40 emotion heads plus a dozen
             attributes, each an MLP over a flattened BUD-E-Whisper encoder
             output (1500 x 768 -> 1.15M -> 64 -> 64 -> 32 -> 16 -> 1)
  voicenet   laion/voicenet-dimension-predictors-commercial — 57 dimensions,
             each a small MLP over one L2-normalised 768-d VoiceCLAP embedding,
             so all 57 come from a single encode

Neither repo documents the activation between the MLP layers, and the state
dicts hold no parameters for those positions, so it has to be parameter-free.
ACT below is chosen by measuring against the reference corpus, whose clips carry
ground-truth emotion labels — see `python score_engine.py --validate`.
"""
import glob
import os
import re
import threading
import time

import numpy as np
import torch
import torch.nn as nn

import config

ACT = os.environ.get("MOSS_SCORE_ACT", "relu")


def _act():
    return nn.GELU() if ACT == "gelu" else nn.ReLU()


class _EIVHead(nn.Module):
    """proj to 64, then 64 -> 64 -> 32 -> 16 -> 1."""

    def __init__(self, in_dim=1152000):
        super().__init__()
        self.proj = nn.Linear(in_dim, 64)
        self.mlp = nn.Sequential(
            _act(), nn.Dropout(0.0), nn.Linear(64, 64),
            _act(), nn.Dropout(0.0), nn.Linear(64, 32),
            _act(), nn.Dropout(0.0), nn.Linear(32, 16),
            _act(), nn.Dropout(0.0), nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.mlp(self.proj(x))


class _VNHead(nn.Module):
    def __init__(self, d=768, h=64):
        super().__init__()
        self.f1 = nn.Linear(d, h)
        self.f2 = nn.Linear(h, 1)
        self.act = _act()

    def forward(self, x):
        return self.f2(self.act(self.f1(x)))


class VoiceScorer:
    def __init__(self, device=None):
        self.device = device or config.SCORE_DEVICE
        self.lock = threading.Lock()
        self.ready = False
        t0 = time.time()
        self._load_emotion()
        self._load_voicenet()
        self._load_baseline()
        self.ready = True
        print(f"[score] ready in {time.time()-t0:.1f}s "
              f"({len(self.emo)} emotions, {len(self.vn)} dimensions)", flush=True)

    # ------------------------------------------------------------- emotions
    def _load_emotion(self):
        from transformers import WhisperFeatureExtractor, WhisperModel
        root = config.EIV_DIR
        self.wfe = WhisperFeatureExtractor.from_pretrained(config.WHISPER_DIR)
        self.wenc = WhisperModel.from_pretrained(
            config.WHISPER_DIR, dtype=torch.float32).encoder.to(self.device).eval()
        # Each head is dominated by one 64 x 1,152,000 projection: 295 MB in
        # fp32, so all 55 would be 16 GB of weights.  Only the 40 emotions are
        # ranked, and half precision is plenty for a score that is rendered as a
        # bar — together that is 5.9 GB instead.
        self.emo, self.emo_attr = {}, {}
        want_attrs = set(config.SCORE_ATTRS)
        for p in sorted(glob.glob(os.path.join(root, "model_*_best.pth"))):
            name = os.path.basename(p)[len("model_"):-len("_best.pth")]
            is_emo = name in config.EMOTION_NAMES
            if not is_emo and name not in want_attrs:
                continue
            sd = torch.load(p, map_location="cpu", weights_only=False)
            sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
            h = _EIVHead(sd["proj.weight"].shape[1])
            h.load_state_dict(sd, strict=True)
            h = h.to(self.device, dtype=torch.float16).eval()
            (self.emo if is_emo else self.emo_attr)[name] = h

    # ------------------------------------------------------------ voicenet
    def _load_voicenet(self):
        from transformers import AutoModel
        root = config.VN_DIR
        self.vclap = AutoModel.from_pretrained(
            os.path.join(root, "voiceclap_commercial"), trust_remote_code=True,
            dtype=torch.float32).to(self.device).eval()
        self.vn, self.vn_meta = {}, {}
        for p in sorted(glob.glob(os.path.join(root, "regression", "*.pt"))):
            d = torch.load(p, map_location="cpu", weights_only=False)
            dim = d.get("dim") or os.path.basename(p)[:-3]
            a = d.get("arch") or {}
            h = _VNHead(a.get("D", 768), a.get("H", 64))
            h.load_state_dict(d["state_dict"], strict=True)
            self.vn[dim] = (h.to(self.device).eval(),
                            torch.as_tensor(d["mu"]).float().to(self.device),
                            torch.as_tensor(d["sd"]).float().to(self.device))
            lv = d.get("levels") or {}
            self.vn_meta[dim] = {"name": d.get("name") or dim,
                                 "levels": len(lv) or 7}

    def _load_baseline(self):
        """Mean/spread of each dimension over ordinary speech, cached to disk."""
        import json
        path = config.VN_BASELINE
        if os.path.exists(path):
            self.baseline = json.load(open(path, encoding="utf-8"))
            return
        self.baseline = {}
        try:
            import subprocess, tempfile
            import soundfile as sf
            idx = json.load(open(os.path.join(config.REF_DIR, "index.json"), encoding="utf-8"))
            clips = []
            for emo, by_i in list(idx.get("emotion", {}).items()):
                for lang in ("en", "de"):
                    leaf = by_i.get("moderate", {}).get("free", {}).get(lang)
                    if leaf:
                        clips.append(os.path.join(config.REF_DIR, leaf["original"]))
            clips = clips[:60]
            acc = {}
            for c in clips:
                with tempfile.NamedTemporaryFile(suffix=".wav") as f:
                    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", c, "-ac", "1",
                                    "-ar", "16000", f.name],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    w, _ = sf.read(f.name, dtype="float32")
                v = self._vclap_embed(w)
                for dim, (h, mu, sd) in self.vn.items():
                    zz = (v - mu) / torch.clamp(sd, min=1e-6)
                    acc.setdefault(dim, []).append(float(h(zz).squeeze()))
            self.baseline = {d: [float(np.mean(a)), float(np.std(a) or 1.0)]
                             for d, a in acc.items()}
            os.makedirs(os.path.dirname(path), exist_ok=True)
            json.dump(self.baseline, open(path, "w", encoding="utf-8"))
            print(f"[score] voicenet baseline built from {len(clips)} clips",
                  flush=True)
        except Exception as ex:
            print(f"[score] baseline failed ({ex}); ranking will be crude",
                  flush=True)

    # -------------------------------------------------------------- scoring
    @torch.inference_mode()
    def _whisper_embed(self, wav16):
        f = self.wfe(wav16, sampling_rate=16000, return_tensors="pt")
        x = f.input_features.to(self.device)
        h = self.wenc(x).last_hidden_state          # (1, 1500, 768)
        return h.reshape(1, -1).to(torch.float16)

    @torch.inference_mode()
    def _vclap_embed(self, wav16):
        # encode_waveform does the model's own training-time mel preprocessing;
        # the heads were fitted on L2-normalised outputs of exactly this path
        t = torch.as_tensor(wav16, dtype=torch.float32, device=self.device)[None]
        e = self.vclap.encode_waveform(t, sample_rate=16000)
        if isinstance(e, (tuple, list)):
            e = e[0]
        if hasattr(e, "last_hidden_state"):
            e = e.last_hidden_state.mean(1)
        e = e.reshape(1, -1).float()
        return torch.nn.functional.normalize(e, dim=-1)

    @torch.inference_mode()
    def score(self, wav16, top_emotions=3, top_dims=5):
        """16 kHz mono float32 in, ranked emotions and dimensions out."""
        if wav16 is None or len(wav16) < 1600:
            return None
        wav16 = np.asarray(wav16, dtype=np.float32)
        out = {}
        with self.lock:
            t0 = time.time()
            try:
                e = self._whisper_embed(wav16)
                sc = {n: float(h(e).squeeze()) for n, h in self.emo.items()}
                attrs = {n: float(h(e).squeeze()) for n, h in self.emo_attr.items()}
                ranked = sorted(sc.items(), key=lambda kv: -kv[1])
                out["emotions"] = [
                    {"name": n.replace("_", " "), "score": round(v, 3)}
                    for n, v in ranked[:top_emotions]]
                out["emotions_all"] = {n: round(v, 3) for n, v in ranked}
                out["attributes"] = {n.replace("_", " "): round(v, 3)
                                     for n, v in attrs.items()}
            except Exception as ex:
                print(f"[score] emotion failed: {ex}", flush=True)
            try:
                v = self._vclap_embed(wav16)
                dims = {}
                for dim, (h, mu, sd) in self.vn.items():
                    z = (v - mu) / torch.clamp(sd, min=1e-6)
                    dims[dim] = float(h(z).squeeze())
                # Rank by how unusual the value is *for speech*, not by distance
                # from the middle of the scale.  Half the dimensions sit near zero
                # for every clip — nobody is a newsreader — so a raw ranking just
                # lists the same handful every time.  The baseline below is the
                # mean and spread over a sample of the reference corpus, so what
                # surfaces is what actually sets this voice apart.
                base = self.baseline
                def z(d, v):
                    b = base.get(d)
                    return abs(v - b[0]) / max(b[1], 1e-3) if b else 0.0
                ranked = sorted(dims.items(), key=lambda kv: -z(*kv))
                out["voicenet"] = [
                    {"dim": d, "name": self.vn_meta[d]["name"],
                     "score": round(sv, 3), "z": round(z(d, sv), 2),
                     "levels": self.vn_meta[d]["levels"],
                     "direction": ("high" if sv >= base.get(d, [sv])[0] else "low")}
                    for d, sv in ranked[:top_dims]]
                out["voicenet_all"] = {d: round(s, 3) for d, s in dims.items()}
            except Exception as ex:
                print(f"[score] voicenet failed: {ex}", flush=True)
            out["score_ms"] = round((time.time() - t0) * 1000, 1)
        return out


# --------------------------------------------------------------- validation
def _validate():
    """Does the reconstruction actually hear what the corpus says is there?

    The reference corpus labels each clip with the emotion it was generated for,
    so a correct scorer should rank that emotion near the top.  Run with each
    activation and keep the one that wins.
    """
    import json
    import subprocess
    import tempfile
    import soundfile as sf

    root = config.REF_DIR
    idx = json.load(open(os.path.join(root, "index.json"), encoding="utf-8"))
    tests = []
    for emo in ["Anger", "Sadness", "Amusement", "Fear", "Contentment"]:
        leaf = idx["emotion"][emo]["intense"]["free"]["en"]
        tests.append((emo, os.path.join(root, leaf["original"])))

    sc = VoiceScorer()
    hits = ranks = 0
    for want, path in tests:
        with tempfile.NamedTemporaryFile(suffix=".wav") as f:
            subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", path, "-ac", "1",
                            "-ar", "16000", f.name],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            w, _ = sf.read(f.name, dtype="float32")
        r = sc.score(w, top_emotions=5)
        names = [e["name"].replace(" ", "_") for e in r["emotions"]]
        allr = list(r["emotions_all"])
        pos = allr.index(want) + 1 if want in allr else 99
        ranks += pos
        hits += want in names
        print(f"  {want:14s} rank {pos:2d}  top: {', '.join(names[:3])}")
    print(f"\n  activation={ACT}: {hits}/{len(tests)} in top-5, mean rank {ranks/len(tests):.1f}")


if __name__ == "__main__":
    import sys
    if "--validate" in sys.argv:
        _validate()
