"""The vocal-burst detector as its own service — the OPTIONAL large-v2 path.

    NOT NEEDED FOR THE DEFAULT.  `bestofn.Judge.burst_scores` scores bursts
    in-process on the `commercial/` heads, using the VoiceCLAP-commercial tower
    that retrieval already has loaded: five MLPs of about a megabyte, no second
    encoder, and the local language model keeps its card.  That is the shipped
    path.

    This service exists for `laion/voiceclap-large-v2`, which wins 12 of 17
    classes on real speech — mostly the quiet sustained ones — and loses on
    Scream, 0.576 against 0.768.  It costs 5.7 GB in 4-bit and the local
    director, so it is worth starting only when those classes matter.


It answers one question the best-of-N reward could not: **is the sound the
script names actually in the audio?**  See `docs/BURST_REWARD.md` for why that
term exists and what it is worth.

It runs in a separate process for the same reason SIDON does, and one more.
The encoder is `laion/voiceclap-large-v2`, a 8.93 B Qwen2.5-Omni thinker —
16.6 GB in bf16, about 9 GB in 8-bit — which does not fit on the card the speech
model is on.  Started with `CUDA_VISIBLE_DEVICES` pointing at the language
model's card, it has that card to itself.

**The quantisation is checked, not assumed.**  The head was trained on bf16
embeddings, so this had to be measured before the term could be trusted.
Against a bf16 CPU reference over 36 windows:

    8-bit   cosine median 0.9959, min 0.9810   top-1 agrees 35/36   ~10 GB
    4-bit   cosine median 0.9753, min 0.9517   top-1 agrees 35/36   ~5.7 GB

**4-bit ships.**  It blurs the embedding measurably more and its DECISIONS are
exactly as good — the same 35 of 36, the same one borderline window that was
already undecided in bf16 (0.519 against 0.634).  What it buys is headroom: with
8-bit the encoder and the app's scorers together came to 21.4 GB of 23.68 and a
turn that allocated during alignment ran out of memory.  Fidelity nobody can
measure in the ranking is not worth an out-of-memory error in generation.
"""
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

import config

app = FastAPI()
M = {"enc": None, "proc": None, "tpl": None, "heads": None, "classes": None,
     "family": None, "ms": [], "n": 0}

# Our burst labels are prose; the detector knows 17 names.  An exact match gives
# the strict term; anything else is scored at the family level only, which is
# the honest thing to do for a label the detector was never trained to separate.
ALIAS = {
    "scream": "Scream", "shriek": "Scream", "chuckle": "Chuckle",
    "laugh": "Chuckle", "giggle": "Breathy Giggle",
    "breathy giggle": "Breathy Giggle", "snicker": "Chuckle",
    "sharp inhale": "Sharp Inhale", "gasp": "Sharp Inhale",
    "fearful gasp": "Sharp Inhale", "sharp intake": "Sharp Inhale",
    "deep breath": "Deep Breath", "heavy breathing": "Heavy Breathing",
    "panting": "Panting", "yawn": "Yawn", "humming": "Humming",
    "soft hum": "Soft Hum", "hum": "Soft Hum",
    "relief sigh": "Relief Sigh", "sigh of relief": "Relief Sigh",
    "wistful sigh": "Wistful Sigh", "contented sigh": "Wistful Sigh",
    "exasperated sigh": "Exasperated Sigh", "sigh": "Exasperated Sigh",
    "frustrated groan": "Frustrated Groan", "groan": "Frustrated Groan",
    "exhausted groan": "Exhausted Groan", "grunt": "Affirmative Grunt",
    "affirmative grunt": "Affirmative Grunt",
}
# When no class matches, place the label in a family by its own words.
FAMILY_WORDS = {
    "scream": "scream", "shriek": "scream", "wail": "scream", "cry": "scream",
    "laugh": "laugh", "giggle": "laugh", "chuckle": "laugh", "snort": "laugh",
    "sigh": "sigh", "breath": "breath", "inhale": "breath", "exhale": "breath",
    "gasp": "breath", "pant": "breath", "hum": "hum", "groan": "groan",
    "grunt": "groan", "moan": "groan", "yawn": "yawn",
}


def load():
    if M["enc"] is not None:
        return
    from transformers import (BitsAndBytesConfig, Qwen2_5OmniProcessor,
                              Qwen2_5OmniThinkerForConditionalGeneration)
    t0 = time.time()
    p = glob.glob(config.BURST_ENCODER_PATH)[0]
    M["proc"] = Qwen2_5OmniProcessor.from_pretrained(p)
    M["tpl"] = open(os.path.join(p, "additional_chat_templates",
                                 "sentence_transformers.jinja")).read()
    if config.BURST_QUANT == "8bit":
        q = BitsAndBytesConfig(load_in_8bit=True)
    else:
        q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                               bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True)
    M["enc"] = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        p, quantization_config=q, device_map={"": 0},
        dtype=torch.bfloat16).eval()
    snap = glob.glob(config.BURST_DETECTOR_PATH)[0]

    class Head(nn.Module):
        def __init__(self, D, H, C, dropout):
            super().__init__()
            # BatchNorm, not LayerNorm — the checkpoint carries running_mean.
            self.net = nn.Sequential(nn.Linear(D, H), nn.BatchNorm1d(H),
                                     nn.GELU(), nn.Dropout(dropout),
                                     nn.Linear(H, C))

        def forward(self, x):
            return self.net(x)

    heads = []
    for f in sorted(glob.glob(os.path.join(snap, "production",
                                           "vocal_burst_mlp_prod_s*.pt"))):
        ck = torch.load(f, map_location="cpu", weights_only=False)
        a = ck["arch"]
        h = Head(a["D"], a["H"], a["C"], a["dropout"])
        h.load_state_dict(ck["state_dict"])
        heads.append(h.eval().to("cuda:0"))
        M["classes"], M["family"] = ck["classes"], ck["family"]
    M["heads"] = heads
    print(f"[burst] {len(heads)} heads, {len(M['classes'])} classes, "
          f"encoder 8-bit on cuda:0, ready in {time.time()-t0:.0f}s", flush=True)


@torch.no_grad()
def embed(segs):
    """One 3584-d embedding per 16 kHz segment."""
    out = []
    for seg in segs:
        msg = [{"role": "user", "content": [{"type": "audio", "audio": seg}]}]
        txt = M["proc"].apply_chat_template(msg, chat_template=M["tpl"],
                                            tokenize=False,
                                            add_generation_prompt=False)
        inp = M["proc"](text=txt, audio=[seg], sampling_rate=16000,
                        return_tensors="pt")
        inp = {k: v.to("cuda:0") for k, v in inp.items()}
        h = M["enc"](**inp, output_hidden_states=True).hidden_states[-1]
        i = inp["attention_mask"].sum(1) - 1
        e = h[torch.arange(h.shape[0]), i].float()
        out.append((e / e.norm(dim=-1, keepdim=True))[0])
    return torch.stack(out)


@torch.no_grad()
def classify(embs):
    x = embs.to("cuda:0")
    if x.shape[0] == 1:                      # BatchNorm needs more than one row
        x = torch.cat([x, x], 0)
        p = torch.stack([torch.softmax(h(x), -1) for h in M["heads"]]).mean(0)
        return p[:1].cpu().numpy()
    return torch.stack([torch.softmax(h(x), -1)
                        for h in M["heads"]]).mean(0).cpu().numpy()


def resolve(label):
    """(class index or None, family name) for one of our burst labels."""
    s = str(label or "").strip().lower().replace("_", " ")
    cls = ALIAS.get(s)
    if cls is None:
        for k, v in ALIAS.items():           # longest containing alias
            if k in s and (cls is None or len(k) > len(cls)):
                cls = v
    idx = M["classes"].index(cls) if cls in (M["classes"] or []) else None
    fam = M["family"][idx] if idx is not None else None
    if fam is None:
        for w, f in FAMILY_WORDS.items():
            if w in s:
                fam = f
                break
    return idx, fam


def score_one(pcm16, bursts, sr):
    """The burst term for one candidate: the spec in docs/BURST_REWARD.md."""
    import scipy.signal as ss
    if not bursts:
        return None
    x = np.asarray(pcm16, dtype=np.float32)
    x16 = ss.resample_poly(x, 16000, sr).astype(np.float32)
    win = int(config.BON_BURST_WIN_S * 16000)
    hop = int(config.BON_BURST_HOP_S * 16000)
    vals = []
    for b in bursts:
        t = float(b.get("onset") or 0.0)
        lo = max(0, int((t - 1.0) * 16000))
        hi = min(len(x16), int((t + 2.0) * 16000) + win)
        segs = [x16[a:a + win] for a in range(lo, max(lo + 1, hi - win + 1), hop)
                if a + win <= len(x16)]
        if not segs:
            continue
        p = classify(embed(segs))
        idx, fam = resolve(b.get("label"))
        fam_idx = [i for i, f in enumerate(M["family"] or []) if f == fam] \
            if fam else []
        nb = M["classes"].index("no_burst")
        s_strict = float(p[:, idx].max()) if idx is not None else 0.0
        s_fam = float(p[:, fam_idx].sum(1).max()) if fam_idx else s_strict
        s_pres = float((1.0 - p[:, nb]).max())
        top = p.argmax(1)
        agree = 1.0 if any(int(t_) in fam_idx for t_ in top) else 0.0
        vals.append(s_strict + 0.5 * (s_fam - s_strict)
                    + 0.25 * s_pres + 0.5 * agree)
    return float(np.mean(vals)) if vals else None


@app.on_event("startup")
def _startup():
    load()


@app.get("/health")
def health():
    ms = M["ms"][-20:]
    return {"ok": M["enc"] is not None, "clips": M["n"],
            "classes": len(M["classes"] or []),
            "ms_per_clip": round(sum(ms) / len(ms), 1) if ms else None}


@app.post("/score")
async def score(req: Request):
    """Raw little-endian int16 for N candidates, one burst score each.

    `x-lengths` gives the sample count of each candidate, `x-bursts` the
    label/onset pairs the script asked for — the same for every candidate,
    because they are all renderings of one script.
    """
    raw = await req.body()
    if not raw:
        return JSONResponse({"error": "empty"}, status_code=400)
    sr = int(req.query_params.get("sr", 48000))
    try:
        lens = [int(x) for x in req.headers.get("x-lengths", "").split(",") if x]
        bursts = json.loads(req.headers.get("x-bursts") or "[]")
    except Exception as e:
        return JSONResponse({"error": f"bad headers: {e}"}, status_code=400)
    try:
        load()
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        out, at, t0 = [], 0, time.time()
        for n in lens:
            out.append(score_one(pcm[at:at + n], bursts, sr))
            at += n
        M["ms"].append((time.time() - t0) * 1000 / max(len(lens), 1))
        M["n"] += len(lens)
        return {"burst": out, "bursts_seen": len(bursts)}
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"{type(e).__name__}: {e}"},
                            status_code=500)
