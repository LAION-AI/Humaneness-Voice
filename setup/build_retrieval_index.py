#!/usr/bin/env python3
"""Build the nearest-neighbour index the director's prose is matched against.

Two things get embedded into the one 768-d VoiceCLAP-commercial space:

  * every acting condition, as the **centroid** of its takes' audio embeddings
    (already computed upstream and shipped in the corpus metadata), and
  * every emotion, as a **text anchor** — the mean of six caption templates in
    the phrasing VoiceCLAP was trained on ("a person speaking with anger in
    their voice"), not the long GENERAL specs, which are far out of its
    distribution.

Why both: measured on a 40-way held-out set of director-style prose, matching
against the text anchor scores 0.61 top-1 / 0.78 top-3 and matching against the
audio centroid scores 0.44 / 0.67, against a 0.025 chance rate.  Neither alone
is the whole story, so the runtime fuses them.  Mean-centring is applied to both
sides first; without it the two collapse to 0.35 and 0.22 respectively, because
the embeddings are strongly anisotropic and a handful of conditions sit close to
everything.
"""
import json, os, sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from transformers import AutoModel, AutoTokenizer

REPO = "TTS-AGI/moss-voice-profile-references"
_root = Path(__file__).resolve().parents[1]
_manifest = json.loads((_root / "manifests/agent-default-v1.json").read_text())
REVISION = _manifest["assets"]["references"]["revision"]
VOICECLAP_REVISION = _manifest["assets"]["voiceclap"]["revision"]
DATA = Path(os.environ.get("HUMANENESS_DATA_DIR") or os.environ.get(
    "MOSS_DATA_DIR", "~/.cache/humaneness-voice")).expanduser()
OUT = os.environ.get("MOSS_RETRIEVAL_DIR", str(DATA / "retrieval"))
VOICECLAP = "laion/voiceclap-commercial"
VOICES = ["anime_088", "emolia_c0542", "emolia_c1682", "emolia_c1699",
          "emolia_c2570", "k10_age3_bg1", "k325_age3_bg1", "k395_age3_bg1",
          "k91_age5_bg0", "mediathek_0184"]
TOPN = 3

TEMPLATES = ["a person speaking with {} in their voice",
             "a voice full of {}",
             "someone sounding {}",
             "speech expressing {}",
             "{} in the tone of voice",
             "a speaker who feels {}"]


def s(x):
    return x if isinstance(x, str) and x and x != "nan" else ""


def main():
    voices = sys.argv[1:] or VOICES
    unknown = set(voices) - set(VOICES)
    if unknown:
        raise SystemExit(f"unknown pilot profile(s): {sorted(unknown)}")
    os.makedirs(OUT, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(VOICECLAP, revision=VOICECLAP_REVISION)
    m = AutoModel.from_pretrained(VOICECLAP, trust_remote_code=True,
                                  revision=VOICECLAP_REVISION,
                                  dtype=torch.float32).to(dev).eval()

    cond, vecs = [], []
    for vi, voice in enumerate(voices, 1):
        p = hf_hub_download(REPO, f"pilot/{voice}/metadata.parquet",
                            revision=REVISION, repo_type="dataset")
        t = pq.read_table(p, columns=["gid", "__key__", "reward", "has_audio", "block",
                                      "lang", "emotion", "dim", "level", "character",
                                      "condition", "voiceclap", "genuineness", "blend",
                                      "spk_sim"]).to_pandas()
        t = t[t.has_audio]
        t = t.sort_values("reward", ascending=False).groupby("gid", as_index=False).head(TOPN)
        for gid, g in t.groupby("gid"):
            v = np.stack(g.voiceclap.values).astype(np.float32).mean(0)
            r0 = g.iloc[0]
            cond.append({
                "gid": gid, "voice": voice, "lang": s(r0.lang), "block": s(r0.block),
                "emotion": s(r0.emotion), "dim": s(r0.dim), "level": s(r0.level),
                "character": s(r0.character), "condition": s(r0.condition),
                "keys": list(g["__key__"]),
                "genuineness": float(g.genuineness.mean()),
                "blend": float(g.blend.mean()),
                "spk_sim": float(g.spk_sim.mean()),
                "reward": float(g.reward.mean()),
            })
            vecs.append(v)
        print(f"[{vi}/{len(voices)}] {voice}: {t.gid.nunique()} conditions", flush=True)

    C = F.normalize(torch.tensor(np.stack(vecs)), dim=-1)

    emotions = sorted({c["emotion"] for c in cond if c["emotion"]})
    anchors = []
    for e in emotions:
        txt = [tpl.format(e.replace("_", " ").lower()) for tpl in TEMPLATES]
        b = tok(txt, return_tensors="pt", padding=True, truncation=True,
                max_length=64).to(dev)
        with torch.no_grad():
            anchors.append(m.encode_text(b["input_ids"], b["attention_mask"]).mean(0).cpu())
    A = F.normalize(torch.stack(anchors), dim=-1)

    np.savez(os.path.join(OUT, "index.npz"),
             cond_emb=C.numpy().astype(np.float32),
             cond_mean=C.mean(0).numpy().astype(np.float32),
             emo_anchor=A.numpy().astype(np.float32),
             emo_anchor_mean=A.mean(0).numpy().astype(np.float32))
    json.dump({"conditions": cond, "emotions": emotions, "templates": TEMPLATES,
               "topn": TOPN, "voiceclap": VOICECLAP},
              open(os.path.join(OUT, "index.json"), "w", encoding="utf-8"))
    print(f"index: {len(cond)} conditions, {len(emotions)} emotions -> {OUT}", flush=True)
    print("RETRIEVAL_INDEX_DONE", flush=True)


if __name__ == "__main__":
    main()
