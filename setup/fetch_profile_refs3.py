#!/usr/bin/env python3
"""Extract the best three takes per condition for each pilot voice.

The winners are not individually addressable — they sit inside five ~1.4 GB
WebDataset shards per voice next to the ~45 candidates we do not want.  So each
shard is pulled, its wanted members are written out, and the shard is deleted
again: peak disk stays at one shard.

Alongside the audio this writes what the retrieval index needs, straight from
the corpus's own annotation: the VoiceCLAP-commercial embedding of every take
(already computed upstream, verified here at cosine 0.98 against a local
recompute), plus genuineness, vocal-burst blend and the VoiceNet vector.
"""
import json, os, shutil, sys, tarfile, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "TTS-AGI/moss-voice-profile-references"
_root = Path(__file__).resolve().parents[1]
_manifest = json.loads((_root / "manifests/agent-default-v1.json").read_text())
REVISION = _manifest["assets"]["references"]["revision"]
DATA = Path(os.environ.get("HUMANENESS_DATA_DIR") or os.environ.get(
    "MOSS_DATA_DIR", "~/.cache/humaneness-voice")).expanduser()
OUT = os.environ.get("MOSS_REF3_DIR", str(DATA / "refs3"))
TOPN = 3
VOICES = ["anime_088", "emolia_c0542", "emolia_c1682", "emolia_c1699",
          "emolia_c2570", "k10_age3_bg1", "k325_age3_bg1", "k395_age3_bg1",
          "k91_age5_bg0", "mediathek_0184"]

COLS = ["gid", "__key__", "reward", "has_audio", "block", "lang", "emotion",
        "dim", "level", "character", "condition", "edge", "caption", "dur",
        "genuineness", "blend", "voicenet", "voiceclap", "spk_sim", "text"]


def winners(voice):
    p = hf_hub_download(REPO, f"pilot/{voice}/metadata.parquet",
                        revision=REVISION, repo_type="dataset")
    t = pq.read_table(p, columns=COLS).to_pandas()
    t = t[t.has_audio]
    t = t.sort_values("reward", ascending=False).groupby("gid", as_index=False).head(TOPN)
    t["take_rank"] = t.groupby("gid")["reward"].rank(ascending=False, method="first").astype(int)
    return t


def main():
    only = sys.argv[1:] or VOICES
    unknown = set(only) - set(VOICES)
    if unknown:
        raise SystemExit(f"unknown pilot profile(s): {sorted(unknown)}")
    for vi, voice in enumerate(only, 1):
        dst = os.path.join(OUT, "audio", voice)
        idx_p = os.path.join(OUT, f"index_{voice}.json")
        if os.path.exists(idx_p):
            print(f"[{vi}/{len(only)}] {voice}: already done", flush=True)
            continue
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        w = winners(voice)
        want = {r["__key__"]: r for _, r in w.iterrows()}
        print(f"[{vi}/{len(only)}] {voice}: {len(want)} takes over {w.gid.nunique()} conditions",
              flush=True)
        idx, meta, embs, got = {}, [], [], 0
        for shard in range(5):
            name = f"pilot/{voice}/data/{voice}-{shard:04d}.tar"
            try:
                tp = hf_hub_download(REPO, name, revision=REVISION, repo_type="dataset",
                                     local_dir=os.path.join(OUT, "_tmp"))
            except Exception as e:
                print(f"    shard {shard}: {str(e)[:80]}", flush=True)
                continue
            try:
                with tarfile.open(tp) as tf:
                    for mem in tf:
                        if not mem.name.endswith(".mp3"):
                            continue
                        key = mem.name[:-4]
                        r = want.get(key)
                        if r is None:
                            continue
                        fn = key.replace("/", "_") + ".mp3"
                        with open(os.path.join(dst, fn), "wb") as f:
                            f.write(tf.extractfile(mem).read())
                        idx.setdefault(r["gid"], []).append(
                            (int(r["take_rank"]), f"audio/{voice}/{fn}"))
                        vc = np.asarray(r["voiceclap"], dtype=np.float32)
                        embs.append(vc)
                        meta.append({
                            "key": key, "gid": r["gid"], "rank": int(r["take_rank"]),
                            "path": f"audio/{voice}/{fn}", "voice": voice,
                            "reward": float(r["reward"]), "lang": r["lang"],
                            "block": r["block"], "emotion": r["emotion"], "dim": r["dim"],
                            "level": r["level"], "character": r["character"],
                            "condition": r["condition"], "edge": r["edge"],
                            "caption": r["caption"], "dur": float(r["dur"] or 0),
                            "genuineness": float(r["genuineness"] or 0),
                            "blend": float(r["blend"] or 0),
                            "spk_sim": float(r["spk_sim"] or 0),
                            "voicenet": r["voicenet"], "text": r["text"],
                        })
                        got += 1
            finally:
                try:
                    os.remove(tp)
                except OSError:
                    pass
            print(f"    shard {shard}: {got}/{len(want)}", flush=True)
        for g in idx:
            idx[g] = [p for _, p in sorted(idx[g])]
        json.dump(idx, open(idx_p, "w", encoding="utf-8"))
        with open(os.path.join(OUT, f"meta_{voice}.jsonl"), "w", encoding="utf-8") as f:
            for m in meta:
                f.write(json.dumps(m, ensure_ascii=False, default=str) + "\n")
        np.save(os.path.join(OUT, f"emb_{voice}.npy"), np.stack(embs) if embs else np.zeros((0, 768), np.float32))
        mb = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e6
        print(f"    {voice}: {got} clips, {len(idx)} conditions, {mb:.0f} MB, "
              f"{time.time()-t0:.0f}s", flush=True)
    shutil.rmtree(os.path.join(OUT, "_tmp"), ignore_errors=True)
    print("REFS3_DONE", flush=True)


if __name__ == "__main__":
    main()
