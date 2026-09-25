#!/usr/bin/env python3
"""Fetch the 832-condition reference matrix for the ten pilot voices.

The winners are not individually addressable: they live inside five ~1.4 GB
WebDataset shards per voice, alongside ~48 candidate takes per condition that we
do not want.  So each shard is pulled, the winning members are extracted, and the
shard is deleted again — peak disk stays at one shard, and what remains is about
170 MB of audio per voice instead of 7 GB.

Winner = highest reward per gid, which is the same rule the corpus itself used to
pick its published take.
"""
import json, os, shutil, sys, tarfile, time
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

REPO = "TTS-AGI/moss-voice-profile-references"
_root = Path(__file__).resolve().parents[1]
_manifest = json.loads((_root / "manifests/agent-default-v1.json").read_text())
REVISION = _manifest["assets"]["references"]["revision"]
DATA = Path(os.environ.get("HUMANENESS_DATA_DIR") or os.environ.get(
    "MOSS_DATA_DIR", "~/.cache/humaneness-voice")).expanduser()
OUT = os.environ.get("MOSS_REF_DIR_OVERRIDE") or os.environ.get(
    "MOSS_REF_DIR", str(DATA / "refs"))
VOICES = ["anime_088", "emolia_c0542", "emolia_c1682", "emolia_c1699",
          "emolia_c2570", "k10_age3_bg1", "k325_age3_bg1", "k395_age3_bg1",
          "k91_age5_bg0", "mediathek_0184"]


def winners(voice):
    p = hf_hub_download(REPO, f"pilot/{voice}/metadata.parquet",
                        revision=REVISION, repo_type="dataset")
    t = pq.read_table(p, columns=["gid", "reward", "__key__", "has_audio",
                                  "block", "lang"]).to_pandas()
    t = t[t.has_audio]
    best = t.sort_values("reward", ascending=False).groupby("gid", as_index=False).first()
    return {r["__key__"]: r["gid"] for _, r in best.iterrows()}


def main():
    only = sys.argv[1:] or VOICES
    unknown = set(only) - set(VOICES)
    if unknown:
        raise SystemExit(f"unknown pilot profile(s): {sorted(unknown)}")
    # Download only the small index, selected voice cards/anchors, and winning
    # audio. The full upstream dataset is nearly 100 GB and is not required.
    for name in ("index.json", "schema.json", "reference/reference_target.mp3"):
        hf_hub_download(REPO, name, revision=REVISION,
                        repo_type="dataset", local_dir=OUT)
    for vi, voice in enumerate(only, 1):
        for name in (f"pilot/{voice}/reference.wav", f"pilot/{voice}/voice.json"):
            hf_hub_download(REPO, name, revision=REVISION,
                            repo_type="dataset", local_dir=OUT)
        dst = os.path.join(OUT, "audio", "profiles", voice)
        idx_p = os.path.join(OUT, f"index_profile_{voice}.json")
        if os.path.exists(idx_p):
            print(f"[{vi}/{len(only)}] {voice}: already done", flush=True)
            continue
        os.makedirs(dst, exist_ok=True)
        t0 = time.time()
        want = winners(voice)
        print(f"[{vi}/{len(only)}] {voice}: {len(want)} winners to extract", flush=True)
        idx, got = {}, 0
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
                    for m in tf:
                        if not m.name.endswith(".mp3"):
                            continue
                        key = m.name[:-4]
                        gid = want.get(key)
                        if gid is None:
                            continue
                        fn = key.replace("/", "_") + ".mp3"
                        with open(os.path.join(dst, fn), "wb") as f:
                            f.write(tf.extractfile(m).read())
                        idx[gid] = f"audio/profiles/{voice}/{fn}"
                        got += 1
            finally:
                try:
                    os.remove(tp)
                except OSError:
                    pass
            print(f"    shard {shard}: {got}/{len(want)} extracted", flush=True)
        if not idx:
            raise RuntimeError(f"No reference audio extracted for {voice}; check the upstream shards")
        with open(idx_p, "w", encoding="utf-8") as f:
            json.dump(idx, f)
        mb = sum(os.path.getsize(os.path.join(dst, f)) for f in os.listdir(dst)) / 1e6
        print(f"    {voice}: {len(idx)} clips, {mb:.0f} MB, {time.time()-t0:.0f}s",
              flush=True)
    shutil.rmtree(os.path.join(OUT, "_tmp"), ignore_errors=True)
    print("PROFILEREFS_DONE", flush=True)


if __name__ == "__main__":
    main()
