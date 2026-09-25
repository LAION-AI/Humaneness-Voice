#!/usr/bin/env python3
"""Measure whether the assistant keeps one identity across a conversation.

Runs a fixed conversation through /api/turn, then scores every reply with ECAPA:
  - vs anchor : cosine to the corpus target speaker (reference_target.mp3)
  - pairwise  : cosine between replies, i.e. does it stay the same person

The manual's thresholds for this model family: regenerate below 0.58, attempt
repair below 0.45, reject below 0.40.  Those are for spoken material — screams
and sobs legitimately score lower and are excluded from the verdict.

  python eval_consistency.py [tag]
"""
import glob, itertools, json, os, struct, subprocess, sys, warnings

import httpx
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

API = os.environ.get("MOSS_API", "http://127.0.0.1:8792")
TAG = sys.argv[1] if len(sys.argv) > 1 else "run"
OUT = os.path.join("/tmp/moss_eval", TAG)

CONVO = [
    "Hallo, wer bist du denn?",
    "Ich hab heute meinen Job verloren.",
    "Aber morgen hab ich ein neues Vorstellungsgespraech!",
    "Tell me a joke.",
    "Now say that again but annoyed.",
    "What do you think about the weather?",
    "I am really proud of you.",
    "Explain quantum physics in one sentence.",
]


def turn(msg, path, hist):
    buf, pcm, evs = b"", [], []
    with httpx.stream("POST", f"{API}/api/turn",
                      json={"message": msg, "history": hist}, timeout=300) as r:
        for c in r.iter_bytes():
            buf += c
            while len(buf) >= 5:
                tag = buf[0]
                ln = struct.unpack(">I", buf[1:5])[0]
                if len(buf) < 5 + ln:
                    break
                p, buf = buf[5:5 + ln], buf[5 + ln:]
                if tag == 0:
                    evs.append(json.loads(p))
                else:
                    pcm.append(np.frombuffer(p, "<i2").astype(np.float32) / 32768)
    w = np.concatenate(pcm) if pcm else np.zeros(1, np.float32)
    sf.write(path, w, 48000)
    llm = [e for e in evs if e["type"] == "llm"][0]
    c = llm.get("chosen") or {}
    label = (c.get("emotion") or c.get("character") or c.get("dimension")
             or c.get("edge_case") or "none")
    cond = f"{label}/{c.get('level') or ''}{c.get('intensity') or ''}{c.get('containment') or ''}"
    return llm, cond, c.get("block")


def main():
    os.makedirs(OUT, exist_ok=True)
    hist, rows = [], []
    for i, m in enumerate(CONVO):
        path = f"{OUT}/t{i}.wav"
        llm, cond, block = turn(m, path, hist)
        hist += [{"role": "user", "content": m},
                 {"role": "assistant", "content": llm["reply"]}]
        rows.append((path, cond, block))
        print(f"{i}: {cond:46s} | {llm['reply'][:58]}")

    from speechbrain.inference.speaker import EncoderClassifier
    import torch, torchaudio
    enc = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        savedir="/tmp/moss_eval/ecapa", run_opts={"device": "cpu"})

    def emb(p):
        w, sr = sf.read(p, dtype="float32")
        if w.ndim > 1:
            w = w.mean(1)
        t = torch.from_numpy(w)[None]
        if sr != 16000:
            t = torchaudio.functional.resample(t, sr, 16000)
        with torch.no_grad():
            e = enc.encode_batch(t).squeeze()
        return torch.nn.functional.normalize(e, dim=-1)

    cos = lambda a, b: float((a * b).sum())
    ref = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/datasets--TTS-AGI--moss-voice-profile-references"
        "/snapshots/*/reference/reference_target.mp3"))[0]
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", ref, "-ac", "1",
                    "-ar", "16000", f"{OUT}/anchor.wav"], capture_output=True)
    A = emb(f"{OUT}/anchor.wav")

    # non-verbal takes break ECAPA by design; score them but keep them out of the verdict
    spoken = [(p, c, b) for p, c, b in rows if b != "edge_case"]
    E = {p: emb(p) for p, _, _ in rows}
    to_a = [cos(A, E[p]) for p, _, _ in spoken]
    pair = [cos(E[a[0]], E[b[0]]) for a, b in itertools.combinations(spoken, 2)]

    print(f"\n{'condition':46s} vs anchor")
    for p, c, _ in rows:
        print(f"{c:46s} {cos(A, E[p]):.3f}")
    print(f"\nspoken clips        : {len(spoken)}/{len(rows)}")
    print(f"vs anchor           : mean {np.mean(to_a):.3f}  min {min(to_a):.3f}")
    print(f"turn-to-turn        : mean {np.mean(pair):.3f}  min {min(pair):.3f}")
    print(f"below 0.40 (reject) : {sum(1 for x in to_a if x < 0.40)}/{len(to_a)}")
    print(f"unique conditions   : {len({c for _, c, _ in rows})}/{len(rows)}")


if __name__ == "__main__":
    main()
