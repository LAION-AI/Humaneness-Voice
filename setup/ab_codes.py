#!/usr/bin/env python3
"""Dump the generated RVQ code tensor for one fixed take, so two trees can be byte-compared.

This exists to falsify one specific claim: that `adapter` mode is **bit-identical to what
this server did before generation modes existed**.  `setup/check_levers.py` shows the
injector is a true no-op object and that the loop shape is unchanged, which is good evidence
and is not the same thing as a byte comparison.

Compare the CODES, not the audio.  The codec decode is deterministic given identical codes,
so identical codes is the stronger claim and it avoids every float-comparison argument about
the decoder, the crossfade and the sliding window.

USAGE -- run it once on each tree, then diff the two hashes:

    git checkout <commit before generation modes>
    python setup/ab_codes.py --out /tmp/before.npz
    git checkout <this branch>
    python setup/ab_codes.py --out /tmp/after.npz
    python setup/ab_codes.py --compare /tmp/before.npz /tmp/after.npz

It deliberately imports NOTHING that generation modes added -- only `config` and
`TTSEngine`, which exist on both sides with the same signatures -- so the same file can be
copied to the older checkout and run there unchanged.  Copy it, do not rely on it being in
the older tree.

It calls `_stream_frames` directly rather than `stream_pcm`, because that is the loop the
claim is about and it hands back the codes without going near the decoder.
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config                                                        # noqa: E402
from tts_engine import TTSEngine                                     # noqa: E402

# One fixed line, one fixed instruction.  Nothing here is sampled from the corpus or the
# director, so the take depends only on the model, the adapters and the seed.
TEXT = ("[0.3 seconds pause] [3.4 seconds duration] I told you this would happen, and you "
        "did not want to hear it. [0.4 seconds pause] [2.6 seconds duration] Now look "
        "where we have ended up.")
INSTRUCTION = (
    "GENERAL: " + config.SPEAKER_IDENTITY + " quietly furious, holding it down hard; "
    "close conversational volume, unforced; same speaker throughout; 6.7s, EN.\n"
    "SCRIPT:\n" + TEXT)


def run(args):
    tts = TTSEngine(device=args.device)
    specs = []
    if args.loras:
        from lora_bank import LoraBank
        lb = LoraBank(tts.model, tts.device)
        lb.discover(config.LORA_ROOTS)
        tts.lora = lb
        for item in args.loras.split(","):
            if not item.strip():
                continue
            name, _, w = item.partition("=")
            specs.append((name.strip(), float(w or 1.0)))
        print(f"[ab] adapters: {specs}", flush=True)

    p = dict(config.DEFAULTS)
    p["seed"] = args.seed
    p["max_new_tokens"] = args.max_frames
    p["stop_bias"] = config.STOP_BIAS

    applied = []
    with tts.lock:
        if tts.lora is not None:
            applied = tts.lora.apply(specs)
        try:
            iid, am = tts.build_inputs(TEXT, INSTRUCTION, "English", None,
                                       p["seed"], tokens=args.tokens)
            frames = None
            # chunk larger than any take -> one yield, at the end
            for fr in tts._stream_frames(iid, am, p, 10 ** 9,
                                         min_frames=int(args.tokens * 0.55)):
                frames = fr
        finally:
            if tts.lora is not None:
                tts.lora.clear()

    codes = torch.stack(frames, dim=1)[0].cpu().numpy().astype(np.int32)
    h = hashlib.sha256(codes.tobytes()).hexdigest()
    np.savez(args.out, codes=codes, seed=np.int64(args.seed),
             loras=np.array([f"{n}={w}" for n, w in applied]))
    print(f"[ab] frames {codes.shape[0]}  channels {codes.shape[1]}")
    print(f"[ab] sha256 {h}")
    print(f"[ab] wrote  {args.out}")
    return 0


def compare(a, b):
    A, B = np.load(a), np.load(b)
    ca, cb = A["codes"], B["codes"]
    ha = hashlib.sha256(ca.tobytes()).hexdigest()
    hb = hashlib.sha256(cb.tobytes()).hexdigest()
    print(f"  A {a}: {ca.shape} sha256 {ha}")
    print(f"  B {b}: {cb.shape} sha256 {hb}")
    if list(A.get("loras", [])) != list(B.get("loras", [])):
        print(f"  !! different adapter sets: {list(A['loras'])} vs {list(B['loras'])}")
    if ca.shape != cb.shape:
        print(f"  DIFFERENT: shapes {ca.shape} vs {cb.shape} — the takes diverged and one "
              "ended earlier")
        n = min(ca.shape[0], cb.shape[0])
        d = np.nonzero((ca[:n] != cb[:n]).any(axis=1))[0]
        print(f"  first differing frame: {int(d[0]) if d.size else 'none before the cut'}")
        return 1
    if ha == hb:
        print("  IDENTICAL — every one of "
              f"{ca.shape[0]} frames x {ca.shape[1]} channels matches")
        return 0
    diff = np.nonzero((ca != cb).any(axis=1))[0]
    print(f"  DIFFERENT: {diff.size} of {ca.shape[0]} frames differ, first at frame "
          f"{int(diff[0])} ({int(diff[0]) / config.FRAME_RATE:.2f} s in)")
    ch = np.nonzero((ca != cb).any(axis=0))[0]
    print(f"  channels affected: {ch.tolist()}")
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/tmp/ab_codes.npz")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tokens", type=int, default=84)
    ap.add_argument("--max-frames", type=int, default=280)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--loras", default="",
                    help="comma-separated name=weight, e.g. "
                         "'sft3_dpo:p2=1.0,sft3_emotion:Anger=1.0'. Empty = bare model.")
    a = ap.parse_args()
    if a.compare:
        return compare(*a.compare)
    return run(a)


if __name__ == "__main__":
    sys.exit(main())
