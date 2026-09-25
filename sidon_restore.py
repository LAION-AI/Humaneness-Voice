#!/usr/bin/env python3
"""Level and bandwidth-restore one clip, for use as a voice-conversion target.

    python sidon_restore.py <in> <out> [target_dbfs]

Runs as its own process on purpose.  Sidon ships as TorchScript with its
constant tensors baked onto cuda:0 at export time; loading it onto any other
device fails inside the interpreter when it indexes a positional-embedding
table.  Giving it a process whose only visible card *is* cuda:0 sidesteps that,
and the ~1 GB of weights disappear with the process instead of sitting in the
server for the rest of the session.
"""
import os
import subprocess
import sys

import numpy as np
import soundfile as sf


def normalise(w, target_dbfs=-16.0, peak_ceiling=0.97):
    rms = float(np.sqrt((w ** 2).mean())) or 1e-9
    g = 10 ** ((target_dbfs - 20 * np.log10(rms)) / 20)
    peak = float(np.abs(w).max()) or 1e-9
    return (w * min(g, peak_ceiling / peak)).astype(np.float32)


def main():
    src, dst = sys.argv[1], sys.argv[2]
    dbfs = float(sys.argv[3]) if len(sys.argv) > 3 else -16.0
    ck = os.environ.get("MOSS_SIDON_CKPTS", "/mnt/nvme/moss-15-v2-assets/sidon-ckpts")
    srcdir = os.environ.get("MOSS_SIDON_SRC", "/mnt/nvme/moss-15-v2-assets/sidon/src")
    out_sr = int(os.environ.get("MOSS_SIDON_OUT_SR", "48000"))

    tmp = dst + ".16k.wav"
    subprocess.run(["ffmpeg", "-nostdin", "-y", "-i", src, "-ac", "1",
                    "-ar", "16000", tmp],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    w, _ = sf.read(tmp, dtype="float32")
    if w.ndim > 1:
        w = w.mean(1)
    # level first: the model should restore speech, not a whisper
    w = normalise(w, dbfs)

    import torch
    sys.path.insert(0, srcdir)
    from sidon.cleansing.audio import extract_seamless_m4t_features

    fe = torch.jit.load(os.path.join(ck, "feature_extractor_cuda.pt"),
                        map_location="cuda").eval()
    de = torch.jit.load(os.path.join(ck, "decoder_cuda.pt"),
                        map_location="cuda").eval()
    with torch.inference_mode():
        f = extract_seamless_m4t_features([torch.from_numpy(w)],
                                          return_tensors="pt",
                                          padding_value=1.0, device="cuda")
        h = fe(f["input_features"].cuda())["last_hidden_state"]
        o = de(h.transpose(1, 2)).cpu().squeeze().view(-1)
    # and level again: restoration changes the level
    o = normalise(o.numpy().astype(np.float32), dbfs)
    sf.write(dst, o, out_sr)
    try:
        os.remove(tmp)
    except OSError:
        pass
    print(f"OK {len(o)/out_sr:.2f}s {out_sr}Hz peak={np.abs(o).max():.3f}")


if __name__ == "__main__":
    main()
