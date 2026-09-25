"""SIDON restoration as its own small service.

It runs in a separate process for one reason: the TorchScript pair has its
constant tensors baked in at trace time, and constants move with neither
`map_location` nor `.to()`, so the CUDA build runs on `cuda:0` and nowhere else.
Loading it inside the app would therefore put it on the TTS card, where it costs
1.67 GiB and leaves 0.25 GiB of slack — not enough for a guided best-of-N batch,
which has run out of memory on this box before.

Started with `CUDA_VISIBLE_DEVICES` pointing at the language model's card, its
`cuda:0` is that card, and generation never sees the memory disappear.
"""
import io
import time

import numpy as np
import torch
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

import config

app = FastAPI()
M = {"fe": None, "dec": None, "proc": None, "ms": [], "n": 0}


def load():
    if M["fe"] is not None:
        return
    import glob
    from transformers import AutoFeatureExtractor
    snap = glob.glob(config.SIDON_SNAPSHOT)[0]
    t0 = time.time()
    M["fe"] = torch.jit.load(f"{snap}/feature_extractor_cuda.pt",
                             map_location="cuda:0").eval()
    M["dec"] = torch.jit.load(f"{snap}/decoder_cuda.pt",
                              map_location="cuda:0").eval()
    M["proc"] = AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")
    print(f"[sidon] ready on cuda:0 in {time.time()-t0:.1f}s", flush=True)


@torch.no_grad()
def restore(x, sr):
    import scipy.signal as ss
    x16 = ss.resample_poly(x, 16000, sr).astype(np.float32)
    feat = M["proc"](x16, sampling_rate=16000,
                     return_tensors="pt")["input_features"].to("cuda:0")
    h = M["fe"](feat)["last_hidden_state"]
    w = M["dec"](h.transpose(1, 2)).squeeze().float().cpu().numpy()
    # match the level of the take it replaces, so a restored clip never arrives
    # louder than the original sitting next to it in the UI
    pin = float(np.abs(x).max()) or 0.9
    pout = float(np.abs(w).max()) or 1.0
    return (w / pout * pin).astype(np.float32)


@app.on_event("startup")
def _startup():
    load()


@app.get("/health")
def health():
    ms = M["ms"][-20:]
    return {"ok": M["fe"] is not None, "clips": M["n"],
            "ms_per_clip": round(sum(ms) / len(ms), 1) if ms else None}


@app.post("/enhance")
async def enhance(req: Request):
    """Raw little-endian int16 in, raw little-endian int16 out, 48 kHz.

    `sr` and `n` are query parameters so the body stays a bare buffer: a batch
    of eight 15-second takes is 23 MB, and base64 on both sides of that is
    pointless work.
    """
    raw = await req.body()
    if not raw:
        return JSONResponse({"error": "empty"}, status_code=400)
    sr = int(req.query_params.get("sr", 48000))
    n = int(req.query_params.get("n", 1))
    try:
        load()
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        parts = np.array_split(pcm, n) if n > 1 else [pcm]
        out, t0 = [], time.time()
        for p in parts:
            if p.size < sr // 8:
                out.append(p)
                continue
            out.append(restore(p, sr))
        M["ms"].append((time.time() - t0) * 1000 / max(n, 1))
        M["n"] += n
        lens = [len(o) for o in out]
        buf = np.concatenate(out) if len(out) > 1 else out[0]
        i16 = np.clip(buf * 32767, -32768, 32767).astype("<i2").tobytes()
        return Response(content=i16, media_type="application/octet-stream",
                        headers={"x-lengths": ",".join(map(str, lens)),
                                 "x-sr": "48000"})
    except Exception as e:
        print(f"[sidon] failed: {type(e).__name__}: {e}", flush=True)
        return JSONResponse({"error": f"{type(e).__name__}: {e}"},
                            status_code=500)
