"""Client for the SIDON restoration service.

Every failure is soft: if the service is down, slow or unhappy, `enhance`
returns exactly what it was given and the caller cannot tell the difference
except from the log line.
"""
import time

import numpy as np
import requests

import config

_STATE = {"up": None, "checked": 0.0, "fails": 0}


def health():
    try:
        r = requests.get(f"{config.SIDON_BASE}/health", timeout=2.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def up():
    """Cached liveness, rechecked every few seconds rather than every clip."""
    now = time.time()
    if _STATE["up"] is None or now - _STATE["checked"] > 15:
        _STATE["up"] = bool(health())
        _STATE["checked"] = now
    return _STATE["up"]


def enhance_many(waves, sr=48000):
    """Restore a list of float32 waveforms in one request."""
    waves = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
    if not waves or not config.SIDON_ON:
        return waves, False
    if not up():
        return waves, False
    try:
        flat = np.concatenate(waves) if len(waves) > 1 else waves[0]
        body = np.clip(flat * 32767, -32768, 32767).astype("<i2").tobytes()
        t0 = time.time()
        r = requests.post(f"{config.SIDON_BASE}/enhance",
                          params={"sr": sr, "n": len(waves)}, data=body,
                          timeout=config.SIDON_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:120]}")
        lens = [int(x) for x in r.headers.get("x-lengths", "").split(",") if x]
        pcm = np.frombuffer(r.content, dtype="<i2").astype(np.float32) / 32768.0
        out, at = [], 0
        for n in lens:
            out.append(pcm[at:at + n])
            at += n
        if len(out) != len(waves):
            raise RuntimeError(f"got {len(out)} clips for {len(waves)}")
        print(f"[sidon] restored {len(out)} clip(s) in "
              f"{(time.time()-t0)*1000:.0f} ms", flush=True)
        _STATE["fails"] = 0
        return out, True
    except Exception as e:
        _STATE["fails"] += 1
        _STATE["up"] = False
        _STATE["checked"] = time.time()
        print(f"[sidon] unavailable, keeping the originals: "
              f"{type(e).__name__}: {str(e)[:140]}", flush=True)
        return waves, False


def enhance(wave, sr=48000):
    out, ok = enhance_many([wave], sr)
    return out[0], ok
