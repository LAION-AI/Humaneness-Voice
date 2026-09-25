"""Client for the vocal-burst detector service.

Soft in every direction: if the service is down, slow or unhappy, the caller
gets `None` for every candidate, `rank` then forces the burst term to exactly 0
and the ranking is what it was before the term existed.
"""
import json
import time

import numpy as np
import requests

import config

_STATE = {"up": None, "checked": 0.0}


def health():
    try:
        r = requests.get(f"{config.BURST_BASE}/health", timeout=2.0)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def up():
    now = time.time()
    if _STATE["up"] is None or now - _STATE["checked"] > 15:
        h = health()
        _STATE["up"] = bool(h and h.get("ok"))
        _STATE["checked"] = now
    return _STATE["up"]


def score(waves, sr, bursts):
    """One burst score per wave, or all None.

    `bursts` is [(label, onset_seconds), ...] from `timed_script.burst_onsets`,
    the same for every candidate because they render one script.
    """
    n = len(waves)
    if not waves or not bursts or not config.BON_BURST_READY or not up():
        return [None] * n
    try:
        arrs = [np.asarray(w, dtype=np.float32).reshape(-1) for w in waves]
        flat = np.concatenate(arrs) if len(arrs) > 1 else arrs[0]
        body = np.clip(flat * 32767, -32768, 32767).astype("<i2").tobytes()
        t0 = time.time()
        r = requests.post(
            f"{config.BURST_BASE}/score", params={"sr": sr}, data=body,
            headers={"x-lengths": ",".join(str(len(a)) for a in arrs),
                     "x-bursts": json.dumps([{"label": l, "onset": t}
                                             for l, t in bursts])},
            timeout=config.BURST_TIMEOUT)
        if r.status_code != 200:
            raise RuntimeError(f"{r.status_code} {r.text[:120]}")
        got = r.json().get("burst") or []
        if len(got) != n:
            raise RuntimeError(f"got {len(got)} scores for {n} candidates")
        print(f"[burst] scored {n} candidate(s) over {len(bursts)} burst(s) "
              f"in {(time.time()-t0)*1000:.0f} ms", flush=True)
        return [None if v is None else float(v) for v in got]
    except Exception as e:
        _STATE["up"] = False
        _STATE["checked"] = time.time()
        print(f"[burst] unavailable, ranking without the term: "
              f"{type(e).__name__}: {str(e)[:130]}", flush=True)
        return [None] * n
