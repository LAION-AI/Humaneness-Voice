#!/usr/bin/env python3
"""Distil the research steering-vector library into the ~5 MB the server needs.

WHY THIS EXISTS RATHER THAN A COMMITTED FILE.  The research library
(`p3_vectors_ext.npz`) is **112 MB**: three difference tables of 99 attributes x 38 taps x
2560 dimensions, in float32.  None of it belongs in git, and the server needs a small
fraction of it — one table (`d_hi_lo`, the high-minus-low difference), and only at each
attribute's own best few layers.  That is 99 x 5 x 2560 x 4 bytes = **5.1 MB**, which fits
alongside the other assets on /mnt/nvme with everything else that does not go in the
repository.

float32 is deliberate and is not negotiable down to float16: a steering direction is a
*difference* of means, one to two orders of magnitude smaller than the means themselves, and
half precision quantises it to a few significant bits.

WHICH LAYERS.  Per attribute, from `tap_rank.json` — the per-dimension ranking of where that
attribute is most decodable.  They differ: Anger peaks at h21 h20 h19, genuineness at
h12 h13 h21, blend at h25 h22 h20.  Five covers every k the shipped recipes use (emotion
k = 1, delivery k = 3-5, quality never).

Run it on a machine that has the research artefacts — a login node of the cluster the study
ran on — and copy the result to the demo box:

    python setup/build_steering_pack.py \\
        --vectors  $SC/out/actforensics/vectors/p3_vectors_ext.npz \\
        --tap-rank $SC/work_vb/tap_rank.json \\
        --out      /mnt/nvme/moss-15-v2-assets/steering/p3_vectors_server.npz

The tap ranking is embedded in the output, so the server needs exactly one file.
"""
import argparse
import json
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vectors", required=True,
                    help="p3_vectors_ext.npz from the layer-forensics study")
    ap.add_argument("--tap-rank", required=True,
                    help="tap_rank.json: the per-dimension layer ranking")
    ap.add_argument("--out", required=True)
    ap.add_argument("-k", type=int, default=5,
                    help="layers kept per attribute (default 5, which covers every "
                         "shipped k)")
    a = ap.parse_args()

    rank = json.load(open(a.tap_rank))["rank"]
    z = np.load(a.vectors, allow_pickle=True)
    names = [f"{k}:{d}" for k, d in zip(z["dim_kind"], z["dim_name"])]
    taps = [int(t) for t in z["taps"]]
    tap_ix = {t: i for i, t in enumerate(taps)}
    A = z["d_hi_lo"]
    assert A.shape[0] == len(names), (A.shape, len(names))

    K = int(a.k)
    V = np.zeros((len(names), K, A.shape[-1]), dtype=np.float32)
    T = np.full((len(names), K), -1, dtype=np.int16)
    n_missing = 0
    for i, nm in enumerate(names):
        dim = nm.split(":", 1)[1]
        r = rank.get(dim)
        if not r:
            n_missing += 1
            continue
        want = [37 if x == "loc" else int(x[1:]) for x in r["ranked"][:K]]
        for j, t in enumerate(want):
            if t not in tap_ix:
                continue
            T[i, j] = t
            V[i, j] = np.asarray(A[i, tap_ix[t]], dtype=np.float32)

    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    np.savez(a.out, names=np.array(names), taps=T, V=V,
             rank=json.dumps(rank),
             source=os.path.basename(a.vectors), k=np.int32(K))
    size = os.path.getsize(a.out) / 1e6
    print(f"wrote {a.out}: {len(names)} dimensions x {K} layers, {size:.1f} MB")
    if n_missing:
        print(f"  {n_missing} dimensions had no entry in the tap ranking and are empty; "
              "the server refuses to steer those rather than guessing a layer")
    kept = int((T >= 0).sum())
    print(f"  {kept} (attribute, layer) vectors kept out of "
          f"{len(names) * len(taps)} in the source file")


if __name__ == "__main__":
    main()
