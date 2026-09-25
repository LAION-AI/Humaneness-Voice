#!/usr/bin/env python3
"""Derive each voice profile's gender, age and timbre from its own recordings.

The shipped profile cards disagreed with the audio: three of the ten are
labelled with the wrong gender, which makes the picker actively misleading.
Rather than trust the cards, this reads the VoiceNet predictions the corpus
already carries for every take and takes the profile's own average.

The scales are not centred on zero and had to be calibrated against the corpus's
own manipulated conditions, which are the only labelled points available:

    GEND  extremely_low 1.45 · moderately_low 2.05 · moderately_high 4.04 ·
          very_high 4.31        -> neutral sits near 3.0, high is masculine
    AGEV  extremely_low 2.26 · moderately_low 3.03 · moderately_high 3.79 ·
          very_high 4.08        -> neutral near 3.4, high is old

Clips whose condition deliberately manipulates GEND or AGEV are excluded, or a
voice would be judged partly on takes designed to push it away from itself.
"""
import json, glob, os, statistics as st, sys

import config

FEMALE_NAMES = ["Nora", "Mira", "Selma", "Juno", "Alba", "Wren"]
MALE_NAMES = ["Anton", "Rasmus", "Cormac", "Idris", "Béla", "Osku"]
NEUTRAL_NAMES = ["Robin", "Andrea", "Kim"]

# timbre keyword: whichever of these dimensions is furthest from the corpus norm
TIMBRE = {"ROUG": ("gravelly", "smooth"), "BRGT": ("bright", "dark"),
          "WARM": ("warm", "cool"), "R_CHST": ("deep", "light"),
          "BRTH": ("breathy", "firm"), "CLRT": ("clear", "muffled"),
          "S_WHIS": ("hushed", "full"), "RANG": ("wide-ranged", "even")}


def _vn(r):
    v = r.get("voicenet")
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except Exception:
            return {}
    return v or {}


def sex_of(g):
    return "f" if g < 2.6 else ("m" if g > 3.4 else "n")


def age_of(a):
    if a < 2.85:
        return "20s"
    if a < 3.30:
        return "30s"
    if a < 3.62:
        return "40s-50s"
    if a < 3.85:
        return "60s"
    return "70s+"


def main():
    out, corpus = {}, {}
    files = sorted(glob.glob(os.path.join(config.REF3_DIR, "meta_*.jsonl")))
    per = {}
    for f in files:
        voice = os.path.basename(f)[len("meta_"):-len(".jsonl")]
        acc = {}
        for line in open(f, encoding="utf-8"):
            r = json.loads(line)
            if r.get("dim") in ("GEND", "AGEV"):
                continue
            v = _vn(r)
            for k, val in v.items():
                if isinstance(val, (int, float)):
                    acc.setdefault(k, []).append(val)
                    corpus.setdefault(k, []).append(val)
        per[voice] = {k: st.mean(vs) for k, vs in acc.items() if vs}
    norm = {k: (st.mean(vs), st.pstdev(vs) or 1.0) for k, vs in corpus.items()}

    fi = mi = ni = 0
    for voice in sorted(per, key=lambda v: -per[v].get("GEND", 0)):
        m = per[voice]
        g, a = m.get("GEND", 3.0), m.get("AGEV", 3.4)
        sex, age = sex_of(g), age_of(a)
        best, bz = "", 0.0
        for dim, (hi, lo) in TIMBRE.items():
            if dim not in m or dim not in norm:
                continue
            mu, sd = norm[dim]
            z = (m[dim] - mu) / sd
            if abs(z) > abs(bz):
                best, bz = (hi if z > 0 else lo), z
        if sex == "f":
            name = FEMALE_NAMES[fi % len(FEMALE_NAMES)]; fi += 1
        elif sex == "m":
            name = MALE_NAMES[mi % len(MALE_NAMES)]; mi += 1
        else:
            name = NEUTRAL_NAMES[ni % len(NEUTRAL_NAMES)]; ni += 1
        out[voice] = {"name": name, "sex": sex, "age": age, "timbre": best,
                      "gend": round(g, 2), "agev": round(a, 2),
                      "timbre_z": round(bz, 2),
                      "short": f"{sex}, {age}, {best}".replace("n, ", "")}
    p = os.path.join(config.RETRIEVAL_DIR, "profile_traits.json")
    json.dump(out, open(p, "w", encoding="utf-8"), indent=1)
    for v, d in out.items():
        print(f'{v:16s} GEND {d["gend"]:+5.2f} AGEV {d["agev"]:+5.2f}  '
              f'-> {d["name"]:8s} {d["short"]}')
    print("wrote", p)


if __name__ == "__main__":
    main()
