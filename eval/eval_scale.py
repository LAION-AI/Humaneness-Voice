#!/usr/bin/env python3
"""How far can each adapter be turned up before it costs intelligibility?

The first sweep established that SFT3 + DPO-p2 is clean (word error 0.013, no
invented words in ten of ten takes) and that every further adapter degrades it.
This one walks four adapters up their scale one at a time, on the same ten
sentences, and measures four things per take:

  wer / extra_w   intelligibility, and words transcribed past the end of the line
  genuineness     laion/voiceclap-commercial-genuineness, 0-6
  blend           laion/voiceclap-commercial-vocalburst-blend, 0-10
  spk_sim         ECAPA cosine against the voice's own reference recording

The last one answers a question worth asking before tuning anything: the prompt
already carries that recording as a reference clip, so how much identity does
the voice adapter actually add on top of it?
"""
import argparse, json, os, sys, tempfile, time

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import config
from eval_tail import (ASR, PROMPTS, GENERAL, say, wer, trailing_words,
                       resample_48k_to_16k)
import timed_script

DPO = "sft3_dpo:p2"
ANCHOR = "/mnt/nvme/moss-15-v2-assets/refs2/pilot/emolia_c1699/reference.wav"

SCALES = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5]
LADDERS = [
    ("genuineness", "sft3_quality:genuineness_high", SCALES),
    ("blend",       "sft3_quality:blend_high",       SCALES),
    ("esthetics",   "sft3_quality:esthetics_high",   SCALES),
    ("voice",       "sft3_voice:emolia_c1699",       SCALES),
]


class Scorers:
    def __init__(self, device="cuda:0"):
        sys.path.insert(0, "/mnt/nvme/moss-15-v2-assets/vc_genuineness")
        sys.path.insert(0, "/mnt/nvme/moss-15-v2-assets/vc_blend")
        from genuineness_scorer import GenuinenessScorer
        from blend_model import CommercialBlendScorer
        self.gen = GenuinenessScorer(pkg_dir="/mnt/nvme/moss-15-v2-assets/vc_genuineness",
                                     model="full", device=device)
        self.bl = CommercialBlendScorer(pkg_dir="/mnt/nvme/moss-15-v2-assets/vc_blend",
                                        device=device)
        from sim_engine import SpeakerSim
        self.sim = SpeakerSim(ANCHOR, device=device)

    def score(self, pcm48):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            sf.write(f.name, np.asarray(pcm48, np.float32), 48000)
            p = f.name
        try:
            g = float(self.gen.score(p))
            b = float(self.bl.score(p))
            try:
                s = float(self.sim.score(np.asarray(pcm48, np.float32), 48000))
            except Exception:
                s = float("nan")
        finally:
            os.unlink(p)
        return g, b, s


def run_condition(name, loras, asr, sc, n, seed=1234):
    rows = []
    for pi, (line, cue) in enumerate(PROMPTS[:n]):
        tagged, frames, _ = timed_script.render(f"{cue} {line}")
        try:
            pcm, _ = say(tagged,
                         f"GENERAL: {GENERAL}; {frames/12.5:.1f}s, EN.\nSCRIPT:\n{tagged}",
                         frames, loras, seed=seed, anchor_path=ANCHOR)
        except Exception as e:
            print(f"  [{name}] prompt {pi}: {str(e)[:100]}", flush=True)
            continue
        hyp, last_t = asr.run(resample_48k_to_16k(pcm))
        extra, xw = trailing_words(line, hyp)
        g, b, s = sc.score(pcm)
        rows.append({"prompt": pi, "wer": wer(line, hyp), "extra_w": extra,
                     "tail_s": len(pcm) / 48000.0 - last_t,
                     "genuineness": g, "blend": b, "spk_sim": s,
                     "hyp": hyp, "extra_words": " ".join(xw)[:100]})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--out", default="/mnt/nvme/moss-15-v2-assets/scale_eval.json")
    a = ap.parse_args()
    asr, sc = ASR(), Scorers()
    out = []

    def emit(name, loras):
        rows = run_condition(name, loras, asr, sc, a.n)
        if not rows:
            return
        m = lambda k: float(np.nanmean([r[k] for r in rows]))
        rec = {"condition": name, "loras": loras, "n": len(rows),
               "wer": m("wer"), "extra_w": m("extra_w"), "tail_s": m("tail_s"),
               "genuineness": m("genuineness"), "blend": m("blend"),
               "spk_sim": m("spk_sim"),
               "pct_extra": float(np.mean([r["extra_w"] > 0 for r in rows])),
               "takes": rows}
        out.append(rec)
        print(f'{name:26s} wer={rec["wer"]:.3f} extra={rec["extra_w"]:4.1f} '
              f'({rec["pct_extra"]:.0%})  genuine={rec["genuineness"]:.2f} '
              f'blend={rec["blend"]:.2f} spk={rec["spk_sim"]:.3f}', flush=True)
        json.dump(out, open(a.out, "w"), indent=1)

    emit("baseline sft3+dpo", [(DPO, 1.0)])
    for label, nm, scales in LADDERS:
        for w in scales:
            emit(f"{label} @{w}", [(DPO, 1.0), (nm, w)])
    print("SCALE_DONE", flush=True)


if __name__ == "__main__":
    main()
