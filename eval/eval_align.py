#!/usr/bin/env python3
"""How often does the end-trim fire, and how much does it remove?

Generates each line once with trimming off, then applies the trim to that same
audio locally.  Both halves are therefore the identical take — the only
difference is the edit — which is a cleaner comparison than generating twice and
is half the GPU time.

Three things are reported per take: whether either edge fired, how much audio
went away, and whether the words the script asked for are still all there.  The
last one is the guard rail: a trim that removes filler is the point, a trim that
eats the last real word is a regression, and `extra_w` / `wer` from the existing
harness measure exactly those two.
"""
import argparse, base64, json, sys, time

import numpy as np
import requests

sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import align_engine as ae
import config
import timed_script
from eval_tail import ASR, GENERAL, wer, trailing_words, resample_48k_to_16k
from eval_scale import ANCHOR, DPO

URL = "http://127.0.0.1:8792/api/say"

# The fixed ten from the other harnesses do not carry bursts and are all calm.
# These add an opening burst, a closing burst and some louder registers, because
# the lead-in rule turns on exactly the burst case and the tail is the part that
# fails when the model is pushed.
LINES = [
    ("(clearly amused, warm) I still cannot believe the cat opened that door by herself.", "plain"),
    ("(quietly, almost hushed) There was a moment this afternoon when the whole street went silent.", "plain"),
    ("(clearly tired) I read the same paragraph four times and understood it on none of them.", "plain"),
    ("(warmly) She sent me a photograph of the sea and said nothing else, and it was enough.", "plain"),
    ("(dryly amused) The lift has been broken since March and I have never been fitter.", "plain"),
    ("(sharp inhale) (intensely surprised) You did what with the entire budget for this quarter?", "opens_burst"),
    ("(chuckle) (clearly amused) He actually wore the costume to the wedding reception.", "opens_burst"),
    ("(clearly delighted) That is the best news I have heard all week. (breathy giggle)", "ends_burst"),
    ("(strongly indignant, voice rising) Nobody asked me before they moved the whole meeting.", "loud"),
    ("(intensely upset, barely holding it) I do not know how to tell them what happened.", "loud"),
    ("(clearly conspiratorial, low) Between the two of us, that report was never finished.", "plain"),
    ("(warmly, unhurried) The bread came out right for the first time in about a year.", "plain"),
]


def say(tagged, frames, loras, seed, anchor=ANCHOR):
    body = {"text": tagged, "tokens": int(frames), "language": "English",
            "seed": int(seed), "anchor_path": anchor,
            "loras": [[n, l] for n, l in loras],
            "instruction": f"GENERAL: {GENERAL}; {frames/12.5:.1f}s, EN.\nSCRIPT:\n{tagged}"}
    r = requests.post(URL, json=body, timeout=900)
    r.raise_for_status()
    j = r.json()
    pcm = np.frombuffer(base64.b64decode(j["pcm"]), "<i2").astype(np.float32) / 32768.0
    return pcm, j.get("sr", 48000)


STACK = [(DPO, 1.0), ("sft3_quality:genuineness_high", 0.25),
         ("sft3_quality:blend_high", 0.5), ("sft3_quality:esthetics_high", 0.5),
         ("sft3_voice:emolia_c1699", 0.25), ("sft3_qdpo:quality_dpo", 1.0),
         ("sft3_emotion:Amusement", 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bsdpo", type=float, default=0.0,
                    help="also load the burst+stop preference adapter at this weight")
    ap.add_argument("--backend", default="mms", choices=["mms", "qwen"])
    ap.add_argument("--out", default="/mnt/nvme/moss-15-v2-assets/align_eval.json")
    a = ap.parse_args()
    stack = list(STACK)
    if a.bsdpo > 0:
        stack.append(("sft3_qdpo:burst_stop_dpo", a.bsdpo))
    asr = ASR()
    aligner = (ae.QwenAligner(device="cuda:0") if a.backend == "qwen"
               else ae.Aligner(device="cpu"))   # the server holds the MMS GPU session
    rows = []
    for li, (script, kind) in enumerate(LINES):
        tagged, frames, plain = timed_script.render(script)
        for s in range(a.seeds):
            seed = 1234 + 1000 * s
            try:
                pcm, sr = say(tagged, frames, stack, seed)
            except Exception as e:
                print(f"  line {li} seed {seed}: {str(e)[:110]}", flush=True)
                continue
            hyp_a, last_a = asr.run(resample_48k_to_16k(pcm))
            extra_a, xw_a = trailing_words(plain, hyp_a)
            y, rep = ae.trim(pcm, sr, tagged, plain, aligner)
            hyp_b, _ = asr.run(resample_48k_to_16k(y))
            extra_b, xw_b = trailing_words(plain, hyp_b)
            rows.append({
                "line": li, "kind": kind, "seed": seed,
                "sec_before": round(len(pcm) / sr, 3),
                "sec_after": round(len(y) / sr, 3),
                "applied": rep["applied"], "lead_s": rep["lead_s"],
                "cut_s": rep["cut_s"], "removed_s": rep["removed_s"],
                "note": rep["note"],
                "extra_before": extra_a, "extra_after": extra_b,
                "wer_before": wer(plain, hyp_a), "wer_after": wer(plain, hyp_b),
                "xw_before": " ".join(xw_a)[:80], "xw_after": " ".join(xw_b)[:80],
            })
            r = rows[-1]
            print(f'[{len(rows)}] line {li} {kind:11s} seed {seed}  '
                  f'{"EDIT" if r["applied"] else "----"} '
                  f'lead={str(r["lead_s"]):>5} cut={str(r["cut_s"]):>6} '
                  f'-{r["removed_s"]:.2f}s  extra {r["extra_before"]}->{r["extra_after"]}  '
                  f'wer {r["wer_before"]:.3f}->{r["wer_after"]:.3f}', flush=True)
            json.dump(rows, open(a.out, "w"), indent=1)
    if rows:
        n = len(rows)
        f = lambda k: float(np.mean([r[k] for r in rows]))
        print(f'\n{n} takes | edited {sum(r["applied"] for r in rows)}/{n} '
              f'({sum(r["applied"] for r in rows)/n:.0%})')
        print(f'  lead-in fired  {sum(r["lead_s"] is not None for r in rows)}/{n}')
        print(f'  tail cut fired {sum(r["cut_s"] is not None for r in rows)}/{n}')
        print(f'  removed        mean {f("removed_s"):.3f}s  '
              f'max {max(r["removed_s"] for r in rows):.3f}s')
        print(f'  extra words    {f("extra_before"):.2f} -> {f("extra_after"):.2f}')
        print(f'  word error     {f("wer_before"):.3f} -> {f("wer_after"):.3f}')
    print("ALIGNEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
