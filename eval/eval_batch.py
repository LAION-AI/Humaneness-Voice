#!/usr/bin/env python3
"""Adapter sweeps over /api/say_batch — one forward pass per condition.

Every condition is the same ten utterances on top of SFT3 + DPO-p2, with one
adapter added at one weight.  Four measurements per take:

  wer, extra_w   intelligibility and words invented past the end of the line
  genuineness    laion/voiceclap-commercial-genuineness, 0-6
  blend          laion/voiceclap-commercial-vocalburst-blend, 0-10
  spk_sim        ECAPA cosine against the voice's own reference recording

The point of the batch endpoint is throughput: the streaming path is batch 1
because audio has to start before the line ends, which a sweep does not need.
"""
import argparse, base64, json, os, sys, time

import numpy as np
import requests

sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import config
import timed_script
from eval_tail import ASR, PROMPTS, GENERAL, wer, trailing_words, resample_48k_to_16k
from eval_scale import Scorers, ANCHOR, DPO

URL = "http://127.0.0.1:8792/api/say_batch"


def build_items(n, fpw=None):
    if fpw is not None:
        config.TIMED_FRAMES_PER_WORD = float(fpw)
    return _build_items(n)


def _build_items(n):
    items, refs = [], []
    for line, cue in PROMPTS[:n]:
        tagged, frames, _ = timed_script.render(f"{cue} {line}")
        items.append({"text": tagged, "tokens": frames, "language": "English",
                      "instruction": f"GENERAL: {GENERAL}; {frames/12.5:.1f}s, EN."
                                     f"\nSCRIPT:\n{tagged}"})
        refs.append(line)
    return items, refs


def run(cond, loras, items, refs, asr, sc, seed=1234, stop_bias=None):
    body = {"items": items, "loras": [[n, l] for n, l in loras],
            "seed": seed, "anchor_path": ANCHOR}
    if stop_bias is not None:
        body["stop_bias"] = float(stop_bias)
    r = requests.post(URL, json=body, timeout=3600)
    r.raise_for_status()
    j = r.json()
    rows = []
    for i, b64 in enumerate(j["pcm"]):
        pcm = np.frombuffer(base64.b64decode(b64), "<i2").astype(np.float32) / 32768.0
        if len(pcm) < 4800:
            continue
        hyp, last_t = asr.run(resample_48k_to_16k(pcm))
        extra, xw = trailing_words(refs[i], hyp)
        g, bl, s = sc.score(pcm)
        rows.append({"prompt": i, "wer": wer(refs[i], hyp), "extra_w": extra,
                     "tail_s": len(pcm) / 48000.0 - last_t,
                     "genuineness": g, "blend": bl, "spk_sim": s,
                     "hyp": hyp, "extra_words": " ".join(xw)[:100]})
    if not rows:
        return None
    m = lambda k: float(np.nanmean([x[k] for x in rows]))
    return {"condition": cond, "loras": loras, "n": len(rows),
            "gpu_ms": j.get("gpu_ms"),
            "wer": m("wer"), "extra_w": m("extra_w"), "tail_s": m("tail_s"),
            "genuineness": m("genuineness"), "blend": m("blend"),
            "spk_sim": m("spk_sim"),
            "pct_extra": float(np.mean([x["extra_w"] > 0 for x in rows])),
            "takes": rows}


def adapters_of(kind):
    import glob
    root = config.LORA_ROOTS.get(kind, "")
    if not root or not os.path.isdir(root):
        return []
    return sorted(os.path.basename(os.path.dirname(f))
                  for f in glob.glob(os.path.join(root, "*", "adapter_model.safetensors")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--set", default="quality",
                    help="quality | voice | voicenet | emotion | baseline")
    ap.add_argument("--scales", default="0.25,0.5,0.75,1.0,1.25,1.5")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    scales = [float(x) for x in a.scales.split(",") if x.strip()]
    out_p = a.out or f"/mnt/nvme/moss-15-v2-assets/sweep_{a.set}.json"

    items, refs = build_items(a.n)
    asr, sc = ASR(), Scorers()
    if a.set == "fmt":
        # Does the duration tag itself cause the padding?  Four prompt shapes,
        # same words, same adapters.
        import re as _re
        G, B, E = ("sft3_quality:genuineness_high", "sft3_quality:blend_high",
                   "sft3_quality:esthetics_high")
        stack = [(DPO, 1.0), (G, 0.25), (B, 0.5), (E, 0.5),
                 ("sft3_voice:emolia_c1699", 0.25),
                 ("sft3_emotion:Amusement", 1.0)]
        def mk(mode, fpw):
            config.TIMED_FRAMES_PER_WORD = fpw
            its, rfs = [], []
            for line, cue in PROMPTS[:a.n]:
                tagged, frames, plain = timed_script.render(f"{cue} {line}")
                if mode in ("no_tail", "no_edges"):
                    # drop the closing [N.N seconds pause] the renderer appends
                    tagged = _re.sub(r"\s*\[[0-9.]+ seconds pause\]\s*$", "", tagged)
                    if mode == "no_edges":
                        tagged = _re.sub(r"^\s*\[[0-9.]+ seconds pause\]\s*", "", tagged)
                    secs = sum(float(m) for m in
                               _re.findall(r"\[([0-9.]+) seconds (?:duration|pause)\]", tagged))
                    secs += sum(float(m) for m in
                                _re.findall(r",\s*([0-9.]+) seconds\)", tagged))
                    frames = max(24, int(round(secs * 12.5)))
                elif mode == "no_duration":
                    tagged = _re.sub(r"\[[0-9.]+ seconds duration\]\s*", "", tagged)
                elif mode == "plain":
                    tagged = f"{cue} {line}"
                    frames = int(len(line.split()) * config.TOKENS_PER_WORD)
                its.append({"text": tagged, "tokens": frames, "language": "English",
                            "instruction": f"GENERAL: {GENERAL}; {frames/12.5:.1f}s, EN."
                                           f"\nSCRIPT:\n{tagged}"})
                rfs.append(line)
            return its, rfs
        res = []
        for label, mode, fpw in [("no trailing pause @4.5", "no_tail", 4.5),
                                 ("no trailing pause @4.0", "no_tail", 4.0),
                                 ("no trailing pause, no lead @4.0", "no_edges", 4.0),
                                 ("timed tags @4.5", "timed", 4.5),
                                 ("timed tags @4.0", "timed", 4.0),
                                 ("pauses+bursts, no duration tags", "no_duration", 4.5),
                                 ("plain text, tokens=words*6", "plain", 4.5)]:
            for sd in (1234, 777, 42):
                it, rf = mk(mode, fpw)
                rec = run(f"{label} | seed {sd}", stack, it, rf, asr, sc, seed=sd)
                if not rec:
                    continue
                res.append(rec)
                json.dump(res, open(out_p, "w"), indent=1)
                print(f'{label:34s} seed {sd}  wer={rec["wer"]:.3f} '
                      f'extra={rec["extra_w"]:4.1f}({rec["pct_extra"]:.0%}) '
                      f'tail={rec["tail_s"]:.2f}s', flush=True)
        print("SWEEP_DONE", flush=True)
        return
    if a.set == "fpw":
        # the duration budget itself: seconds per word asked for in the prompt
        G, B, E = ("sft3_quality:genuineness_high", "sft3_quality:blend_high",
                   "sft3_quality:esthetics_high")
        stack = [(DPO, 1.0), (G, 0.25), (B, 0.5), (E, 0.5),
                 ("sft3_voice:emolia_c1699", 0.25),
                 ("sft3_emotion:Amusement", 1.0)]
        res = []
        for fpw in [float(x) for x in os.environ.get("FPW_LIST","3.0,3.5,4.0,4.5,5.0,5.5").split(",")]:
            it, rf = build_items(a.n, fpw)
            rec = run(f"frames/word {fpw}", stack, it, rf, asr, sc, seed=a.seed)
            if not rec:
                continue
            rec["fpw"] = fpw
            res.append(rec)
            json.dump(res, open(out_p, "w"), indent=1)
            print(f'frames/word {fpw:<4} wer={rec["wer"]:.3f} '
                  f'extra={rec["extra_w"]:4.1f}({rec["pct_extra"]:.0%}) '
                  f'tail={rec["tail_s"]:.2f}s genuine={rec["genuineness"]:.2f}',
                  flush=True)
        print("SWEEP_DONE", flush=True)
        return
    todo = []
    if a.set == "baseline":
        todo = [("baseline sft3+dpo", [])]
    elif a.set == "quality":
        for n in adapters_of("sft3_quality"):
            todo += [(f"{n} @{w}", [(f"sft3_quality:{n}", w)]) for w in scales]
    elif a.set == "voice":
        for w in scales:
            todo.append((f"voice emolia_c1699 @{w}", [("sft3_voice:emolia_c1699", w)]))
    elif a.set == "voicenet":
        for n in adapters_of("sft3_voicenet"):
            todo += [(f"{n} @{w}", [(f"sft3_voicenet:{n}", w)]) for w in scales]
    elif a.set == "emotion":
        for n in adapters_of("sft3_emotion"):
            todo += [(f"{n} @{w}", [(f"sft3_emotion:{n}", w)]) for w in scales]
    elif a.set == "stopbias":
        # the brake was built against lines that stopped too early; the current
        # complaint is the opposite, so this walks it through zero into negative
        G, B, E = ("sft3_quality:genuineness_high", "sft3_quality:blend_high",
                   "sft3_quality:esthetics_high")
        stack = [(G, 0.25), (B, 0.5), (E, 0.5),
                 ("sft3_voice:emolia_c1699", 0.25),
                 ("sft3_emotion:Amusement", 1.0)]
        todo = [(f"seed {sd} stop_bias {sb:+.1f}", stack)
                for sb in (0.0, 1.0, 2.0, 3.0, 4.0) for sd in (1234, 777, 42)]
    elif a.set == "combo":
        # built from the single-adapter results: genuineness is only safe low,
        # blend is safe anywhere, esthetics costs genuineness, voice buys
        # identity cheaply at 0.25.
        G, B, E = ("sft3_quality:genuineness_high", "sft3_quality:blend_high",
                   "sft3_quality:esthetics_high")
        V = "sft3_voice:emolia_c1699"
        EM = "sft3_emotion:Amusement"
        q_lo = [(G, 0.25), (B, 0.5), (E, 0.5)]
        q_hi = [(G, 1.0), (B, 1.0), (E, 1.0)]
        todo = [
            ("quality trio @0.25/0.5/0.5", q_lo),
            ("quality trio @1.0 (old default)", q_hi),
            ("q_lo + voice 0.25", q_lo + [(V, 0.25)]),
            ("q_lo + voice 1.0", q_lo + [(V, 1.0)]),
            ("q_lo + voice 0.25 + emo 0.5", q_lo + [(V, 0.25), (EM, 0.5)]),
            ("q_lo + voice 0.25 + emo 1.0", q_lo + [(V, 0.25), (EM, 1.0)]),
            ("q_lo + voice 0.25 + emo 1.5", q_lo + [(V, 0.25), (EM, 1.5)]),
            ("q_hi + voice 1.0 + emo 1.5 (live)", q_hi + [(V, 1.0), (EM, 1.5)]),
            ("q_lo + v0.25 + emo0.5 + 1 axis",
             q_lo + [(V, 0.25), (EM, 0.5), ("sft3_voicenet:VALS_high", 0.5)]),
            ("q_lo + v0.25 + emo0.5 + 2 axes",
             q_lo + [(V, 0.25), (EM, 0.5), ("sft3_voicenet:VALS_high", 0.5),
                     ("sft3_voicenet:VFLX_high", 0.5)]),
            ("q_lo + v0.25 + emo0.5 + 3 axes",
             q_lo + [(V, 0.25), (EM, 0.5), ("sft3_voicenet:VALS_high", 0.5),
                     ("sft3_voicenet:VFLX_high", 0.5), ("sft3_voicenet:EMPH_high", 0.5)]),
            ("q_lo + v0.25 + emo0.5 + burst",
             q_lo + [(V, 0.25), (EM, 0.5), ("burst:chuckle", 0.25)]),
            ("proposed default",
             [(G, 0.25), (B, 0.5), (V, 0.25), (EM, 0.5)]),
            ("proposed default + axis",
             [(G, 0.25), (B, 0.5), (V, 0.25), (EM, 0.5),
              ("sft3_voicenet:VALS_high", 0.5)]),
        ]
    else:
        raise SystemExit("unknown --set")

    res = []
    if os.path.exists(out_p):
        try:
            res = json.load(open(out_p))
        except Exception:
            res = []
    done = {r["condition"] for r in res}
    t0 = time.time()
    for i, (cond, extra) in enumerate(todo, 1):
        if cond in done:
            continue
        try:
            sb = sd = None
            if a.set == "stopbias":
                parts = cond.split()
                sd, sb = int(parts[1]), float(parts[-1])
            rec = run(cond, [(DPO, 1.0)] + extra, items, refs, asr, sc,
                      seed=sd if sd is not None else a.seed, stop_bias=sb)
        except Exception as e:
            print(f"  {cond}: {str(e)[:140]}", flush=True)
            continue
        if not rec:
            continue
        res.append(rec)
        json.dump(res, open(out_p, "w"), indent=1)
        el = time.time() - t0
        print(f'[{i}/{len(todo)}] {cond:28s} wer={rec["wer"]:.3f} '
              f'extra={rec["extra_w"]:4.1f}({rec["pct_extra"]:.0%}) '
              f'genuine={rec["genuineness"]:.2f} blend={rec["blend"]:.2f} '
              f'spk={rec["spk_sim"]:.3f}  [{el/60:.0f}m]', flush=True)
    print("SWEEP_DONE", flush=True)


if __name__ == "__main__":
    main()
