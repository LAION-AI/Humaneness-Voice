#!/usr/bin/env python3
"""Why does it keep talking after the line is finished?

Drives the running server's /api/say — exact text, exact instruction, exact
adapter list, no language model in the loop — and measures three things per
take with Parakeet TDT v3:

  over_s    audio length minus the length the prompt asked for
  tail_s    audio after the last recognised speech token (trailing noise)
  extra_w   words transcribed beyond the end of the reference text

`extra_w` is the direct measurement of the complaint: the line is spoken, and
then more words follow.  Timestamps come from the TDT duration head — the
cumulative sum of per-token durations at 0.08 s per encoder frame, verified
against a known file (last token 10.24 s in a 10.80 s recording).
"""
import argparse, difflib, io, json, re, struct, sys, time

import numpy as np
import requests
import torch

sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import config, timed_script

URL = "http://127.0.0.1:8792/api/say"
FRAME_S = 0.08          # Parakeet encoder frame

PROMPTS = [
    ("The kettle boiled twice before I remembered why I walked into the kitchen.",
     "(clearly amused, easy and conversational)"),
    ("There is a fox that crosses the yard at the same hour every evening. I have started waiting for it.",
     "(warmly, quietly pleased)"),
    ("I finally fixed the squeaking door. It took eleven minutes and four years of complaining.",
     "(dryly amused, matter of fact)"),
    ("The train was late, the coffee was cold, and somehow it was still a good morning.",
     "(clearly content, unhurried)"),
    ("My neighbour plays the trumpet on Sundays. He is not good, but he is committed.",
     "(clearly amused, affectionate)"),
    ("I planted the basil too close together and now it is a small green argument.",
     "(lightly amused, conversational)"),
    ("She sent me a photograph of the sea and said nothing else. It was enough.",
     "(warmly, quietly moved)"),
    ("The lift has been broken since March. I have never been fitter in my life.",
     "(dryly amused, resigned)"),
    ("There was a moment this afternoon when the whole street went completely quiet.",
     "(quietly, almost hushed)"),
    ("I read the same paragraph four times and understood it on none of them.",
     "(clearly tired, self-deprecating)"),
]

GENERAL = ("a woman's voice, in their thirties, speaking with Standard American; "
           "close conversational volume, unforced; genuine, not acted; "
           "clean studio recording")

DPO = "sft3_dpo:p2"
GEN_ = "sft3_quality:genuineness_high"
BLE = "sft3_quality:blend_high"
EST = "sft3_quality:esthetics_high"
VOI = "sft3_voice:emolia_c1699"

CONDITIONS = [
    ("bare SFT3",              []),
    ("+dpo",                   [(DPO, 1.0)]),
    ("+dpo+genuineness",       [(DPO, 1.0), (GEN_, 1.0)]),
    ("+dpo+blend",             [(DPO, 1.0), (BLE, 1.0)]),
    ("+dpo+esthetics",         [(DPO, 1.0), (EST, 1.0)]),
    ("+dpo+gen+blend",         [(DPO, 1.0), (GEN_, 1.0), (BLE, 1.0)]),
    ("+dpo+gen+esth",          [(DPO, 1.0), (GEN_, 1.0), (EST, 1.0)]),
    ("+dpo+blend+esth",        [(DPO, 1.0), (BLE, 1.0), (EST, 1.0)]),
    ("+dpo+all three quality", [(DPO, 1.0), (GEN_, 1.0), (BLE, 1.0), (EST, 1.0)]),
    ("+voice",                 [(DPO, 1.0), (GEN_, 1.0), (BLE, 1.0), (EST, 1.0), (VOI, 1.0)]),
    ("+emotion 1.5",           [(DPO, 1.0), (GEN_, 1.0), (BLE, 1.0), (EST, 1.0), (VOI, 1.0),
                                ("sft3_emotion:Amusement", 1.5)]),
    ("live default",           [(DPO, 1.0), (GEN_, 1.0), (BLE, 1.0), (EST, 1.0), (VOI, 1.0),
                                ("sft3_voicenet:VALS_high", 0.5), ("burst:chuckle", 0.25),
                                ("sft3_emotion:Amusement", 1.5)]),
]

# follow-up: the first sweep put the damage on the voice and emotion adapters,
# so this walks their weight down instead of adding more of them
Q3 = [(GEN_, 1.0), (BLE, 1.0), (EST, 1.0)]
Q3h = [(GEN_, 0.5), (BLE, 0.5), (EST, 0.5)]
EM = "sft3_emotion:Amusement"
CONDITIONS += [
    ("dose: voice 0.5",        [(DPO, 1.0)] + Q3 + [(VOI, 0.5)]),
    ("dose: voice 0.75",       [(DPO, 1.0)] + Q3 + [(VOI, 0.75)]),
    ("dose: emo 1.5",          [(DPO, 1.0)] + Q3 + [(VOI, 1.0), (EM, 1.5)]),
    ("dose: emo 1.0",          [(DPO, 1.0)] + Q3 + [(VOI, 1.0), (EM, 1.0)]),
    ("dose: emo 0.5",          [(DPO, 1.0)] + Q3 + [(VOI, 1.0), (EM, 0.5)]),
    ("dose: q0.5 voice0.75 emo1.0",
                               [(DPO, 1.0)] + Q3h + [(VOI, 0.75), (EM, 1.0)]),
    ("dose: no quality, emo1.0", [(DPO, 1.0), (VOI, 1.0), (EM, 1.0)]),
]

_W = re.compile(r"[a-z0-9']+")


def words(s):
    return _W.findall(str(s).lower())


class ASR:
    def __init__(self, device="cuda:0"):
        from transformers import ParakeetForTDT, ParakeetProcessor
        self.proc = ParakeetProcessor.from_pretrained(config.ASR_MODEL)
        self.model = ParakeetForTDT.from_pretrained(
            config.ASR_MODEL, dtype=torch.bfloat16).to(device).eval()
        self.device = device

    @torch.inference_mode()
    def run(self, wav16k):
        inp = self.proc(np.asarray(wav16k, np.float32), sampling_rate=16000,
                        return_tensors="pt")
        inp = {k: (v.to(self.device).to(torch.bfloat16)
                   if v.dtype.is_floating_point else v.to(self.device))
               for k, v in inp.items()}
        out = self.model.generate(**inp, max_new_tokens=512)
        seq, dur = out.sequences[0], out.durations[0]
        starts = (dur.cumsum(0) - dur).float().cpu().numpy() * FRAME_S
        # decode the whole sequence for the words (per-token decode drops the
        # SentencePiece space marker and glues everything together), and use the
        # per-token times only to find where speech stops
        text = self.proc.batch_decode(out.sequences)[0]
        text = re.sub(r"\s{2,}", " ", text.replace("<blank>", " ")).strip()
        toks = self.proc.tokenizer.convert_ids_to_tokens([int(t) for t in seq])
        last_t = 0.0
        for t, st in zip(toks, starts):
            if t and "blank" not in str(t).lower():
                last_t = float(st)
        return text, last_t


def resample_48k_to_16k(pcm):
    x = np.asarray(pcm, np.float32)
    n = len(x) // 3 * 3
    return x[:n].reshape(-1, 3).mean(1)


def say(text, instruction, tokens, loras, seed, stop_bias=None, anchor_path=None):
    body = {"text": text, "instruction": instruction, "language": "English",
            "tokens": int(tokens), "seed": int(seed),
            "loras": [[n, l] for n, l in loras]}
    if stop_bias is not None:
        body["stop_bias"] = float(stop_bias)
    if anchor_path:
        body["anchor_path"] = anchor_path
    r = requests.post(URL, json=body, timeout=600)
    r.raise_for_status()
    j = r.json()
    import base64
    raw = base64.b64decode(j["pcm"] if isinstance(j, dict) and "pcm" in j else j["audio"])
    return np.frombuffer(raw, "<i2").astype(np.float32) / 32768.0, j


def trailing_words(ref, hyp):
    """Words the transcript has after the reference text has run out."""
    a, b = words(ref), words(hyp)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    blocks = [m for m in sm.get_matching_blocks() if m.size]
    if not blocks:
        return len(b), b
    end = blocks[-1].b + blocks[-1].size
    return len(b) - end, b[end:]


def wer(ref, hyp):
    a, b = words(ref), words(hyp)
    if not a:
        return 0.0
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    same = sum(m.size for m in sm.get_matching_blocks())
    return (len(a) - same + max(0, len(b) - same)) / len(a)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=len(PROMPTS))
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--stop-bias", type=float, default=None)
    ap.add_argument("--out", default="/mnt/nvme/moss-15-v2-assets/tail_eval.json")
    a = ap.parse_args()

    asr = ASR()
    conds = [c for c in CONDITIONS if not a.only or a.only in c[0]]
    rows = []
    for name, loras in conds:
        agg = []
        for pi, (line, cue) in enumerate(PROMPTS[:a.n]):
            script = f"{cue} {line}"
            tagged, frames, plain = timed_script.render(script)
            for s in range(a.seeds):
                t0 = time.time()
                try:
                    pcm, meta = say(tagged, f"GENERAL: {GENERAL}; {frames/12.5:.1f}s, EN.\nSCRIPT:\n{tagged}",
                                    frames, loras, seed=1234 + s)
                except Exception as e:
                    print(f"  [{name}] prompt {pi}: {str(e)[:120]}", flush=True)
                    continue
                got = len(pcm) / 48000.0
                hyp, last_t = asr.run(resample_48k_to_16k(pcm))
                extra, xw = trailing_words(line, hyp)
                agg.append({"prompt": pi, "seed": s,
                            "req_s": frames / 12.5, "got_s": got,
                            "over_s": got - frames / 12.5,
                            "tail_s": got - last_t,
                            "extra_w": extra, "wer": wer(line, hyp),
                            "hyp": hyp, "extra_words": " ".join(xw)[:120],
                            "gen_s": time.time() - t0})
        if not agg:
            continue
        m = lambda k: float(np.mean([r[k] for r in agg]))
        row = {"condition": name, "n": len(agg), "loras": loras,
               "over_s": m("over_s"), "tail_s": m("tail_s"),
               "extra_w": m("extra_w"), "wer": m("wer"),
               "pct_with_extra": float(np.mean([r["extra_w"] > 0 for r in agg])),
               "takes": agg}
        rows.append(row)
        print(f'{name:26s} n={row["n"]:3d}  over={row["over_s"]:+5.2f}s  '
              f'tail={row["tail_s"]:5.2f}s  extra_words={row["extra_w"]:5.2f}  '
              f'wer={row["wer"]:.3f}  with_extra={row["pct_with_extra"]:.0%}', flush=True)
        json.dump(rows, open(a.out, "w"), indent=1)
    print("TAILEVAL_DONE", flush=True)


if __name__ == "__main__":
    main()
