"""What does guidance buy inside a best-of-8 candidate set?

The script is held fixed per prompt, so the three guidance settings see the
identical line and the comparison is about generation and not about the director
writing something different each call.  Everything else is the shipped stack.
"""
import json, sys, time
import numpy as np, requests
sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import bestofn, config, timed_script

URL = "http://127.0.0.1:8792/api/say_batch"
GEN = ("a woman's voice, in their thirties, speaking with Standard American; "
       "close conversational volume, unforced; genuine, not acted; clean studio recording")
GEN_UNC = ("a woman's voice, in their thirties, speaking with Standard American; "
           "close conversational volume, unforced; clean studio recording")
SCRIPTS = [
 "(clearly amused, warm) I still cannot believe the cat opened that door by herself. (chuckle) she is far too clever.",
 "(intensely upset, barely holding it together) I do not know how to tell them what actually happened. (sharp inhale) it is going to hurt.",
 "(clearly delighted) that is the best news I have heard all week, and I am not even exaggerating.",
 "(quietly, almost conspiratorial) between the two of us, that report was never actually finished.",
]
STACK = [("sft3_dpo:p2",1.0), ("sft3_voice:emolia_c1699",0.5),
         ("sft3_quality:genuineness_high",0.25), ("sft3_quality:blend_high",0.5),
         ("sft3_quality:esthetics_high",0.5), ("sft3_qdpo:quality_dpo",1.5),
         ("sft3_emotion:Amusement",1.0)]
N = 8

def run(script, g, judge, seed=1234):
    tagged, frames, plain = timed_script.render(script)
    lc = "EN"
    gl = timed_script.general_line(GEN, frames/12.5, lc, None)
    it = {"text": tagged, "tokens": frames, "language": "English",
          "instruction": f"GENERAL: {gl}\nSCRIPT:\n{tagged}"}
    if g > 1.0001:
        gu = timed_script.general_line(GEN_UNC, frames/12.5, lc, None)
        tu = timed_script.neutralise(tagged)
        it["instruction_unc"] = f"GENERAL: {gu}\nSCRIPT:\n{tu}"
        it["text_unc"] = tu
    t0 = time.time()
    r = requests.post(URL, json={"items": [dict(it) for _ in range(N)],
                                 "loras": [[n,l] for n,l in STACK],
                                 "seed": seed, "guidance": g}, timeout=3600)
    r.raise_for_status()
    j = r.json()
    import base64
    waves = [np.frombuffer(base64.b64decode(x), "<i2").astype(np.float32)/32768.0
             for x in j["pcm"]]
    el = time.time() - t0
    cands = judge.score(waves, j.get("sr", 48000), plain, general=GEN, script=script)
    bestofn.rank(cands)
    return cands, el

if __name__ == "__main__":
    from eval_tail import ASR
    class _A:
        def __init__(self): self.a = ASR(device="cuda:0")
        def transcribe(self, w): return self.a.run(w)[0]
    judge = bestofn.Judge(device="cuda:0", asr=_A())
    import retrieval
    judge.attach_clap(retrieval.Retriever(device="cuda:0"))
    rows = {}
    for g in (1.0, 3.0, 4.0):
        best, mean, wers, claps, gens, bls, secs = [], [], [], [], [], [], []
        for sc in SCRIPTS:
            c, el = run(sc, g, judge)
            top = min(c, key=lambda x: x["rank"])
            best.append(top["reward"]); mean.append(np.mean([x["reward"] for x in c]))
            wers.append(np.mean([x["wer"] for x in c]))
            claps.append(np.mean([x["clap"] for x in c]))
            gens.append(np.mean([x["genuineness"] for x in c]))
            bls.append(np.mean([x["blend"] for x in c]))
            secs.append(el)
        rows[g] = (np.mean(best), np.mean(mean), np.mean(wers), np.mean(claps),
                   np.mean(gens), np.mean(bls), np.mean(secs))
        print(f"g={g:<4} best_reward={rows[g][0]:.3f} mean_reward={rows[g][1]:.3f} "
              f"wer={rows[g][2]:.3f} clap={rows[g][3]:+.3f} gen={rows[g][4]:.2f} "
              f"blend={rows[g][5]:.2f}  {rows[g][6]:.0f}s/set", flush=True)
    print("\n(reward ist innerhalb jedes Satzes normalisiert -- vergleichbar sind "
          "die ROHWERTE wer/clap/gen/blend, nicht der reward zwischen Saetzen)")
