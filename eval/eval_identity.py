"""Speaker identity against intelligibility, script held fixed."""
import sys, numpy as np
sys.path.insert(0,'/mnt/nvme/moss-15-v2')
import timed_script
from eval_tail import ASR, wer, resample_48k_to_16k
from eval_align import say
from eval_scale import DPO, ANCHOR, Scorers
STACK = [(DPO,1.0), ("sft3_quality:genuineness_high",0.25),
         ("sft3_quality:blend_high",0.5), ("sft3_quality:esthetics_high",0.5),
         ("sft3_qdpo:quality_dpo",1.5), ("sft3_emotion:Contentment",1.0)]
SCRIPTS = [
 "(clearly amused, letting it out, warm and unguarded) I still cannot believe the cat opened that door by herself.",
 "(quietly, held close) There was a moment this afternoon when the whole street went completely silent.",
 "(clearly tired, self-deprecating) I read the same paragraph four times and understood it on none of them.",
]
asr = ASR(device="cuda:0"); sc = Scorers(device="cuda:0")
print(f"{'w':>5} {'spk_sim':>8} {'wer':>7}   (3 skripte x 2 seeds)")
for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.25):
    sims, wers = [], []
    for s in SCRIPTS:
        tg, fr, pl = timed_script.render(s)
        loras = list(STACK) + ([("sft3_voice:emolia_c1699", w)] if w > 0 else [])
        for seed in (1234, 777):
            try:
                pcm, sr = say(tg, fr, loras, seed, anchor=ANCHOR)
            except Exception as e:
                print("  ", str(e)[:80]); continue
            _, _, sim = sc.score(pcm)
            hyp, _ = asr.run(resample_48k_to_16k(pcm))
            sims.append(sim); wers.append(wer(pl, hyp))
    if sims:
        print(f"{w:5.2f} {np.nanmean(sims):8.3f} {np.mean(wers):7.3f}   n={len(sims)}")
