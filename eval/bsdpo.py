"""Does burst+stop DPO help or hurt when burst adapters are already merged?

Two axes: the burst-stop preference adapter at four weights, crossed with the
burst adapter dose that is safe (2 x 0.5) and the one that is known to break the
line (2 x 1.5).  Word error is the intelligibility axis; the gap between the
requested duration and the audio that comes back is the stop axis, which is what
the adapter was trained for.
"""
import sys, statistics as st
sys.path.insert(0,'/tmp/claude-1000/-mnt-nvme-moss-15-v2/9ad1e4fd-e08b-4e60-b3e4-9683236600b1/scratchpad')
import why_babble as W
sys.path.insert(0,'/mnt/nvme/moss-15-v2')
import timed_script, config

SEEDS=[11,22,33,44]
BS="sft3_qdpo:burst_stop_dpo"
_,want_frames,_ = timed_script.render(W.VAR["b_ein_scream"])
WANT_S = want_frames/config.FRAME_RATE
print(f"angefordert: {WANT_S:.2f}s\n")
print(f"{'burst-dosis':<16}{'bs-dpo':>8}" + "".join(f"{s:>7}" for s in SEEDS)
      + f"{'wer med':>9}{'kaputt':>8}{'ueberzug s':>12}")
for dose_tag, dose in (("2 x 0.5", [(n,0.5) for n,_ in W.BURST]),
                       ("2 x 1.5", W.BURST),
                       ("keine",   [])):
    for bs in (0.0, 0.5, 1.0, 1.5):
        lor = W.BASE + dose + ([(BS,bs)] if bs>0 else [])
        ws, secs = [], []
        for s in SEEDS:
            _,w,sec,_ = W.run("x", W.VAR["b_ein_scream"], lor, seed=s)
            ws.append(w if w is not None else float('nan'))
            secs.append(sec or 0)
        bad=sum(1 for w in ws if w and w>0.25)
        over = st.mean(secs)-WANT_S
        print(f"{dose_tag:<16}{bs:>8.1f}" + "".join(f"{w:>7.2f}" for w in ws)
              + f"{st.median(ws):>9.3f}{bad:>6}/{len(SEEDS)}{over:>+12.2f}", flush=True)
