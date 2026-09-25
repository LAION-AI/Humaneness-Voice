import sys, statistics as st
sys.path.insert(0,'/tmp/claude-1000/-mnt-nvme-moss-15-v2/9ad1e4fd-e08b-4e60-b3e4-9683236600b1/scratchpad')
import why_babble as W
SEEDS=[11,22,33,44,55,66]
THIRD=("burst:sharp inhale",0)
def two(w):  return [(n,w) for n,_ in W.BURST]
def three(ws): return [(W.BURST[0][0],ws[0]),(W.BURST[1][0],ws[1]),(THIRD[0],ws[2])]
conds=[
 ("1 adapter @1.25",      [ (W.BURST[0][0],1.25) ]),
 ("2 adapter @0.5  (sum 1.0)",  two(0.5)),
 ("2 adapter @0.75 (sum 1.5)",  two(0.75)),
 ("2 adapter @0.875(sum 1.75)", two(0.875)),
 ("2 adapter @1.0  (sum 2.0)",  two(1.0)),
 ("2 adapter @1.25 (sum 2.5)",  two(1.25)),
 ("3 adapter @0.5  (sum 1.5)",  three([0.5,0.5,0.5])),
 ("3 adapter 1.0/0.5/0.5 (2.0)",three([1.0,0.5,0.5])),
 ("3 adapter @0.67 (sum 2.0)",  three([0.67,0.67,0.66])),
]
print(f"{'bedingung':<30}" + "".join(f"{s:>7}" for s in SEEDS) + f"{'median':>8}{'mittel':>8}{'kaputt':>8}")
for tag,burst in conds:
    ws=[]
    for s in SEEDS:
        _,w,_,_ = W.run(tag, W.VAR["b_ein_scream"], W.BASE+burst, seed=s)
        ws.append(w if w is not None else float('nan'))
    bad=sum(1 for w in ws if w and w>0.25)
    print(f"{tag:<30}" + "".join(f"{w:>7.2f}" for w in ws) +
          f"{st.median(ws):>8.3f}{st.mean(ws):>8.3f}{bad:>6}/{len(SEEDS)}", flush=True)
