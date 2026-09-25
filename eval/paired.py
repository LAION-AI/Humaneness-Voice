"""Comparisons that survive the design: paired within (task, director)."""
import collections, json, math, sys
RUB = ("pleasant", "fit", "natural")
rows = [json.loads(l) for l in open("runs/full/takes.jsonl") if l.strip()]
rows = [r for r in rows if r.get("scores")]

def mean(v): return sum(v)/len(v) if v else float("nan")
def ttest_paired(d):
    n=len(d)
    if n<3: return float('nan'),float('nan')
    m=mean(d); s=math.sqrt(sum((x-m)**2 for x in d)/(n-1)) if n>1 else 0
    if s==0: return float('inf') if m else 0.0, 0.0 if m else 1.0
    t=m/(s/math.sqrt(n)); return t, math.erfc(abs(t)/math.sqrt(2))

# cell = (task, brain, gen, variant); its value = mean and best over 3 takes
cell = collections.defaultdict(list)
for r in rows: cell[(r["task"], r["brain"], r["gen"], r["variant"])].append(r)

print("=" * 78)
print("A. GENERATION 1 ONLY — the five seeded blocks, every task, both directors")
print("   (balanced: each block ran on exactly the same cells)")
print("=" * 78)
seeds = ["control","imperfection","breath","body","subtext"]
base = {}
for (t,b,g,v),rs in cell.items():
    if g==1: base.setdefault((t,b),{})[v]=rs
pairs = {v: [] for v in seeds if v!="control"}
absol = {v: [] for v in seeds}
for k,d in base.items():
    if "control" not in d: continue
    c = mean([x["total"] for x in d["control"]])
    for v in seeds:
        if v in d: absol[v].append(mean([x["total"] for x in d[v]]))
        if v!="control" and v in d:
            pairs[v].append(mean([x["total"] for x in d[v]]) - c)
print(f"{'block':<16}{'cells':>7}{'mean total':>12}{'vs control':>12}{'t':>8}{'p':>8}")
for v in seeds:
    if v=="control":
        print(f"{v:<16}{len(absol[v]):>7}{mean(absol[v]):>12.2f}{'—':>12}{'':>8}{'':>8}")
    else:
        t,p = ttest_paired(pairs[v])
        print(f"{v:<16}{len(pairs[v]):>7}{mean(absol[v]):>12.2f}"
              f"{mean(pairs[v]):>+12.2f}{t:>8.2f}{p:>8.3f}")

print()
print("=" * 78)
print("B. DID BREEDING HELP? best variant per (task, director) in each generation")
print("=" * 78)
bg = collections.defaultdict(dict)
for (t,b,g,v),rs in cell.items():
    bg[(t,b)].setdefault(g,[]).append((mean([x["total"] for x in rs]), max(x["total"] for x in rs), v))
gens = collections.defaultdict(list); gensb = collections.defaultdict(list)
for k,d in bg.items():
    if len(d)<3: continue
    for g in (1,2,3):
        gens[g].append(max(x[0] for x in d[g])); gensb[g].append(max(x[1] for x in d[g]))
print(f"{'generation':<14}{'arms':>7}{'best cell mean':>17}{'best single take':>19}")
for g in (1,2,3):
    print(f"gen {g:<10}{len(gens[g]):>7}{mean(gens[g]):>17.2f}{mean(gensb[g]):>19.2f}")
for g in (2,3):
    d=[a-b for a,b in zip(gens[g],gens[1])]; t,p=ttest_paired(d)
    print(f"  gen{g} - gen1: {mean(d):+.2f}  t {t:+.2f}  p {p:.3f}")

print()
print("=" * 78)
print("C. WHAT A BRED BLOCK BEAT — each child against the control of its own cell")
print("=" * 78)
kids=[]
for (t,b,g,v),rs in cell.items():
    if v in seeds or g==1: continue
    ctrl = base.get((t,b),{}).get("control")
    if not ctrl: continue
    kids.append((mean([x["total"] for x in rs]) - mean([x["total"] for x in ctrl]),
                 mean([x["total"] for x in rs]), v, t, b))
kids.sort(reverse=True)
d=[k[0] for k in kids]; t,p=ttest_paired(d)
print(f"{len(kids)} bred blocks, mean gain over the control of their own cell: "
      f"{mean(d):+.2f}  t {t:+.2f}  p {p:.4f}")
print(f"beat their control: {sum(1 for x in d if x>0)}/{len(d)}")
print(f"\n{'block':<22}{'gain':>7}{'mean':>7}  task / director")
for g_,m_,v,t_,b_ in kids[:12]:
    print(f"{v[:21]:<22}{g_:>+7.1f}{m_:>7.1f}  {t_} / {b_}")

print()
print("=" * 78)
print("D. PER RUBRIC — gen-1 seeds, paired difference from control")
print("=" * 78)
print(f"{'block':<16}" + "".join(f"{r:>12}" for r in RUB))
for v in seeds[1:]:
    out=[]
    for rub in RUB:
        dd=[]
        for k,d in base.items():
            if "control" in d and v in d:
                dd.append(mean([x["scores"][rub]["score"] for x in d[v]])
                          - mean([x["scores"][rub]["score"] for x in d["control"]]))
        tt,pp=ttest_paired(dd)
        out.append(f"{mean(dd):+.2f}{'*' if pp<0.05 else ' '}")
    print(f"{v:<16}" + "".join(f"{o:>12}" for o in out))
