"""The corrected run, and whether it replicates the first."""
import collections, json, math, statistics as st

RUB = ("pleasant","fit","natural")

def load(run):
    rows=[json.loads(l) for l in open(f"runs/{run}/takes.jsonl") if l.strip()]
    return [r for r in rows if r.get("scores")]

def cells(rows):
    c=collections.defaultdict(list)
    for r in rows: c[(r["task"],r["brain"],r["gen"],r["variant"])].append(r)
    return {k: st.mean([x["total"] for x in v]) for k,v in c.items()}, c

def paired(cm, gen, name, base="control"):
    d=[]
    arms={(t,b) for (t,b,g,v) in cm if g==gen}
    for t,b in arms:
        a=cm.get((t,b,gen,name)); c=cm.get((t,b,gen,base))
        if a is not None and c is not None: d.append(a-c)
    if len(d)<3: return None
    m=st.mean(d); sd=st.stdev(d)
    t_=m/(sd/len(d)**0.5) if sd else 0.0
    return m, t_, math.erfc(abs(t_)/math.sqrt(2)), len(d)

fx=load("fixed"); cm,cells_raw=cells(fx)
print(f"korrigierter lauf: {len(fx)} clips, {len({r['key'] for r in fx})} zellen, "
      f"{len({r['task'] for r in fx})} aufgaben")
for k in RUB:
    v=[r["scores"][k]["score"] for r in fx]
    h=collections.Counter(v)
    print(f"  {k:<9} mittel {st.mean(v):.2f}   " + " ".join(f"{s}:{h.get(s,0)}" for s in range(6)))

print("\n" + "="*76)
print("JEDE GENERATION, gepaart gegen die kontrolle, n=40 arme")
print("="*76)
for gen in (1,2,3):
    vs=sorted({v for (t,b,g,v) in cm if g==gen})
    print(f"\n--- generation {gen}")
    print(f"{'block':<22}{'mittel':>9}{'delta':>8}{'t':>7}{'p':>8}{'arme':>6}")
    rowsg=[]
    for v in vs:
        vals=[m for (t,b,g,n),m in cm.items() if g==gen and n==v]
        pr=paired(cm,gen,v)
        rowsg.append((pr[0] if pr else 0, v, st.mean(vals), pr))
    for delta,v,mean,pr in sorted(rowsg, reverse=True):
        if v=="control":
            print(f"{v:<22}{mean:>9.2f}{'—':>8}{'':>7}{'':>8}{'':>6}")
        elif pr:
            print(f"{v:<22}{mean:>9.2f}{pr[0]:>+8.2f}{pr[1]:>+7.2f}{pr[2]:>8.3f}{pr[3]:>6}")

print("\n" + "="*76)
print("HAT DIE ZUCHT GEHOLFEN? bester block je generation (gepaart gg. kontrolle)")
print("="*76)
for gen in (1,2,3):
    vs={v for (t,b,g,v) in cm if g==gen and v!="control"}
    best=max((paired(cm,gen,v) or (-9,0,1,0), v) for v in vs) if vs else None
    res=[(paired(cm,gen,v), v) for v in vs]
    res=[(r,v) for r,v in res if r]
    r,v=max(res, key=lambda x: x[0][0])
    print(f"  gen {gen}: bester block '{v}'  delta {r[0]:+.2f}  t {r[1]:+.2f}  p {r[2]:.3f}")

print("\n" + "="*76)
print("REPLIKATION: generation 1 in beiden laeufen (identisches design)")
print("="*76)
try:
    f1=load("full"); cm1,_=cells(f1)
    print(f"{'block':<16}{'lauf1 delta':>14}{'p':>8}{'lauf2 delta':>14}{'p':>8}{'gepoolt':>10}")
    for v in ("imperfection","breath","body","subtext"):
        a=paired(cm1,1,v); b=paired(cm,1,v)
        if a and b:
            pooled=(a[0]*a[3]+b[0]*b[3])/(a[3]+b[3])
            print(f"{v:<16}{a[0]:>+14.2f}{a[2]:>8.3f}{b[0]:>+14.2f}{b[2]:>8.3f}{pooled:>+10.2f}")
except FileNotFoundError:
    print("  lauf 1 nicht gefunden")

print("\n" + "="*76)
print("PRO RUBRIK, generation 1, gepaart")
print("="*76)
craw=collections.defaultdict(lambda: collections.defaultdict(dict))
for (t,b,g,v),rs in cells_raw.items():
    if g!=1: continue
    for rub in RUB: craw[rub][(t,b)][v]=st.mean([x["scores"][rub]["score"] for x in rs])
print(f"{'block':<16}" + "".join(f"{r:>13}" for r in RUB))
for v in ("imperfection","breath","body","subtext"):
    out=[]
    for rub in RUB:
        d=[d_[v]-d_["control"] for d_ in craw[rub].values() if v in d_ and "control" in d_]
        m=st.mean(d); sd=st.stdev(d) if len(d)>1 else 0
        t_=m/(sd/len(d)**0.5) if sd else 0
        p=math.erfc(abs(t_)/math.sqrt(2)) if sd else 1
        out.append(f"{m:+.2f}{'*' if p<0.05 else ''}")
    print(f"{v:<16}" + "".join(f"{o:>13}" for o in out))

print("\n" + "="*76)
print("RAUSCHEN, jetzt mit mittelwert-fitness")
print("="*76)
w=[st.pstdev([x["total"] for x in v]) for v in cells_raw.values() if len(v)>=3]
arm=collections.defaultdict(list)
for (t,b,g,v),m in cm.items(): arm[(t,b,g)].append(m)
bw=[st.pstdev(v) for v in arm.values() if len(v)>=5]
print(f"  zwischen den 3 takes einer zelle    SD {st.mean(w):.2f}")
print(f"  zwischen den bloecken eines arms    SD {st.mean(bw):.2f}")
print(f"  zellmittel-rauschen (SD/sqrt3)      SD {st.mean(w)/3**0.5:.2f}")
