"""What the search found: which prompt additions moved which rubric."""
import collections
import json
import math
import os
import sys

ROOT = "/mnt/nvme/arena"
RUB = ("pleasant", "fit", "natural")


def load(run):
    p = os.path.join(ROOT, "runs", run, "takes.jsonl")
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("scores"):
            rows.append(r)
    return rows


def mean(v):
    return sum(v) / len(v) if v else float("nan")


def welch(a, b):
    """t and a rough two-sided p, without scipy."""
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan")
    ma, mb = mean(a), mean(b)
    va = sum((x - ma) ** 2 for x in a) / (len(a) - 1)
    vb = sum((x - mb) ** 2 for x in b) / (len(b) - 1)
    se = math.sqrt(va / len(a) + vb / len(b))
    if se == 0:
        return 0.0, 1.0
    t = (ma - mb) / se
    # normal approximation is fine at these n
    p = math.erfc(abs(t) / math.sqrt(2))
    return t, p


def by(rows, key):
    d = collections.defaultdict(list)
    for r in rows:
        d[key(r)].append(r)
    return d


def table(rows, key, label, floor=6):
    groups = by(rows, key)
    base = [r["scores"]["pleasant"]["score"] + r["scores"]["fit"]["score"]
            + r["scores"]["natural"]["score"]
            for r in groups.get("control", [])]
    out = []
    for name, rs in groups.items():
        tot = [r["total"] for r in rs]
        row = {"name": name, "n": len(rs), "total": mean(tot)}
        for k in RUB:
            row[k] = mean([r["scores"][k]["score"] for r in rs])
        if base and name != "control":
            row["t"], row["p"] = welch(tot, base)
        out.append(row)
    out.sort(key=lambda r: -r["total"])
    print(f"\n=== {label} ===")
    print(f"{'':<24}{'n':>5}{'pleasant':>10}{'fit':>7}{'natural':>9}"
          f"{'total':>8}{'t vs control':>14}")
    for r in out:
        if r["n"] < floor:
            continue
        t = f"{r['t']:+.2f} (p {r['p']:.3f})" if "t" in r else ""
        print(f"{str(r['name'])[:23]:<24}{r['n']:>5}{r['pleasant']:>10.2f}"
              f"{r['fit']:>7.2f}{r['natural']:>9.2f}{r['total']:>8.2f}{t:>14}")
    return out


def main():
    run = sys.argv[1] if len(sys.argv) > 1 else "full"
    rows = load(run)
    if not rows:
        print("no scored takes yet")
        return
    print(f"run {run}: {len(rows)} scored clips, "
          f"{len({r['task'] for r in rows})} tasks, "
          f"{len({r['key'] for r in rows})} cells")
    for k in RUB:
        v = [r["scores"][k]["score"] for r in rows]
        hist = collections.Counter(v)
        print(f"  {k:<9} mean {mean(v):.2f}  " +
              " ".join(f"{s}:{hist.get(s,0)}" for s in range(6)))

    table(rows, lambda r: r["variant"], "prompt addition (all generations)")
    table(rows, lambda r: r["brain"], "director")
    table(rows, lambda r: f"gen {r['gen']}", "generation")
    table(rows, lambda r: r["track"], "benchmark track")

    # seeded blocks only, split by director
    for b in sorted({r["brain"] for r in rows}):
        table([r for r in rows if r["brain"] == b],
              lambda r: r["variant"], f"prompt addition — {b}")

    # what the winning blocks actually say
    groups = by(rows, lambda r: r["variant"])
    best = sorted(((mean([x["total"] for x in v]), k, v)
                   for k, v in groups.items() if len(v) >= 6), reverse=True)
    print("\n=== the three best-scoring blocks, in full ===")
    for score, name, rs in best[:3]:
        txt = next((r["variant_text"] for r in rs if r["variant_text"]), "")
        print(f"\n--- {name}  (mean total {score:.2f} of 15, n={len(rs)})")
        print(txt or "(the standing prompt alone)")


if __name__ == "__main__":
    main()
