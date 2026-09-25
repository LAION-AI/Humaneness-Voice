"""The corrected run: measure the prompt, not the luck.

Three things differ from `harness.py`, and each fixes a specific way the first
run fooled itself.

1. **Every block is evaluated on every arm.**  The first run bred children
   inside a single (task, director) arm, so a block existed at n = 1 arm and was
   compared against a control pooled over twenty.  That comparison measured task
   difficulty.  Here the population is global: all five blocks run on all forty
   arms in every generation, so every comparison is paired and equally powered.

2. **Fitness is the mean, not the maximum.**  Take-to-take spread inside a cell
   was SD 1.57 of 15 — as large as the spread between variants — so best-of-three
   bought +2.78 points by luck alone, and the search climbed that gradient.

3. **Breeding sees arm-level statistics**, not a single lucky clip: a block is
   described to the mutator by its mean against control across all arms.
"""
import argparse
import asyncio
import collections
import json
import os
import statistics
import sys
import time

sys.path.insert(0, "/mnt/nvme/arena")
from harness import (Run, SEEDS, MUTATOR, RUBRICS, pick_tasks, ROOT)  # noqa


def cell_mean(rows):
    got = [r["total"] for r in rows if r.get("scores")]
    return statistics.mean(got) if got else None


async def breed_global(run, stats, n):
    """New blocks from arm-level evidence, written by the judge's own family."""
    lines = []
    for s in sorted(stats, key=lambda x: -x["delta"])[:6]:
        lines.append(
            f'--- block "{s["name"]}" scored {s["mean"]:.2f} of 15 across '
            f'{s["arms"]} tasks, which is {s["delta"]:+.2f} against the '
            f'unmodified prompt (pleasant {s["pleasant"]:+.2f}, fit '
            f'{s["fit"]:+.2f}, natural {s["natural"]:+.2f})\n'
            f'{s["text"] or "(no extra block — the standing prompt alone)"}\n'
            f'What listeners said about its weakest takes: {s["why"]}')
    body = {"model": "gemini-3.8-flash",
            "messages": [{"role": "user",
                          "content": MUTATOR % ("\n\n".join(lines), n)}],
            "response_format": {"type": "json_object"}}
    import re
    for _ in range(3):
        try:
            r = await run.http.post(
                f"{__import__('config').LUNA_BASE}/v1/chat/completions",
                json=body, headers={"Authorization": f"Bearer {run.key}"})
            r.raise_for_status()
            txt = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", txt, re.S)
            out = []
            for b in json.loads(m.group(0) if m else txt)["blocks"][:n]:
                nm = re.sub(r"[^a-z0-9-]", "", str(b["name"]).lower())[:24]
                out.append({"name": nm or f"child{len(out)}",
                            "text": str(b["text"]).strip()})
            if out:
                return out
        except Exception as e:
            run.log(f"  ! breed failed: {type(e).__name__} {str(e)[:110]}")
            await asyncio.sleep(4)
    return []


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="fixed")
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--brains", default="local,luna")
    ap.add_argument("--bench", default=os.path.join(ROOT, "arena.json"))
    ap.add_argument("--gen-concurrency", type=int, default=2)
    ap.add_argument("--judge-concurrency", type=int, default=12)
    a = ap.parse_args()

    r = Run(a)
    tasks = pick_tasks(a.bench, a.tasks)
    brains = a.brains.split(",")
    arms = [(t, b) for t in tasks for b in brains]
    json.dump(tasks, open(os.path.join(r.dir, "tasks.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump({"seeds": SEEDS, "args": vars(a), "design": "global population, "
               "fitness = mean over takes, every block on every arm"},
              open(os.path.join(r.dir, "setup.json"), "w"),
              ensure_ascii=False, indent=1)
    r.log(f"corrected run: {len(arms)} arms x {a.variants} blocks x "
          f"{a.generations} generations x {a.takes} takes = "
          f"{len(arms)*a.variants*a.generations*a.takes} clips")
    st = os.statvfs(ROOT)
    r.log(f"disk free {st.f_bavail*st.f_frsize/2**30:.0f} GiB")

    variants = [{"name": n, "text": t} for n, t in SEEDS][:a.variants]
    history = []
    for gen in range(1, a.generations + 1):
        r.log(f"=== generation {gen}: {[v['name'] for v in variants]}")
        cells = collections.defaultdict(dict)      # arm -> variant -> mean
        rubric = collections.defaultdict(lambda: collections.defaultdict(list))
        whys = collections.defaultdict(list)
        for i, (task, brain) in enumerate(arms, 1):
            for v in variants:
                rows = await r.one_cell(task, brain, v, gen)
                m = cell_mean(rows)
                if m is None:
                    continue
                cells[(task["id"], brain)][v["name"]] = m
                for rub in RUBRICS:
                    rubric[v["name"]][rub] += [
                        x["scores"][rub]["score"] for x in rows
                        if x.get("scores")]
                worst = min((x for x in rows if x.get("scores")),
                            key=lambda x: x["total"], default=None)
                if worst:
                    whys[v["name"]].append(worst["scores"]["natural"]["why"])
            if i % 4 == 0 or i == len(arms):
                r.log(f"  arm {i}/{len(arms)} done ({len(r.done)} cells total)")
            free = os.statvfs(ROOT)
            if free.f_bavail * free.f_frsize / 2**30 < 20:
                r.log("stopping: disk below 20 GiB")
                return

        # ---- arm-level, paired against control -------------------------
        stats = []
        for v in variants:
            nm = v["name"]
            pairs = [(d[nm] - d["control"]) for d in cells.values()
                     if nm in d and "control" in d]
            vals = [d[nm] for d in cells.values() if nm in d]
            if not vals:
                continue
            s = {"name": nm, "text": v["text"], "arms": len(vals),
                 "mean": statistics.mean(vals),
                 "delta": statistics.mean(pairs) if pairs else 0.0,
                 "why": " | ".join(whys[nm][:3])[:500]}
            for rub in RUBRICS:
                base = rubric["control"][rub]
                s[rub] = (statistics.mean(rubric[nm][rub]) -
                          (statistics.mean(base) if base else 0.0))
            if len(pairs) > 2:
                sd = statistics.stdev(pairs)
                s["t"] = (s["delta"] / (sd / len(pairs) ** 0.5)) if sd else 0.0
            stats.append(s)
        stats.sort(key=lambda x: -x["delta"])
        r.log(f"  --- generation {gen} result (paired against control, "
              f"n={stats[0]['arms'] if stats else 0} arms)")
        for s in stats:
            r.log(f"      {s['name']:<22} mean {s['mean']:5.2f}  "
                  f"delta {s['delta']:+5.2f}  t {s.get('t', float('nan')):+5.2f}")
        history += stats
        json.dump(history, open(os.path.join(r.dir, "generations.json"), "w"),
                  ensure_ascii=False, indent=1)
        if gen == a.generations:
            break
        elite = stats[:2]
        kids = await breed_global(r, history, a.variants - len(elite))
        seen = {e["name"] for e in elite}
        for i, k in enumerate(kids):
            while k["name"] in seen:
                k["name"] += f"-{i}"
            seen.add(k["name"])
        # the control always stays in the population: without it the paired
        # comparison has no baseline in the next generation
        variants = [{"name": e["name"], "text": e["text"]} for e in elite] + kids
        if not any(v["name"] == "control" for v in variants):
            variants = [{"name": "control", "text": ""}] + variants[:-1]
        r.log(f"  -> next: {[v['name'] for v in variants]}")
    r.log("done")


if __name__ == "__main__":
    asyncio.run(main())
