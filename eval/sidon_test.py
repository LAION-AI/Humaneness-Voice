"""Does restoring the audio with SIDON change what the listener scores?

Paired: the same clip judged before and after, in the same session, with the
same rubric prompt.  Both sides are judged fresh — the stored score is not
reused, because judge-to-judge noise would otherwise be charged entirely to the
enhancement.
"""
import asyncio, base64, glob, json, os, random, statistics, sys, time
import numpy as np, soundfile as sf, scipy.signal as ss, torch, httpx
sys.path.insert(0, "/mnt/nvme/moss-15-v2"); sys.path.insert(0, "/mnt/nvme/arena")
import config, harness

ROOT = "/mnt/nvme/arena"
OUT = os.path.join(ROOT, "audio", "sidon")
os.makedirs(OUT, exist_ok=True)
DEV = os.environ.get("SIDON_DEV", "cpu")


def load_sidon():
    snap = glob.glob("/mnt/nvme/hf_cache/hub/models--sarulab-speech--sidon-v0.1/snapshots/*")[0]
    tag = "cuda" if DEV.startswith("cuda") else "cpu"
    fe = torch.jit.load(f"{snap}/feature_extractor_{tag}.pt", map_location=DEV).eval()
    dec = torch.jit.load(f"{snap}/decoder_{tag}.pt", map_location=DEV).eval()
    from transformers import AutoFeatureExtractor
    return fe, dec, AutoFeatureExtractor.from_pretrained("facebook/w2v-bert-2.0")


@torch.no_grad()
def enhance(path, fe, dec, proc):
    x, sr = sf.read(path)
    x = np.asarray(x, dtype="float32")
    x16 = ss.resample_poly(x, 16000, sr).astype("float32")
    inp = proc(x16, sampling_rate=16000, return_tensors="pt")["input_features"].to(DEV)
    h = fe(inp)["last_hidden_state"].transpose(1, 2)
    w = dec(h).squeeze().float().cpu().numpy()
    peak = float(np.abs(w).max()) or 1.0
    return (w / peak * float(np.abs(x).max() or 0.9)).astype("float32"), 48000


async def judge(http, key, path, task, tag):
    tgt, ins = task.get("target") or {}, task.get("instruction") or {}
    sc = task.get("script"); text = sc.get("text") if isinstance(sc, dict) else sc
    prompt = harness.JUDGE_PROMPT % (ins.get("context",""),
        ins.get("performance_direction",""), tgt.get("label",""),
        tgt.get("intensity",""), text)
    raw = open(path, "rb").read()
    body = {"model": harness.JUDGE, "messages": [{"role":"user","content":[
        {"type":"text","text":prompt},
        {"type":"input_audio","input_audio":{
            "data": base64.b64encode(raw).decode(), "format":"wav"}}]}],
        "response_format":{"type":"json_object"}}
    import re
    for att in range(4):
        try:
            r = await http.post(f"{config.LUNA_BASE}/v1/chat/completions",
                                json=body, headers={"Authorization": f"Bearer {key}"})
            if r.status_code >= 400: raise RuntimeError(f"{r.status_code}")
            t = r.json()["choices"][0]["message"]["content"] or ""
            m = re.search(r"\{.*\}", t, re.S)
            d = json.loads(m.group(0) if m else t)
            o = {k: {"score": max(0, min(5, int(round(float(d[k]["score"]))))),
                     "why": str(d[k].get("why",""))[:300]} for k in harness.RUBRICS}
            o["total"] = sum(o[k]["score"] for k in harness.RUBRICS)
            o["tag"] = tag
            return o
        except Exception:
            if att == 3: return None
            await asyncio.sleep(2 + 3*att)


async def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    rows = [json.loads(l) for l in open(f"{ROOT}/runs/full/takes.jsonl") if l.strip()]
    rows = [r for r in rows if r.get("scores")]
    tasks = {t["id"]: t for t in json.load(open(f"{ROOT}/runs/full/tasks.json"))}
    # spread across the score range so the answer is not "it helps bad clips"
    rnd = random.Random(11)
    by = {}
    for r in rows: by.setdefault(r["total"] // 3, []).append(r)
    pick = []
    for k in sorted(by):
        rnd.shuffle(by[k]); pick += by[k][:max(1, n // len(by))]
    pick = pick[:n]
    print(f"{len(pick)} clips, totals {sorted(set(p['total'] for p in pick))}")

    fe, dec, proc = load_sidon()
    t0 = time.time()
    pairs = []
    for i, r in enumerate(pick, 1):
        src = os.path.join(ROOT, "audio", "full", r["audio"])
        dst = os.path.join(OUT, r["audio"])
        if not os.path.exists(dst):
            w, sr = enhance(src, fe, dec, proc)
            sf.write(dst, w, sr, subtype="PCM_16")
        pairs.append((r, src, dst))
        if i % 6 == 0: print(f"  enhanced {i}/{len(pick)}  "
                             f"{(time.time()-t0)/i:.1f} s/clip", flush=True)

    http = httpx.AsyncClient(timeout=600.0); key = config.luna_key()
    sem = asyncio.Semaphore(8)
    async def both(r, src, dst):
        async with sem:
            a, b = await asyncio.gather(
                judge(http, key, src, tasks[r["task"]], "orig"),
                judge(http, key, dst, tasks[r["task"]], "sidon"))
        return {"row": r["audio"], "task": r["task"], "stored": r["total"],
                "orig": a, "sidon": b}
    res = await asyncio.gather(*[both(*p) for p in pairs])
    json.dump(res, open(f"{ROOT}/runs/sidon_test.json","w"), ensure_ascii=False, indent=1)

    ok = [x for x in res if x["orig"] and x["sidon"]]
    print(f"\n{len(ok)} paare bewertet\n")
    print(f"{'rubrik':<12}{'original':>10}{'sidon':>10}{'delta':>9}{'t':>8}{'p':>8}")
    import math
    for k in list(harness.RUBRICS) + ["total"]:
        a = [x["orig"][k]["score"] if k != "total" else x["orig"][k] for x in ok]
        b = [x["sidon"][k]["score"] if k != "total" else x["sidon"][k] for x in ok]
        d = [y - z for y, z in zip(b, a)]
        m = statistics.mean(d)
        sd = statistics.stdev(d) if len(d) > 1 else 0
        t = m / (sd / len(d) ** 0.5) if sd else 0.0
        p = math.erfc(abs(t) / math.sqrt(2)) if sd else 1.0
        print(f"{k:<12}{statistics.mean(a):>10.2f}{statistics.mean(b):>10.2f}"
              f"{m:>+9.2f}{t:>8.2f}{p:>8.3f}")
    better = sum(1 for x in ok if x["sidon"]["total"] > x["orig"]["total"])
    worse = sum(1 for x in ok if x["sidon"]["total"] < x["orig"]["total"])
    print(f"\nsidon besser bei {better}/{len(ok)}, schlechter bei {worse}")
    print("\nbeispiel-begruendungen (natural):")
    for x in ok[:3]:
        print(f"  orig : {x['orig']['natural']['why'][:120]}")
        print(f"  sidon: {x['sidon']['natural']['why'][:120]}\n")

asyncio.run(main())
