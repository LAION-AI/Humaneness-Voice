"""Evolutionary search over prompt additions, scored by a listening model.

For each benchmark task and each of two directors (the local gemma-4-12B and the
hosted luna), three generations of five prompt variants, three takes per variant
in one batched forward pass — 45 clips per task per director.  Every clip is
rated 0-5 by gemini-3.8-flash on three rubrics with a sentence of justification,
and the generation that follows is bred from what scored well.

Everything is written to disk as it happens: the audio, the exact prompt
addition, the script the director wrote, the hyper-parameters of the take, the
rubric scores and their justifications.  The run resumes where it stopped.
"""
import argparse
import asyncio
import base64
import hashlib
import json
import os
import random
import re
import struct
import sys
import time

import httpx

APP = os.environ.get("ARENA_APP", "http://127.0.0.1:8792")
ROOT = os.environ.get("ARENA_ROOT", "/mnt/nvme/arena")
JUDGE = "gemini-3.8-flash"
SR = 48000

sys.path.insert(0, "/mnt/nvme/moss-15-v2")
import config                                                   # noqa: E402

RUBRICS = ("pleasant", "fit", "natural")

# ---------------------------------------------------------------- variants --
# Generation 1.  Five hypotheses about what a director is not being told
# strongly enough, plus the untouched prompt as the control every later
# generation is measured against.
SEEDS = [
    ("control", ""),
    ("imperfection",
     "ONE MORE THING FOR THIS REPLY. Real speech is never clean. Build in the "
     "small failures a person makes and does not notice: a word started and "
     "restarted, a breath taken in the wrong place, a thought that arrives "
     "before the sentence is ready for it. Mark them with directions and "
     "pauses where they fall. A take that is slightly untidy in the way a "
     "person is untidy beats a take that is smooth."),
    ("breath",
     "ONE MORE THING FOR THIS REPLY. Put more silence in it than feels correct "
     "on the page. Between two and four pauses, most of them INSIDE sentences "
     "rather than between them, placed where the speaker is thinking, "
     "hesitating, choosing a word, or deciding whether to go on. Vary their "
     "length: 0.2 for a breath, 0.5 for a beat, 0.9 when the thought genuinely "
     "stalls."),
    ("body",
     "ONE MORE THING FOR THIS REPLY. This voice has a body. Use vocal bursts "
     "for what the body does while the mind talks — the intake before bad "
     "news, the laugh that escapes before it is approved, the sigh that lands "
     "after the sentence rather than before it. At least one burst, given its "
     "own bracket and its own length, placed where the body would act rather "
     "than where it would be tidy."),
    ("subtext",
     "ONE MORE THING FOR THIS REPLY. Play what is underneath, not what is on "
     "top. The feeling should be visibly held rather than performed: contained, "
     "leaking at the edges of phrases, showing in the timing and the breath "
     "before it shows in the volume. Trust the listener to hear a small signal. "
     "Under-play by one notch on the adverb scale rather than over-play."),
]

MUTATOR = """You are tuning the system prompt of a voice director. The director
writes a timed script — delivery directions in round brackets, vocal bursts with
a length, pauses in square brackets — which a text-to-speech model then performs.

An extra block of instruction is appended to the director's standing rules. We
are searching for the block that produces the best-sounding performance.

Each block below was tried and the resulting audio was rated 0-5 by a listener
on three rubrics: PLEASANT (how pleasant it sounds), FIT (how well it matches
the task) and NATURAL (how much it sounds like a real, spontaneous moment,
including imperfections and micro-expressions).

%s

Write %d NEW blocks. Each should be a different bet, not a rewording of the
winners. You may combine what worked, push a winning idea further, or try
something none of them tried. Keep each under 90 words, addressed to the
director in the second person, and phrased as instructions for THIS reply.
Do not mention scores, rubrics, experiments or this message.

Return JSON: {"blocks": [{"name": "<two words, lowercase, hyphenated>",
"text": "<the block>"}, ...]}"""

JUDGE_PROMPT = """You are listening to a single take from a voice-acting
benchmark. Rate it on three rubrics, each 0 to 5, each with one sentence of
justification.

THE TASK THE ACTOR WAS GIVEN
Situation: %s
Direction: %s
Emotion to convey: %s (%s intensity)
The words, which are fixed: "%s"

THE RUBRICS
- pleasant: how pleasant this is to listen to. 0 is grating, distorted or
  painful; 5 is a voice you would happily hear for an hour.
- fit: how well the performance matches the task above — the emotion, its
  intensity, the situation and the direction. 0 is unrelated or contradictory;
  5 is exactly what was asked for.
- natural: how much this sounds like a real person in a real, spontaneous
  moment, with the imperfections and micro-expressions that come with it —
  breath, hesitation, a word caught, timing that is not metronomic. 0 is
  obviously synthetic or read aloud; 5 is indistinguishable from a candid
  recording.

Judge only what you hear. Ignore recording quality differences that are not the
performance. Be willing to use the whole range: most takes are not 4s.

Return JSON: {"pleasant": {"score": n, "why": "..."},
"fit": {"score": n, "why": "..."}, "natural": {"score": n, "why": "..."}}"""


def wav_bytes(pcm16, sr=SR):
    n = len(pcm16)
    return (b"RIFF" + struct.pack("<I", 36 + n) + b"WAVEfmt " +
            struct.pack("<IHHIIHH", 16, 1, 1, sr, sr * 2, 2, 16) +
            b"data" + struct.pack("<I", n) + pcm16)


def pick_tasks(path, n, seed=20260905):
    """A spread across tracks, intensities and emotions rather than the first n."""
    tasks = json.load(open(path))["tasks"]
    rnd = random.Random(seed)
    buckets = {}
    for t in tasks:
        key = (t.get("track"), (t.get("target") or {}).get("intensity"))
        buckets.setdefault(key, []).append(t)
    for v in buckets.values():
        rnd.shuffle(v)
    out, seen_label, keys = [], set(), sorted(buckets, key=lambda k: str(k))
    while len(out) < n:
        progress = False
        for k in keys:
            if len(out) >= n:
                break
            for i, t in enumerate(buckets[k]):
                lab = (t.get("target") or {}).get("label")
                if lab in seen_label:
                    continue
                out.append(buckets[k].pop(i))
                seen_label.add(lab)
                progress = True
                break
        if not progress:                       # labels exhausted; take any
            for k in keys:
                if buckets[k] and len(out) < n:
                    out.append(buckets[k].pop(0))
            break
    return out[:n]


class Run:
    def __init__(self, args):
        self.a = args
        self.dir = os.path.join(ROOT, "runs", args.run)
        self.audio = os.path.join(ROOT, "audio", args.run)
        os.makedirs(self.dir, exist_ok=True)
        os.makedirs(self.audio, exist_ok=True)
        self.rec_path = os.path.join(self.dir, "takes.jsonl")
        self.done = set()
        if os.path.exists(self.rec_path):
            for line in open(self.rec_path, encoding="utf-8"):
                try:
                    r = json.loads(line)
                    self.done.add(r["key"])
                except Exception:
                    pass
        self.rec = open(self.rec_path, "a", encoding="utf-8")
        self.log_f = open(os.path.join(self.dir, "log.txt"), "a",
                          encoding="utf-8")
        self.http = httpx.AsyncClient(timeout=900.0)
        self.key = config.luna_key()
        self.sem_judge = asyncio.Semaphore(args.judge_concurrency)
        self.sem_gen = asyncio.Semaphore(args.gen_concurrency)

    def log(self, msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    # ------------------------------------------------------------ generate --
    async def take(self, task, brain, variant, gen):
        """One variant: the director writes once, three takes in one pass."""
        key = f"{task['id']}|{brain}|g{gen}|{variant['name']}"
        body = {"message": json.dumps(task, ensure_ascii=False),
                "brain": brain, "session": f"arena-{abs(hash(key)) % 10**8}",
                "best_of": self.a.takes, "best_of_audio": True,
                "prompt_extra": variant["text"] or None,
                # Both directors get the same style.  The codes style emits no
                # neutralised branch, so it silently ran at guidance 1 while
                # prose ran at 4 — the director and the guidance would have
                # been confounded.
                "prompt_style": "prose"}
        async with self.sem_gen:
            t0 = time.time()
            try:
                r = await self.http.post(f"{APP}/api/turn", json=body)
                r.raise_for_status()
                raw = r.content
            except Exception as e:
                self.log(f"  ! generate failed {key}: {type(e).__name__} {e}")
                return []
        ev, i = [], 0
        while i < len(raw):
            t = raw[i]
            n = struct.unpack(">I", raw[i + 1:i + 5])[0]
            p = raw[i + 5:i + 5 + n]
            i += 5 + n
            if t == 0:
                try:
                    ev.append(json.loads(p))
                except Exception:
                    pass
        llm = next((e for e in ev if e.get("type") == "llm"), {})
        bon = next((e for e in ev if e.get("type") == "best_of"), {})
        end = next((e for e in ev if e.get("type") == "end"), {})
        err = next((e for e in ev if e.get("type") == "error"), None)
        if err:
            self.log(f"  ! {key}: {str(err.get('message'))[:140]}")
            return []
        cands = bon.get("candidates") or []
        if not cands:
            self.log(f"  ! {key}: no candidates (best_of missing)")
            return []
        rows = []
        for c in cands:
            if not c.get("pcm"):
                continue
            pcm = base64.b64decode(c["pcm"])
            h = hashlib.sha1(pcm).hexdigest()[:16]
            fn = f"{task['id']}_{brain}_g{gen}_{variant['name']}_{c.get('index', 0)}_{h}.wav"
            with open(os.path.join(self.audio, fn), "wb") as f:
                f.write(wav_bytes(pcm))
            rows.append({
                "key": key, "task": task["id"], "track": task.get("track"),
                "brain": brain, "gen": gen, "variant": variant["name"],
                "variant_text": variant["text"], "take": c.get("index", 0),
                "audio": fn, "sec": round(len(pcm) / 2 / SR, 3),
                "reward": c.get("reward"), "wer": c.get("wer"),
                "extra_w": c.get("extra_w"), "gate": c.get("gate"),
                "rank": c.get("rank"),
                "script": llm.get("script"), "general": llm.get("general"),
                "reply": llm.get("reply"), "voice": llm.get("voice"),
                "chosen": llm.get("chosen"), "language": llm.get("language"),
                "speed": llm.get("speed"), "style": llm.get("style"),
                "prompt_style": llm.get("prompt_style"),
                "loras": end.get("loras"), "align": end.get("align"),
                "guidance": bon.get("guidance"), "bon_n": bon.get("n"),
                "gpu_ms": bon.get("ms"), "llm_ms": llm.get("llm_ms"),
                "wall_ms": round((time.time() - t0) * 1000, 1),
            })
        return rows

    # --------------------------------------------------------------- judge --
    async def judge(self, row, task):
        tgt, ins = task.get("target") or {}, task.get("instruction") or {}
        sc = task.get("script")
        text = sc.get("text") if isinstance(sc, dict) else sc
        prompt = JUDGE_PROMPT % (
            ins.get("context", ""), ins.get("performance_direction", ""),
            tgt.get("label", ""), tgt.get("intensity", ""), text)
        raw = open(os.path.join(self.audio, row["audio"]), "rb").read()
        body = {"model": JUDGE, "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "input_audio", "input_audio": {
                "data": base64.b64encode(raw).decode(), "format": "wav"}}]}],
            "response_format": {"type": "json_object"}}
        async with self.sem_judge:
            for attempt in range(4):
                try:
                    r = await self.http.post(
                        f"{config.LUNA_BASE}/v1/chat/completions", json=body,
                        headers={"Authorization": f"Bearer {self.key}"})
                    if r.status_code >= 400:
                        raise RuntimeError(f"{r.status_code} {r.text[:120]}")
                    txt = r.json()["choices"][0]["message"]["content"] or ""
                    m = re.search(r"\{.*\}", txt, re.S)
                    d = json.loads(m.group(0) if m else txt)
                    out = {}
                    for k in RUBRICS:
                        v = d.get(k) or {}
                        s = int(round(float(v.get("score"))))
                        out[k] = {"score": max(0, min(5, s)),
                                  "why": str(v.get("why", ""))[:400]}
                    out["total"] = sum(out[k]["score"] for k in RUBRICS)
                    return out
                except Exception as e:
                    if attempt == 3:
                        self.log(f"  ! judge failed {row['audio']}: "
                                 f"{type(e).__name__} {str(e)[:110]}")
                        return None
                    await asyncio.sleep(2 + 3 * attempt)

    # ----------------------------------------------------------- evolution --
    async def breed(self, history, n):
        """New blocks from what scored well, written by the judge's own family."""
        lines = []
        for h in sorted(history, key=lambda x: -x["fitness"])[:6]:
            lines.append(
                f'--- block "{h["name"]}" scored pleasant {h["pleasant"]:.1f}, '
                f'fit {h["fit"]:.1f}, natural {h["natural"]:.1f} '
                f'(total {h["fitness"]:.1f} of 15)\n'
                f'{h["text"] or "(no extra block — the standing prompt alone)"}\n'
                f'A listener said: {h["why"]}')
        body = {"model": JUDGE,
                "messages": [{"role": "user",
                              "content": MUTATOR % ("\n\n".join(lines), n)}],
                "response_format": {"type": "json_object"}}
        for attempt in range(3):
            try:
                r = await self.http.post(
                    f"{config.LUNA_BASE}/v1/chat/completions", json=body,
                    headers={"Authorization": f"Bearer {self.key}"})
                r.raise_for_status()
                txt = r.json()["choices"][0]["message"]["content"] or ""
                m = re.search(r"\{.*\}", txt, re.S)
                blocks = json.loads(m.group(0) if m else txt)["blocks"]
                out = []
                for b in blocks[:n]:
                    nm = re.sub(r"[^a-z0-9-]", "", str(b["name"]).lower())[:24]
                    out.append({"name": nm or f"child{len(out)}",
                                "text": str(b["text"]).strip()})
                if out:
                    return out
            except Exception as e:
                self.log(f"  ! breed failed: {type(e).__name__} {str(e)[:110]}")
                await asyncio.sleep(3)
        return []

    # ------------------------------------------------------------- the run --
    async def one_cell(self, task, brain, variant, gen):
        """Generate the takes for one variant and rate every one of them."""
        key = f"{task['id']}|{brain}|g{gen}|{variant['name']}"
        if key in self.done:
            rows = [json.loads(l) for l in open(self.rec_path, encoding="utf-8")]
            return [r for r in rows if r.get("key") == key]
        rows = await self.take(task, brain, variant, gen)
        if not rows:
            return []
        scores = await asyncio.gather(*[self.judge(r, task) for r in rows])
        for r, s in zip(rows, scores):
            r["scores"] = s
            r["total"] = s["total"] if s else None
            self.rec.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.rec.flush()
        self.done.add(key)
        got = [r["total"] for r in rows if r.get("total") is not None]
        self.log(f"    {variant['name']:<16} n={len(rows)} "
                 f"best={max(got) if got else '—'} "
                 f"mean={sum(got)/len(got):.1f}" if got else
                 f"    {variant['name']:<16} unrated")
        return rows

    @staticmethod
    def fitness(rows):
        """Best of the takes: the search is for a prompt that can produce a
        good performance, not one that is safe on average."""
        got = [r for r in rows if r.get("scores")]
        if not got:
            return None
        best = max(got, key=lambda r: r["total"])
        return {"fitness": float(best["total"]),
                **{k: float(sum(r["scores"][k]["score"] for r in got) / len(got))
                   for k in RUBRICS},
                "why": " ".join(best["scores"][k]["why"] for k in RUBRICS)[:600]}

    async def task_arm(self, task, brain):
        """Three generations of five variants for one task and one director."""
        self.log(f"  {task['id']} / {brain}")
        variants = [{"name": n, "text": t} for n, t in SEEDS][:self.a.variants]
        history, per_gen = [], []
        for gen in range(1, self.a.generations + 1):
            results = []
            for v in variants:
                rows = await self.one_cell(task, brain, v, gen)
                f = self.fitness(rows)
                if f:
                    results.append({"name": v["name"], "text": v["text"], **f})
            per_gen.append({"gen": gen, "variants": results})
            history += results
            if gen == self.a.generations:
                break
            elite = sorted(results, key=lambda r: -r["fitness"])[:2]
            n_new = self.a.variants - len(elite)
            kids = await self.breed(history, n_new) if n_new else []
            seen = {e["name"] for e in elite}
            for i, k in enumerate(kids):
                while k["name"] in seen:
                    k["name"] = f"{k['name']}-{i}"
                seen.add(k["name"])
            variants = [{"name": e["name"], "text": e["text"]} for e in elite] \
                + kids
            while len(variants) < self.a.variants:      # breeding fell short
                variants.append({"name": f"repeat{len(variants)}",
                                 "text": elite[0]["text"] if elite else ""})
            self.log(f"    -> gen {gen+1}: elite "
                     f"{[e['name'] for e in elite]} + "
                     f"{[k['name'] for k in kids]}")
        return {"task": task["id"], "brain": brain, "generations": per_gen}


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=time.strftime("%Y%m%d-%H%M"))
    ap.add_argument("--tasks", type=int, default=20)
    ap.add_argument("--variants", type=int, default=5)
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--takes", type=int, default=3)
    ap.add_argument("--brains", default="local,luna")
    ap.add_argument("--bench", default=os.path.join(ROOT, "arena.json"))
    ap.add_argument("--gen-concurrency", type=int, default=2)
    ap.add_argument("--judge-concurrency", type=int, default=12)
    ap.add_argument("--only", default="")
    a = ap.parse_args()

    r = Run(a)
    tasks = pick_tasks(a.bench, a.tasks)
    if a.only:
        tasks = [t for t in tasks if t["id"] in a.only.split(",")]
    brains = a.brains.split(",")
    json.dump(tasks, open(os.path.join(r.dir, "tasks.json"), "w"),
              ensure_ascii=False, indent=1)
    json.dump({"seeds": SEEDS, "judge": JUDGE, "args": vars(a),
               "app": APP}, open(os.path.join(r.dir, "setup.json"), "w"),
              ensure_ascii=False, indent=1)
    total = len(tasks) * len(brains) * a.variants * a.generations * a.takes
    r.log(f"run {a.run}: {len(tasks)} tasks x {len(brains)} directors x "
          f"{a.generations} generations x {a.variants} variants x {a.takes} "
          f"takes = {total} clips")
    r.log(f"already on disk: {len(r.done)} cells")
    st = os.statvfs(ROOT)
    r.log(f"disk free: {st.f_bavail * st.f_frsize / 2**30:.0f} GiB "
          f"(the run needs about {total * 1.4 / 1024:.1f} GiB of audio)")

    out = []
    t0 = time.time()
    for i, task in enumerate(tasks, 1):
        for brain in brains:
            r.log(f"[{i}/{len(tasks)}] {task['id']} {task.get('title','')} "
                  f"({task.get('track')}) — {brain}")
            out.append(await r.task_arm(task, brain))
            json.dump(out, open(os.path.join(r.dir, "arms.json"), "w"),
                      ensure_ascii=False, indent=1)
        st = os.statvfs(ROOT)
        free = st.f_bavail * st.f_frsize / 2**30
        if free < 20:
            r.log(f"stopping: only {free:.1f} GiB left on disk")
            break
        done_cells = len(r.done)
        want = len(tasks) * len(brains) * a.variants * a.generations
        el = time.time() - t0
        if done_cells:
            r.log(f"  progress {done_cells}/{want} cells, {el/60:.0f} min "
                  f"elapsed, ~{(want-done_cells)*el/done_cells/60:.0f} min left")
    r.log("done")


if __name__ == "__main__":
    asyncio.run(main())
