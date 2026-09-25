# The WikiSkill knowledge layer — first draft

> **⚠ Superseded in part — see [REVISION_2026-09-02.md](REVISION_2026-09-02.md)
> (latest) and [REVISION_2026-08-31.md](REVISION_2026-08-31.md).**
> The general-quality DPO adapter ships at weight **1.5**, chosen by ear because the
> statistics could not separate the candidates. Style-axis steering measures strongly and
> for free at low alpha, but has never been listened to; the emotion-axis listening report
> that switched steering off was collected at alpha 0.1, inside the good range.
> Of 140 burst class x placement cells only 37 ever produce the requested burst.
> The emotion adapters' "contrast compression" was measured on endpoint contrast, which is
> not what a transition is for: that result is **open**, not negative.
> Steering recommendations must not be acted on (listener evidence; the server default
> is `adapter`). Guidance is usable to g = 4.0. Emotion adapters compress the contrast
> in a *transition* — use the two-take splice there. Burst doses 0.25 / 0.5 confirmed.

**29 August 2026.** The wiki layer specified in `whitepaper/WIKISKILL_ACTING_SYSTEM.md` §4.2,
built for the first time. Sixty attribute pages, an index, a cross-cutting interactions page,
and the machine-readable `coefficients.json` the actor reads at run time.

Six weeks of experiments in this project have produced exactly what WikiSkill (arXiv
2608.27454) diagnoses: *insights that guide skill development remain scattered across
optimization histories, limiting their systematic reuse*. A large, correct and almost unusable
pile. This is the first attempt to make it usable — and it is deliberately a *layer over the
measurements*, not a second copy of them.

## Everything here is generated

`code/build_wikiskills.py` reads four files and writes everything else. **Nothing in
`patterns/`, `index.md`, `interactions.md` or `coefficients.json` is typed by hand.**

```bash
python code/build_wikiskills.py
```

| input | what it contributes |
|---|---|
| `combination-study/stats/comb_recommendations.json` | the balanced and high-effect recipe for each of 60 attributes, with every measured Δ |
| `combination-study/stats/analysis.json` | the 2×2×2 factorial: main effects and the three pairwise interactions, pooled and per attribute |
| `lora-dose/coefficients.json` | 79 adapters: dose-response shape, safe and strong weight, the derived guardrail thresholds |
| `work_vb/tap_rank.json` | the per-dimension layer ranking that `top1` / `top3` resolve to |

The reason is drift. A hand-typed table stops agreeing with the data the moment either
changes, and the whole point of the layer is that the actor and the maintainer read the same
numbers. There are exactly **four hand-carried figures** in the generator, in a block called
`CITED` at the top, each with the study it comes from; they are the ones whose aggregates do
not live in those four files (the α = 0.3 break point, the k ≥ 2 quality break, the 1.93×
guidance cost, and the numbness subtraction).

## What is in it

| file | what |
|---|---|
| `index.md` | the catalogue: every attribute, its balanced and high-effect mode, Δ target, t, and adapter weight |
| `patterns/*.md` | one page per attribute — coefficient block, both operating points with the measured cost on all five guardrails, the adapter's own dose ladder, interactions, a `never` list, provenance |
| `interactions.md` | the cross-cutting page: which levers combine, which cancel, and the controls |
| `coefficients.json` | the same thing for machines; the demo server loads this file directly |
| `code/build_wikiskills.py` | the generator |

**60 attributes**: 40 emotions, 17 delivery axes, 3 quality axes.

## The headline, and why the layer is shaped this way

**The best single lever flips by family.** Target in SD units, from the factorial:

| family | adapter | steering | guidance |
|---|--:|--:|--:|
| emotion | +0.077 (t 2.8) | **+0.384** (t 9.4) | +0.050 (t 1.8) |
| delivery | +0.377 (t 7.1) | **+0.614** (t 9.6) | +0.026 (t 0.4) |
| quality | **+0.399** (t 6.0) | +0.006 (t 0.0) | +0.062 (t 0.9) |

A single global recipe would therefore be wrong for two of the three families, which is what
makes a per-attribute table worth having rather than a paragraph of advice.

## Where an attribute has no usable setting, the page says so

**A balanced operating point exists for 53 of 60 attributes and a high-effect point for 56.**
Seven have neither: `emo/Bitterness`, `emo/Disappointment`, `emo/Disgust`,
`emo/Emotional_Numbness`, `emo/Relief`, `emo/Shame`, `vn/ARSH_low`. Their pages say **No
usable setting** and explain that no candidate cleared the guardrails.

Twenty-eight of sixty attributes have a usable adapter merge weight; the rest are `none`.
That is not the same as harmful — **no adapter in the 5,740-cell sweep was harmful at any
weight**; every failure was a failure to move the target.

This is the one thing the layer exists to get right. An absent recommendation is a finding.
Filling it with something plausible would make the table worse than no table, because a
consumer cannot tell an interpolation from a measurement once it is written down.

## How to read a recommendation

* **Balanced** clears every guardrail: word error ≤ 0.15 absolute and ≤ +0.05 over baseline,
  genuineness ≥ −0.6 of 6, blend ≥ −1.0 of 10, burst realisation ≥ −0.1, |duration error|
  ≤ 0.3 s.
* **High effect** relaxes those to word error ≤ 0.30, genuineness ≥ −1.5, blend ≥ −3.0, burst
  realisation ≥ −0.25, |duration error| ≤ 1.0 s.
* Every Δ is **paired**: same prompts, same seed, a prompt's clip samples averaged before
  averaging across prompts, *n* = number of prompts, with a t statistic and how many prompts
  improved.
* A recommendation only counts if it beats a **random direction of matched norm at the same
  operating point**. That control is null on its own (−0.033 pooled) and **not** null at the
  combined point (+0.106, t 2.78). Each recipe carries a `beats_random_floor` flag.

## Who consumes it

`LAION-AI/Humaneness-Voice-Demo-Server` loads `coefficients.json` at start-up
(`MOSS_WIKI_COEFFICIENTS`) and resolves each turn's generation mode against it — see that
repository's `docs/LEVERS.md` and `levers.py`. Absent the file, it refuses steering and
guidance rather than guessing a setting.

## What is not here

* **Vocal-burst classes.** 71 adapters exist and 19 are in the dose sweep, but the combination
  study did not cover them, so there is no operating point to write down and no page is
  generated.
* **`logs.md` and `skill-impact.md`.** In the whitepaper layout those are written by the
  consolidation cycle, which has not run. This draft is cycle zero: the maintainer, the
  proposer and the gate do not exist yet.
* **Provenance down to take ids.** The pages name the file and the key each number comes from,
  which is enough to re-derive any of them, but not the individual takes.
* **Listening tests.** Every number in every page is one model's judgement of another model's
  output. Nobody has listened to any of it.
* **Anything but English and German**, and almost everything measured so far is English.

## Two limits worth stating twice

**Ten prompts per cell.** The family-level rows pool hundreds of cells and are solid. A single
attribute's interaction row is ten prompts and is not — the pages say so where they print one.

**The recipes were scored without the numbness subtraction.** Subtracting `Emotional_Numbness`
at α = −0.10 returns +0.60 of genuineness (t 9.64, on 67 of 80 prompts) at no cost in emotion
when the adapter carries the emotion, and the actor attaches it to every steered emotion. It
is a separate, separately measured component and is deliberately *not* folded into the
coefficient blocks, which would misreport what was scored.

## The steering pack

`vectors/p3_vectors_server.npz` (**5.3 MB**, in the research-log mirror only) is the distilled
form of the steering library the coefficient blocks refer to: 99 attributes × their own top 5
layers × 2560, float32, plus the layer ranking embedded as JSON so the consumer needs one
file. `code/build_steering_pack.py` produces it from `p3_vectors_ext.npz` (**112 MB**, three
difference tables of 99 × 38 × 2560) and `work_vb/tap_rank.json`, both of which live on the
cluster at `$SC/out/actforensics/vectors/` and `$SC/work_vb/` and are **not** in this
repository — `actforensics/` here holds the write-up, not the arrays.

The distilled pack is committed deliberately and it is worth saying why, since it is a binary:
it is the only channel by which the demo server — which cannot reach the cluster — gets the
vectors, it is a twentieth of the size of the file it comes from, and this repository already
tracks `layer-forensics/w3/archetypes.npy` at 58 MB and `mid_reference.npy` at 37 MB. **The
112 MB source is not being added.**

float32 rather than float16 is deliberate: a steering direction is a *difference* of means,
one to two orders of magnitude smaller than the means themselves, and half precision would
quantise it to a few significant bits.
