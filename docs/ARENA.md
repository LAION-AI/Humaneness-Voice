# Searching for a better director prompt, and not finding one

An evolutionary search over additions to the system prompt, scored by a
listening model. Run 5 September 2026 against `voice_acting_arena_v0.2.json`
(374 tasks). **The headline is a null result, and the reason it is null is worth
more than the ranking it replaced.**

---

## The design

Twenty tasks, drawn stratified — every one a different emotion label, spread
across `emotion`, `voicenet` and `acting_challenge` and across intensities. The
pick is deterministic (seed 20260905) and saved with the run.

For each task and each of two directors — local `gemma-4-12B-it-qat` and hosted
`gpt-5.6-luna` — three generations of five prompt additions, three takes per
addition in one batched forward pass: **45 clips per task per director**.

A *prompt addition* is a block appended to the standing system prompt.
Generation 1 is five fixed hypotheses. Later generations keep the two best
verbatim and breed three new ones from the scores and the listener's own
justifications, written by `gemini-3.8-flash`.

Every clip was rated by `gemini-3.8-flash`, 0–5 on three rubrics with one
sentence of justification each: **pleasant** (how pleasant to listen to),
**fit** (how well it matches task, emotion, intensity, direction) and
**natural** (how much it sounds like a real spontaneous moment, with the
imperfections and micro-expressions that come with one).

Fitness was the **best of three takes**. That choice is the flaw the run
uncovered; see below.

## The judge discriminates

Rating our own output is worthless if the scale is pinned, so the judge was
calibrated first on audio it had not produced. A real candid recording came back
`natural 5` — *"the hesitation, breathing, spontaneous laughter and interactive
timing sound entirely candid and unforced"*. A read-aloud narration from the same
corpus came back `natural 1`. The scale works, and the `natural 1` our clips
often earn is an honest verdict with four points of headroom above it.

Across 1,140 clips the judge used the range: `fit` returned 83 fives and 51
zeros, `natural` 30 fives and 49 zeros. **`pleasant` returned five 5s in 1,140
clips.** The ceiling in this system is the sound, not the direction.

## What the naive ranking said, and why it was wrong

Pooled over everything, bred blocks looked spectacular:

| block | n | pleasant | fit | natural | total | t vs control |
|---|--:|--:|--:|--:|--:|--:|
| shifting-focus | 6 | 3.67 | 4.17 | 3.83 | **11.67** | +3.09 (p 0.002) |
| shared-room | 6 | 4.00 | 3.83 | 3.33 | 11.17 | +5.11 (p 0.000) |
| mental-friction | 6 | 4.00 | 3.33 | 3.50 | 10.83 | +3.58 (p 0.000) |
| control | 144 | 2.82 | 2.37 | 2.12 | 7.31 | — |

Every one of those p-values is an artefact. **Each task-arm breeds its own
children**, so a block with n = 6 was tried on exactly one task with one
director, while `control` is pooled over all of them. The comparison measures
task difficulty, not prompt quality.

## The comparisons that survive the design

**Generation 1 is balanced** — all five seeded blocks ran on the same cells —
so it is the clean test of the five hypotheses, paired within (task, director):

| block | cells | mean total | vs control | t | p |
|---|--:|--:|--:|--:|--:|
| control | 26 | 6.90 | — | | |
| imperfection | 26 | 7.38 | **+0.49** | 1.04 | 0.300 |
| subtext | 26 | 7.36 | **+0.46** | 1.03 | 0.305 |
| body | 26 | 6.78 | −0.12 | −0.21 | 0.830 |
| breath | 26 | 6.63 | −0.27 | −0.52 | 0.606 |

**None of the five beats the untouched prompt.** Two point the right way and
neither reaches significance at n = 26 arms.

**Breeding did not help.** Best variant per arm, by generation:

| generation | arms | best cell mean | best single take |
|---|--:|--:|--:|
| gen 1 | 25 | 8.88 | 11.40 |
| gen 2 | 25 | 9.28 | 11.40 |
| gen 3 | 25 | 8.93 | 10.96 |

gen2 − gen1 = +0.40 (t 1.12, p 0.27); gen3 − gen1 = +0.05 (t 0.16, p 0.87).

**And the children were coin flips.** Each of the 176 bred blocks against the
control *of its own cell*: mean gain **−0.01** (t −0.07, p 0.94), and **84 of
176 beat their control**.

## Why: the fitness function was selecting luck

| source of spread | SD (scale 0–15) |
|---|--:|
| between the three takes of **one cell** | **1.57** |
| between the variants of one arm | 1.75 |
| between tasks | 1.69 |

Take-to-take noise is as large as everything the search was trying to measure.
With fitness defined as *best of three*, drawing three clips at random from the
pool and keeping the maximum scores **+2.78 points** over a single take — before
any prompt has done anything. The search was climbing that gradient.

The direct check: the same block, run twice in different generations, correlates
at **r = +0.55** and differs by **2.16 points** on average. A single cell cannot
resolve a difference smaller than about two points, and no real prompt effect
here is that large.

## What is solid

* **Track differences are large and real.** `emotion` 7.78, `acting_challenge`
  7.57, `voicenet` **4.87** over 1,140 clips. The VoiceNet items — axis targets
  like `ARSH_low`, `S_DRAM_high` — are where the system is weakest, and no
  prompt addition closed that gap. This points at the adapters, not the prompt.
* **Luna edges out the local model**, 6.92 to 6.60 — smaller than expected, and
  the local model only became competitive once it was given the same `prose`
  style and the same guidance.
* **`breath` was the only block with a consistently negative sign on all three
  rubrics.** It asks for more pauses, which is exactly what `docs/PROMPTING.md`
  was changed to encourage the week before. That is a warning, not a refutation:
  it is not significant, and the block asks for *more* pauses than the standing
  prompt already produces.

## Recommendation, today

**Do not add any of these blocks to the shipped prompt.** The standing prompt
was not beaten by anything that survived a paired comparison.

If one must be chosen — for a demo where a single reply matters more than an
average — `imperfection` and `subtext` are the two with a positive sign, at
+0.49 and +0.46 of 15, both p ≈ 0.30. That is a preference, not a measurement.

**Do not push pauses further** until `breath` is retested. It is the only signed
warning in the run.

**The next run should measure the prompt, not the luck.** That run is now
complete; see below. It did not change the recommendation — it hardened it, and
it turned one of the five blocks from a hint into a finding.

## Everything is on disk

`/mnt/nvme/arena/runs/full/takes.jsonl`, one line per clip: task and track,
director, generation, variant, **the full text of the prompt addition**, the
script the director wrote, the chosen voice and reference clip, adapters and
weights, guidance, internal reward, word error, extra words, timings, and the
three rubric scores with their justifications. Audio beside it in
`audio/full/`, named so each clip traces back to its cell. `analyse.py` prints
the pooled tables, `paired.py` the ones above.

The exact prompts — the five seeded blocks, the breeding prompt and the judge's
rubric prompt — are in [`ARENA_PROMPTS.md`](ARENA_PROMPTS.md).


---

# The corrected run

Completed 6 September 2026, 05:19. **600 cells, 1,800 clips, 20 tasks, nothing
dropped.** Three things differ from the run above, each fixing a way the first
one fooled itself:

* **Every block is evaluated on every arm.** The population is global rather
  than bred inside a single (task, director) arm, so every block carries n = 40
  paired comparisons instead of one.
* **Fitness is the mean of the takes, not the maximum** — the maximum was
  buying +2.78 points from luck alone.
* **The control stays in the population in every generation**, so the paired
  comparison never loses its baseline.

## Generation 1 — the five hypotheses, now at n = 40

| block | mean | vs control | t | p |
|---|--:|--:|--:|--:|
| body | 7.97 | +0.03 | +0.07 | 0.947 |
| **control** | **7.94** | — | | |
| imperfection | 7.49 | −0.45 | −0.86 | 0.389 |
| subtext | 7.39 | −0.55 | −1.24 | 0.214 |
| breath | 6.91 | **−1.03** | −2.34 | **0.019** |

**The control is at the top.** Nothing beat the shipped prompt, and one block is
measurably worse than it.

## The replication is the result

Generation 1 is the same experiment in both runs, so the two are independent
replicates of one measurement:

| block | run 1 delta | p | run 2 delta | p | pooled |
|---|--:|--:|--:|--:|--:|
| imperfection | **+0.49** | 0.300 | **−0.45** | 0.389 | −0.08 |
| subtext | **+0.46** | 0.305 | **−0.55** | 0.214 | −0.15 |
| body | −0.12 | 0.830 | +0.03 | 0.947 | −0.03 |
| breath | −0.27 | 0.606 | **−1.03** | 0.019 | −0.73 |

**Both of run 1's promising blocks changed sign.** The two we would have shipped
on a preference — `imperfection` and `subtext` — came back negative when
measured again. Only `breath` replicated, and it replicated as harm.

That is the whole lesson of these two runs in one table: at this effect size, a
single well-run experiment is not enough to justify a prompt change. It takes a
replication to tell +0.5 from −0.5.

## Breeding, again, did not help

| generation | best block | delta | t | p |
|---|---|--:|--:|--:|
| gen 1 | body | +0.03 | +0.07 | 0.947 |
| gen 2 | anchored-burst | +0.21 | +0.46 | 0.647 |
| gen 3 | anchored-burst | +0.64 | +1.65 | 0.099 |

`anchored-burst` is the only block with a consistent positive sign across two
generations, and it is the closest thing to a candidate this study produced. It
is still not significant, and **twelve block-versus-control comparisons were
computed**, so the Bonferroni threshold here is p < 0.0042. It does not come
close.

Its text, for the record:

> ONE MORE THING FOR THIS REPLY. Ground the voice in physical action, but keep
> it strictly fused to speech. Include a single vocal burst — a scoff, a sharp
> intake, or a dry chuckle — placed immediately ahead of the line that triggers
> it. Never place bursts or breaths after the final word. Direct the physical
> posture in round brackets: speaking through a grin, ribs tight, or leaning
> back. Let posture shape the vocal timbre so the performance sounds produced by
> muscle.

## More pauses make it worse, and it is the one thing worth acting on

`breath` — *"between two and four pauses, most of them INSIDE sentences"* — is
negative on **all three rubrics individually** at generation 1:

| rubric | delta | |
|---|--:|:--|
| pleasant | −0.28 | p < 0.05 |
| fit | −0.38 | p < 0.05 |
| natural | −0.38 | p < 0.05 |

And it is almost entirely the hosted director: **luna −1.83 (t −3.04, p 0.002)**
against **local −0.23 (t −0.38, p 0.70)**. Luna already writes dense direction,
and pushing it for more silence degrades the result on every axis.

**What this does and does not say.** `breath` asks for *more* pauses than the
standing prompt, which already carries the pause rules added on 5 September and
the `breathe()` floor. So this measures the marginal push beyond the current
default, and finds it harmful. It does **not** show that the current default is
itself wrong — that needs a block which *removes* the emphasis, run against the
same control. **That experiment has not been run.** Until it is, the honest
position is: do not push pauses further, and treat last week's pause work as
unvalidated rather than as confirmed.

## Standing differences

| | mean of 15 |
|---|--:|
| emotion track | 8.88 |
| acting_challenge | 7.93 |
| **voicenet** | **5.93** |
| luna | 7.62 |
| local gemma-4-12B | 7.40 |

The VoiceNet gap is nearly three points and no prompt addition in either run
touched it. That is an adapter problem, not a prompting one.

## Recommendation after both runs

**Ship nothing from this study.** The prompt as it stands was not beaten by any
of the nineteen blocks tried across two runs and 2,949 rated clips.

**Do not push pauses further** — the only replicated, signed effect in either
run, significant on all three rubrics and strongest on the hosted director.

**If a follow-up is run**, the two questions worth the GPU time are: does
*removing* the pause emphasis beat the current prompt, and why does the VoiceNet
track sit three points below the others.

**And the lever that did work was not a prompt at all.** Nineteen prompt
additions moved nothing that survived a paired comparison; SIDON restoration
moved `pleasant` by +0.43 of 5 at p < 0.001 on the first attempt, with no search
and no tuning. It shipped on for that reason and was turned back off a day
later, when a listener found it distorts on screams and loud bursts — the clips
behind the measurement contained none. See [`SIDON.md`](SIDON.md), including the
hypothesis that the adapter stack is what it is compensating for.
