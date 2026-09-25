# The scream that came back as babble

A VoiceNet benchmark item — a parent screaming a child away from a window —
came back unintelligible, word error **1.82**. Four separate faults, three of
them fixed here and one of them mine from the week before.

## What it was not

The obvious suspects were wrong, and it is worth saying so before the one that
was right.

The script opened with **two** screams where the director wrote one — a real
bug, below — but removing the duplicate did not help. Nor did moving the scream
after the first words, nor the wrong emotion adapter the retrieval had chosen
(`Jealousy_and_Envy`, for a horror scene). Each of those was measured
individually and each came back broken.

## What it was: the sum of the burst adapters

Isolating the brackets from the adapters settles it in four rows, one seed each:

| condition | word error |
|---|--:|
| bursts in the script + 2 burst adapters @1.5 | 0.520 |
| bursts in the script, **no** burst adapter | **0.000** |
| **no** burst in the script, 2 burst adapters @1.5 | 0.840 |
| bursts in the script, **one** adapter @1.5 | **0.000** |

The brackets are innocent. **Two burst adapters merged at 1.5 each destroy the
line whether or not the script asks for a burst at all.** Replicated over five
seeds: 2 × 1.5 broke 5 of 5 takes at a median word error of 0.82, while one at
1.5 broke none.

That is the opposite of what `config.py` said. The comment there recorded a
measurement — *"two adapters cost nothing over one (1.5 × 2 → 0.320 against
0.322)"* — and cited it to justify a budget of 3.0. The difference is the rest
of the stack, and **the stack is part of the measurement**: a single-adapter
ladder on a bare model gives a ceiling of 1.5 and is not wrong, it is answering
a different question. Every take here carries 5.75 to 7.25 of merged weight
before a burst adapter is considered at all — the exact list is in
[`DEPENDENCIES.md`](DEPENDENCIES.md), and `eval/why_babble.py` sets it at the
top as `BASE` so it can be reproduced or deliberately changed: that ladder measured burst adapters on a bare model, while a turn
here already carries a voice adapter, three quality adapters, a preference
adapter, an emotion adapter **and a delivery axis at 1.5** before a burst
arrives. It is not the single weight that breaks a line here; it is the sum.

## The sweet spot, measured

Bursts carry the drama of a line, so the fix is a budget rather than a ban. Six
seeds per cell, same script, full shipped stack:

| arrangement | sum | median WER | mean | broken |
|---|--:|--:|--:|--:|
| 2 × 0.5 | 1.0 | 0.000 | 0.000 | 0/6 |
| 3 × 0.5 | 1.5 | 0.000 | 0.007 | 0/6 |
| 2 × 0.75 | 1.5 | 0.020 | 0.057 | 1/6 |
| 2 × 0.875 | 1.75 | 0.030 | 0.060 | 0/6 |
| 1 × 1.25 | 1.25 | 0.000 | 0.027 | 0/6 |
| **2 × 1.0** | **2.0** | **0.010** | **0.013** | **0/6** |
| 3 × 0.67 | 2.0 | 0.000 | 0.003 | 0/6 |
| 3 × 1.0/0.5/0.5 | 2.0 | 0.000 | 0.010 | 0/6 |
| 2 × 1.25 | 2.5 | 0.110 | 0.103 | 0/6 |
| 2 × 1.5 | 3.0 | 0.820 | — | 5/5 |

Everything at or under a sum of **2.0** is clean on every arrangement tried.
2.5 is visibly degrading and 3.0 is unusable. So:

* `BURST_LAM_BUDGET` 3.0 → **2.0** — the total across a turn
* `BURST_LAM_MAX` 1.5 → **1.25** — any single adapter

**And over budget the weights are now scaled, not dropped.** The old code
dropped whichever adapter pushed past the budget, which silently changed what a
reply could do depending on tag order — a scream quietly not merged is a worse
outcome than a scream at two thirds weight. Three adapters wanting 1.0 each now
run at 0.67, and the log says so: `burst budget 2: 2 adapters want 2.5, scaling
by 0.80`.

## Burst+stop DPO does not help, at either dose

Asked directly, crossed against the burst dose, four seeds per cell:

| burst dose | bs-dpo 0.0 | 0.5 | 1.0 | 1.5 |
|---|--:|--:|--:|--:|
| 2 × 0.5 (safe) | 0.000 | 0.000 | 0.000 | 0.030 |
| 2 × 1.5 (broken) | 0.600 | 0.950 | 0.860 | 0.900 |
| none | 0.000 | 0.000 | 0.000 | — |

At a safe dose it changes nothing up to 1.0 and costs a little at 1.5. **It does
not rescue the broken dose** — if anything it makes it worse. And the axis it
was trained for is not in play: the audio matched the requested duration to
within ±0.04 s in every cell, so there was no over-running to stop. It stays at
**0.0**, now for a measured reason rather than for want of a listening test.

## The duplicate scream: a regression from the pause work

`skills.repair_script` gives a burst its own bracket when the director only
*names* it inside a delivery direction. It has a guard against doing that when
the burst is already written — and the guard matched only brackets with **no
comma and no digit**:

```python
already = {... for m in re.finditer(r"\(([^),0-9]+)\)", script)}   # old
```

`(scream)` matches. `(scream, 0.7 seconds)` does not. And the prompt was changed
on 5 September to ask the director for exactly the second form. So from that
day, a director who wrote the burst *correctly* and then mentioned it in the
following direction got a second bare `(scream)` inserted, and the line opened
on two screams. The guard now reads the label off either form. There is a test.

## Why end-trimming "did not seem to work"

It was working. It was being shown the wrong copy.

Candidate audio was base64'd into the best-of-N payload **before** the trim,
which ran only on the take that got streamed. So a reply showed a player saying
18.4 s next to a stream of 16.5 s — the trimmed 1.9 seconds were real, and every
player in the list still contained them.

Every candidate is now trimmed before it is judged and before it is sent, and
the per-candidate report travels with it. Ranking on the trimmed audio also
means the reward scores what will actually be heard.

## Result

The item that started this, through the running server with all four fixes:

```
screams in the script: 1        (was 2)
[skills] burst budget 2: 2 adapters want 2.5, scaling by 0.80
rank0 wer=0.000  rank1 wer=0.000  rank2 wer=0.000
```

Word error **1.82 → 0.00**.

Reproduce with `eval/why_babble.py` (isolation), `eval/sweetspot.py` (the dose
table) and `eval/bsdpo.py` (the interaction).

---

# The German turn, which babbled for a different reason

The same item performed in German came back as babble again, and the burst
budget was not the cause this time. Four seeds a cell, word error against the
intended German text:

| condition | median WER | unusable |
|---|--:|--:|
| German cues + German GENERAL | 0.267 | 2/4 |
| German cues + English GENERAL | 0.389 | 3/4 |
| English cues + German GENERAL | 0.033 | 0/4 |
| **English cues + English GENERAL** | **0.000** | 0/4 |
| English/English + `reads as jealousy and envy` | 0.044 | 0/4 |
| English/English + the wrong emotion adapter | 0.067 | 0/4 |
| English/English + the right one (`Fear`) | 0.011 | 0/4 |

**It is the brackets.** Translating only the cues — leaving every German word of
the line untouched — took it from babble to clean. The corpus is captioned in
English; its German lines read *"Das zerreisst einen einfach, weisst du?
(relief sigh)"*. A German cue is outside the distribution the voice model
learned and it takes the words down with it.

The wrong emotion adapter, which looked like the obvious culprit —
`Jealousy_and_Envy` conditioning a horror scene — costs 0.067 against 0.011.
Real, and an order of magnitude smaller than the cues.

## The fix, since the prompt does not hold

The rule *"EVERY BRACKET IS WRITTEN IN ENGLISH"* has been in the system prompt
since the corpus was described, and the director writes German cues on German
turns anyway. So `cues.py` rewrites them server-side, **before retrieval** —
which matters, because the retriever falls back to the director's named emotion
whenever the cues are German, and that fallback is how a horror scene ended up
conditioned on `Jealousy_and_Envy` in the first place. English cues let the
retriever read the cues it was built to read.

The language model does the rewriting, not a lexicon. A lexicon over the
prescribed vocabulary was tried first and produced half-translated cues —
`(clearly amused, frei heraus, warm and offen)` — which is worse than either
language alone. It survives only as the offline fallback when no model can be
reached.

`MOSS_ENGLISH_CUES=0` turns it off. Measured cost: **1.3–3.6 s** on
`gemini-3.5-flash-lite`. The local 12B took **15.6 s** for the same three
brackets, which is a third of a turn spent translating, so the hosted model is
used when a key is present and the local one is the fallback.

## Three smaller faults found on the way

**The budget scaling was being undone.** After scaling, the code re-derived
every burst weight from the recipe. Adapters whose name tail *is* the class
(`burst_v2:fearful_gasp`) got their full recipe weight back, while
`burst_abl:ablation_d2_matched__scream` kept its scaled value because that tail
is not a class name. The two then summed to 2.42 against a budget of 2.0. It
now caps only, and a final clamp bounds the total whatever put an adapter in the
list.

**The UI slider outranked the new ceiling.** `blmax` still defaulted to 1.5 and
is passed as `burst_lam_max`, which overrides `BURST_LAM_MAX`. Now 1.25.

**A closure could not see `re`.** The translation helper referenced `re` while
the same handler imports `re` locally further down, which makes the name local
for the whole function — so the closure raised `name 're' is not defined`, fell
through to the lexicon, and produced the half-translated cues above. It uses its
own alias now.
