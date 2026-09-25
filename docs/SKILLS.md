# The measured knowledge layer, and what it changed

`wikiskills/` is a generated corpus of everything six weeks of experiments
established: sixty attribute pages, a machine-readable `coefficients.json` that
the lever planner already reads, and `VOCAL_BURSTS.md`, which is the part that
changes what the **director** writes.

`skills.py` parses those burst tables at boot and uses them in three places. The
tables are parsed rather than transcribed, for the same reason the wiki is
generated rather than typed: a copy stops agreeing with its source the moment
either moves.

## It corrects two things this server was getting wrong

**We were telling the director to do the one thing measured to hurt.** The
shipped instruction said to put bursts *mid-sentence*. Measured, mid-clause
placement is worse on **15 of 15** classes tested — hit rate −0.07 to −0.12, miss
rate +0.31 to +0.37, t 8–10. The single inversion is `clears_throat`, which is
genuinely better mid-sentence, and the new block says so.

**We were offering sounds that do not exist.** Of the 71 burst adapters on disk,
**36 are measured as never realising at any dose, or below the 0.15 shipping
bar** — every mouth class and every whistle class in the bank is among them. The
old prompt offered all 71. Asking for one produces a silent gap and no warning.
The new block offers 36 and names the rest as unavailable.

And one thing it did not know: **the useful merge weight is per class**, running
0.25 to 2.3 against the flat 0.25 / 0.5 shipped here. `chuckle` alone goes from
0.25 to its measured 2.0.

## What the director is now told

Beyond the list, five rules, each with the measurement behind it:

| rule | measured |
|---|---|
| put the burst between sentences, not inside one | −0.07…−0.12 hit, 15/15 classes worse |
| name the sound's **cause** in the GENERAL line | +0.026 hit |
| give a burst that matters a longer stated duration | +0.022, and +0.044 with the cause sentence |
| write the sound, never the action that makes it | "(he chuckles)" degrades to silence, −0.08…−0.11 |
| do not substitute a neighbouring class | null on family, a significant *harm* on strict (−0.021, t −2.9) |

The list itself is ordered by measured hit rate and split at 0.40, so the
director reaches for the reliable sounds first.

## The v2 adapter release, served per class

The 105-adapter release adds five serving arms beside the original 71. Which arm
serves a class is **not** a global setting: each class's own page names the arm
that won for it, and the merge rule moved some while most stayed. Read from the
pages at boot, that resolves to

| arm | classes |
|---|--:|
| the original set | 35 |
| `bulk_mix_full` (per class) | 12 |
| `ablation_d2_matched` | 2 |
| `bulk_mix_top1` | 1 |

— which is the merge table's own count, and a useful check that the parser is
reading what the study decided rather than what it would like.

All 105 are registered as adapter sets regardless, so every one is addressable
by name from the overlay and from `adapter_overrides`, including the ablation and
dose arms that exist as evidence rather than for serving. 274 adapters across 17
sets are now loadable.

The selector in the page pins one set for a whole turn instead, which is what
makes an A/B possible; `recipe` is the default and the only setting that follows
the measurements.

## Why a scream did not happen, and what changed

A real turn wrote `(a raw, tearing scream, completely overwhelmed by sudden
shock)` and produced no scream and loaded no scream adapter. Traced on that exact
script, there were three separate causes.

**1. The bracket was a direction, not a burst.** A round bracket without a number
is an instruction about how to *speak* the next sentence. `scream` is in the
vocabulary; `a raw, tearing scream, completely overwhelmed by sudden shock` is
not. So no `(scream, N.N seconds)` tag reached the model, no time was budgeted
for the sound, and no adapter was pulled. The prompt already told the director to
write the sound rather than describe it, and it described it anyway.

*Fixed by repairing rather than only instructing.* When a direction contains an
offerable burst label as a whole phrase, the direction is **kept** and a bare
burst bracket is inserted after it. The match is narrow on purpose — whole word,
and only labels with a measured recipe above the bar — which is what stops
`(spitting the words out)` from becoming a spit: `spitting` never realises, so it
is not offerable, so it is not repaired.

**2. Only one burst adapter was ever loaded per turn.** `detect_burst` returns a
single best match over the whole reply, chosen by *longest string*. That line
tagged both a scream and an exasperated sigh; `exasperated sigh` is the longer
name, so it won, and the scream lost — not on merit, on spelling. Now the tags
are read in order and each gets its own adapter, up to `BURST_MAX_ADAPTERS`.

**3. The weight was looked up by adapter name, not by class.** Found while
verifying the fix: two classes are served by ablation arms filed as
`ablation_d2_matched__scream`, whose tail is not the class name, so the lookup
missed and fell back to the flat 0.25 instead of the measured 1.5.

Verified end to end afterwards: the same request now produces
`(a raw, tearing scream …) (scream)` in the script, and loads
`ablation_d2_matched__scream @1.5` together with `sharp_inhale @2.3` — two
adapters, each at its own measured weight.

## The recipe weights do not survive this stack

Serving the measured per-class weights made the demo babble. Swept with the
**script held fixed** — three burst-carrying scripts, two seeds, only the weight
varying, because comparing across `/api/turn` calls measures the director writing
a different line each time rather than the adapter:

| w | 0.00 | 0.25 | 0.50 | 1.00 | 1.50 | **2.00** | **2.30** |
|---|--:|--:|--:|--:|--:|--:|--:|
| word error | 0.285 | 0.319 | 0.304 | 0.322 | 0.341 | **0.859** | **1.337** |
| invented words | 0.67 | 0.67 | 0.67 | 0.67 | 0.67 | **1.83** | **3.17** |

It is a **cliff, not a slope**: flat to 1.5, then intelligibility collapses. Nine
recipes ask for 2.0 or 2.3 — `chuckle`, `sharp_inhale`, `soft_hum` among them —
and every one is on the far side of it here. `BURST_LAM_MAX` was therefore 1.5, and is **1.25** since 6 September under a
total budget of 2.0 — see [`BABBLE.md`](BABBLE.md), where two adapters at 1.5
each destroyed the line in 5 of 5 seeds. It is the sum that breaks a line here,
with a slider in the page to hear the recipe weights.

This does not contradict the study. Its ladder measured **one** adapter, for
several classes with no production stack under it at all, while a turn here
merges it on top of a voice adapter, three quality adapters, a preference
adapter, an emotion adapter and often a delivery axis. The study saw the same
edge from its own side: nine of 400 ladder cells produced no decodable audio,
every one at w ≥ 2.3.

**Stacking is not the problem; weight is.** Two adapters cost nothing over one
(1.0 ×2 → 0.285 against 0.319 for ×1; 1.5 ×2 → 0.320 against 0.322), so the
total budget is generous and only bounds the merge cost.

> An earlier version of this section put the ceiling at 1.0 and the budget at
> 1.0, from turn-to-turn comparisons. Those were dominated by the director
> writing a different script each call — the same configuration measured 0.000
> once and 0.737 the next time — and both numbers were wrong. The fixed-script
> sweep is the one to trust.

## Directions in the shape the model was trained on

The round-3 prompting scheme builds a delivery direction from four pieces in a
fixed order, and the director now writes them the same way:

```
intensity adverb + emotion name (+ second emotion) + how it is held + manner
```

Three of those were missing and are now spelled out in the prompt.

**The adverb comes from a band, not from taste.** The same four bands labelled
every training clip — `barely/faintly` for held down, `clearly/plainly` for
plainly audible, `strongly/intensely` for running hard,
`overwhelmingly/utterly` for at the limit.

**The emotion is named in the direction**, in words, not only in GENERAL. A
direction that says how the voice moves without saying what it feels arrives as
manner rather than as feeling — which is exactly the failure round 2 measured
when it dropped directions altogether and emotional control collapsed to the
corpus median.

**Whether it is let out or held in is a fork in the training data**, not a
nuance: *"letting it out, not hiding it, unguarded"* against *"fought down
rather than shown, held in and only leaking at the edges of phrases"*.

And structurally: **only the first sentence gets the full direction**, later ones
get a short reminder — `(still clearly amused)`, `(malicious, still kept under)`.
A thirty-word note in front of a 0.6 second line buries the line it was meant to
shape.

Observed after the change, unprompted:

```
(clearly amused, letting it out, warm and unguarded; bright, relaxed)
[6.1 seconds duration] The most annoying thing this week was …
(exasperated sigh, 0.1 seconds)
[4.2 seconds duration] It took ten minutes to fix, and three days to stop hearing about.
```

### Burst length without a number

Burst durations are computed, because a number inside a round bracket is what
separates a burst from a direction and the director cannot be given one to write.
It can now write **`short`** or **`long`** in front of the label instead —
`(short chuckle)`, `(long exhausted groan)` — which map to the corpus's own 10th
and 90th percentiles, 0.14 s and 0.48 s, against a 0.28 s default. The prompt
says plainly that long means half a second: the longest burst ever observed in
the data is 2.46 s and the median is 0.28, so a sigh asked for at three seconds
is outside everything the model has seen.

Two bugs surfaced while checking this against the scheme:

* **The language code was wrong on English lines.** `GENERAL` ended `11.8s, DE`
  on an English take, because the German detector counted `was` — an ordinary
  English word that is also German — and two of them were enough. `die` had the
  same problem. The marker list now holds only words that are not also English,
  and umlauts count double since no English word carries one.
* **The repair inserted a second burst** when the director used the new length
  syntax: `(short chuckle)` was not recognised as an already-complete burst, so
  a bare `(chuckle)` was added after it and the line got two in a row.

## Cue language

Cues are written in **English even when the spoken line is German**. This is the
corpus convention, not a preference: the German training lines read
`Das zerreisst einen einfach, weisst du? (relief sigh)` — German words, English
cue. A German cue is out of distribution.

This is now in both the director prompt and the skills block. Observed
afterwards: a German turn wrote `(exhausted groan)` for the burst — the rule
taking effect — while its delivery cue stayed German. So the rule is followed
for burst labels and only partly for directions, which is worth knowing before
anyone reads a single German cue as evidence that it does not work.

## The switch

Checkbox **Neue Skills** in the chat page, on by default; `MOSS_SKILLS=0` for the
server default. Off restores the previous hand-written block *and* the flat
0.25 / 0.5 dose, so the two are comparable rather than half-mixed. Agents are
built for both settings at boot — the flag changes only the system prompt, so it
costs a string each — and the choice is per request, without a restart.

## Measured so far

Ten prompts each way, one seed. Small, and reported as such:

| | skills on | skills off |
|---|--:|--:|
| bursts written | 9 | 9 |
| of those, from the reliable tier (hit ≥ 0.40) | **5** | 2 |
| impossible classes asked for | 0 | 0 |

The director writes about as many bursts either way but shifts toward the ones
that actually come out. Neither run asked for an impossible class, so the
36-class exclusion is unproven in behaviour here — it is a guard whose value
shows up in the tail, and ten prompts is not the tail.

**What is not measured yet:** whether the audio is better. Everything above is
what the director *writes* and which adapter is loaded at which weight. Whether
more bursts land, and whether the higher per-class doses cost intelligibility,
needs the detector and the transcription harness, and needs the listening the
recipes themselves were chosen with:
[laion/moss-vocal-burst-recipes](https://huggingface.co/spaces/laion/moss-vocal-burst-recipes).

## One correction from the same study

The burst+stop preference adapter is a **null on burst realisation**:
+0.007 / +0.017 / +0.006 at its three recommended checkpoints, none significant,
and the two adapters are redundant rather than additive — all of the gain is the
burst adapter. It is not harmful and step 896 buys genuineness (+0.044) on top,
so ship it for that if at all, never for bursts. It remains off by default with
its own checkbox, which the earlier preliminary signal in `ALIGNMENT.md` was
already too weak to justify changing.
