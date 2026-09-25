# Field notes — what using it actually taught us

Everything else in `docs/` is measurement. This file is the other half: what a
person reported after chatting with the thing for a few days. It is weaker
evidence in every respect except the one that matters — it is about how the
output *sounds*, which no scorer in this project can tell you.

Keep the two apart when reading. A number here means "one listener said so",
not "n = 30, three seeds".

---

## Guidance is usable up to about 4.0

Reported after playing with the CFG sweep in the chat page (`docs/LEVERS.md`
explains the lever; the sweep renders one line at several guidance values so
they can be compared by ear).

> CFG seems fine up to about 4.0. Above that it degenerates.

The measured side had nothing to say about this. The CFG study picked family
defaults of 3.0 for emotion and 2.5 for delivery by word error rate, and it never
ran a listening test at all. So: the ceiling is roughly where the family defaults
already sit, one notch higher, and the interface lets you go to 5.0 — which is
now known to be too far.

**What follows from it.** The strength slider stays at 1–5 rather than being
clamped to 4, because a control that silently refuses is worse than one that
lets you hear why the limit exists. But 4.0 is the honest top of the useful
range and this is the first evidence anyone has about it.

## Steering measures better and sounds worse

Recorded in full in `docs/LEVERS.md`. Short version: with `GEN_MODE=auto` the
resolver chose steering on nearly every turn, the attribute scores went the right
way, and a listener described the result as *more emotional, with strange
artefacts, and the timbre off*. Rolled back to `adapter`, which is the shipped
default. The lever is still there and still one environment variable away.

## The vocal-burst dose — answered, and the answer is "leave it"

**Superseded by measurement.** The dose was swept: ten weights, 2,420 cells under
this exact stack (`MEASURED_2026-08-31.md`). No weight beats the shipped 0.25 /
0.5 on the pre-registered metric, and switching the adapter off is significantly
worse (−0.061, t −2.32). So the carried-over values were a lucky guess rather
than a wrong one, and they stay.

The caveat points the other way and is worth keeping in view: on plain hit-rate
the dose *does* move, +0.061 at 0.5 rising to +0.091 at 1.5 — but with its
maximum at the **top of the swept range**, which is an edge and not an optimum.
A forced-choice listening test at 0.5 / 1.0 / 1.5 would settle it.

**What outranks the dose entirely:** three of six tested burst classes essentially
never realise at any weight — `frustrated_groan` 0 of 1188 prompted cues,
`shriek` 11 of 1045. The model substitutes reliably *down the arousal axis*
(frustrated groan → contented sigh, 127×). No merge weight lifts a floor of zero.
When a shriek fails to appear, this dial is not the fix; the burst+stop
preference adapter might be, and it is now available behind a checkbox.

### The original entry, for the record

## The vocal-burst dose is an open question

The burst adapters merge at **0.25** inline and **0.5** when a burst stands as
its own beat. Those numbers were chosen as conservative guesses when the set was
still the seven v2-era adapters, and they were never swept.

Since then the set has been replaced entirely: 71 adapters from
`laion/moss-va-sft3-vocal-burst-lora-adapters`, trained against SFT3 itself
(`rank 16, alpha 32`, base `…-voice-acting-v2-sft3`). The dose was carried over
unchanged, which is not a decision, it is an oversight.

Two things make this harder than the other dose questions:

* In our own scale sweep the burst family was **the weakest of the four**: 4 of
  19 adapters moved their target measurably. So a sweep may find nothing, and
  "nothing" would be an honest result rather than a failed experiment.
* The obvious target metric — the vocal-burst blend score — is also what one of
  the *quality* adapters optimises, so a sweep has to hold `blend_high` fixed or
  it measures the wrong thing.

**Proposed experiment**, same harness as everything else in
[`EXPERIMENTS.md`](EXPERIMENTS.md): ten lines each *containing a burst*, which the
existing ten do not, swept at 0.25 / 0.5 / 0.75 / 1.0 / 1.5, scored for word
error, blend, and whether the burst is audible at all in the transcript.

---

## The residual improvisation at the end

Rarer since the duration fix (`TIMED_FRAMES_PER_WORD` 4.0, no closing pause —
see `EXPERIMENTS.md` §8), but not gone. Occasionally a take still adds something
after the line is finished.

### Why a time-based cut will not fix it

The tempting fix is to stop the audio at the duration the prompt asked for. That
does not work here, and the reason is the most useful thing in this section:
**the audio already matches the requested duration to within 0.02 s**, in every
condition ever measured, clean or degraded. The model does not overrun. When it
improvises, it speaks the actual line slightly faster and puts the filler
*inside* the same budget. There is no overhang to trim.

So any fix has to be **content-based**: something has to notice that the words
that were asked for have been said.

### Four ways to do that, cheapest first

**1. Offline, after the take: ASR with word timestamps, then trim.** We already
do exactly this in the harness — `eval_tail.ASR` runs `parakeet-tdt-0.6b-v3` and
derives per-token times from the TDT duration head at 0.08 s per encoder frame.
The `extra_w` metric already identifies the words that came after the last one
matching the script. Cutting at that word's end and fading 200 ms is a small
amount of code on top of machinery that exists and is tested. **This is the one
worth doing first**, and it costs one ASR pass per take.

**2. Forced alignment rather than free recognition.** We know exactly which words
were supposed to be spoken, so this is an alignment problem, not a transcription
problem — cheaper, and it cannot hallucinate a word that was never asked for. CTC
segmentation against a small acoustic model would do it. Worth it only if (1)
turns out to be too slow.

**3. Streaming, with a detector rather than an ASR.** The stated idea: run
something lightweight alongside generation that watches for the last three or
four expected words, and fade out over ~200 ms once they have been said with
confidence. The hard part is not the detector, it is the latency budget — audio
is already playing, so a detector that decides 400 ms late has already let the
filler through. A WavLM- or HuBERT-sized model on 80 ms hops is plausible;
nothing here has measured it.

**4. Do nothing, and let the budget do the work.** The rate is the lever that
actually moved this: 5.5 frames per word produced 2.2 invented words per take,
4.0 with no closing pause produced none across thirty takes. If the residue is
rare enough, tightening to 3.5 is one config line and no new machinery. It costs
speech that is slightly faster.

### What was actually built

(4) and (1). The rate was tightened to 4.0 frames per word with no closing pause
(`EXPERIMENTS.md` §8), and the content-based trim is now implemented with forced
alignment for both edges, streaming and offline — see
[`ALIGNMENT.md`](ALIGNMENT.md). Measured over 36 takes it removes 86 % of the
invented words and lowers word error rather than raising it, so it is not eating
real speech.

(3), the streaming detector, turned out not to need a separate detector: the same
aligner runs on the audio generated so far, every 0.5 s, at 50 ms per pass on
CUDA. What it did need was two guards that the offline path does not — a longer
first look for the lead-in, and a refusal to look for the tail before most of the
requested duration exists. Both were bugs before they were guards.

### The original ordering, for the record

(4) then (1). (4) is free and already understood; (1) is a bounded piece of work
on top of tested code and would settle the offline case completely. (3) is the
interesting one and the only one that helps a live conversation, but it is a
research question wearing an engineering hat, and it should not be started until
somebody has measured how often this still happens with the current settings —
which nobody has, because the harness measures the fixed ten lines and the
complaint arrives from free conversation.

---

## What would make these notes better

Every entry above is one person's ear. The cheapest improvement is not a better
model, it is **counting**: how often, out of fifty free-conversation turns, does
a take improvise at the end? Nobody knows. The `extra_w` metric already answers
it for scripted lines and would answer it for live ones with a log line.
