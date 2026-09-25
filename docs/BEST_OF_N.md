# Best-of-N

Generate the turn N times, score every candidate, keep one. The case for it is
already written into the material this server loads: the burst recipes quote an
`N` per class — *"at a hit rate of 0.27 it takes 8 candidates for 90 %
confidence"* — and the emotion adapter card says to generate several and rank
them because the base model drifts under strong emotion. Without this, that
column is advice nobody can act on.

All N candidates are **one batched forward pass**. The streaming path is batch 1
by necessity, because audio has to start before the line ends; choosing between
candidates is the opposite situation. Measured: **8 candidates in 14.9 s without
guidance, 29.2 s with it** — about one take's worth of wall clock per two or
three candidates, not per candidate.

## The reward

```
R = ( norm(genuineness) + norm(blend) + 2 · norm(clap) ) · (1 − WER)
```

| term | what it is |
|---|---|
| `genuineness` | `laion/voiceclap-commercial-genuineness`, 0–6 |
| `blend` | `laion/voiceclap-commercial-vocalburst-blend`, 0–10 |
| `clap` | cosine between the take's VoiceCLAP **audio** embedding and the **text** embedding of GENERAL plus every round bracket in the script |
| `1 − WER` | the inverse word error rate, straight — no threshold |

`clap` carries double weight because it is the only term that asks whether this
is *the performance that was requested*. The other two ask whether it is a good
take of anything.

Normalisation is **within the candidate set**, not against an absolute scale.
Only the ranking matters, the three scorers have unrelated ranges, and their
absolute values are not calibrated against human judgement anyway.

The intelligibility factor is the **raw inverse word error rate**, multiplied
straight in. It was originally flattened to 1.0 above 0.85, on the argument that
intelligibility is a threshold and not a preference. That was removed: the flat
region covered most of a candidate set — six of eight in one run — so the factor
stopped separating exactly where the candidates were closest, and the ranking
there fell to the three perceptual terms alone.

## Guidance

Independent of N. `1.0` is off; the default when the box is ticked is **3.0**,
the family default for emotion and the value the crossfade study separated
emotions best at. The batched loop runs the same two-branch guided decode as the
streaming path, with the neutralised branch built as its own batch so both halves
pad to the same width.

## What guidance buys inside a candidate set

Four scripts, each held fixed, best-of-8 at three guidance settings, shipped
stack. Averaged over the four sets:

| g | word error | clap | genuineness | blend | per set |
|--:|--:|--:|--:|--:|--:|
| 1.0 (off) | **0.006** | +0.089 | 0.98 | 5.80 | 8 s |
| 3.0 | 0.049 | **+0.108** | 1.23 | **6.07** | 11 s |
| 4.0 | 0.029 | **+0.108** | **1.30** | 5.49 | 11 s |

Guidance moves the two things it is supposed to: agreement with the direction
that was asked for (+0.019 clap, about a fifth of the ungiuded value) and
genuineness (+0.25 to +0.32). It costs word error — from 0.006 to 0.049 at g=3,
still low in absolute terms — and about 40 % more wall clock.

**3.0 and 4.0 are not distinguishable here**, but not for the reason the table
first appeared to give. Printed to three decimals both showed `clap +0.108`,
which looks like a broken switch. It is not: the generated audio differs between
the two settings (verified by hashing it) and reproduces exactly when a setting
is repeated. At six decimals the same measurement reads

| g | word error | clap | genuineness | blend |
|--:|--:|--:|--:|--:|
| 3.0 | 0.01953 | +0.103553 | 1.0807 | 5.4840 |
| 4.0 | 0.02148 | +0.086408 | 1.1882 | 6.0212 |

so it was rounding, and the two settings are simply close.

What separates them is **smaller than the run-to-run variation**: the same g=3.0
measured `clap +0.108, genuineness 1.23` in one run and `+0.1036, 1.08` in the
next. Four scripts at one seed cannot resolve a difference of that size. What
the comparison does establish is that both differ from guidance off, in the
intended direction.

The `reward` column is deliberately not compared across settings: it is
normalised *within* a candidate set, so a mean reward at one guidance value says
nothing about another. Only the raw terms are comparable.

> Found while running this: `/api/say_batch` was dropping the `guidance` field
> entirely. The first version of this table showed g=3 and g=4 producing
> byte-identical numbers, which is impossible if guidance is doing anything —
> all three rows were measuring the unguided path. The endpoint now forwards it,
> along with the neutralised prompt each item needs.

## Memory

Guidance doubles the batch — two branches per candidate — so eight guided
candidates are sixteen sequences, and that does not fit beside the model on a
24 GB card. It used to fail the whole block and fall back to streaming, which
produced **exactly one take and no explanation**: indistinguishable from an
ordinary turn unless you read the server log.

Candidates now go through in chunks (`BON_BATCH` 8, `BON_BATCH_CFG` 4), and an
out-of-memory error halves the chunk and retries rather than giving up — a
slower best-of-N is still best-of-N. If it does fail, the turn now says so in the
page instead of quietly looking normal.

## What this does not do

* **It does not stream.** Nothing plays until every candidate is finished, which
  is the whole point: choosing needs all of them. Time to first audio is the
  full batch time.
* **It has never been listened to.** Every number in the reward is a model
  scoring another model's output. Whether the top-ranked candidate is the one a
  person would pick is unmeasured, and it is the obvious next experiment: the
  ranking is reported per turn precisely so it can be checked against an ear.
* **N is not free of the director.** Every candidate speaks the same script; the
  variation is in the performance, not the words.

## Checking the ranking by ear

**Every candidate is playable, best first.** The winner streams and plays by
itself, as any turn does; the rest appear underneath it as their own players,
each labelled with its rank and its four numbers.

This is deliberately not a debug switch that has to be turned on. The whole
reward is one model scoring another model's output, and the question that
matters — whether rank 0 is the take a person would have chosen — cannot be
answered from the numbers at all. Hearing rank 3 beat rank 0 is the finding; a
ranking nobody can listen to is an assertion.

Cost: the candidates travel as PCM in the `best_of` event, about 14 MB for eight
takes of 13 seconds. `best_of_audio: false` in the request drops them and leaves
the ranking, for a caller that only wants the winner.

## Using it

Checkbox and an N slider (2–12) in the chat page, off by default;
`MOSS_BON=1` and `MOSS_BON_N` for the server default. The full ranking — reward,
gate, word error and all three terms per candidate — arrives as a `best_of`
event and is shown in the chat, so a bad choice can be traced to its numbers
rather than guessed at.
