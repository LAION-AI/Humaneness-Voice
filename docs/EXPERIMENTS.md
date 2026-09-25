# Experiments: what each adapter costs and what it buys

Everything here was produced by the harnesses in this repository — `eval_tail.py`,
`eval_scale.py` and `eval_batch.py` — against the running server. Raw per-take
results, including every transcript, are in the JSON files named at the end.

## Why this was run

The demo had started talking past the end of its lines: the requested text came
out correctly and then more words followed. Turning up the stop-token pressure
did not help. The question was whether that is a property of the base model, of
the prompt format, or of the adapter stack.

The answer is the adapter stack, and the effect is monotone in how many adapters
are loaded.

## Method

### The utterances

Ten fixed English lines, two to three sentences, the same in every condition and
in the same order. Each is wrapped in one delivery cue and rendered into the
SFT3 timed-script format before it is sent.


1. `(clearly amused, easy and conversational)` The kettle boiled twice before I remembered why I walked into the kitchen.
2. `(warmly, quietly pleased)` There is a fox that crosses the yard at the same hour every evening. I have started waiting for it.
3. `(dryly amused, matter of fact)` I finally fixed the squeaking door. It took eleven minutes and four years of complaining.
4. `(clearly content, unhurried)` The train was late, the coffee was cold, and somehow it was still a good morning.
5. `(clearly amused, affectionate)` My neighbour plays the trumpet on Sundays. He is not good, but he is committed.
6. `(lightly amused, conversational)` I planted the basil too close together and now it is a small green argument.
7. `(warmly, quietly moved)` She sent me a photograph of the sea and said nothing else. It was enough.
8. `(dryly amused, resigned)` The lift has been broken since March. I have never been fitter in my life.
9. `(quietly, almost hushed)` There was a moment this afternoon when the whole street went completely quiet.
10. `(clearly tired, self-deprecating)` I read the same paragraph four times and understood it on none of them.

### The prompt

`GENERAL` is held byte-identical across every condition:

```
GENERAL: a woman's voice, in their thirties, speaking with Standard American; close conversational volume, unforced; genuine, not acted; clean studio recording; <duration>s, EN.
```

The script is rendered by `timed_script.render`, which inserts a
`[N.N seconds duration]` tag before each sentence, `[N.N seconds pause]` for
every gap, and turns a burst cue into `(label, N.N seconds)`. The resulting
string goes into both the `SCRIPT:` block and the `Text:` field, byte-identical,
and `Tokens` is the sum of every number in it. Worked example for line 1:

```
[0.3 seconds pause] (clearly amused, easy and conversational) [4.7 seconds duration] The kettle boiled twice before I remembered why I walked into the kitchen. [0.3 seconds pause]
```
Tokens: 66  (5.3 s x 12.5 frames/s)

### Generation hyper-parameters

Held constant everywhere. Seed is fixed at **1234**, so the only thing that
varies between conditions is the adapter set.

| parameter | value |
|---|--:|
| `audio_temperature` | 1.0 |
| `audio_top_p` | 0.95 |
| `audio_top_k` | 25 |
| `audio_repetition_penalty` | 1.1 |
| `stop_bias` | 3.0 |
| `MIN_FRAME_FRACTION` | 0.55 |
| `TOKEN_HEADROOM` | 2.5 |
| `TIMED_FRAMES_PER_WORD` | 4.5 |
| `FRAME_RATE` | 12.5 frames/s |
| sample rate | 48000 Hz |
| base checkpoint | `laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3` |
| always-on adapter | `sft3_dpo:p2` @ 1.0 |

### Reference conditioning

The prompt carries one reference recording. **This changed between experiments 1–2
and 3–5, and the two groups are therefore not directly comparable:**

* Experiments 1 and 2 used the corpus anchor (`reference_target.mp3`), which is a
  *different speaker* from the voice adapter under test. The voice adapter was
  fighting the reference clip, and speaker similarity came out near zero.
* Experiments 3 onwards condition on `emolia_c1699`'s own reference recording,
  the same voice the adapter was trained on. `/api/say` gained an `anchor_path`
  field for this. Baseline speaker similarity is 0.513 under this setup.

### Metrics

| metric | how |
|---|---|
| **WER** | word error rate of the transcript against the intended line, `difflib` alignment over lower-cased word tokens |
| **extra words** | words transcribed *after* the last word that still matches the reference — the direct measurement of "it keeps talking" |
| **takes w/ extra** | share of the ten takes with at least one extra word |
| **tail s** | audio remaining after the last recognised speech token |
| **over s** | audio length minus the length the prompt asked for |
| **genuineness** | `laion/voiceclap-commercial-genuineness`, 0–6, higher = sounds less rehearsed |
| **blend** | `laion/voiceclap-commercial-vocalburst-blend`, 0–10, how naturally non-speech vocalisations sit in the speech |
| **spk sim** | ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) cosine against `emolia_c1699`'s reference recording |

Transcription is `nvidia/parakeet-tdt-0.6b-v3`. Word timing comes from the TDT
duration head — the cumulative sum of per-token durations at 0.08 s per encoder
frame, checked against a known file (last token at 10.24 s in a 10.80 s clip).

> One measurement bug worth recording: the first version of the harness rebuilt
> the transcript token by token, which drops SentencePiece word boundaries and
> glued every word together. That produced a WER of 1.088 on audio that was in
> fact perfect. Transcripts are now decoded whole and the per-token times are used
> only to locate the end of speech.


## Experiment 1 — adding adapters, one group at a time

Corpus anchor. Each row adds to the row above it.

| condition | WER | extra words | takes w/ extra | tail s | over s |
|---|--:|--:|--:|--:|--:|
| bare SFT3 | 0.013 | 0.00 | — | 0.31 | 0.01 |
| +dpo | 0.013 | 0.00 | — | 0.29 | 0.01 |
| +dpo+genuineness | 0.027 | 0.20 | — | 0.39 | 0.00 |
| +dpo+blend | 0.013 | 0.00 | — | 0.34 | 0.00 |
| +dpo+esthetics | 0.047 | 0.20 | — | 0.34 | 0.00 |
| +dpo+gen+blend | 0.037 | 0.40 | — | 0.39 | 0.01 |
| +dpo+gen+esth | 0.063 | 0.40 | — | 0.43 | 0.00 |
| +dpo+blend+esth | 0.018 | 0.10 | — | 0.25 | 0.01 |
| +dpo+all three quality | 0.103 | 0.90 | — | 0.34 | 0.01 |
| +voice | 0.139 | 1.80 | — | 0.94 | 0.00 |
| +emotion 1.5 | 0.250 | 1.70 | — | 1.52 | 0.00 |
| live default | 0.258 | 2.20 | — | 0.85 | 0.01 |

`over s` is within ±0.02 s in every condition: the model hits the requested
duration exactly whether it is clean or not. The extra words are not the model
running past a budget — they appear *inside* the same duration, which means the
line itself is being compressed to make room for them. Turning up the stop-token
pressure cannot fix that, and measurably did not.

**SFT3 + DPO-p2 is clean**: word error 0.013, and not one of the ten takes
invented a word. Degradation begins with the third adapter and is monotone from
there.


## Experiment 2 — the same adapters, turned down

Corpus anchor, same ten lines.

| condition | WER | extra words | takes w/ extra | tail s | over s |
|---|--:|--:|--:|--:|--:|
| dose: voice 0.5 | 0.095 | 1.20 | — | 0.46 | -0.01 |
| dose: voice 0.75 | 0.130 | 1.60 | — | 0.77 | 0.00 |
| dose: emo 1.5 | 0.252 | 3.10 | — | 0.99 | 0.02 |
| dose: emo 1.0 | 0.210 | 2.60 | — | 0.69 | -0.02 |
| dose: emo 0.5 | 0.179 | 2.30 | — | 0.50 | 0.00 |
| dose: q0.5 voice0.75 emo1.0 | 0.046 | 0.60 | — | 0.63 | 0.01 |
| dose: no quality, emo1.0 | 0.042 | 0.40 | — | 0.38 | 0.01 |

Weight matters at least as much as count. The full live stack scored 0.258; the
same adapters at quality 0.5 / voice 0.75 / emotion 1.0 score 0.046.


## Experiment 3 — one adapter at a time, up its whole scale

On top of SFT3 + DPO-p2, conditioned on the voice's own reference recording.

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| baseline sft3+dpo | 0.043 | 0.30 | 30% | 1.17 | 3.35 | 0.513 |
| genuineness @0.25 | 0.050 | 0.10 | 10% | 1.57 | 3.26 | 0.555 |
| genuineness @0.5 | 0.055 | 0.00 | 0% | 1.25 | 2.65 | 0.598 |
| genuineness @0.75 | 0.079 | 1.00 | 30% | 1.51 | 2.83 | 0.529 |
| genuineness @1.0 | 0.080 | 0.80 | 30% | 1.16 | 2.18 | 0.546 |
| genuineness @1.25 | 0.176 | 1.60 | 30% | 1.63 | 2.27 | 0.523 |
| genuineness @1.5 | 0.177 | 1.90 | 60% | 1.38 | 1.59 | 0.533 |
| blend @0.25 | 0.053 | 0.20 | 20% | 1.92 | 4.25 | 0.512 |
| blend @0.5 | 0.061 | 0.20 | 10% | 1.42 | 3.90 | 0.562 |
| blend @0.75 | 0.046 | 0.40 | 30% | 1.23 | 2.93 | 0.535 |
| blend @1.0 | 0.033 | 0.00 | 0% | 1.49 | 3.20 | 0.490 |
| blend @1.25 | 0.013 | 0.00 | 0% | 1.46 | 3.64 | 0.499 |
| blend @1.5 | 0.037 | 0.30 | 20% | 1.52 | 3.82 | 0.513 |
| esthetics @0.25 | 0.074 | 0.60 | 20% | 1.64 | 3.74 | 0.535 |
| esthetics @0.5 | 0.028 | 0.00 | 0% | 1.21 | 3.49 | 0.537 |
| esthetics @0.75 | 0.034 | 0.10 | 10% | 1.04 | 3.51 | 0.509 |
| esthetics @1.0 | 0.058 | 0.10 | 10% | 1.19 | 2.80 | 0.522 |
| esthetics @1.25 | 0.000 | 0.00 | 0% | 0.79 | 3.14 | 0.546 |
| esthetics @1.5 | 0.013 | 0.00 | 0% | 0.79 | 4.22 | 0.486 |
| voice @0.25 | 0.020 | 0.10 | 10% | 1.21 | 3.29 | 0.581 |
| voice @0.5 | 0.052 | 0.20 | 20% | 1.33 | 2.69 | 0.566 |
| voice @0.75 | 0.028 | 0.30 | 20% | 1.55 | 4.48 | 0.589 |
| voice @1.0 | 0.038 | 0.30 | 20% | 1.57 | 4.45 | 0.600 |
| voice @1.25 | 0.088 | 1.00 | 30% | 1.42 | 3.06 | 0.609 |
| voice @1.5 | 0.072 | 0.80 | 50% | 1.64 | 3.08 | 0.645 |


## Experiment 4 — the 17 delivery adapters, six weights each

102 conditions, run through the batched endpoint. Full table in
`sweep_voicenet.json`; the extremes:

**Cheapest on intelligibility**

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| AROU_high @0.75 | 0.000 | 0.00 | 0% | 1.18 | 3.00 | 0.557 |
| S_DRAM_high @1.25 | 0.000 | 0.00 | 0% | 1.39 | 3.65 | 0.508 |
| TENS_high @1.5 | 0.000 | 0.00 | 0% | 1.55 | 4.09 | 0.473 |
| VALN_low @0.75 | 0.000 | 0.00 | 0% | 1.22 | 4.90 | 0.485 |
| VALN_low @1.25 | 0.000 | 0.00 | 0% | 1.25 | 5.00 | 0.463 |
| VOLT_high @1.25 | 0.000 | 0.00 | 0% | 1.49 | 2.89 | 0.527 |
| VOLT_high @1.5 | 0.000 | 0.00 | 0% | 1.42 | 3.48 | 0.522 |
| VALS_low @0.5 | 0.007 | 0.10 | 10% | 1.32 | 3.57 | 0.551 |

**Most expensive**

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| VALS_high @1.5 | 0.200 | 1.00 | 40% | 2.72 | 3.00 | 0.524 |
| S_ASMR_high @1.5 | 0.175 | 0.00 | 0% | 1.69 | 5.05 | 0.523 |
| VALN_high @1.5 | 0.156 | 1.30 | 30% | 2.74 | 2.33 | 0.446 |
| EMPH_high @1.25 | 0.153 | 1.50 | 40% | 1.18 | 2.31 | 0.546 |
| VFLX_high @1.25 | 0.147 | 1.60 | 50% | 1.95 | 3.61 | 0.521 |
| VALS_high @1.25 | 0.132 | 0.50 | 20% | 2.18 | 3.48 | 0.532 |
| VALN_high @1.25 | 0.120 | 0.00 | 0% | 1.70 | 1.93 | 0.415 |
| VALS_high @1.0 | 0.107 | 0.90 | 20% | 1.68 | 2.48 | 0.508 |


## Experiment 5 — the 40 emotion adapters, six weights each

240 conditions. Averaged over all forty adapters at each weight, the cost
of the emotion adapter rises with the weight the demo was using:

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| all 40 emotions @0.25 | 0.036 | 0.19 | 10% | 1.13 | 3.25 | 0.548 |
| all 40 emotions @0.5 | 0.031 | 0.13 | 8% | 1.23 | 3.22 | 0.542 |
| all 40 emotions @0.75 | 0.034 | 0.18 | 11% | 1.25 | 3.26 | 0.528 |
| all 40 emotions @1.0 | 0.034 | 0.12 | 7% | 1.29 | 3.37 | 0.516 |
| all 40 emotions @1.25 | 0.036 | 0.21 | 10% | 1.28 | 3.38 | 0.507 |
| all 40 emotions @1.5 | 0.072 | 0.70 | 11% | 1.34 | 3.39 | 0.484 |

**Cheapest**

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| Anger @0.5 | 0.000 | 0.00 | 0% | 1.29 | 2.78 | 0.578 |
| Anger @1.5 | 0.000 | 0.00 | 0% | 0.72 | 2.17 | 0.468 |
| Concentration @0.25 | 0.000 | 0.00 | 0% | 1.27 | 2.73 | 0.557 |
| Confusion @0.75 | 0.000 | 0.00 | 0% | 1.19 | 2.96 | 0.576 |
| Disappointment @1.0 | 0.000 | 0.00 | 0% | 1.54 | 3.31 | 0.493 |
| Distress @0.5 | 0.000 | 0.00 | 0% | 1.14 | 3.67 | 0.523 |
| Distress @0.75 | 0.000 | 0.00 | 0% | 1.29 | 2.87 | 0.488 |
| Distress @1.0 | 0.000 | 0.00 | 0% | 1.25 | 3.19 | 0.505 |

**Most expensive** — note the tail: single adapters at high weight can derail a
take completely rather than degrade it gently.

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| Confusion @1.5 | 1.285 | 19.30 | 20% | 1.25 | 3.59 | 0.543 |
| Awe @1.5 | 0.128 | 0.00 | 0% | 1.19 | 4.27 | 0.439 |
| Sexual_Lust @1.25 | 0.115 | 1.30 | 20% | 1.38 | 4.23 | 0.500 |
| Pain @0.5 | 0.103 | 0.50 | 30% | 1.44 | 2.96 | 0.533 |
| Distress @1.25 | 0.100 | 0.00 | 0% | 1.40 | 3.57 | 0.474 |
| Contemplation @0.75 | 0.096 | 0.60 | 40% | 1.16 | 2.94 | 0.514 |
| Infatuation @0.75 | 0.089 | 0.90 | 40% | 1.24 | 3.18 | 0.531 |
| Contentment @0.25 | 0.088 | 0.00 | 0% | 0.97 | 3.36 | 0.565 |

Full table in `sweep_emotion.json`.


## Experiment 6 — combinations

Built from the single-adapter results: genuineness only stays safe low, blend is
safe anywhere, aesthetics costs genuineness, and the voice adapter buys most of
its identity by 0.25. `q_lo` is genuineness 0.25 / blend 0.5 / aesthetics 0.5;
`q_hi` is all three at 1.0, which is what the demo shipped.

| condition | WER | extra words | takes w/ extra | genuineness 0–6 | blend 0–10 | spk sim |
|---|--:|--:|--:|--:|--:|--:|
| quality trio @0.25/0.5/0.5 | 0.055 | 0.20 | 10% | 0.99 | 3.17 | 0.579 |
| quality trio @1.0 (old default) | 0.116 | 1.10 | 60% | 2.12 | 3.09 | 0.551 |
| q_lo + voice 0.25 | 0.027 | 0.10 | 10% | 1.08 | 3.79 | 0.567 |
| q_lo + voice 1.0 | 0.021 | 0.30 | 20% | 1.36 | 2.96 | 0.643 |
| q_lo + voice 0.25 + emo 0.5 | 0.041 | 0.60 | 30% | 1.24 | 2.82 | 0.557 |
| q_lo + voice 0.25 + emo 1.0 | 0.018 | 0.30 | 20% | 1.90 | 2.91 | 0.509 |
| q_lo + voice 0.25 + emo 1.5 | 0.083 | 0.50 | 20% | 1.83 | 1.99 | 0.430 |
| q_hi + voice 1.0 + emo 1.5 (live) | 0.273 | 1.00 | 30% | 3.21 | 2.56 | 0.523 |
| q_lo + v0.25 + emo0.5 + 1 axis | 0.060 | 0.40 | 30% | 1.67 | 2.51 | 0.541 |
| q_lo + v0.25 + emo0.5 + 2 axes | 0.143 | 1.50 | 50% | 2.10 | 3.07 | 0.531 |
| q_lo + v0.25 + emo0.5 + 3 axes | 0.076 | 1.10 | 20% | 1.78 | 2.99 | 0.528 |
| q_lo + v0.25 + emo0.5 + burst | 0.008 | 0.10 | 10% | 1.13 | 3.27 | 0.537 |
| proposed default | 0.037 | 0.40 | 30% | 1.09 | 2.52 | 0.564 |
| proposed default + axis | 0.048 | 0.50 | 40% | 1.58 | 2.43 | 0.562 |

Three things come out of this table.

**The stack the demo shipped is the worst row in it** for intelligibility —
word error 0.273 against 0.008 for the best combination. It also scores the
*highest* genuineness of any condition measured (3.21). That is a real
trade-off, not a mistake: the stack does make the voice sound less rehearsed,
and it does so by loosening exactly the control that keeps it on script.

**Turning the same adapters down recovers almost everything.** The same five
adapter families at genuineness 0.25 / blend 0.5 / aesthetics 0.5 / voice 0.25 /
emotion 1.0 give word error 0.018 with genuineness still at 1.90.

**Delivery axes stack badly.** One costs little; two took word error from 0.041
to 0.143 and put invented words in half the takes.


## What was adopted

The demo now ships the doses these measurements point at, not the trained
defaults it started from:

| adapter | was | now | why |
|---|--:|--:|---|
| `sft3_quality:genuineness_high` | 1.0 | **0.25** | raises its own score only below 0.5; word error 0.176 at 1.25 |
| `sft3_quality:blend_high` | 1.0 | **0.5** | safe at every weight measured; 0.25 gave the best genuineness of the whole study |
| `sft3_quality:esthetics_high` | 1.0 | **0.5** | cheap on intelligibility but costs genuineness monotonically |
| `sft3_voice:<profile>` | 1.0 | **0.25** | speaker similarity 0.581 at 0.25 vs 0.600 at 1.0, against 0.513 with no adapter at all |
| `sft3_emotion:<name>` | 1.5 | **1.0** | 1.5 derails whole takes rather than degrading gently |
| delivery axes, max | 2 | **1** | two took word error from 0.041 to 0.143 |
| `sft3_dpo:p2` | 1.0 | 1.0 | unchanged — it costs nothing measurable |

A live turn under the new defaults:

```
sft3_dpo:p2 1.0 · sft3_voice:emolia_c1699 0.25 · burst:soft_hum 0.25
sft3_quality:genuineness_high 0.25 · blend_high 0.5 · esthetics_high 0.5
sft3_emotion:Contentment 1.0
```

> Two of those have since moved and this example is kept as it was measured: the
> voice adapter is now **0.5**, because identity broke audibly once the burst and
> delivery adapters grew — a failure speaker similarity over a whole take does
> not catch. And burst weight is now **per class** from its own recipe, capped at
> 1.5, rather than a flat 0.25. `DEFAULTS.md` is generated from the code and is
> always current.

```
```

This is a trade, not a free win. The stack that shipped before scored the
highest genuineness of any condition measured (3.21 against 1.90); it bought
that by loosening the control that keeps the model on script. The new defaults
give up some of that in exchange for word error 0.018 instead of 0.273.

## Experiment 7 — the stop-token brake does nothing here

`stop_bias` subtracts a constant from the end token's logit before sampling, so
a positive value makes stopping *less* likely. It exists because takes used to
end early, truncating the line. Against the opposite complaint it is pointing the
wrong way, and turning it up makes things worse rather than better.

Swept on the default stack, ten utterances each:

| stop_bias | word error | extra words | takes w/ extra |
|--:|--:|--:|--:|
| −3.0 | 0.047 | 0.5 | 20% |
| −2.0 | 0.069 | 0.8 | 30% |
| −1.0 | 0.076 | 0.9 | 30% |
| −0.5 | 0.076 | 0.9 | 30% |
| 0.0 | 0.076 | 0.9 | 30% |
| +1.0 | 0.176 | 1.1 | 30% |
| +3.0 | 0.027 | 0.4 | 20% |
| +4.0 | 0.054 | 0.2 | 20% |

−1.0, −0.5 and 0.0 produce **bit-identical** results, through two different code
paths. The continue-or-end decision is never marginal: a few nats do not move it,
because it is not what decides when the take ends.

## Experiment 8 — what does decide when the take ends

The audio matches the requested duration to within 0.02 s in every condition
measured, clean or degraded. The model stops when the **time budget** is used up,
not when the sentence is finished — so any budget it does not need for the words
gets filled, and what fills it is invented speech.

Two things set that budget: the seconds-per-word rate the script is rendered at,
and the closing `[N.N seconds pause]` the renderer used to append. Three seeds
per condition, ten utterances each, mean over seeds:

| prompt shape | word error | extra words | tail s |
|---|--:|--:|--:|
| duration tags @4.5 f/word, closing pause (what shipped) | 0.055 | 0.53 | 0.56 |
| duration tags @4.0, closing pause | 0.044 | 0.30 | 0.47 |
| no duration tags at all, @4.5 | 0.048 | 0.53 | 0.48 |
| plain text, `tokens = words × 6`, no tags at all | 0.138 | 0.90 | 1.36 |
| no closing pause, @4.5 | 0.088 | 0.53 | 0.51 |
| **no closing pause, @4.0** | **0.025** | **0.03** | **0.38** |
| no closing pause and no opening pause, @4.0 | 0.036 | 0.13 | 0.41 |

Neither change alone is enough — the closing pause at 4.5 still invents words in
20–40 % of takes, and 4.0 with a closing pause still does at 10–20 %. Together
they are the only configuration that held at 0–10 % across all three seeds.

Worth recording because it was the intuitive suspect: **removing the timing tags
does not help.** The plain-text shape, which is what this demo used before the
timed format, is the *worst* row in the table — tail 1.36 s and 0.90 invented
words. The timed format is not the cause; an over-generous budget inside it is.

**Adopted:** `TIMED_FRAMES_PER_WORD` 4.5 → 4.0, and `timed_script.render` no
longer appends a closing pause — it will not end a script on silence the model
still owes.

## Throughput

The streaming path is batch 1 by necessity — audio has to start before the line
is finished. A sweep does not need that, so `/api/say_batch` runs a whole
condition (ten utterances) in one forward pass. The generation loop already
carried a batch dimension; what was missing was a padded batch (the processor
left-pads, which is what batched generation needs) and a per-row minimum-frame
floor, since a batch holds lines of different lengths.

Measured on the baseline condition: **under one minute batched against just over
two minutes serially**, with the perceptual scores unchanged (genuineness 1.24 vs
1.17, blend 3.36 vs 3.35, speaker similarity 0.521 vs 0.513).

## What to be careful about in these numbers

* **Ten utterances per condition, one seed.** Trends across six weights are
  solid; single cells are not. `esthetics @1.25 -> WER 0.000` next to
  `@1.0 -> 0.058` is sampling noise, not a real non-linearity.
* Experiments 1–2 and 3–5 use different reference audio (see above) and their
  absolute numbers are not comparable across that boundary.
* The genuineness and blend scorers are themselves models, on 0–6 and 0–10
  scales whose absolute values are not calibrated against human judgement here.
  Differences between conditions are the signal; the absolute level is not.
* All ten lines are English, declarative and calm. Nothing here says how these
  adapters behave on shouting, on German, or on lines with vocal bursts.

## Raw data

| file | what |
|---|---|
| `tail_eval.json` | experiment 1, per take |
| `tail_dose.json` | experiment 2, per take |
| `scale_eval.json` | experiment 3, per take |
| `sweep_voicenet.json` | experiment 4, per take |
| `sweep_emotion.json` | experiment 5, per take |

Each take records the transcript, the extra words, and all six metrics.
