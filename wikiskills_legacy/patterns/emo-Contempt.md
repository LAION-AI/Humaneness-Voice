# Contempt

`emo/Contempt` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Contempt`. Target metric `emo_pct`, baseline 0.111 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Contempt
steering_key: emo:Contempt
target_metric: emo_pct
adapter:
  name: sft3_emotion:Contempt
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Contempt", w: 1.0}
  steer:
    - {key: "emo:Contempt", alpha: 0.1, taps: top1}   # h19
  steer_branch: both
  cfg: {g: 2.0}
  measured: {d_target: +0.143, t: 1.99, n_prompts: 10, n_up: 5,
             d_wer_parakeet: +0.016, d_genuineness: -0.414,
             d_blend: +0.012, d_r_burst: -0.059, d_dur_err_abs_s: +0.001}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Contempt", w: 1.5}
  steer:
    - {key: "emo:Contempt", alpha: 0.1, taps: top1}   # h19
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.306, t: 3.14, n_prompts: 10, n_up: 6,
             d_wer_parakeet: +0.048, d_genuineness: -1.303,
             d_blend: -0.433, d_r_burst: -0.164, d_dur_err_abs_s: +0.040}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Contempt` at w = 1.0. Guidance g = 2.0. Steering on the both branch. 

Target moves **+0.143** (t 1.99, better on 5 of 10 prompts), from 0.111 to 0.254. This clears the matched random-direction floor of +0.026.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.079 | 0.095 | +0.016 |
| genuineness, raw of 6 | 3.947 | 3.534 | -0.414 |
| burst blend, raw of 10 | 4.794 | 4.806 | +0.012 |
| burst realisation | 0.444 | 0.385 | -0.059 |
| |duration error|, s | 0.083 | 0.085 | +0.001 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Contempt` at w = 1.5. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.306** (t 3.14, better on 6 of 10 prompts), from 0.111 to 0.417. This clears the matched random-direction floor of +0.026.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.079 | 0.127 | +0.048 |
| genuineness, raw of 6 | 3.947 | 2.644 | -1.303 |
| burst blend, raw of 10 | 4.794 | 4.361 | -0.433 |
| burst realisation | 0.444 | 0.279 | -0.164 |
| |duration error|, s | 0.083 | 0.123 | +0.040 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.111 | -0.007 | -0.19 | 2/10 | -0.023 | -0.024 | -0.042 | yes |
| 0.5 | 0.139 | +0.021 | 0.58 | 2/10 | -0.011 | -0.077 | +0.094 | yes |
| 0.75 | 0.150 | +0.033 | 1.27 | 4/10 | -0.035 | -0.010 | +0.046 | yes |
| 1.0 | 0.150 | +0.032 | 2.14 | 4/10 | -0.027 | +0.065 | +0.070 | yes |
| 1.25 | 0.164 | +0.047 | 1.45 | 4/10 | -0.038 | +0.137 | +0.081 | yes |
| 1.5 | 0.116 | -0.002 | -0.11 | 2/10 | -0.007 | +0.114 | +0.057 | yes |

## Interactions

Pooled over the emotion family (target in SD units, n = 399 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | +0.038 | 1.36 | additive — the two combine predictably |
| adapter × guidance | -0.031 | -1.22 | additive — the two combine predictably |
| steering × guidance | +0.277 | 7.51 | **super-additive — and it carries a cost, see below** |

Cumulativity ratio for this family: **1.52** (observed with all three levers, divided by the sum of the three alone).

Steering × guidance is the only real synergy in the study, and it is a coupled package. On the emotion family the same interaction term carries:

| carried by steering × guidance | value | t |
|---|--:|--:|
| target (SD) | +0.277 | 7.51 |
| word error | +0.078 | 11.80 |
| genuineness | -0.862 | -21.53 |
| burst realisation | -0.070 | -6.73 |

Every damage term has a larger |t| than the gain. When both are on, steer **both** CFG branches: that keeps 82 % of the effect and returns 0.209 of word error and 0.75 of genuineness.

For this attribute specifically (n = 10 prompts, so read the family row first and this one second):

| pair | interaction | t |
|---|--:|--:|
| adapter × steering | -0.011 | -0.08 |
| adapter × guidance | +0.123 | 1.08 |
| steering × guidance | +0.631 | 3.92 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.378 | 0.381 | 0.815 | 1.310 | 0.459 | 0.447 | 0.746 | 1.503 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: below_resolution). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Contempt`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Contempt`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Contempt`.
