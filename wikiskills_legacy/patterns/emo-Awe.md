# Awe

`emo/Awe` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Awe`. Target metric `emo_pct`, baseline 0.083 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Awe
steering_key: emo:Awe
target_metric: emo_pct
adapter:
  name: sft3_emotion:Awe
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_emotion:Awe", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.012, t: 0.72, n_prompts: 10, n_up: 2,
             d_wer_parakeet: +0.004, d_genuineness: +0.063,
             d_blend: -0.274, d_r_burst: -0.055, d_dur_err_abs_s: +0.007}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Awe", w: 1.0}
  steer:
    - {key: "emo:Awe", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.080, t: 1.62, n_prompts: 10, n_up: 3,
             d_wer_parakeet: +0.020, d_genuineness: -1.293,
             d_blend: +0.104, d_r_burst: -0.083, d_dur_err_abs_s: +0.037}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_emotion:Awe` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.012** (t 0.72, better on 2 of 10 prompts), from 0.083 to 0.095. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.078 | 0.082 | +0.004 |
| genuineness, raw of 6 | 3.862 | 3.925 | +0.063 |
| burst blend, raw of 10 | 5.014 | 4.740 | -0.274 |
| burst realisation | 0.531 | 0.476 | -0.055 |
| |duration error|, s | 0.047 | 0.054 | +0.007 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Awe` at w = 1.0. Steering on the cond branch. 

Target moves **+0.080** (t 1.62, better on 3 of 10 prompts), from 0.083 to 0.163. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.078 | 0.098 | +0.020 |
| genuineness, raw of 6 | 3.862 | 2.569 | -1.293 |
| burst blend, raw of 10 | 5.014 | 5.118 | +0.104 |
| burst realisation | 0.531 | 0.448 | -0.083 |
| |duration error|, s | 0.047 | 0.085 | +0.037 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.059 | +0.006 | 1.31 | 2/10 | -0.011 | +0.182 | +0.048 | yes |
| 0.5 | 0.058 | +0.005 | 0.19 | 2/10 | -0.049 | +0.108 | -0.013 | yes |
| 0.75 | 0.088 | +0.035 | 1.24 | 4/10 | -0.051 | +0.068 | +0.156 | yes |
| 1.0 | 0.109 | +0.055 | 1.93 | 3/10 | -0.048 | +0.043 | +0.018 | yes |
| 1.25 | 0.061 | +0.008 | 0.49 | 3/10 | -0.044 | -0.022 | -0.063 | yes |
| 1.5 | 0.076 | +0.023 | 0.94 | 2/10 | -0.041 | +0.189 | +0.079 | yes |

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
| adapter × steering | +0.001 | 0.01 |
| adapter × guidance | -0.049 | -0.75 |
| steering × guidance | +0.029 | 0.43 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.311 | 0.299 | 0.503 | 0.596 | 0.340 | 0.356 | 0.609 | 0.578 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Awe`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Awe`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Awe`.
