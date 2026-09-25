# Thankfulness Gratitude

`emo/Thankfulness_Gratitude` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Thankfulness_Gratitude`. Target metric `emo_pct`, baseline 0.532 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Thankfulness_Gratitude
steering_key: emo:Thankfulness_Gratitude
target_metric: emo_pct
adapter:
  name: sft3_emotion:Thankfulness_Gratitude
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: noisy
balanced:
  mode: steer
  lora: null
  steer:
    - {key: "emo:Thankfulness_Gratitude", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.079, t: 1.81, n_prompts: 10, n_up: 5,
             d_wer_parakeet: +0.023, d_genuineness: -0.443,
             d_blend: +0.353, d_r_burst: -0.069, d_dur_err_abs_s: +0.024}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Thankfulness_Gratitude", w: 1.0}
  steer:
    - {key: "emo:Thankfulness_Gratitude", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.127, t: 3.70, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.032, d_genuineness: -0.896,
             d_blend: -0.378, d_r_burst: -0.144, d_dur_err_abs_s: +0.027}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`steer`**. No adapter. Steering on the cond branch. 

Target moves **+0.079** (t 1.81, better on 5 of 10 prompts), from 0.532 to 0.610. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.081 | 0.104 | +0.023 |
| genuineness, raw of 6 | 3.834 | 3.390 | -0.443 |
| burst blend, raw of 10 | 4.770 | 5.123 | +0.353 |
| burst realisation | 0.470 | 0.400 | -0.069 |
| |duration error|, s | 0.063 | 0.087 | +0.024 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Thankfulness_Gratitude` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.127** (t 3.70, better on 10 of 10 prompts), from 0.532 to 0.658. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.081 | 0.113 | +0.032 |
| genuineness, raw of 6 | 3.834 | 2.937 | -0.896 |
| burst blend, raw of 10 | 4.770 | 4.391 | -0.378 |
| burst realisation | 0.470 | 0.325 | -0.144 |
| |duration error|, s | 0.063 | 0.090 | +0.027 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `noisy`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.532 | +0.037 | 1.18 | 6/10 | +0.012 | -0.089 | -0.005 | yes |
| 0.5 | 0.585 | +0.089 | 1.57 | 7/10 | +0.009 | -0.043 | -0.044 | yes |
| 0.75 | 0.521 | +0.026 | 0.49 | 5/10 | +0.003 | +0.098 | +0.095 | yes |
| 1.0 | 0.552 | +0.057 | 1.51 | 8/10 | +0.009 | +0.104 | +0.047 | yes |
| 1.25 | 0.462 | -0.033 | -0.79 | 3/10 | +0.021 | +0.093 | -0.057 | yes |
| 1.5 | 0.577 | +0.082 | 1.54 | 7/10 | +0.007 | -0.017 | +0.042 | yes |

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
| adapter × steering | +0.016 | 0.17 |
| adapter × guidance | +0.177 | 0.95 |
| steering × guidance | +0.204 | 1.07 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.135 | 1.984 | 2.451 | 2.366 | 2.221 | 2.109 | 2.416 | 2.645 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: noisy). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Thankfulness_Gratitude`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Thankfulness_Gratitude`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Thankfulness_Gratitude`.
