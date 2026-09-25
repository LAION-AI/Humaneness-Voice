# Hope Enthusiasm Optimism

`emo/Hope_Enthusiasm_Optimism` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Hope_Enthusiasm_Optimism`. Target metric `emo_pct`, baseline 0.392 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Hope_Enthusiasm_Optimism
steering_key: emo:Hope_Enthusiasm_Optimism
target_metric: emo_pct
adapter:
  name: sft3_emotion:Hope_Enthusiasm_Optimism
  usable: true
  safe_w: 0.75
  strong_w: 0.75
  dose_shape: below_resolution
balanced:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Hope_Enthusiasm_Optimism", w: 1.0}
  steer:
    - {key: "emo:Hope_Enthusiasm_Optimism", alpha: 0.1, taps: top1}   # h19
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.220, t: 5.09, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.005, d_genuineness: +0.093,
             d_blend: +0.416, d_r_burst: +0.061, d_dur_err_abs_s: +0.007}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Hope_Enthusiasm_Optimism", w: 1.0}
  steer:
    - {key: "emo:Hope_Enthusiasm_Optimism", alpha: 0.1, taps: top1}   # h19
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.343, t: 5.68, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.055, d_genuineness: -0.273,
             d_blend: -0.155, d_r_burst: -0.075, d_dur_err_abs_s: +0.012}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Hope_Enthusiasm_Optimism` at w = 1.0. Steering on the cond branch. 

Target moves **+0.220** (t 5.09, better on 9 of 10 prompts), from 0.392 to 0.612. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.085 | 0.080 | -0.005 |
| genuineness, raw of 6 | 3.799 | 3.892 | +0.093 |
| burst blend, raw of 10 | 5.012 | 5.427 | +0.416 |
| burst realisation | 0.383 | 0.444 | +0.061 |
| |duration error|, s | 0.050 | 0.057 | +0.007 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Hope_Enthusiasm_Optimism` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.343** (t 5.68, better on 10 of 10 prompts), from 0.392 to 0.735. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.085 | 0.140 | +0.055 |
| genuineness, raw of 6 | 3.799 | 3.526 | -0.273 |
| burst blend, raw of 10 | 5.012 | 4.857 | -0.155 |
| burst realisation | 0.383 | 0.309 | -0.075 |
| |duration error|, s | 0.050 | 0.062 | +0.012 |

## The adapter on its own

Dose-response shape `below_resolution`. Safe weight **0.75**, strong weight **0.75**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.277 | +0.070 | 2.62 | 7/10 | -0.001 | +0.042 | +0.030 | yes |
| 0.5 | 0.243 | +0.036 | 0.98 | 4/10 | -0.004 | +0.014 | -0.018 | yes |
| 0.75 | 0.297 | +0.089 | 3.04 | 7/10 | -0.006 | +0.021 | +0.056 | yes |
| 1.0 | 0.261 | +0.054 | 2.09 | 6/10 | +0.008 | +0.031 | +0.005 | yes |
| 1.25 | 0.274 | +0.067 | 2.05 | 7/10 | +0.079 | +0.155 | -0.077 | yes |
| 1.5 | 0.279 | +0.071 | 1.40 | 6/10 | -0.006 | +0.078 | -0.077 | yes |

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
| adapter × steering | +0.150 | 1.05 |
| adapter × guidance | -0.233 | -1.25 |
| steering × guidance | +0.382 | 1.91 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.276 | 1.526 | 1.652 | 2.264 | 1.486 | 1.483 | 1.992 | 2.391 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Hope_Enthusiasm_Optimism`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Hope_Enthusiasm_Optimism`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Hope_Enthusiasm_Optimism`.
