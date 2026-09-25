# Affection

`emo/Affection` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Affection`. Target metric `emo_pct`, baseline 0.210 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Affection
steering_key: emo:Affection
target_metric: emo_pct
adapter:
  name: sft3_emotion:Affection
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: below_resolution
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_emotion:Affection", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.161, t: 1.92, n_prompts: 10, n_up: 5,
             d_wer_parakeet: +0.042, d_genuineness: +0.121,
             d_blend: -0.062, d_r_burst: +0.054, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_emotion:Affection", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.161, t: 1.92, n_prompts: 10, n_up: 5,
             d_wer_parakeet: +0.042, d_genuineness: +0.121,
             d_blend: -0.062, d_r_burst: +0.054, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_emotion:Affection` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.161** (t 1.92, better on 5 of 10 prompts), from 0.210 to 0.372. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.076 | 0.118 | +0.042 |
| genuineness, raw of 6 | 3.861 | 3.982 | +0.121 |
| burst blend, raw of 10 | 4.953 | 4.891 | -0.062 |
| burst realisation | 0.459 | 0.513 | +0.054 |
| |duration error|, s | 0.079 | 0.078 | -0.001 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_emotion:Affection` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.161** (t 1.92, better on 5 of 10 prompts), from 0.210 to 0.372. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.076 | 0.118 | +0.042 |
| genuineness, raw of 6 | 3.861 | 3.982 | +0.121 |
| burst blend, raw of 10 | 4.953 | 4.891 | -0.062 |
| burst realisation | 0.459 | 0.513 | +0.054 |
| |duration error|, s | 0.079 | 0.078 | -0.001 |

## The adapter on its own

Dose-response shape `below_resolution`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.685 | +0.095 | 0.99 | 8/10 | -0.024 | +0.001 | +0.070 | yes |
| 0.5 | 0.637 | +0.047 | 0.65 | 6/10 | -0.030 | -0.114 | +0.107 | yes |
| 0.75 | 0.640 | +0.049 | 0.96 | 8/10 | +0.005 | -0.059 | +0.109 | yes |
| 1.0 | 0.678 | +0.088 | 1.15 | 7/10 | +0.004 | +0.082 | +0.058 | yes |
| 1.25 | 0.681 | +0.090 | 2.33 | 9/10 | -0.026 | +0.029 | +0.074 | yes |
| 1.5 | 0.629 | +0.039 | 0.59 | 6/10 | +0.001 | +0.224 | +0.058 | yes |

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
| adapter × steering | +0.070 | 0.44 |
| adapter × guidance | -0.055 | -0.20 |
| steering × guidance | +0.174 | 2.10 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.595 | 0.896 | 1.239 | 1.766 | 0.752 | 1.050 | 1.519 | 1.939 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Affection`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Affection`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Affection`.
