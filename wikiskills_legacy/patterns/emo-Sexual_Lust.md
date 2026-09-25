# Sexual Lust

`emo/Sexual_Lust` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Sexual_Lust`. Target metric `emo_pct`, baseline 0.082 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Sexual_Lust
steering_key: emo:Sexual_Lust
target_metric: emo_pct
adapter:
  name: sft3_emotion:Sexual_Lust
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_emotion:Sexual_Lust", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.150, t: 2.42, n_prompts: 10, n_up: 6,
             d_wer_parakeet: +0.016, d_genuineness: -0.134,
             d_blend: +0.051, d_r_burst: -0.073, d_dur_err_abs_s: -0.013}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Sexual_Lust", w: 1.0}
  steer:
    - {key: "emo:Sexual_Lust", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.802, t: 21.43, n_prompts: 9, n_up: 9,
             d_wer_parakeet: +0.092, d_genuineness: -1.459,
             d_blend: +0.771, d_r_burst: -0.159, d_dur_err_abs_s: -0.015}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_emotion:Sexual_Lust` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.150** (t 2.42, better on 6 of 10 prompts), from 0.082 to 0.232. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.078 | 0.094 | +0.016 |
| genuineness, raw of 6 | 3.915 | 3.781 | -0.134 |
| burst blend, raw of 10 | 5.186 | 5.237 | +0.051 |
| burst realisation | 0.490 | 0.417 | -0.073 |
| |duration error|, s | 0.077 | 0.063 | -0.013 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Sexual_Lust` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.802** (t 21.43, better on 9 of 9 prompts), from 0.082 to 0.892. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.078 | 0.171 | +0.092 |
| genuineness, raw of 6 | 3.915 | 2.456 | -1.459 |
| burst blend, raw of 10 | 5.186 | 5.957 | +0.771 |
| burst realisation | 0.490 | 0.331 | -0.159 |
| |duration error|, s | 0.077 | 0.062 | -0.015 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.533 | +0.047 | 1.20 | 7/10 | -0.049 | +0.044 | +0.037 | yes |
| 0.5 | 0.522 | +0.035 | 0.85 | 7/10 | -0.042 | -0.159 | -0.010 | yes |
| 0.75 | 0.562 | +0.075 | 1.42 | 7/10 | -0.048 | +0.063 | -0.011 | yes |
| 1.0 | 0.546 | +0.059 | 1.12 | 7/10 | -0.041 | -0.174 | +0.039 | yes |
| 1.25 | 0.651 | +0.164 | 2.35 | 7/10 | -0.049 | -0.151 | -0.078 | yes |
| 1.5 | 0.591 | +0.104 | 1.80 | 7/10 | -0.012 | +0.044 | -0.120 | yes |

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
| adapter × steering | +0.192 | 1.11 |
| adapter × guidance | -0.247 | -2.10 |
| steering × guidance | +0.725 | 2.71 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.256 | 0.435 | 1.145 | 2.353 | 0.498 | 0.734 | 1.883 | 2.541 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Sexual_Lust`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Sexual_Lust`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Sexual_Lust`.
