# Pleasure Ecstasy

`emo/Pleasure_Ecstasy` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Pleasure_Ecstasy`. Target metric `emo_pct`, baseline 0.199 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Pleasure_Ecstasy
steering_key: emo:Pleasure_Ecstasy
target_metric: emo_pct
adapter:
  name: sft3_emotion:Pleasure_Ecstasy
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: turnover
balanced:
  mode: adapter
  lora: {name: "sft3_emotion:Pleasure_Ecstasy", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.031, t: 0.65, n_prompts: 10, n_up: 4,
             d_wer_parakeet: +0.022, d_genuineness: +0.170,
             d_blend: +0.192, d_r_burst: +0.027, d_dur_err_abs_s: +0.007}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Pleasure_Ecstasy", w: 1.0}
  steer:
    - {key: "emo:Pleasure_Ecstasy", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.076, t: 1.18, n_prompts: 10, n_up: 6,
             d_wer_parakeet: -0.000, d_genuineness: -1.262,
             d_blend: -0.194, d_r_burst: -0.081, d_dur_err_abs_s: +0.023}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_emotion:Pleasure_Ecstasy` at w = 1.0. 

Target moves **+0.031** (t 0.65, better on 4 of 10 prompts), from 0.199 to 0.230. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.082 | 0.104 | +0.022 |
| genuineness, raw of 6 | 3.889 | 4.060 | +0.170 |
| burst blend, raw of 10 | 5.026 | 5.218 | +0.192 |
| burst realisation | 0.398 | 0.425 | +0.027 |
| |duration error|, s | 0.071 | 0.078 | +0.007 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Pleasure_Ecstasy` at w = 1.0. Steering on the cond branch. 

Target moves **+0.076** (t 1.18, better on 6 of 10 prompts), from 0.199 to 0.276. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.082 | 0.082 | -0.000 |
| genuineness, raw of 6 | 3.889 | 2.627 | -1.262 |
| burst blend, raw of 10 | 5.026 | 4.832 | -0.194 |
| burst realisation | 0.398 | 0.318 | -0.081 |
| |duration error|, s | 0.071 | 0.094 | +0.023 |

## The adapter on its own

Dose-response shape `turnover`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.226 | +0.015 | 0.72 | 5/10 | +0.006 | -0.068 | +0.066 | yes |
| 0.5 | 0.228 | +0.017 | 0.46 | 5/10 | -0.019 | -0.002 | +0.057 | yes |
| 0.75 | 0.174 | -0.037 | -0.85 | 4/10 | -0.002 | +0.076 | -0.045 | yes |
| 1.0 | 0.252 | +0.041 | 0.75 | 5/10 | +0.031 | +0.101 | -0.013 | yes |
| 1.25 | 0.324 | +0.114 | 2.82 | 7/10 | +0.004 | -0.068 | -0.077 | yes |
| 1.5 | 0.207 | -0.004 | -0.08 | 4/10 | -0.015 | -0.074 | -0.062 | yes |

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
| adapter × steering | -0.107 | -0.53 |
| adapter × guidance | -0.137 | -0.63 |
| steering × guidance | +0.654 | 2.61 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.859 | 0.781 | 1.184 | 1.738 | 0.993 | 0.756 | 1.189 | 1.628 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Pleasure_Ecstasy`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Pleasure_Ecstasy`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Pleasure_Ecstasy`.
