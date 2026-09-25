# Amusement

`emo/Amusement` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Amusement`. Target metric `emo_pct`, baseline 0.254 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Amusement
steering_key: emo:Amusement
target_metric: emo_pct
adapter:
  name: sft3_emotion:Amusement
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Amusement", w: 1.0}
  steer:
    - {key: "emo:Amusement", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.379, t: 5.32, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.025, d_genuineness: +0.596,
             d_blend: -0.155, d_r_burst: -0.098, d_dur_err_abs_s: +0.001}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Amusement", w: 1.0}
  steer:
    - {key: "emo:Amusement", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.5}
  measured: {d_target: +0.630, t: 8.35, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.094, d_genuineness: +0.911,
             d_blend: -0.706, d_r_burst: -0.152, d_dur_err_abs_s: +0.009}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Amusement` at w = 1.0. Steering on the cond branch. 

Target moves **+0.379** (t 5.32, better on 10 of 10 prompts), from 0.254 to 0.633. This clears the matched random-direction floor of +0.070.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.092 | 0.117 | +0.025 |
| genuineness, raw of 6 | 3.800 | 4.396 | +0.596 |
| burst blend, raw of 10 | 4.760 | 4.605 | -0.155 |
| burst realisation | 0.474 | 0.376 | -0.098 |
| |duration error|, s | 0.065 | 0.066 | +0.001 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Amusement` at w = 1.0. Guidance g = 1.5. Steering on the cond branch. 

Target moves **+0.630** (t 8.35, better on 10 of 10 prompts), from 0.254 to 0.884. This clears the matched random-direction floor of +0.070.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.092 | 0.186 | +0.094 |
| genuineness, raw of 6 | 3.800 | 4.710 | +0.911 |
| burst blend, raw of 10 | 4.760 | 4.054 | -0.706 |
| burst realisation | 0.474 | 0.322 | -0.152 |
| |duration error|, s | 0.065 | 0.074 | +0.009 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.330 | -0.006 | -0.08 | 2/10 | +0.052 | +0.128 | -0.114 | yes |
| 0.5 | 0.309 | -0.027 | -0.49 | 3/10 | +0.029 | +0.045 | -0.013 | yes |
| 0.75 | 0.350 | +0.014 | 0.36 | 4/10 | +0.014 | +0.055 | -0.147 | yes |
| 1.0 | 0.368 | +0.033 | 0.50 | 4/10 | +0.013 | +0.048 | -0.065 | yes |
| 1.25 | 0.375 | +0.039 | 0.66 | 4/10 | +0.043 | +0.006 | -0.167 | yes |
| 1.5 | 0.334 | -0.002 | -0.05 | 4/10 | +0.053 | +0.141 | -0.141 | yes |

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
| adapter × steering | -0.078 | -0.50 |
| adapter × guidance | -0.289 | -2.65 |
| steering × guidance | +0.824 | 3.02 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.678 | 1.008 | 1.493 | 2.573 | 1.030 | 0.996 | 1.693 | 2.557 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Amusement`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Amusement`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Amusement`.
