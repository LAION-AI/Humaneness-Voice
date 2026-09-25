# Infatuation

`emo/Infatuation` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Infatuation`. Target metric `emo_pct`, baseline 0.030 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Infatuation
steering_key: emo:Infatuation
target_metric: emo_pct
adapter:
  name: sft3_emotion:Infatuation
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: noisy
balanced:
  mode: cfg
  lora: null
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.049, t: 0.81, n_prompts: 10, n_up: 3,
             d_wer_parakeet: +0.008, d_genuineness: +0.335,
             d_blend: +0.474, d_r_burst: -0.012, d_dur_err_abs_s: -0.004}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Infatuation", w: 1.0}
  steer:
    - {key: "emo:Infatuation", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.203, t: 3.31, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.005, d_genuineness: -1.069,
             d_blend: +0.873, d_r_burst: -0.020, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`cfg`**. No adapter. Guidance g = 2.0. 

Target moves **+0.049** (t 0.81, better on 3 of 10 prompts), from 0.030 to 0.079. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.086 | 0.093 | +0.008 |
| genuineness, raw of 6 | 3.661 | 3.996 | +0.335 |
| burst blend, raw of 10 | 4.324 | 4.798 | +0.474 |
| burst realisation | 0.455 | 0.443 | -0.012 |
| |duration error|, s | 0.063 | 0.059 | -0.004 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Infatuation` at w = 1.0. Steering on the cond branch. 

Target moves **+0.203** (t 3.31, better on 7 of 10 prompts), from 0.030 to 0.233. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.086 | 0.091 | +0.005 |
| genuineness, raw of 6 | 3.661 | 2.592 | -1.069 |
| burst blend, raw of 10 | 4.324 | 5.197 | +0.873 |
| burst realisation | 0.455 | 0.435 | -0.020 |
| |duration error|, s | 0.063 | 0.061 | -0.003 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `noisy`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.320 | +0.011 | 0.19 | 3/10 | +0.006 | +0.102 | -0.007 | yes |
| 0.5 | 0.335 | +0.026 | 0.33 | 6/10 | +0.021 | +0.064 | -0.041 | yes |
| 0.75 | 0.378 | +0.070 | 1.07 | 4/10 | +0.006 | +0.145 | +0.021 | yes |
| 1.0 | 0.393 | +0.084 | 1.74 | 4/10 | +0.045 | +0.098 | +0.020 | yes |
| 1.25 | 0.291 | -0.018 | -0.28 | 4/10 | +0.007 | +0.154 | +0.100 | yes |
| 1.5 | 0.328 | +0.019 | 0.26 | 6/10 | +0.006 | +0.254 | +0.100 | yes |

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
| adapter × steering | +0.230 | 3.08 |
| adapter × guidance | -0.028 | -0.15 |
| steering × guidance | +0.690 | 3.11 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.118 | 0.310 | 0.856 | 1.521 | 0.166 | 0.112 | 0.918 | 1.771 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Infatuation`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Infatuation`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Infatuation`.
