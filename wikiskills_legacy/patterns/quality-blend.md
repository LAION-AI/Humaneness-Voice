# blend_high

`qual/blend_high` — quality axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `q:blend`. Target metric `blend`, baseline 5.169 over 10 prompts.

## Coefficient block

```yaml
attribute: qual/blend_high
steering_key: q:blend
target_metric: blend
adapter:
  name: sft3_quality:blend_high
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter+steer
  lora: {name: "sft3_quality:blend_high", w: 1.0}
  steer:
    - {key: "q:blend", alpha: 0.1, taps: top1}   # h25
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +1.847, t: 2.73, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.042, d_genuineness: +0.593,
             d_blend: +1.847, d_r_burst: -0.098, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_quality:blend_high", w: 1.0}
  steer:
    - {key: "q:blend", alpha: 0.1, taps: top1}   # h25
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +1.847, t: 2.73, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.042, d_genuineness: +0.593,
             d_blend: +1.847, d_r_burst: -0.098, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+steer`**. Adapter `sft3_quality:blend_high` at w = 1.0. Steering on the cond branch. 

Target moves **+1.847** (t 2.73, better on 9 of 10 prompts), from 5.169 to 7.016. This clears the matched random-direction floor of -0.752.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.130 | +0.042 |
| genuineness, raw of 6 | 3.845 | 4.438 | +0.593 |
| burst blend, raw of 10 | 5.169 | 7.016 | +1.847 |
| burst realisation | 0.468 | 0.370 | -0.098 |
| |duration error|, s | 0.081 | 0.073 | -0.008 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_quality:blend_high` at w = 1.0. Steering on the cond branch. 

Target moves **+1.847** (t 2.73, better on 9 of 10 prompts), from 5.169 to 7.016. This clears the matched random-direction floor of -0.752.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.130 | +0.042 |
| genuineness, raw of 6 | 3.845 | 4.438 | +0.593 |
| burst blend, raw of 10 | 5.169 | 7.016 | +1.847 |
| burst realisation | 0.468 | 0.370 | -0.098 |
| |duration error|, s | 0.081 | 0.073 | -0.008 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.470) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 6.497 | +0.032 | 0.07 | 5/10 | +0.018 | -0.004 | +0.098 | yes |
| 0.5 | 6.885 | +0.420 | 1.02 | 6/10 | -0.007 | +0.063 | +0.017 | yes |
| 0.75 | 7.290 | +0.825 | 2.14 | 8/10 | -0.005 | -0.070 | -0.014 | yes |
| 1.0 | 7.028 | +0.562 | 2.00 | 8/10 | +0.002 | +0.120 | +0.013 | yes |
| 1.25 | 7.048 | +0.583 | 1.63 | 7/10 | -0.013 | +0.065 | -0.038 | yes |
| 1.5 | 7.321 | +0.856 | 1.99 | 8/10 | -0.005 | +0.054 | -0.011 | yes |

## Interactions

Pooled over the quality axis family (target in SD units, n = 30 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | +0.259 | 2.05 | **super-additive — and it carries a cost, see below** |
| adapter × guidance | -0.060 | -0.63 | additive — the two combine predictably |
| steering × guidance | -0.334 | -2.24 | **sub-additive — pick one** |

Cumulativity ratio for this family: **0.42** (observed with all three levers, divided by the sum of the three alone).

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
| adapter × steering | +0.272 | 1.38 |
| adapter × guidance | -0.039 | -0.17 |
| steering × guidance | -0.579 | -1.64 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.370 | 2.213 | 2.282 | 1.607 | 2.637 | 2.502 | 2.882 | 2.108 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **the `steer` lever, for this attribute** — steering does nothing on the quality axes: pooled +0.006 (t 0.04, n 30); and genuineness and blend break at k >= 2 (genuineness -2.81 at k = 5)  
  *combination-study 2^3 factorial; steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: below_resolution). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `qual/blend_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `qual/blend_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `q:blend`.
