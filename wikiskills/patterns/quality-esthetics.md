# esthetics_high

`qual/esthetics_high` — quality axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:ESTH`. Target metric `vn:ESTH`, baseline 2.606 over 10 prompts.

## Coefficient block

```yaml
attribute: qual/esthetics_high
steering_key: vn:ESTH
target_metric: vn:ESTH
adapter:
  name: sft3_quality:esthetics_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter
  lora: {name: "sft3_quality:esthetics_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.215, t: 1.44, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.005, d_genuineness: -0.495,
             d_blend: -0.229, d_r_burst: -0.039, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_quality:esthetics_high", w: 1.5}
  steer:
    - {key: "vn:ESTH", alpha: 0.08, taps: top1}   # h12
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.323, t: 1.33, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.070, d_genuineness: -1.375,
             d_blend: -0.502, d_r_burst: -0.040, d_dur_err_abs_s: +0.005}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_quality:esthetics_high` at w = 1.0. 

Target moves **+0.215** (t 1.44, better on 5 of 10 prompts), from 2.606 to 2.821. This clears the matched random-direction floor of +0.021.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.083 | -0.005 |
| genuineness, raw of 6 | 3.845 | 3.349 | -0.495 |
| burst blend, raw of 10 | 5.169 | 4.941 | -0.229 |
| burst realisation | 0.468 | 0.429 | -0.039 |
| |duration error|, s | 0.081 | 0.073 | -0.008 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_quality:esthetics_high` at w = 1.5. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.323** (t 1.33, better on 7 of 10 prompts), from 2.606 to 2.929. This clears the matched random-direction floor of +0.021.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.158 | +0.070 |
| genuineness, raw of 6 | 3.845 | 2.469 | -1.375 |
| burst blend, raw of 10 | 5.169 | 4.667 | -0.502 |
| burst realisation | 0.468 | 0.428 | -0.040 |
| |duration error|, s | 0.081 | 0.086 | +0.005 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.861 | +0.131 | 1.46 | 6/10 | +0.006 | -0.097 | -0.044 | yes |
| 0.5 | 2.729 | -0.001 | -0.01 | 5/10 | -0.015 | -0.167 | -0.082 | yes |
| 0.75 | 3.122 | +0.393 | 2.37 | 7/10 | -0.017 | -0.147 | -0.048 | yes |
| 1.0 | 3.040 | +0.311 | 1.73 | 5/10 | -0.007 | -0.314 | -0.015 | yes |
| 1.25 | 3.185 | +0.456 | 2.16 | 7/10 | -0.011 | -0.357 | +0.026 | no |
| 1.5 | 3.301 | +0.571 | 2.42 | 7/10 | -0.020 | -0.126 | -0.109 | yes |

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
| adapter × steering | +0.235 | 0.91 |
| adapter × guidance | -0.079 | -0.50 |
| steering × guidance | -0.118 | -0.86 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 5.829 | 6.077 | 5.860 | 5.689 | 6.311 | 6.178 | 6.276 | 6.326 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **the `steer` lever, for this attribute** — steering does nothing on the quality axes: pooled +0.006 (t 0.04, n 30); and genuineness and blend break at k >= 2 (genuineness -2.81 at k = 5)  
  *combination-study 2^3 factorial; steering study*
* **compose with vn/S_RANT_high** — ESTH and S_RANT cancel: +0.464 alone (t 7.01, 12 of 12), -0.012 together. Asking for both produces neither.  
  *layer-forensics w3 arm G; encoded as config.QUALITY_CONFLICTS in the demo server*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `qual/esthetics_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `qual/esthetics_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:ESTH`.
