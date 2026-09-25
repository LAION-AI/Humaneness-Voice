# S_RANT_high

`vn/S_RANT_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:S_RANT`. Target metric `vn:S_RANT`, baseline 0.579 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/S_RANT_high
steering_key: vn:S_RANT
target_metric: vn:S_RANT
adapter:
  name: sft3_voicenet:S_RANT_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:S_RANT_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +1.183, t: 5.18, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.023, d_genuineness: +0.151,
             d_blend: +0.119, d_r_burst: -0.016, d_dur_err_abs_s: +0.011}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_voicenet:S_RANT_high", w: 1.0}
  steer:
    - {key: "vn:S_RANT", alpha: 0.05, taps: top3}   # h12,h13,h26
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +3.112, t: 6.75, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.161, d_genuineness: -1.399,
             d_blend: -1.485, d_r_burst: -0.220, d_dur_err_abs_s: +0.016}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:S_RANT_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+1.183** (t 5.18, better on 9 of 10 prompts), from 0.579 to 1.763. This clears the matched random-direction floor of +0.294.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.111 | +0.023 |
| genuineness, raw of 6 | 3.845 | 3.996 | +0.151 |
| burst blend, raw of 10 | 5.169 | 5.288 | +0.119 |
| burst realisation | 0.468 | 0.452 | -0.016 |
| |duration error|, s | 0.081 | 0.091 | +0.011 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_voicenet:S_RANT_high` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+3.112** (t 6.75, better on 9 of 10 prompts), from 0.579 to 3.691. This clears the matched random-direction floor of +0.294.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.249 | +0.161 |
| genuineness, raw of 6 | 3.845 | 2.445 | -1.399 |
| burst blend, raw of 10 | 5.169 | 3.685 | -1.485 |
| burst realisation | 0.468 | 0.248 | -0.220 |
| |duration error|, s | 0.081 | 0.097 | +0.016 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.200 | +0.162 | 1.14 | 6/10 | -0.005 | +0.101 | -0.031 | yes |
| 0.5 | 1.470 | +0.432 | 3.07 | 9/10 | -0.017 | +0.011 | +0.026 | yes |
| 0.75 | 1.822 | +0.783 | 4.17 | 9/10 | -0.012 | +0.001 | +0.039 | yes |
| 1.0 | 1.892 | +0.853 | 3.46 | 9/10 | +0.051 | -0.126 | -0.093 | yes |
| 1.25 | 2.014 | +0.976 | 4.16 | 8/10 | +0.000 | -0.015 | -0.026 | yes |
| 1.5 | 2.404 | +1.366 | 7.08 | 10/10 | +0.021 | -0.079 | -0.174 | yes |

## Interactions

Pooled over the delivery axis family (target in SD units, n = 170 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | -0.164 | -3.75 | **sub-additive — pick one** |
| adapter × guidance | -0.125 | -3.50 | **sub-additive — pick one** |
| steering × guidance | +0.144 | 2.02 | **super-additive — and it carries a cost, see below** |

Cumulativity ratio for this family: **0.97** (observed with all three levers, divided by the sum of the three alone).

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
| adapter × steering | -0.375 | -4.59 |
| adapter × guidance | -0.297 | -6.10 |
| steering × guidance | +0.499 | 2.59 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.318 | 0.361 | 1.879 | 2.709 | 0.934 | 0.967 | 2.406 | 2.652 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **stack delivery adapter + delivery steering vector** — on a delivery axis the two levers do the same job and are significantly sub-additive: interaction -0.164 (t -3.75). Pick one.  
  *combination-study 2^3 factorial*
* **stack delivery adapter + guidance** — also sub-additive on delivery: interaction -0.125 (t -3.50)  
  *combination-study 2^3 factorial*
* **compose with qual/esthetics_high** — ESTH cancels S_RANT: +0.464 alone (t 7.01, 12 of 12), -0.012 together.  
  *layer-forensics w3 arm G*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/S_RANT_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/S_RANT_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:S_RANT`.
