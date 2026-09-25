# TENS_high

`vn/TENS_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:TENS`. Target metric `vn:TENS`, baseline 1.544 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/TENS_high
steering_key: vn:TENS
target_metric: vn:TENS
adapter:
  name: sft3_voicenet:TENS_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter
  lora: {name: "sft3_voicenet:TENS_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.312, t: 2.10, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.005, d_genuineness: -0.043,
             d_blend: -0.552, d_r_burst: -0.038, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_voicenet:TENS_high", w: 1.0}
  steer:
    - {key: "vn:TENS", alpha: 0.1, taps: top3}   # h12,h13,h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +2.152, t: 9.20, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.023, d_genuineness: -1.425,
             d_blend: -2.626, d_r_burst: -0.181, d_dur_err_abs_s: +0.007}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_voicenet:TENS_high` at w = 1.0. 

Target moves **+0.312** (t 2.10, better on 8 of 10 prompts), from 1.544 to 1.856. This clears the matched random-direction floor of +0.091.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.083 | -0.005 |
| genuineness, raw of 6 | 3.845 | 3.801 | -0.043 |
| burst blend, raw of 10 | 5.169 | 4.618 | -0.552 |
| burst realisation | 0.468 | 0.430 | -0.038 |
| |duration error|, s | 0.081 | 0.078 | -0.003 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_voicenet:TENS_high` at w = 1.0. Steering on the cond branch. 

Target moves **+2.152** (t 9.20, better on 10 of 10 prompts), from 1.544 to 3.697. This clears the matched random-direction floor of +0.091.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.111 | +0.023 |
| genuineness, raw of 6 | 3.845 | 2.420 | -1.425 |
| burst blend, raw of 10 | 5.169 | 2.543 | -2.626 |
| burst realisation | 0.468 | 0.287 | -0.181 |
| |duration error|, s | 0.081 | 0.087 | +0.007 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.842 | +0.255 | 1.93 | 8/10 | -0.007 | -0.157 | -0.039 | yes |
| 0.5 | 1.893 | +0.306 | 2.05 | 9/10 | -0.011 | +0.121 | +0.031 | yes |
| 0.75 | 2.041 | +0.453 | 4.23 | 9/10 | +0.001 | +0.050 | -0.037 | yes |
| 1.0 | 2.021 | +0.433 | 2.25 | 8/10 | +0.037 | +0.004 | -0.123 | yes |
| 1.25 | 2.247 | +0.660 | 3.81 | 9/10 | +0.050 | -0.014 | -0.088 | yes |
| 1.5 | 2.671 | +1.084 | 6.03 | 10/10 | +0.054 | +0.126 | -0.198 | yes |

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
| adapter × steering | +0.009 | 0.08 |
| adapter × guidance | -0.147 | -0.96 |
| steering × guidance | +0.798 | 7.38 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.193 | 1.204 | 2.521 | 3.415 | 1.434 | 1.384 | 2.856 | 3.518 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **stack delivery adapter + delivery steering vector** — on a delivery axis the two levers do the same job and are significantly sub-additive: interaction -0.164 (t -3.75). Pick one.  
  *combination-study 2^3 factorial*
* **stack delivery adapter + guidance** — also sub-additive on delivery: interaction -0.125 (t -3.50)  
  *combination-study 2^3 factorial*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/TENS_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/TENS_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:TENS`.
