# AROU_high

`vn/AROU_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:AROU`. Target metric `vn:AROU`, baseline 1.903 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/AROU_high
steering_key: vn:AROU
target_metric: vn:AROU
adapter:
  name: sft3_voicenet:AROU_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:AROU_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.562, t: 3.57, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.013, d_genuineness: +0.137,
             d_blend: -0.618, d_r_burst: +0.007, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_voicenet:AROU_high", w: 1.0}
  steer:
    - {key: "vn:AROU", alpha: 0.05, taps: top3}   # h12,h13,h18
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +2.287, t: 10.24, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.198, d_genuineness: -0.399,
             d_blend: -0.699, d_r_burst: -0.192, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:AROU_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.562** (t 3.57, better on 9 of 10 prompts), from 1.903 to 2.465. This clears the matched random-direction floor of -0.267.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.075 | -0.013 |
| genuineness, raw of 6 | 3.845 | 3.982 | +0.137 |
| burst blend, raw of 10 | 5.169 | 4.551 | -0.618 |
| burst realisation | 0.468 | 0.475 | +0.007 |
| |duration error|, s | 0.081 | 0.070 | -0.011 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_voicenet:AROU_high` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+2.287** (t 10.24, better on 10 of 10 prompts), from 1.903 to 4.190. This clears the matched random-direction floor of -0.267.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.286 | +0.198 |
| genuineness, raw of 6 | 3.845 | 3.445 | -0.399 |
| burst blend, raw of 10 | 5.169 | 4.470 | -0.699 |
| burst realisation | 0.468 | 0.276 | -0.192 |
| |duration error|, s | 0.081 | 0.079 | -0.001 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.196 | +0.289 | 2.16 | 6/10 | +0.031 | -0.051 | -0.095 | yes |
| 0.5 | 2.369 | +0.461 | 6.01 | 10/10 | +0.016 | -0.146 | -0.002 | yes |
| 0.75 | 2.566 | +0.659 | 4.34 | 10/10 | +0.041 | -0.105 | +0.007 | yes |
| 1.0 | 2.865 | +0.958 | 4.52 | 10/10 | +0.021 | -0.008 | -0.019 | yes |
| 1.25 | 3.187 | +1.280 | 4.53 | 10/10 | +0.015 | -0.056 | -0.143 | yes |
| 1.5 | 3.203 | +1.296 | 5.26 | 10/10 | +0.075 | -0.109 | -0.059 | yes |

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
| adapter × steering | -0.172 | -0.72 |
| adapter × guidance | -0.308 | -2.42 |
| steering × guidance | +0.233 | 1.01 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.590 | 1.675 | 2.983 | 3.539 | 2.045 | 2.060 | 3.504 | 3.514 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/AROU_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/AROU_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:AROU`.
