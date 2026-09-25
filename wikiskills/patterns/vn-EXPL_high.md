# EXPL_high

`vn/EXPL_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:EXPL`. Target metric `vn:EXPL`, baseline 0.052 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/EXPL_high
steering_key: vn:EXPL
target_metric: vn:EXPL
adapter:
  name: sft3_voicenet:EXPL_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:EXPL_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.125, t: 3.66, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.009, d_genuineness: -0.154,
             d_blend: -0.234, d_r_burst: -0.057, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:EXPL_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.125, t: 3.66, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.009, d_genuineness: -0.154,
             d_blend: -0.234, d_r_burst: -0.057, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:EXPL_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.125** (t 3.66, better on 9 of 10 prompts), from 0.052 to 0.177. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.078 | -0.009 |
| genuineness, raw of 6 | 3.845 | 3.691 | -0.154 |
| burst blend, raw of 10 | 5.169 | 4.935 | -0.234 |
| burst realisation | 0.468 | 0.411 | -0.057 |
| |duration error|, s | 0.081 | 0.073 | -0.008 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:EXPL_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.125** (t 3.66, better on 9 of 10 prompts), from 0.052 to 0.177. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.078 | -0.009 |
| genuineness, raw of 6 | 3.845 | 3.691 | -0.154 |
| burst blend, raw of 10 | 5.169 | 4.935 | -0.234 |
| burst realisation | 0.468 | 0.411 | -0.057 |
| |duration error|, s | 0.081 | 0.073 | -0.008 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.211 | +0.107 | 3.82 | 8/10 | -0.008 | -0.066 | -0.022 | yes |
| 0.5 | 0.179 | +0.075 | 1.77 | 7/10 | -0.002 | -0.065 | -0.057 | yes |
| 0.75 | 0.207 | +0.103 | 2.03 | 7/10 | -0.016 | +0.052 | -0.077 | yes |
| 1.0 | 0.237 | +0.133 | 1.55 | 7/10 | +0.006 | +0.069 | -0.037 | yes |
| 1.25 | 0.243 | +0.139 | 2.15 | 7/10 | +0.001 | +0.100 | -0.112 | yes |
| 1.5 | 0.296 | +0.192 | 2.67 | 6/10 | +0.020 | +0.110 | -0.063 | yes |

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
| adapter × steering | +0.170 | 1.08 |
| adapter × guidance | -0.013 | -0.08 |
| steering × guidance | -1.310 | -7.97 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.241 | 0.301 | -0.078 | -1.295 | 0.749 | 0.829 | 0.632 | -0.631 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/EXPL_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/EXPL_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:EXPL`.
