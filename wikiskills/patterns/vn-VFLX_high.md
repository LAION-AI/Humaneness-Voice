# VFLX_high

`vn/VFLX_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VFLX`. Target metric `vn:VFLX`, baseline 1.506 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VFLX_high
steering_key: vn:VFLX
target_metric: vn:VFLX
adapter:
  name: sft3_voicenet:VFLX_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter
  lora: {name: "sft3_voicenet:VFLX_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.843, t: 5.82, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.014, d_genuineness: +0.203,
             d_blend: +0.192, d_r_burst: -0.043, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
high_effect:
  mode: adapter
  lora: {name: "sft3_voicenet:VFLX_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.843, t: 5.82, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.014, d_genuineness: +0.203,
             d_blend: +0.192, d_r_burst: -0.043, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_voicenet:VFLX_high` at w = 1.0. 

Target moves **+0.843** (t 5.82, better on 10 of 10 prompts), from 1.506 to 2.349. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.102 | +0.014 |
| genuineness, raw of 6 | 3.845 | 4.047 | +0.203 |
| burst blend, raw of 10 | 5.169 | 5.361 | +0.192 |
| burst realisation | 0.468 | 0.425 | -0.043 |
| |duration error|, s | 0.081 | 0.070 | -0.011 |

## High effect operating point

Mode **`adapter`**. Adapter `sft3_voicenet:VFLX_high` at w = 1.0. 

Target moves **+0.843** (t 5.82, better on 10 of 10 prompts), from 1.506 to 2.349. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.102 | +0.014 |
| genuineness, raw of 6 | 3.845 | 4.047 | +0.203 |
| burst blend, raw of 10 | 5.169 | 5.361 | +0.192 |
| burst realisation | 0.468 | 0.425 | -0.043 |
| |duration error|, s | 0.081 | 0.070 | -0.011 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.421 | -0.002 | -0.02 | 6/10 | +0.024 | +0.008 | -0.017 | yes |
| 0.5 | 1.583 | +0.160 | 2.04 | 8/10 | -0.011 | +0.062 | +0.001 | yes |
| 0.75 | 1.802 | +0.379 | 2.07 | 7/10 | +0.011 | +0.036 | -0.099 | yes |
| 1.0 | 1.791 | +0.368 | 2.28 | 7/10 | +0.025 | +0.037 | -0.140 | yes |
| 1.25 | 2.116 | +0.693 | 4.48 | 9/10 | +0.011 | +0.157 | -0.061 | yes |
| 1.5 | 2.374 | +0.951 | 3.81 | 9/10 | +0.014 | +0.011 | -0.041 | yes |

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
| adapter × steering | -0.368 | -1.75 |
| adapter × guidance | -0.666 | -4.71 |
| steering × guidance | +1.228 | 6.30 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.767 | 1.926 | 2.301 | 3.866 | 2.755 | 2.426 | 3.099 | 3.821 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VFLX_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VFLX_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VFLX`.
