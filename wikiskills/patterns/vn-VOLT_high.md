# VOLT_high

`vn/VOLT_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VOLT`. Target metric `vn:VOLT`, baseline 1.933 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VOLT_high
steering_key: vn:VOLT
target_metric: vn:VOLT
adapter:
  name: sft3_voicenet:VOLT_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: saturating
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VOLT_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.270, t: 2.89, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.015, d_genuineness: -0.032,
             d_blend: -0.160, d_r_burst: -0.081, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_voicenet:VOLT_high", w: 1.0}
  steer:
    - {key: "vn:VOLT", alpha: 0.1, taps: top3}   # h12,h13,h25
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +1.997, t: 14.19, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.067, d_genuineness: -0.968,
             d_blend: -1.825, d_r_burst: -0.218, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VOLT_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.270** (t 2.89, better on 9 of 10 prompts), from 1.933 to 2.203. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.073 | -0.015 |
| genuineness, raw of 6 | 3.845 | 3.813 | -0.032 |
| burst blend, raw of 10 | 5.169 | 5.010 | -0.160 |
| burst realisation | 0.468 | 0.387 | -0.081 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_voicenet:VOLT_high` at w = 1.0. Steering on the cond branch. 

Target moves **+1.997** (t 14.19, better on 10 of 10 prompts), from 1.933 to 3.930. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.154 | +0.067 |
| genuineness, raw of 6 | 3.845 | 2.876 | -0.968 |
| burst blend, raw of 10 | 5.169 | 3.344 | -1.825 |
| burst realisation | 0.468 | 0.250 | -0.218 |
| |duration error|, s | 0.081 | 0.079 | -0.001 |

## The adapter on its own

Dose-response shape `saturating`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 3.017 | -0.051 | -0.84 | 4/10 | -0.012 | +0.103 | +0.016 | yes |
| 0.5 | 3.207 | +0.138 | 1.55 | 6/10 | -0.013 | +0.018 | -0.112 | yes |
| 0.75 | 3.235 | +0.167 | 1.76 | 6/10 | -0.014 | +0.082 | -0.051 | yes |
| 1.0 | 3.449 | +0.381 | 3.86 | 9/10 | -0.013 | -0.022 | -0.073 | yes |
| 1.25 | 3.259 | +0.191 | 2.47 | 8/10 | +0.015 | +0.087 | -0.105 | yes |
| 1.5 | 3.497 | +0.428 | 2.85 | 8/10 | -0.006 | +0.124 | -0.037 | yes |

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
| adapter × steering | -0.001 | -0.01 |
| adapter × guidance | +0.021 | 0.18 |
| steering × guidance | +0.776 | 5.52 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.404 | 1.403 | 2.680 | 3.524 | 1.512 | 1.600 | 2.854 | 3.651 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VOLT_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VOLT_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VOLT`.
