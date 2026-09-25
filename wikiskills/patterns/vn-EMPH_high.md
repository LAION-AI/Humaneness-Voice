# EMPH_high

`vn/EMPH_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:EMPH`. Target metric `vn:EMPH`, baseline 2.751 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/EMPH_high
steering_key: vn:EMPH
target_metric: vn:EMPH
adapter:
  name: sft3_voicenet:EMPH_high
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: monotone
balanced:
  mode: adapter
  lora: {name: "sft3_voicenet:EMPH_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.431, t: 2.46, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.020, d_genuineness: +0.199,
             d_blend: -0.481, d_r_burst: -0.043, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
high_effect:
  mode: adapter
  lora: {name: "sft3_voicenet:EMPH_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.431, t: 2.46, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.020, d_genuineness: +0.199,
             d_blend: -0.481, d_r_burst: -0.043, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_voicenet:EMPH_high` at w = 1.0. 

Target moves **+0.431** (t 2.46, better on 8 of 10 prompts), from 2.751 to 3.182. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.068 | -0.020 |
| genuineness, raw of 6 | 3.845 | 4.043 | +0.199 |
| burst blend, raw of 10 | 5.169 | 4.689 | -0.481 |
| burst realisation | 0.468 | 0.425 | -0.043 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## High effect operating point

Mode **`adapter`**. Adapter `sft3_voicenet:EMPH_high` at w = 1.0. 

Target moves **+0.431** (t 2.46, better on 8 of 10 prompts), from 2.751 to 3.182. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.068 | -0.020 |
| genuineness, raw of 6 | 3.845 | 4.043 | +0.199 |
| burst blend, raw of 10 | 5.169 | 4.689 | -0.481 |
| burst realisation | 0.468 | 0.425 | -0.043 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.521 | +0.119 | 1.05 | 6/10 | -0.008 | -0.040 | -0.089 | yes |
| 0.5 | 2.545 | +0.143 | 1.53 | 7/10 | +0.013 | +0.078 | -0.023 | yes |
| 0.75 | 2.688 | +0.286 | 2.15 | 8/10 | +0.003 | +0.010 | -0.059 | yes |
| 1.0 | 2.908 | +0.506 | 4.97 | 10/10 | -0.003 | -0.017 | -0.065 | yes |
| 1.25 | 3.071 | +0.669 | 4.21 | 8/10 | +0.023 | +0.135 | -0.028 | yes |
| 1.5 | 3.066 | +0.664 | 7.84 | 10/10 | -0.008 | +0.084 | -0.049 | yes |

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
| adapter × steering | -0.587 | -2.33 |
| adapter × guidance | -0.559 | -5.53 |
| steering × guidance | +0.303 | 1.12 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.823 | 3.837 | 4.793 | 5.646 | 4.422 | 4.414 | 5.343 | 5.099 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/EMPH_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/EMPH_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:EMPH`.
