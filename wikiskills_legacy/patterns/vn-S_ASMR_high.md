# S_ASMR_high

`vn/S_ASMR_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:S_ASMR`. Target metric `vn:S_ASMR`, baseline 2.115 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/S_ASMR_high
steering_key: vn:S_ASMR
target_metric: vn:S_ASMR
adapter:
  name: sft3_voicenet:S_ASMR_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:S_ASMR_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.696, t: 2.48, n_prompts: 10, n_up: 8,
             d_wer_parakeet: +0.003, d_genuineness: +0.044,
             d_blend: -0.100, d_r_burst: +0.006, d_dur_err_abs_s: +0.005}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:S_ASMR_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.696, t: 2.48, n_prompts: 10, n_up: 8,
             d_wer_parakeet: +0.003, d_genuineness: +0.044,
             d_blend: -0.100, d_r_burst: +0.006, d_dur_err_abs_s: +0.005}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:S_ASMR_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.696** (t 2.48, better on 8 of 10 prompts), from 2.115 to 2.811. This clears the matched random-direction floor of -0.064.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.091 | +0.003 |
| genuineness, raw of 6 | 3.845 | 3.889 | +0.044 |
| burst blend, raw of 10 | 5.169 | 5.069 | -0.100 |
| burst realisation | 0.468 | 0.474 | +0.006 |
| |duration error|, s | 0.081 | 0.086 | +0.005 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:S_ASMR_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.696** (t 2.48, better on 8 of 10 prompts), from 2.115 to 2.811. This clears the matched random-direction floor of -0.064.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.091 | +0.003 |
| genuineness, raw of 6 | 3.845 | 3.889 | +0.044 |
| burst blend, raw of 10 | 5.169 | 5.069 | -0.100 |
| burst realisation | 0.468 | 0.474 | +0.006 |
| |duration error|, s | 0.081 | 0.086 | +0.005 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.071 | +0.041 | 0.20 | 4/10 | -0.014 | -0.009 | -0.033 | yes |
| 0.5 | 2.508 | +0.479 | 1.98 | 9/10 | -0.011 | +0.019 | -0.031 | yes |
| 0.75 | 2.360 | +0.330 | 1.27 | 8/10 | -0.007 | +0.100 | -0.024 | yes |
| 1.0 | 2.923 | +0.893 | 3.45 | 10/10 | -0.011 | -0.012 | +0.013 | yes |
| 1.25 | 3.009 | +0.980 | 3.10 | 8/10 | -0.018 | +0.001 | +0.053 | yes |
| 1.5 | 3.352 | +1.322 | 4.42 | 9/10 | -0.014 | -0.069 | -0.023 | yes |

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
| adapter × steering | -0.162 | -0.77 |
| adapter × guidance | -0.209 | -1.85 |
| steering × guidance | -0.855 | -7.08 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.466 | 1.550 | 3.150 | 2.547 | 1.907 | 1.949 | 3.596 | 2.616 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/S_ASMR_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/S_ASMR_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:S_ASMR`.
