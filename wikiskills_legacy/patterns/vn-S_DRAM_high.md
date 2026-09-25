# S_DRAM_high

`vn/S_DRAM_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:S_DRAM`. Target metric `vn:S_DRAM`, baseline 1.723 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/S_DRAM_high
steering_key: vn:S_DRAM
target_metric: vn:S_DRAM
adapter:
  name: sft3_voicenet:S_DRAM_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter
  lora: {name: "sft3_voicenet:S_DRAM_high", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.812, t: 6.13, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.004, d_genuineness: -0.000,
             d_blend: -0.320, d_r_burst: -0.024, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_voicenet:S_DRAM_high", w: 1.0}
  steer:
    - {key: "vn:S_DRAM", alpha: 0.1, taps: top3}   # h12,h13,h14
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +2.831, t: 9.91, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.025, d_genuineness: -1.004,
             d_blend: -1.788, d_r_burst: -0.133, d_dur_err_abs_s: +0.007}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_voicenet:S_DRAM_high` at w = 1.0. 

Target moves **+0.812** (t 6.13, better on 9 of 10 prompts), from 1.723 to 2.535. This clears the matched random-direction floor of +0.084.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.084 | -0.004 |
| genuineness, raw of 6 | 3.845 | 3.844 | -0.000 |
| burst blend, raw of 10 | 5.169 | 4.849 | -0.320 |
| burst realisation | 0.468 | 0.444 | -0.024 |
| |duration error|, s | 0.081 | 0.078 | -0.003 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_voicenet:S_DRAM_high` at w = 1.0. Steering on the cond branch. 

Target moves **+2.831** (t 9.91, better on 10 of 10 prompts), from 1.723 to 4.554. This clears the matched random-direction floor of +0.084.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.113 | +0.025 |
| genuineness, raw of 6 | 3.845 | 2.840 | -1.004 |
| burst blend, raw of 10 | 5.169 | 3.381 | -1.788 |
| burst realisation | 0.468 | 0.335 | -0.133 |
| |duration error|, s | 0.081 | 0.087 | +0.007 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.988 | +0.281 | 4.82 | 9/10 | +0.010 | -0.033 | -0.148 | yes |
| 0.5 | 3.178 | +0.471 | 2.58 | 8/10 | -0.007 | -0.043 | +0.048 | yes |
| 0.75 | 3.270 | +0.563 | 6.03 | 10/10 | +0.006 | +0.058 | -0.138 | yes |
| 1.0 | 3.438 | +0.731 | 6.18 | 10/10 | +0.001 | -0.013 | -0.101 | yes |
| 1.25 | 3.900 | +1.193 | 7.81 | 10/10 | -0.001 | -0.053 | -0.071 | yes |
| 1.5 | 3.975 | +1.268 | 8.86 | 10/10 | +0.002 | +0.033 | -0.140 | yes |

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
| adapter × steering | -0.081 | -0.65 |
| adapter × guidance | -0.203 | -3.12 |
| steering × guidance | +0.700 | 3.90 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.110 | 1.138 | 2.354 | 3.219 | 1.633 | 1.596 | 2.933 | 3.459 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/S_DRAM_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/S_DRAM_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:S_DRAM`.
