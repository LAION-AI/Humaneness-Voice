# AROU_low

`vn/AROU_low` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:AROU` (this is a low tail of the axis; see *Never*). Target metric `vn:AROU`, baseline -1.903 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/AROU_low
steering_key: vn:AROU
target_metric: vn:AROU
adapter:
  name: sft3_voicenet:AROU_low
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:AROU_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.289, t: 2.32, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.011, d_genuineness: -0.154,
             d_blend: -0.335, d_r_burst: -0.092, d_dur_err_abs_s: +0.000}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:AROU_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.289, t: 2.32, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.011, d_genuineness: -0.154,
             d_blend: -0.335, d_r_burst: -0.092, d_dur_err_abs_s: +0.000}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:AROU_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.289** (t 2.32, better on 7 of 10 prompts), from -1.903 to -1.614. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.099 | +0.011 |
| genuineness, raw of 6 | 3.845 | 3.691 | -0.154 |
| burst blend, raw of 10 | 5.169 | 4.834 | -0.335 |
| burst realisation | 0.468 | 0.376 | -0.092 |
| |duration error|, s | 0.081 | 0.081 | +0.000 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:AROU_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.289** (t 2.32, better on 7 of 10 prompts), from -1.903 to -1.614. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.099 | +0.011 |
| genuineness, raw of 6 | 3.845 | 3.691 | -0.154 |
| burst blend, raw of 10 | 5.169 | 4.834 | -0.335 |
| burst realisation | 0.468 | 0.376 | -0.092 |
| |duration error|, s | 0.081 | 0.081 | +0.000 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.718 | -0.189 | -1.84 | 3/10 | +0.007 | +0.045 | -0.010 | yes |
| 0.5 | 1.707 | -0.200 | -1.62 | 3/10 | +0.006 | +0.018 | +0.029 | yes |
| 0.75 | 1.499 | -0.409 | -3.06 | 1/10 | +0.015 | +0.030 | +0.018 | yes |
| 1.0 | 1.545 | -0.363 | -2.31 | 2/10 | -0.007 | +0.187 | -0.075 | yes |
| 1.25 | 1.382 | -0.525 | -4.16 | 0/10 | -0.006 | +0.089 | -0.022 | yes |
| 1.5 | 1.241 | -0.666 | -4.68 | 0/10 | -0.019 | -0.052 | -0.052 | yes |

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
| adapter × steering | +0.120 | 0.97 |
| adapter × guidance | -0.003 | -0.02 |
| steering × guidance | -0.609 | -2.73 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| -1.572 | -1.657 | -2.950 | -3.499 | -1.390 | -1.333 | -2.504 | -3.201 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **stack delivery adapter + delivery steering vector** — on a delivery axis the two levers do the same job and are significantly sub-additive: interaction -0.164 (t -3.75). Pick one.  
  *combination-study 2^3 factorial*
* **stack delivery adapter + guidance** — also sub-additive on delivery: interaction -0.125 (t -3.50)  
  *combination-study 2^3 factorial*
* **the `steer` lever, for this attribute** — no measured steering route to this tail. The vector table holds the high-minus-low difference, and the two tails of an attribute are orthogonal rather than opposite (median cos -0.0004), so -alpha along it is not 'the low tail'. No balanced or high-effect recipe for any _low axis uses steering.  
  *layer-forensics; combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/AROU_low`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/AROU_low`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:AROU`.
