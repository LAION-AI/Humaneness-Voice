# VALS_low

`vn/VALS_low` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VALS` (this is a low tail of the axis; see *Never*). Target metric `vn:VALS`, baseline -2.894 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VALS_low
steering_key: vn:VALS
target_metric: vn:VALS
adapter:
  name: sft3_voicenet:VALS_low
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: saturating
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALS_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.001, t: 0.01, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.017, d_genuineness: +0.083,
             d_blend: -0.224, d_r_burst: +0.019, d_dur_err_abs_s: -0.009}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALS_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.001, t: 0.01, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.017, d_genuineness: +0.083,
             d_blend: -0.224, d_r_burst: +0.019, d_dur_err_abs_s: -0.009}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALS_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.001** (t 0.01, better on 5 of 10 prompts), from -2.894 to -2.893. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.071 | -0.017 |
| genuineness, raw of 6 | 3.845 | 3.928 | +0.083 |
| burst blend, raw of 10 | 5.169 | 4.945 | -0.224 |
| burst realisation | 0.468 | 0.487 | +0.019 |
| |duration error|, s | 0.081 | 0.071 | -0.009 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALS_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.001** (t 0.01, better on 5 of 10 prompts), from -2.894 to -2.893. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.071 | -0.017 |
| genuineness, raw of 6 | 3.845 | 3.928 | +0.083 |
| burst blend, raw of 10 | 5.169 | 4.945 | -0.224 |
| burst realisation | 0.468 | 0.487 | +0.019 |
| |duration error|, s | 0.081 | 0.071 | -0.009 |

## The adapter on its own

Dose-response shape `saturating`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.967 | +0.045 | 0.38 | 4/10 | +0.001 | -0.009 | -0.081 | yes |
| 0.5 | 1.795 | -0.127 | -1.13 | 4/10 | -0.021 | +0.081 | -0.022 | yes |
| 0.75 | 1.763 | -0.159 | -1.19 | 3/10 | -0.002 | -0.001 | -0.012 | yes |
| 1.0 | 1.725 | -0.197 | -2.92 | 1/10 | -0.000 | -0.033 | -0.120 | yes |
| 1.25 | 1.698 | -0.224 | -2.36 | 3/10 | -0.010 | +0.218 | +0.022 | yes |
| 1.5 | 1.765 | -0.157 | -1.74 | 2/10 | -0.012 | +0.116 | -0.048 | yes |

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
| adapter × steering | +0.031 | 0.19 |
| adapter × guidance | +0.258 | 1.40 |
| steering × guidance | +0.189 | 1.00 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| -4.353 | -4.505 | -2.823 | -2.667 | -4.576 | -4.351 | -2.896 | -2.602 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VALS_low`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VALS_low`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VALS`.
