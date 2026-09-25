# VALN_low

`vn/VALN_low` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VALN` (this is a low tail of the axis; see *Never*). Target metric `vn:VALN`, baseline -2.757 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VALN_low
steering_key: vn:VALN
target_metric: vn:VALN
adapter:
  name: sft3_voicenet:VALN_low
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALN_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.082, t: 0.86, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.009, d_genuineness: -0.029,
             d_blend: +0.303, d_r_burst: -0.056, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALN_low", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.082, t: 0.86, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.009, d_genuineness: -0.029,
             d_blend: +0.303, d_r_burst: -0.056, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALN_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.082** (t 0.86, better on 5 of 10 prompts), from -2.757 to -2.675. This clears the matched random-direction floor of -0.005.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.079 | -0.009 |
| genuineness, raw of 6 | 3.845 | 3.815 | -0.029 |
| burst blend, raw of 10 | 5.169 | 5.472 | +0.303 |
| burst realisation | 0.468 | 0.412 | -0.056 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALN_low` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.082** (t 0.86, better on 5 of 10 prompts), from -2.757 to -2.675. This clears the matched random-direction floor of -0.005.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.079 | -0.009 |
| genuineness, raw of 6 | 3.845 | 3.815 | -0.029 |
| burst blend, raw of 10 | 5.169 | 5.472 | +0.303 |
| burst realisation | 0.468 | 0.412 | -0.056 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.406 | -0.096 | -0.75 | 5/10 | +0.057 | -0.037 | -0.013 | yes |
| 0.5 | 1.491 | -0.011 | -0.10 | 6/10 | -0.014 | -0.038 | -0.005 | yes |
| 0.75 | 1.417 | -0.085 | -0.59 | 6/10 | -0.017 | +0.020 | -0.018 | yes |
| 1.0 | 1.328 | -0.175 | -1.47 | 2/10 | -0.013 | -0.028 | -0.023 | yes |
| 1.25 | 1.251 | -0.251 | -2.29 | 3/10 | -0.018 | -0.017 | -0.012 | yes |
| 1.5 | 1.177 | -0.325 | -1.79 | 4/10 | -0.013 | +0.087 | -0.133 | yes |

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
| adapter × steering | +0.165 | 2.10 |
| adapter × guidance | +0.110 | 2.32 |
| steering × guidance | -0.378 | -4.16 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| -2.183 | -2.165 | -2.681 | -3.095 | -2.192 | -2.118 | -2.579 | -2.829 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VALN_low`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VALN_low`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VALN`.
