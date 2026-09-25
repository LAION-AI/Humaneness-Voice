# VULN_high

`vn/VULN_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VULN`. Target metric `vn:VULN`, baseline 2.469 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VULN_high
steering_key: vn:VULN
target_metric: vn:VULN
adapter:
  name: sft3_voicenet:VULN_high
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: saturating
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VULN_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.171, t: 1.37, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.008, d_genuineness: +0.253,
             d_blend: -0.218, d_r_burst: -0.099, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VULN_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.171, t: 1.37, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.008, d_genuineness: +0.253,
             d_blend: -0.218, d_r_burst: -0.099, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VULN_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.171** (t 1.37, better on 5 of 10 prompts), from 2.469 to 2.640. This clears the matched random-direction floor of +0.035.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.080 | -0.008 |
| genuineness, raw of 6 | 3.845 | 4.097 | +0.253 |
| burst blend, raw of 10 | 5.169 | 4.951 | -0.218 |
| burst realisation | 0.468 | 0.369 | -0.099 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VULN_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.171** (t 1.37, better on 5 of 10 prompts), from 2.469 to 2.640. This clears the matched random-direction floor of +0.035.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.080 | -0.008 |
| genuineness, raw of 6 | 3.845 | 4.097 | +0.253 |
| burst blend, raw of 10 | 5.169 | 4.951 | -0.218 |
| burst realisation | 0.468 | 0.369 | -0.099 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## The adapter on its own

Dose-response shape `saturating`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 3.417 | -0.206 | -3.55 | 1/10 | -0.008 | -0.041 | -0.029 | yes |
| 0.5 | 3.786 | +0.163 | 1.76 | 7/10 | -0.006 | +0.002 | +0.000 | yes |
| 0.75 | 3.771 | +0.148 | 1.25 | 5/10 | -0.010 | +0.141 | +0.015 | yes |
| 1.0 | 3.881 | +0.258 | 2.20 | 7/10 | +0.024 | +0.048 | -0.105 | yes |
| 1.25 | 3.811 | +0.188 | 2.50 | 8/10 | +0.005 | +0.089 | -0.077 | yes |
| 1.5 | 3.911 | +0.288 | 1.89 | 7/10 | -0.019 | +0.226 | -0.068 | yes |

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
| adapter × steering | +0.277 | 2.06 |
| adapter × guidance | -0.055 | -0.69 |
| steering × guidance | +0.107 | 0.68 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.116 | 2.192 | 3.506 | 3.798 | 2.132 | 2.262 | 3.908 | 4.036 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VULN_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VULN_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VULN`.
