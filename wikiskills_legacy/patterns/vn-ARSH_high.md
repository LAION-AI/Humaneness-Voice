# ARSH_high

`vn/ARSH_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:ARSH`. Target metric `vn:ARSH`, baseline 2.187 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/ARSH_high
steering_key: vn:ARSH
target_metric: vn:ARSH
adapter:
  name: sft3_voicenet:ARSH_high
  usable: true
  safe_w: 1.5
  strong_w: 1.5
  dose_shape: monotone
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:ARSH_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.513, t: 4.68, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.011, d_genuineness: +0.229,
             d_blend: -0.203, d_r_burst: +0.047, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:ARSH_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.513, t: 4.68, n_prompts: 10, n_up: 9,
             d_wer_parakeet: -0.011, d_genuineness: +0.229,
             d_blend: -0.203, d_r_burst: +0.047, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:ARSH_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.513** (t 4.68, better on 9 of 10 prompts), from 2.187 to 2.700. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.077 | -0.011 |
| genuineness, raw of 6 | 3.845 | 4.073 | +0.229 |
| burst blend, raw of 10 | 5.169 | 4.966 | -0.203 |
| burst realisation | 0.468 | 0.515 | +0.047 |
| |duration error|, s | 0.081 | 0.070 | -0.011 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:ARSH_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.513** (t 4.68, better on 9 of 10 prompts), from 2.187 to 2.700. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.077 | -0.011 |
| genuineness, raw of 6 | 3.845 | 4.073 | +0.229 |
| burst blend, raw of 10 | 5.169 | 4.966 | -0.203 |
| burst realisation | 0.468 | 0.515 | +0.047 |
| |duration error|, s | 0.081 | 0.070 | -0.011 |

## The adapter on its own

Dose-response shape `monotone`. Safe weight **1.5**, strong weight **1.5**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.817 | +0.034 | 0.44 | 6/10 | -0.005 | +0.077 | +0.005 | yes |
| 0.5 | 1.975 | +0.193 | 1.71 | 6/10 | -0.013 | +0.005 | +0.040 | yes |
| 0.75 | 2.174 | +0.392 | 2.75 | 8/10 | +0.026 | +0.049 | +0.004 | yes |
| 1.0 | 2.444 | +0.661 | 4.32 | 9/10 | -0.003 | -0.081 | -0.068 | yes |
| 1.25 | 2.536 | +0.753 | 4.66 | 9/10 | +0.026 | -0.023 | -0.028 | yes |
| 1.5 | 2.820 | +1.038 | 6.22 | 10/10 | -0.002 | -0.137 | -0.119 | yes |

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
| adapter × steering | -0.736 | -4.13 |
| adapter × guidance | -0.012 | -0.12 |
| steering × guidance | +1.191 | 6.35 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.873 | 4.056 | 4.882 | 6.191 | 4.677 | 4.782 | 4.885 | 6.247 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/ARSH_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/ARSH_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:ARSH`.
