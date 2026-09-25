# VALS_high

`vn/VALS_high` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:VALS`. Target metric `vn:VALS`, baseline 2.894 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/VALS_high
steering_key: vn:VALS
target_metric: vn:VALS
adapter:
  name: sft3_voicenet:VALS_high
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: saturating
balanced:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALS_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.509, t: 5.81, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.001, d_genuineness: +0.224,
             d_blend: +0.117, d_r_burst: -0.059, d_dur_err_abs_s: -0.000}
  beats_random_floor: true
high_effect:
  mode: adapter+cfg
  lora: {name: "sft3_voicenet:VALS_high", w: 1.0}
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.509, t: 5.81, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.001, d_genuineness: +0.224,
             d_blend: +0.117, d_r_burst: -0.059, d_dur_err_abs_s: -0.000}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALS_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.509** (t 5.81, better on 10 of 10 prompts), from 2.894 to 3.403. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.089 | +0.001 |
| genuineness, raw of 6 | 3.845 | 4.068 | +0.224 |
| burst blend, raw of 10 | 5.169 | 5.286 | +0.117 |
| burst realisation | 0.468 | 0.409 | -0.059 |
| |duration error|, s | 0.081 | 0.081 | -0.000 |

## High effect operating point

Mode **`adapter+cfg`**. Adapter `sft3_voicenet:VALS_high` at w = 1.0. Guidance g = 2.0. 

Target moves **+0.509** (t 5.81, better on 10 of 10 prompts), from 2.894 to 3.403. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.089 | +0.001 |
| genuineness, raw of 6 | 3.845 | 4.068 | +0.224 |
| burst blend, raw of 10 | 5.169 | 5.286 | +0.117 |
| burst realisation | 0.468 | 0.409 | -0.059 |
| |duration error|, s | 0.081 | 0.081 | -0.000 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `saturating`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.098) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.871 | -0.051 | -0.58 | 2/10 | -0.012 | -0.028 | -0.029 | yes |
| 0.5 | 1.928 | +0.006 | 0.05 | 4/10 | -0.010 | +0.073 | -0.105 | yes |
| 0.75 | 2.043 | +0.121 | 0.85 | 6/10 | +0.011 | +0.000 | -0.097 | yes |
| 1.0 | 2.067 | +0.145 | 1.22 | 6/10 | +0.062 | +0.063 | -0.077 | yes |
| 1.25 | 2.164 | +0.242 | 1.71 | 7/10 | -0.003 | +0.011 | -0.029 | yes |
| 1.5 | 2.080 | +0.158 | 0.80 | 6/10 | +0.014 | +0.199 | -0.118 | yes |

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
| adapter × steering | -0.540 | -6.42 |
| adapter × guidance | -0.035 | -0.31 |
| steering × guidance | -0.278 | -2.37 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.798 | 3.931 | 2.463 | 2.327 | 4.359 | 4.466 | 2.494 | 2.314 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **stack delivery adapter + delivery steering vector** — on a delivery axis the two levers do the same job and are significantly sub-additive: interaction -0.164 (t -3.75). Pick one.  
  *combination-study 2^3 factorial*
* **stack delivery adapter + guidance** — also sub-additive on delivery: interaction -0.125 (t -3.50)  
  *combination-study 2^3 factorial*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: saturating). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/VALS_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/VALS_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:VALS`.
