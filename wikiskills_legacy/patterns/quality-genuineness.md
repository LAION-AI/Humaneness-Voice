# genuineness_high

`qual/genuineness_high` — quality axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `q:genuineness`. Target metric `genuineness`, baseline 3.845 over 10 prompts.

## Coefficient block

```yaml
attribute: qual/genuineness_high
steering_key: q:genuineness
target_metric: genuineness
adapter:
  name: sft3_quality:genuineness_high
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: monotone
balanced:
  mode: adapter+steer+cfg
  lora: {name: "sft3_quality:genuineness_high", w: 1.0}
  steer:
    - {key: "q:genuineness", alpha: 0.05, taps: top1}   # h12
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.810, t: 6.18, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.016, d_genuineness: +0.810,
             d_blend: -0.147, d_r_burst: -0.097, d_dur_err_abs_s: -0.005}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_quality:genuineness_high", w: 1.5}
  steer:
    - {key: "q:genuineness", alpha: 0.1, taps: top1}   # h12
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.819, t: 3.13, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.157, d_genuineness: +0.819,
             d_blend: -1.228, d_r_burst: -0.143, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
```

## Balanced operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_quality:genuineness_high` at w = 1.0. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.810** (t 6.18, better on 10 of 10 prompts), from 3.845 to 4.654. This clears the matched random-direction floor of -0.108.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.104 | +0.016 |
| genuineness, raw of 6 | 3.845 | 4.654 | +0.810 |
| burst blend, raw of 10 | 5.169 | 5.022 | -0.147 |
| burst realisation | 0.468 | 0.371 | -0.097 |
| |duration error|, s | 0.081 | 0.075 | -0.005 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_quality:genuineness_high` at w = 1.5. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.819** (t 3.13, better on 9 of 10 prompts), from 3.845 to 4.663. This clears the matched random-direction floor of -0.108.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.245 | +0.157 |
| genuineness, raw of 6 | 3.845 | 4.663 | +0.819 |
| burst blend, raw of 10 | 5.169 | 3.942 | -1.228 |
| burst realisation | 0.468 | 0.325 | -0.143 |
| |duration error|, s | 0.081 | 0.078 | -0.003 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `monotone`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.114) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 2.679 | +0.095 | 1.44 | 7/10 | -0.012 | +0.095 | -0.007 | yes |
| 0.5 | 2.711 | +0.126 | 1.06 | 5/10 | -0.006 | +0.126 | +0.001 | yes |
| 0.75 | 2.575 | -0.009 | -0.07 | 5/10 | +0.039 | -0.009 | -0.001 | yes |
| 1.0 | 2.820 | +0.236 | 1.61 | 6/10 | +0.001 | +0.236 | -0.021 | yes |
| 1.25 | 2.864 | +0.280 | 2.12 | 8/10 | +0.025 | +0.280 | -0.167 | yes |
| 1.5 | 2.873 | +0.289 | 1.71 | 8/10 | +0.065 | +0.289 | -0.112 | yes |

## Interactions

Pooled over the quality axis family (target in SD units, n = 30 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | +0.259 | 2.05 | **super-additive — and it carries a cost, see below** |
| adapter × guidance | -0.060 | -0.63 | additive — the two combine predictably |
| steering × guidance | -0.334 | -2.24 | **sub-additive — pick one** |

Cumulativity ratio for this family: **0.42** (observed with all three levers, divided by the sum of the three alone).

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
| adapter × steering | +0.269 | 1.23 |
| adapter × guidance | -0.063 | -0.66 |
| steering × guidance | -0.304 | -1.25 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.519 | 3.524 | 3.595 | 3.182 | 3.969 | 3.797 | 4.199 | 3.837 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **the `steer` lever, for this attribute** — steering does nothing on the quality axes: pooled +0.006 (t 0.04, n 30); and genuineness and blend break at k >= 2 (genuineness -2.81 at k = 5)  
  *combination-study 2^3 factorial; steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: monotone). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `qual/genuineness_high`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `qual/genuineness_high`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `q:genuineness`.
