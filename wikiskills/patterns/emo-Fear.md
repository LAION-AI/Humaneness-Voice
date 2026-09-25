# Fear

`emo/Fear` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Fear`. Target metric `emo_pct`, baseline 0.251 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Fear
steering_key: emo:Fear
target_metric: emo_pct
adapter:
  name: sft3_emotion:Fear
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: cfg
  lora: null
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.059, t: 1.67, n_prompts: 10, n_up: 6,
             d_wer_parakeet: +0.004, d_genuineness: +0.163,
             d_blend: +0.550, d_r_burst: -0.017, d_dur_err_abs_s: -0.011}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Fear", w: 1.0}
  steer:
    - {key: "emo:Fear", alpha: 0.1, taps: top1}   # h21
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.149, t: 2.12, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.011, d_genuineness: -1.333,
             d_blend: +1.923, d_r_burst: -0.080, d_dur_err_abs_s: +0.004}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`cfg`**. No adapter. Guidance g = 2.0. 

Target moves **+0.059** (t 1.67, better on 6 of 10 prompts), from 0.251 to 0.310. This clears the matched random-direction floor of -0.104.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.081 | 0.085 | +0.004 |
| genuineness, raw of 6 | 3.691 | 3.854 | +0.163 |
| burst blend, raw of 10 | 4.495 | 5.045 | +0.550 |
| burst realisation | 0.453 | 0.436 | -0.017 |
| |duration error|, s | 0.057 | 0.046 | -0.011 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Fear` at w = 1.0. Steering on the cond branch. 

Target moves **+0.149** (t 2.12, better on 7 of 10 prompts), from 0.251 to 0.400. This clears the matched random-direction floor of -0.104.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.081 | 0.092 | +0.011 |
| genuineness, raw of 6 | 3.691 | 2.359 | -1.333 |
| burst blend, raw of 10 | 4.495 | 6.418 | +1.923 |
| burst realisation | 0.453 | 0.373 | -0.080 |
| |duration error|, s | 0.057 | 0.061 | +0.004 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.480 | -0.009 | -0.15 | 4/10 | -0.023 | -0.023 | -0.006 | yes |
| 0.5 | 0.470 | -0.019 | -0.64 | 3/10 | -0.042 | -0.121 | +0.065 | yes |
| 0.75 | 0.496 | +0.008 | 0.12 | 4/10 | -0.035 | -0.169 | +0.112 | yes |
| 1.0 | 0.533 | +0.044 | 1.27 | 4/10 | -0.042 | -0.181 | +0.008 | yes |
| 1.25 | 0.484 | -0.005 | -0.07 | 3/10 | -0.035 | -0.062 | -0.014 | yes |
| 1.5 | 0.540 | +0.051 | 1.15 | 5/10 | -0.030 | -0.156 | +0.017 | yes |

## Interactions

Pooled over the emotion family (target in SD units, n = 399 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | +0.038 | 1.36 | additive — the two combine predictably |
| adapter × guidance | -0.031 | -1.22 | additive — the two combine predictably |
| steering × guidance | +0.277 | 7.51 | **super-additive — and it carries a cost, see below** |

Cumulativity ratio for this family: **1.52** (observed with all three levers, divided by the sum of the three alone).

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
| adapter × steering | +0.133 | 0.51 |
| adapter × guidance | -0.166 | -1.26 |
| steering × guidance | +0.254 | 1.45 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.825 | 1.021 | 0.973 | 1.275 | 0.927 | 0.809 | 1.060 | 1.344 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: below_resolution). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Fear`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Fear`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Fear`.
