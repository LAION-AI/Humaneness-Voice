# Embarrassment

`emo/Embarrassment` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Embarrassment`. Target metric `emo_pct`, baseline 0.261 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Embarrassment
steering_key: emo:Embarrassment
target_metric: emo_pct
adapter:
  name: sft3_emotion:Embarrassment
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: monotone
balanced:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Embarrassment", w: 1.0}
  steer:
    - {key: "emo:Embarrassment", alpha: 0.1, taps: top1}   # h21
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.172, t: 3.13, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.003, d_genuineness: -0.157,
             d_blend: +0.325, d_r_burst: -0.042, d_dur_err_abs_s: -0.004}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Embarrassment", w: 1.0}
  steer:
    - {key: "emo:Embarrassment", alpha: 0.1, taps: top1}   # h21
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.172, t: 3.13, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.003, d_genuineness: -0.157,
             d_blend: +0.325, d_r_burst: -0.042, d_dur_err_abs_s: -0.004}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Embarrassment` at w = 1.0. Steering on the cond branch. 

Target moves **+0.172** (t 3.13, better on 8 of 10 prompts), from 0.261 to 0.434. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.079 | 0.076 | -0.003 |
| genuineness, raw of 6 | 3.722 | 3.565 | -0.157 |
| burst blend, raw of 10 | 4.939 | 5.263 | +0.325 |
| burst realisation | 0.455 | 0.413 | -0.042 |
| |duration error|, s | 0.062 | 0.058 | -0.004 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Embarrassment` at w = 1.0. Steering on the cond branch. 

Target moves **+0.172** (t 3.13, better on 8 of 10 prompts), from 0.261 to 0.434. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.079 | 0.076 | -0.003 |
| genuineness, raw of 6 | 3.722 | 3.565 | -0.157 |
| burst blend, raw of 10 | 4.939 | 5.263 | +0.325 |
| burst realisation | 0.455 | 0.413 | -0.042 |
| |duration error|, s | 0.062 | 0.058 | -0.004 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `monotone`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.468 | -0.040 | -0.70 | 4/10 | -0.024 | -0.023 | +0.031 | yes |
| 0.5 | 0.505 | -0.003 | -0.06 | 5/10 | -0.019 | +0.082 | +0.008 | yes |
| 0.75 | 0.498 | -0.011 | -0.22 | 5/10 | -0.023 | +0.009 | +0.040 | yes |
| 1.0 | 0.492 | -0.016 | -0.24 | 5/10 | -0.006 | +0.234 | -0.041 | yes |
| 1.25 | 0.470 | -0.039 | -0.70 | 4/10 | -0.029 | +0.129 | -0.008 | yes |
| 1.5 | 0.593 | +0.085 | 1.63 | 8/10 | -0.009 | +0.164 | -0.048 | yes |

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
| adapter × steering | +0.018 | 0.14 |
| adapter × guidance | -0.078 | -0.77 |
| steering × guidance | +0.047 | 0.42 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.790 | 0.797 | 1.003 | 1.012 | 1.124 | 1.009 | 1.310 | 1.286 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: monotone). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Embarrassment`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Embarrassment`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Embarrassment`.
