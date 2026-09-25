# Triumph

`emo/Triumph` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Triumph`. Target metric `emo_pct`, baseline 0.080 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Triumph
steering_key: emo:Triumph
target_metric: emo_pct
adapter:
  name: sft3_emotion:Triumph
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: steer
  lora: null
  steer:
    - {key: "emo:Triumph", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.289, t: 3.34, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.003, d_genuineness: -0.135,
             d_blend: +0.916, d_r_burst: -0.009, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
high_effect:
  mode: steer+cfg
  lora: null
  steer:
    - {key: "emo:Triumph", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.712, t: 11.26, n_prompts: 10, n_up: 10,
             d_wer_parakeet: +0.168, d_genuineness: -0.403,
             d_blend: +1.579, d_r_burst: -0.088, d_dur_err_abs_s: +0.015}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`steer`**. No adapter. Steering on the cond branch. 

Target moves **+0.289** (t 3.34, better on 9 of 10 prompts), from 0.080 to 0.368. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.093 | 0.096 | +0.003 |
| genuineness, raw of 6 | 3.853 | 3.718 | -0.135 |
| burst blend, raw of 10 | 4.960 | 5.875 | +0.916 |
| burst realisation | 0.448 | 0.438 | -0.009 |
| |duration error|, s | 0.065 | 0.062 | -0.003 |

## High effect operating point

Mode **`steer+cfg`**. No adapter. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.712** (t 11.26, better on 10 of 10 prompts), from 0.080 to 0.792. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.093 | 0.261 | +0.168 |
| genuineness, raw of 6 | 3.853 | 3.450 | -0.403 |
| burst blend, raw of 10 | 4.960 | 6.538 | +1.579 |
| burst realisation | 0.448 | 0.359 | -0.088 |
| |duration error|, s | 0.065 | 0.079 | +0.015 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.020 | -0.014 | -0.54 | 2/10 | -0.036 | +0.016 | -0.012 | yes |
| 0.5 | 0.001 | -0.033 | -1.80 | 0/10 | -0.009 | -0.168 | -0.035 | yes |
| 0.75 | 0.009 | -0.024 | -1.59 | 1/10 | -0.019 | +0.113 | -0.088 | yes |
| 1.0 | 0.029 | -0.005 | -0.20 | 1/10 | -0.017 | +0.063 | -0.087 | yes |
| 1.25 | 0.015 | -0.018 | -0.91 | 2/10 | -0.024 | -0.129 | -0.052 | yes |
| 1.5 | 0.001 | -0.033 | -1.80 | 0/10 | -0.030 | -0.110 | -0.079 | yes |

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
| adapter × steering | -0.185 | -2.41 |
| adapter × guidance | +0.106 | 0.99 |
| steering × guidance | +1.315 | 5.37 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.228 | 0.147 | 1.051 | 2.258 | 0.357 | 0.355 | 0.969 | 2.309 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Triumph`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Triumph`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Triumph`.
