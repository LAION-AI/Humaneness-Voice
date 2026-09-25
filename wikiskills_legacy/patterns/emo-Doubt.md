# Doubt

`emo/Doubt` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Doubt`. Target metric `emo_pct`, baseline 0.570 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Doubt
steering_key: emo:Doubt
target_metric: emo_pct
adapter:
  name: sft3_emotion:Doubt
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter
  lora: {name: "sft3_emotion:Doubt", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.011, t: 0.18, n_prompts: 10, n_up: 3,
             d_wer_parakeet: -0.007, d_genuineness: -0.126,
             d_blend: -0.519, d_r_burst: +0.064, d_dur_err_abs_s: +0.003}
  beats_random_floor: true
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Doubt", w: 1.0}
  steer:
    - {key: "emo:Doubt", alpha: 0.1, taps: top1}   # h21
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.025, t: 0.43, n_prompts: 10, n_up: 5,
             d_wer_parakeet: -0.011, d_genuineness: -1.174,
             d_blend: +0.280, d_r_burst: -0.053, d_dur_err_abs_s: -0.003}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_emotion:Doubt` at w = 1.0. 

Target moves **+0.011** (t 0.18, better on 3 of 10 prompts), from 0.570 to 0.581. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.093 | 0.086 | -0.007 |
| genuineness, raw of 6 | 3.898 | 3.773 | -0.126 |
| burst blend, raw of 10 | 5.289 | 4.770 | -0.519 |
| burst realisation | 0.404 | 0.468 | +0.064 |
| |duration error|, s | 0.078 | 0.081 | +0.003 |

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Doubt` at w = 1.0. Steering on the cond branch. 

Target moves **+0.025** (t 0.43, better on 5 of 10 prompts), from 0.570 to 0.596. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.093 | 0.082 | -0.011 |
| genuineness, raw of 6 | 3.898 | 2.724 | -1.174 |
| burst blend, raw of 10 | 5.289 | 5.570 | +0.280 |
| burst realisation | 0.404 | 0.352 | -0.053 |
| |duration error|, s | 0.078 | 0.075 | -0.003 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.520 | -0.019 | -0.54 | 3/10 | -0.017 | -0.025 | +0.052 | yes |
| 0.5 | 0.537 | -0.002 | -0.04 | 6/10 | -0.014 | -0.018 | +0.000 | yes |
| 0.75 | 0.590 | +0.052 | 1.35 | 6/10 | -0.026 | +0.101 | -0.092 | yes |
| 1.0 | 0.527 | -0.012 | -0.22 | 5/10 | -0.036 | +0.167 | -0.037 | yes |
| 1.25 | 0.555 | +0.017 | 0.27 | 7/10 | -0.001 | +0.206 | +0.021 | yes |
| 1.5 | 0.542 | +0.003 | 0.07 | 4/10 | -0.035 | +0.233 | +0.038 | yes |

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
| adapter × steering | +0.185 | 0.69 |
| adapter × guidance | -0.136 | -0.84 |
| steering × guidance | +0.064 | 0.33 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.477 | 2.469 | 2.401 | 2.411 | 2.525 | 2.334 | 2.587 | 2.507 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Doubt`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Doubt`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Doubt`.
