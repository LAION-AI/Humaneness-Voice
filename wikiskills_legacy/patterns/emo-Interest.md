# Interest

`emo/Interest` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Interest`. Target metric `emo_pct`, baseline 0.583 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Interest
steering_key: emo:Interest
target_metric: emo_pct
adapter:
  name: sft3_emotion:Interest
  usable: true
  safe_w: 1.0
  strong_w: 1.0
  dose_shape: below_resolution
balanced:
  mode: adapter
  lora: {name: "sft3_emotion:Interest", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.051, t: 2.48, n_prompts: 10, n_up: 7,
             d_wer_parakeet: -0.020, d_genuineness: +0.070,
             d_blend: -0.399, d_r_burst: -0.032, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
high_effect:
  mode: adapter
  lora: {name: "sft3_emotion:Interest", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.051, t: 2.48, n_prompts: 10, n_up: 7,
             d_wer_parakeet: -0.020, d_genuineness: +0.070,
             d_blend: -0.399, d_r_burst: -0.032, d_dur_err_abs_s: -0.008}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_emotion:Interest` at w = 1.0. 

Target moves **+0.051** (t 2.48, better on 7 of 10 prompts), from 0.583 to 0.634. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.097 | 0.076 | -0.020 |
| genuineness, raw of 6 | 3.847 | 3.917 | +0.070 |
| burst blend, raw of 10 | 4.963 | 4.564 | -0.399 |
| burst realisation | 0.483 | 0.452 | -0.032 |
| |duration error|, s | 0.070 | 0.062 | -0.008 |

## High effect operating point

Mode **`adapter`**. Adapter `sft3_emotion:Interest` at w = 1.0. 

Target moves **+0.051** (t 2.48, better on 7 of 10 prompts), from 0.583 to 0.634. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.097 | 0.076 | -0.020 |
| genuineness, raw of 6 | 3.847 | 3.917 | +0.070 |
| burst blend, raw of 10 | 4.963 | 4.564 | -0.399 |
| burst realisation | 0.483 | 0.452 | -0.032 |
| |duration error|, s | 0.070 | 0.062 | -0.008 |

## The adapter on its own

Dose-response shape `below_resolution`. Safe weight **1.0**, strong weight **1.0**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.439 | +0.006 | 0.25 | 5/10 | +0.027 | +0.015 | +0.123 | yes |
| 0.5 | 0.453 | +0.021 | 0.69 | 5/10 | -0.028 | +0.107 | +0.033 | yes |
| 0.75 | 0.514 | +0.082 | 2.13 | 9/10 | +0.004 | +0.062 | -0.025 | yes |
| 1.0 | 0.516 | +0.083 | 2.35 | 7/10 | -0.018 | +0.070 | +0.001 | yes |
| 1.25 | 0.506 | +0.073 | 2.34 | 8/10 | -0.009 | +0.157 | +0.071 | yes |
| 1.5 | 0.469 | +0.037 | 0.99 | 7/10 | +0.002 | +0.115 | -0.001 | yes |

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
| adapter × steering | +0.105 | 0.48 |
| adapter × guidance | +0.147 | 0.86 |
| steering × guidance | -1.280 | -4.75 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.470 | 2.497 | 2.429 | 0.954 | 2.686 | 2.637 | 2.528 | 1.422 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Interest`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Interest`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Interest`.
