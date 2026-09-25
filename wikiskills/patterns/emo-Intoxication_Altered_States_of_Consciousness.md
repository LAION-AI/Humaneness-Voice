# Intoxication Altered States of Consciousness

`emo/Intoxication_Altered_States_of_Consciousness` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Intoxication_Altered_States_of_Consciousness`. Target metric `emo_pct`, baseline 0.252 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Intoxication_Altered_States_of_Consciousness
steering_key: emo:Intoxication_Altered_States_of_Consciousness
target_metric: emo_pct
adapter:
  name: sft3_emotion:Intoxication_Altered_States_of_Consciousness
  usable: true
  safe_w: 0.75
  strong_w: 0.75
  dose_shape: below_resolution
balanced:
  mode: adapter
  lora: {name: "sft3_emotion:Intoxication_Altered_States_of_Consciousness", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.089, t: 1.05, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.003, d_genuineness: +0.162,
             d_blend: +0.065, d_r_burst: +0.038, d_dur_err_abs_s: +0.011}
  beats_random_floor: true
high_effect:
  mode: adapter
  lora: {name: "sft3_emotion:Intoxication_Altered_States_of_Consciousness", w: 1.0}
  steer: []
  cfg: {g: 1.0}
  measured: {d_target: +0.089, t: 1.05, n_prompts: 10, n_up: 8,
             d_wer_parakeet: -0.003, d_genuineness: +0.162,
             d_blend: +0.065, d_r_burst: +0.038, d_dur_err_abs_s: +0.011}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter`**. Adapter `sft3_emotion:Intoxication_Altered_States_of_Consciousness` at w = 1.0. 

Target moves **+0.089** (t 1.05, better on 8 of 10 prompts), from 0.252 to 0.341. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.085 | -0.003 |
| genuineness, raw of 6 | 3.745 | 3.907 | +0.162 |
| burst blend, raw of 10 | 4.633 | 4.698 | +0.065 |
| burst realisation | 0.412 | 0.450 | +0.038 |
| |duration error|, s | 0.059 | 0.070 | +0.011 |

## High effect operating point

Mode **`adapter`**. Adapter `sft3_emotion:Intoxication_Altered_States_of_Consciousness` at w = 1.0. 

Target moves **+0.089** (t 1.05, better on 8 of 10 prompts), from 0.252 to 0.341. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.088 | 0.085 | -0.003 |
| genuineness, raw of 6 | 3.745 | 3.907 | +0.162 |
| burst blend, raw of 10 | 4.633 | 4.698 | +0.065 |
| burst realisation | 0.412 | 0.450 | +0.038 |
| |duration error|, s | 0.059 | 0.070 | +0.011 |

## The adapter on its own

Dose-response shape `below_resolution`. Safe weight **0.75**, strong weight **0.75**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.195 | +0.064 | 2.25 | 7/10 | -0.043 | +0.210 | +0.108 | yes |
| 0.5 | 0.187 | +0.056 | 1.21 | 3/10 | -0.025 | +0.164 | -0.000 | yes |
| 0.75 | 0.227 | +0.096 | 2.75 | 6/10 | -0.011 | +0.356 | +0.101 | yes |
| 1.0 | 0.250 | +0.120 | 1.79 | 6/10 | -0.016 | +0.228 | -0.023 | yes |
| 1.25 | 0.227 | +0.097 | 1.79 | 6/10 | -0.023 | +0.237 | -0.079 | yes |
| 1.5 | 0.235 | +0.105 | 2.09 | 6/10 | -0.034 | +0.218 | +0.000 | yes |

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
| adapter × steering | -0.263 | -2.46 |
| adapter × guidance | -0.018 | -0.14 |
| steering × guidance | +0.511 | 2.25 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.701 | 0.781 | 2.136 | 2.541 | 0.949 | 0.825 | 1.935 | 2.508 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Intoxication_Altered_States_of_Consciousness`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Intoxication_Altered_States_of_Consciousness`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Intoxication_Altered_States_of_Consciousness`.
