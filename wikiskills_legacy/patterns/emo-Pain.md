# Pain

`emo/Pain` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Pain`. Target metric `emo_pct`, baseline 0.008 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Pain
steering_key: emo:Pain
target_metric: emo_pct
adapter:
  name: sft3_emotion:Pain
  usable: true
  safe_w: 0.75
  strong_w: 0.75
  dose_shape: saturating
balanced:
  mode: cfg
  lora: null
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.040, t: 1.31, n_prompts: 10, n_up: 3,
             d_wer_parakeet: -0.001, d_genuineness: +0.033,
             d_blend: +0.305, d_r_burst: -0.059, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
high_effect:
  mode: cfg
  lora: null
  steer: []
  cfg: {g: 2.0}
  measured: {d_target: +0.040, t: 1.31, n_prompts: 10, n_up: 3,
             d_wer_parakeet: -0.001, d_genuineness: +0.033,
             d_blend: +0.305, d_r_burst: -0.059, d_dur_err_abs_s: -0.001}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`cfg`**. No adapter. Guidance g = 2.0. 

Target moves **+0.040** (t 1.31, better on 3 of 10 prompts), from 0.008 to 0.047. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.080 | 0.079 | -0.001 |
| genuineness, raw of 6 | 3.903 | 3.936 | +0.033 |
| burst blend, raw of 10 | 5.055 | 5.360 | +0.305 |
| burst realisation | 0.499 | 0.441 | -0.059 |
| |duration error|, s | 0.070 | 0.069 | -0.001 |

## High effect operating point

Mode **`cfg`**. No adapter. Guidance g = 2.0. 

Target moves **+0.040** (t 1.31, better on 3 of 10 prompts), from 0.008 to 0.047. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.080 | 0.079 | -0.001 |
| genuineness, raw of 6 | 3.903 | 3.936 | +0.033 |
| burst blend, raw of 10 | 5.055 | 5.360 | +0.305 |
| burst realisation | 0.499 | 0.441 | -0.059 |
| |duration error|, s | 0.070 | 0.069 | -0.001 |

## The adapter on its own

Dose-response shape `saturating`. Safe weight **0.75**, strong weight **0.75**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.385 | +0.018 | 0.60 | 4/10 | +0.014 | +0.075 | +0.065 | yes |
| 0.5 | 0.443 | +0.075 | 1.33 | 5/10 | +0.008 | +0.068 | +0.042 | yes |
| 0.75 | 0.488 | +0.120 | 3.13 | 7/10 | +0.035 | +0.246 | -0.025 | yes |
| 1.0 | 0.448 | +0.081 | 2.19 | 5/10 | +0.024 | +0.107 | +0.018 | yes |
| 1.25 | 0.433 | +0.066 | 1.52 | 5/10 | +0.006 | +0.178 | +0.008 | yes |
| 1.5 | 0.461 | +0.093 | 1.96 | 7/10 | -0.007 | +0.192 | +0.045 | yes |

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
| adapter × steering | -0.061 | -0.25 |
| adapter × guidance | +0.336 | 2.10 |
| steering × guidance | +0.013 | 0.04 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 0.045 | 0.283 | 0.930 | 0.795 | 0.092 | 0.280 | 0.531 | 1.117 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Pain`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Pain`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Pain`.
