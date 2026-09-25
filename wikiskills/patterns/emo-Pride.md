# Pride

`emo/Pride` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Pride`. Target metric `emo_pct`, baseline 0.636 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Pride
steering_key: emo:Pride
target_metric: emo_pct
adapter:
  name: sft3_emotion:Pride
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Pride", w: 1.0}
  steer:
    - {key: "emo:Pride", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.142, t: 4.91, n_prompts: 10, n_up: 9,
             d_wer_parakeet: +0.047, d_genuineness: +0.718,
             d_blend: +1.254, d_r_burst: -0.002, d_dur_err_abs_s: +0.001}
  beats_random_floor: true
high_effect:
  mode: adapter+steer+cfg
  lora: {name: "sft3_emotion:Pride", w: 0.5}
  steer:
    - {key: "emo:Pride", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 2.0}
  measured: {d_target: +0.214, t: 4.62, n_prompts: 10, n_up: 8,
             d_wer_parakeet: +0.193, d_genuineness: +0.786,
             d_blend: +3.026, d_r_burst: -0.021, d_dur_err_abs_s: +0.012}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Pride` at w = 1.0. Steering on the cond branch. 

Target moves **+0.142** (t 4.91, better on 9 of 10 prompts), from 0.636 to 0.779. This clears the matched random-direction floor of -0.011.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.097 | 0.144 | +0.047 |
| genuineness, raw of 6 | 3.804 | 4.522 | +0.718 |
| burst blend, raw of 10 | 5.005 | 6.259 | +1.254 |
| burst realisation | 0.464 | 0.463 | -0.002 |
| |duration error|, s | 0.063 | 0.065 | +0.001 |

## High effect operating point

Mode **`adapter+steer+cfg`**. Adapter `sft3_emotion:Pride` at w = 0.5. Guidance g = 2.0. Steering on the cond branch. 

Target moves **+0.214** (t 4.62, better on 8 of 10 prompts), from 0.636 to 0.851. This clears the matched random-direction floor of -0.011.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.097 | 0.290 | +0.193 |
| genuineness, raw of 6 | 3.804 | 4.590 | +0.786 |
| burst blend, raw of 10 | 5.005 | 8.031 | +3.026 |
| burst realisation | 0.464 | 0.444 | -0.021 |
| |duration error|, s | 0.063 | 0.075 | +0.012 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.598 | +0.019 | 1.74 | 4/10 | -0.049 | +0.162 | +0.101 | yes |
| 0.5 | 0.592 | +0.013 | 0.94 | 1/10 | -0.019 | +0.051 | +0.010 | yes |
| 0.75 | 0.557 | -0.022 | -1.22 | 1/10 | -0.049 | +0.074 | +0.127 | yes |
| 1.0 | 0.575 | -0.005 | -0.27 | 2/10 | -0.017 | +0.034 | +0.029 | yes |
| 1.25 | 0.581 | +0.002 | 0.14 | 1/10 | -0.033 | +0.018 | +0.119 | yes |
| 1.5 | 0.552 | -0.027 | -1.24 | 2/10 | -0.028 | +0.199 | +0.148 | yes |

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
| adapter × steering | -0.176 | -1.47 |
| adapter × guidance | +0.012 | 0.10 |
| steering × guidance | +0.783 | 2.69 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 3.849 | 3.731 | 4.483 | 5.220 | 3.874 | 3.840 | 4.405 | 5.081 |

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

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Pride`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Pride`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Pride`.
