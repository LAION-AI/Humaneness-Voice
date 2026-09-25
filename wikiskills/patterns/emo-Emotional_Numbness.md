# Emotional Numbness

`emo/Emotional_Numbness` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Emotional_Numbness`. Target metric `emo_pct`, baseline 0.379 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Emotional_Numbness
steering_key: emo:Emotional_Numbness
target_metric: emo_pct
adapter:
  name: sft3_emotion:Emotional_Numbness
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced: null            # no setting cleared the guardrails
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Emotional_Numbness", w: 1.0}
  steer:
    - {key: "emo:Emotional_Numbness", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.106, t: 1.56, n_prompts: 10, n_up: 5,
             d_wer_parakeet: +0.025, d_genuineness: -1.191,
             d_blend: +0.508, d_r_burst: -0.092, d_dur_err_abs_s: +0.004}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

**No usable setting.** No candidate configuration cleared the balanced guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Emotional_Numbness` at w = 1.0. Steering on the cond branch. 

Target moves **+0.106** (t 1.56, better on 5 of 10 prompts), from 0.379 to 0.485. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.072 | 0.097 | +0.025 |
| genuineness, raw of 6 | 3.954 | 2.764 | -1.191 |
| burst blend, raw of 10 | 4.707 | 5.215 | +0.508 |
| burst realisation | 0.513 | 0.421 | -0.092 |
| |duration error|, s | 0.054 | 0.058 | +0.004 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.361 | +0.000 | 0.01 | 4/10 | +0.007 | -0.217 | +0.028 | yes |
| 0.5 | 0.307 | -0.053 | -1.26 | 2/10 | -0.024 | -0.185 | +0.055 | yes |
| 0.75 | 0.295 | -0.065 | -0.82 | 5/10 | -0.026 | -0.048 | +0.132 | yes |
| 1.0 | 0.344 | -0.017 | -0.43 | 4/10 | -0.008 | -0.104 | +0.165 | yes |
| 1.25 | 0.350 | -0.010 | -0.23 | 4/10 | -0.038 | -0.112 | +0.112 | yes |
| 1.5 | 0.303 | -0.057 | -1.00 | 4/10 | -0.043 | -0.024 | +0.068 | yes |

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
| adapter × steering | +0.220 | 1.11 |
| adapter × guidance | -0.042 | -0.26 |
| steering × guidance | +0.005 | 0.02 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.286 | 1.181 | 1.506 | 1.476 | 1.137 | 1.059 | 1.647 | 1.504 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: below_resolution). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*
* **the balanced operating point** — no candidate cleared the balanced guardrails. 7 candidates were scored.  
  *combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Emotional_Numbness`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Emotional_Numbness`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Emotional_Numbness`.
