# Disgust

`emo/Disgust` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Disgust`. Target metric `emo_pct`, baseline 0.456 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Disgust
steering_key: emo:Disgust
target_metric: emo_pct
adapter:
  name: sft3_emotion:Disgust
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: below_resolution
balanced: null            # no setting cleared the guardrails
high_effect:
  mode: steer
  lora: null
  steer:
    - {key: "emo:Disgust", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.100, t: 1.68, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.014, d_genuineness: -1.171,
             d_blend: -0.084, d_r_burst: -0.193, d_dur_err_abs_s: +0.045}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

**No usable setting.** No candidate configuration cleared the balanced guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## High effect operating point

Mode **`steer`**. No adapter. Steering on the cond branch. 

Target moves **+0.100** (t 1.68, better on 7 of 10 prompts), from 0.456 to 0.556. This clears the matched random-direction floor of +0.024.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.096 | 0.110 | +0.014 |
| genuineness, raw of 6 | 3.838 | 2.667 | -1.171 |
| burst blend, raw of 10 | 5.039 | 4.954 | -0.084 |
| burst realisation | 0.518 | 0.325 | -0.193 |
| |duration error|, s | 0.087 | 0.133 | +0.045 |

## The adapter on its own

**No usable merge weight.** Dose-response shape `below_resolution`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.415 | -0.063 | -1.27 | 3/10 | -0.045 | -0.118 | +0.023 | yes |
| 0.5 | 0.499 | +0.021 | 0.48 | 4/10 | -0.064 | +0.070 | -0.051 | yes |
| 0.75 | 0.480 | +0.002 | 0.04 | 5/10 | -0.059 | +0.083 | -0.028 | yes |
| 1.0 | 0.471 | -0.007 | -0.20 | 3/10 | -0.063 | +0.089 | +0.050 | yes |
| 1.25 | 0.450 | -0.028 | -0.67 | 4/10 | -0.058 | +0.131 | +0.025 | yes |
| 1.5 | 0.455 | -0.023 | -0.51 | 5/10 | -0.071 | -0.149 | -0.129 | yes |

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
| adapter × steering | +0.234 | 0.89 |
| adapter × guidance | +0.201 | 1.28 |
| steering × guidance | -0.156 | -0.57 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.971 | 1.869 | 2.157 | 1.836 | 1.901 | 1.936 | 2.258 | 2.201 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: below_resolution). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*
* **the balanced operating point** — no candidate cleared the balanced guardrails. 16 candidates were scored.  
  *combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Disgust`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Disgust`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Disgust`.
