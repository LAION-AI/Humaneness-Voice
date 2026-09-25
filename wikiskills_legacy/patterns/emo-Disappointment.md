# Disappointment

`emo/Disappointment` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Disappointment`. Target metric `emo_pct`, baseline 0.212 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Disappointment
steering_key: emo:Disappointment
target_metric: emo_pct
adapter:
  name: sft3_emotion:Disappointment
  usable: false
  safe_w: null
  strong_w: null
  dose_shape: monotone
balanced: null            # no setting cleared the guardrails
high_effect: null            # no setting cleared the guardrails
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

**No usable setting.** No candidate configuration cleared the balanced guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## High effect operating point

**No usable setting.** No candidate configuration cleared the high-effect guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## The adapter on its own

**No usable merge weight.** Dose-response shape `monotone`. Across the 0.25–1.5 ladder no weight both moved the target above its noise floor (0.042) and stayed inside the safe guardrails. It is **not harmful** at any weight — no adapter in the 5,740-cell sweep was — the failure is a failure to move the target.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.273 | +0.005 | 0.07 | 3/10 | -0.018 | -0.087 | +0.030 | yes |
| 0.5 | 0.339 | +0.071 | 1.20 | 4/10 | -0.031 | -0.052 | -0.068 | yes |
| 0.75 | 0.351 | +0.083 | 1.70 | 4/10 | -0.023 | +0.036 | -0.054 | yes |
| 1.0 | 0.325 | +0.057 | 1.59 | 5/10 | +0.023 | -0.038 | +0.014 | yes |
| 1.25 | 0.357 | +0.089 | 1.70 | 6/10 | -0.007 | +0.001 | -0.120 | yes |
| 1.5 | 0.353 | +0.085 | 1.43 | 5/10 | -0.032 | +0.041 | -0.087 | yes |

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
| adapter × steering | +0.399 | 0.97 |
| adapter × guidance | +0.202 | 1.20 |
| steering × guidance | +0.388 | 2.86 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 1.039 | 0.771 | 0.625 | 0.601 | 0.747 | 0.537 | 0.588 | 0.910 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the `adapter` lever, for this attribute** — no weight in the 0.25-1.5 ladder both moves the target above the noise floor and clears the safe guardrails (shape: monotone). It is not harmful at any weight -- the failure is a failure to move the target.  
  *lora-dose 5,740-cell sweep*
* **the balanced operating point** — no candidate cleared the balanced guardrails. 7 candidates were scored.  
  *combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Disappointment`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Disappointment`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Disappointment`.
