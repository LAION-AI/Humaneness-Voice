# ARSH_low

`vn/ARSH_low` — delivery axis. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `vn:ARSH` (this is a low tail of the axis; see *Never*). Target metric `vn:ARSH`, baseline -2.187 over 10 prompts.

## Coefficient block

```yaml
attribute: vn/ARSH_low
steering_key: vn:ARSH
target_metric: vn:ARSH
adapter:
  name: sft3_voicenet:ARSH_low
  usable: true
  safe_w: 1.25
  strong_w: 1.25
  dose_shape: below_resolution
balanced: null            # no setting cleared the guardrails
high_effect: null            # no setting cleared the guardrails
```

## Balanced operating point

**No usable setting.** No candidate configuration cleared the balanced guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## High effect operating point

**No usable setting.** No candidate configuration cleared the high-effect guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## The adapter on its own

Dose-response shape `below_resolution`. Safe weight **1.25**, strong weight **1.25**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 1.675 | -0.107 | -1.15 | 5/10 | -0.002 | +0.107 | +0.031 | yes |
| 0.5 | 1.730 | -0.052 | -0.68 | 3/10 | -0.014 | +0.044 | +0.030 | yes |
| 0.75 | 1.694 | -0.088 | -1.29 | 3/10 | -0.015 | +0.033 | -0.022 | yes |
| 1.0 | 1.609 | -0.174 | -3.90 | 1/10 | -0.013 | +0.066 | -0.069 | yes |
| 1.25 | 1.541 | -0.241 | -3.19 | 1/10 | -0.013 | +0.185 | -0.102 | yes |
| 1.5 | 1.666 | -0.116 | -1.42 | 2/10 | +0.002 | +0.062 | -0.104 | yes |

## Interactions

Pooled over the delivery axis family (target in SD units, n = 170 attribute×prompt cells):

| pair | interaction | t | reading |
|---|--:|--:|---|
| adapter × steering | -0.164 | -3.75 | **sub-additive — pick one** |
| adapter × guidance | -0.125 | -3.50 | **sub-additive — pick one** |
| steering × guidance | +0.144 | 2.02 | **super-additive — and it carries a cost, see below** |

Cumulativity ratio for this family: **0.97** (observed with all three levers, divided by the sum of the three alone).

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
| adapter × steering | +0.184 | 1.42 |
| adapter × guidance | +0.128 | 0.70 |
| steering × guidance | -0.992 | -5.40 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| -3.801 | -3.980 | -4.791 | -6.076 | -3.902 | -4.066 | -4.822 | -5.865 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **stack delivery adapter + delivery steering vector** — on a delivery axis the two levers do the same job and are significantly sub-additive: interaction -0.164 (t -3.75). Pick one.  
  *combination-study 2^3 factorial*
* **stack delivery adapter + guidance** — also sub-additive on delivery: interaction -0.125 (t -3.50)  
  *combination-study 2^3 factorial*
* **the `steer` lever, for this attribute** — no measured steering route to this tail. The vector table holds the high-minus-low difference, and the two tails of an attribute are orthogonal rather than opposite (median cos -0.0004), so -alpha along it is not 'the low tail'. No balanced or high-effect recipe for any _low axis uses steering.  
  *layer-forensics; combination-study recommendations*
* **the balanced operating point** — no candidate cleared the balanced guardrails. 7 candidates were scored.  
  *combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `vn/ARSH_low`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `vn/ARSH_low`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `vn:ARSH`.
