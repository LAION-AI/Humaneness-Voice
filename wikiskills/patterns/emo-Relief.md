# Relief

`emo/Relief` — emotion. Generated from the measured JSON on 2026-09-02 by `code/build_wikiskills.py`; do not edit by hand, edit the generator or the measurements it reads.

Steering vector key `emo:Relief`. Target metric `emo_pct`, baseline 0.602 over 10 prompts.

## Coefficient block

```yaml
attribute: emo/Relief
steering_key: emo:Relief
target_metric: emo_pct
adapter:
  name: sft3_emotion:Relief
  usable: true
  safe_w: 1.0
  strong_w: 1.0
  dose_shape: noisy
balanced: null            # no setting cleared the guardrails
high_effect:
  mode: adapter+steer
  lora: {name: "sft3_emotion:Relief", w: 1.0}
  steer:
    - {key: "emo:Relief", alpha: 0.1, taps: top1}   # h20
  steer_branch: cond
  cfg: {g: 1.0}
  measured: {d_target: +0.061, t: 1.88, n_prompts: 10, n_up: 7,
             d_wer_parakeet: +0.029, d_genuineness: -1.451,
             d_blend: -0.002, d_r_burst: -0.120, d_dur_err_abs_s: +0.003}
  beats_random_floor: true
```

The recipes above are as measured. **On top of them the actor applies the numbness subtraction automatically** whenever an emotion is pushed — `{key: "emo:Emotional_Numbness", alpha: -0.10, taps: top1}` — because subtracting Emotional_Numbness at alpha = -0.10 returns +0.60 of genuineness (t 9.64, 67 of 80 prompts) at no cost in emotion when the adapter carries the emotion (*combination-study, arm t3*). It is not folded into the blocks above because the combination study scored the recipes without it; it is a separate, separately measured component.

## Balanced operating point

**No usable setting.** No candidate configuration cleared the balanced guardrails for this attribute. That is a finding, not a gap: the actor must not invent one. Reach for a delivery axis instead, or accept the baseline.

## High effect operating point

Mode **`adapter+steer`**. Adapter `sft3_emotion:Relief` at w = 1.0. Steering on the cond branch. 

Target moves **+0.061** (t 1.88, better on 7 of 10 prompts), from 0.602 to 0.663. This clears the matched random-direction floor of -0.033.

| guardrail | baseline | at this point | change |
|---|--:|--:|--:|
| word error (Parakeet) | 0.072 | 0.101 | +0.029 |
| genuineness, raw of 6 | 3.852 | 2.402 | -1.451 |
| burst blend, raw of 10 | 5.137 | 5.135 | -0.002 |
| burst realisation | 0.485 | 0.365 | -0.120 |
| |duration error|, s | 0.066 | 0.069 | +0.003 |

## The adapter on its own

Dose-response shape `noisy`. Safe weight **1.0**, strong weight **1.0**.

| w | target | Δ target | t | n up | Δ WER | Δ genuineness | Δ burst real. | safe |
|--:|--:|--:|--:|--:|--:|--:|--:|:--|
| 0.25 | 0.528 | +0.076 | 1.24 | 6/10 | -0.006 | +0.127 | +0.023 | yes |
| 0.5 | 0.583 | +0.132 | 2.87 | 9/10 | -0.005 | +0.003 | +0.039 | yes |
| 0.75 | 0.499 | +0.047 | 0.73 | 6/10 | -0.019 | +0.024 | +0.044 | yes |
| 1.0 | 0.612 | +0.160 | 2.62 | 9/10 | -0.018 | +0.012 | +0.089 | yes |
| 1.25 | 0.577 | +0.125 | 2.14 | 9/10 | -0.015 | +0.026 | -0.109 | yes |
| 1.5 | 0.583 | +0.131 | 1.90 | 8/10 | +0.011 | +0.149 | +0.036 | yes |

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
| adapter × steering | +0.331 | 1.78 |
| adapter × guidance | +0.140 | 0.99 |
| steering × guidance | +0.074 | 0.37 |

Target in SD units at each corner of the 2×2×2 (bits are adapter, steering, guidance):

| 000 | 001 | 010 | 011 | 100 | 101 | 110 | 111 |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 2.541 | 2.310 | 2.536 | 2.590 | 2.263 | 2.383 | 2.800 | 2.783 |

## Never

* **steering α ≥ 0.3** — steering collapses above alpha = 0.3, and at alpha >= 0.5 a random direction of matched norm does the same damage  
  *steering study, layer-forensics/w3/steer_study.json, 12,136 clips*
* **guidance g < 1.0** — below g = 1 guidance actively hurts (-0.0370 at g = 0.5, t -2.56)  
  *cfg-study*
* **steering at k > 1 layers** — emotion is free only at k = 1; the matched random control is not null at k = 3 (+0.081, t 2.04)  
  *steering study*
* **the balanced operating point** — no candidate cleared the balanced guardrails. 7 candidates were scored.  
  *combination-study recommendations*

## Provenance

* Recipes and every Δ in this page: `combination-study/stats/comb_recommendations.json`, key `emo/Relief`.
* Interaction terms: `combination-study/stats/analysis.json`, `t2`.
* Dose ladder: `lora-dose/coefficients.json`, key `emotion/Relief`.
* Layer ranking behind `topK`: `work_vb/tap_rank.json`.
* Steering vectors: `actforensics/vectors/p3_vectors_ext.npz`, row `emo:Relief`.
