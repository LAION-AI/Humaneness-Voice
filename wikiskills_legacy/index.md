# WikiSkill acting wiki — index

First draft, generated 2026-09-02 from the measured JSON by `code/build_wikiskills.py`. Layout follows `whitepaper/WIKISKILL_ACTING_SYSTEM.md` §4.2.

`coefficients.json` is the machine-readable form of every table below and is what the demo server loads at start-up. The pages are for the maintainer and the proposer; the actor reads the JSON.

**60 attributes.** A balanced operating point exists for **53**, a high-effect point for **56**, and a usable adapter weight for **28**. Where a column says *none*, the measurement says there is no usable setting — that is the finding, and the actor must not substitute a guess.

## How to read a recommendation

* **Balanced** clears every guardrail in `thresholds.balanced`: word error ≤ 0.15 absolute and ≤ +0.05 over baseline, genuineness ≥ −0.6, blend ≥ −1.0, burst realisation ≥ −0.1, |duration error| ≤ 0.3 s.
* **High effect** relaxes those to word error ≤ 0.30 absolute, genuineness ≥ −1.5, blend ≥ −3.0, burst realisation ≥ −0.25, |duration error| ≤ 1.0 s.
* Every Δ is paired: same prompts, same seed, clip samples averaged inside a prompt before averaging across prompts, n = number of prompts.
* A recommendation only counts if it beats a **random direction of matched norm at the same operating point**. That control is null on its own (-0.033) but **not** null at the combined point (+0.106, t 2.78).

## Emotions (40)

| attribute | balanced mode | Δ target | t | high-effect mode | Δ target | adapter w (safe/strong) | page |
|---|---|--:|--:|---|--:|:--|---|
| `emo/Affection` | `adapter+cfg` | +0.161 | 1.92 | `adapter+cfg` | +0.161 | 1.25 / 1.25 | [emo-Affection](patterns/emo-Affection.md) |
| `emo/Amusement` | `adapter+steer` | +0.379 | 5.32 | `adapter+steer+cfg` | +0.630 | none | [emo-Amusement](patterns/emo-Amusement.md) |
| `emo/Anger` | `steer` | +0.283 | 3.94 | `adapter+steer+cfg` | +0.621 | none | [emo-Anger](patterns/emo-Anger.md) |
| `emo/Astonishment_Surprise` | `cfg` | +0.060 | 1.88 | `adapter+steer` | +0.164 | none | [emo-Astonishment_Surprise](patterns/emo-Astonishment_Surprise.md) |
| `emo/Awe` | `adapter+cfg` | +0.012 | 0.72 | `adapter+steer` | +0.080 | none | [emo-Awe](patterns/emo-Awe.md) |
| `emo/Bitterness` | **none** | — | — | **none** | — | none | [emo-Bitterness](patterns/emo-Bitterness.md) |
| `emo/Concentration` | `adapter+cfg` | +0.054 | 2.25 | `adapter+steer` | +0.112 | 0.5 / 0.5 | [emo-Concentration](patterns/emo-Concentration.md) |
| `emo/Confusion` | `adapter+steer` | +0.072 | 1.65 | `adapter+steer` | +0.072 | none | [emo-Confusion](patterns/emo-Confusion.md) |
| `emo/Contemplation` | `adapter` | +0.023 | 0.83 | `steer+cfg` | +0.199 | 1.5 / 1.5 | [emo-Contemplation](patterns/emo-Contemplation.md) |
| `emo/Contempt` | `adapter+steer+cfg` | +0.143 | 1.99 | `adapter+steer+cfg` | +0.306 | none | [emo-Contempt](patterns/emo-Contempt.md) |
| `emo/Contentment` | `cfg` | +0.137 | 2.47 | `steer` | +0.199 | none | [emo-Contentment](patterns/emo-Contentment.md) |
| `emo/Disappointment` | **none** | — | — | **none** | — | none | [emo-Disappointment](patterns/emo-Disappointment.md) |
| `emo/Disgust` | **none** | — | — | `steer` | +0.100 | none | [emo-Disgust](patterns/emo-Disgust.md) |
| `emo/Distress` | `adapter+cfg` | +0.062 | 1.06 | `steer` | +0.135 | none | [emo-Distress](patterns/emo-Distress.md) |
| `emo/Doubt` | `adapter` | +0.011 | 0.18 | `adapter+steer` | +0.025 | none | [emo-Doubt](patterns/emo-Doubt.md) |
| `emo/Elation` | `adapter+steer` | +0.281 | 3.72 | `adapter+steer` | +0.379 | none | [emo-Elation](patterns/emo-Elation.md) |
| `emo/Embarrassment` | `adapter+steer` | +0.172 | 3.13 | `adapter+steer` | +0.172 | none | [emo-Embarrassment](patterns/emo-Embarrassment.md) |
| `emo/Emotional_Numbness` | **none** | — | — | `adapter+steer` | +0.106 | none | [emo-Emotional_Numbness](patterns/emo-Emotional_Numbness.md) |
| `emo/Fatigue_Exhaustion` | `adapter+cfg` | +0.113 | 1.25 | `adapter+cfg` | +0.113 | none | [emo-Fatigue_Exhaustion](patterns/emo-Fatigue_Exhaustion.md) |
| `emo/Fear` | `cfg` | +0.059 | 1.67 | `adapter+steer` | +0.149 | none | [emo-Fear](patterns/emo-Fear.md) |
| `emo/Helplessness` | `adapter+cfg` | +0.007 | 0.32 | `steer` | +0.015 | none | [emo-Helplessness](patterns/emo-Helplessness.md) |
| `emo/Hope_Enthusiasm_Optimism` | `adapter+steer` | +0.220 | 5.09 | `adapter+steer+cfg` | +0.343 | 0.75 / 0.75 | [emo-Hope_Enthusiasm_Optimism](patterns/emo-Hope_Enthusiasm_Optimism.md) |
| `emo/Impatience_and_Irritability` | `adapter+steer` | +0.195 | 3.56 | `steer+cfg` | +0.366 | none | [emo-Impatience_and_Irritability](patterns/emo-Impatience_and_Irritability.md) |
| `emo/Infatuation` | `cfg` | +0.049 | 0.81 | `adapter+steer` | +0.203 | none | [emo-Infatuation](patterns/emo-Infatuation.md) |
| `emo/Interest` | `adapter` | +0.051 | 2.48 | `adapter` | +0.051 | 1.0 / 1.0 | [emo-Interest](patterns/emo-Interest.md) |
| `emo/Intoxication_Altered_States_of_Consciousness` | `adapter` | +0.089 | 1.05 | `adapter` | +0.089 | 0.75 / 0.75 | [emo-Intoxication_Altered_States_of_Consciousness](patterns/emo-Intoxication_Altered_States_of_Consciousness.md) |
| `emo/Jealousy_and_Envy` | `steer+cfg` | +0.145 | 2.98 | `steer+cfg` | +0.145 | none | [emo-Jealousy_and_Envy](patterns/emo-Jealousy_and_Envy.md) |
| `emo/Longing` | `cfg` | +0.077 | 1.41 | `cfg` | +0.077 | none | [emo-Longing](patterns/emo-Longing.md) |
| `emo/Malevolence_Malice` | `steer` | +0.219 | 3.23 | `adapter+steer+cfg` | +0.429 | none | [emo-Malevolence_Malice](patterns/emo-Malevolence_Malice.md) |
| `emo/Pain` | `cfg` | +0.040 | 1.31 | `cfg` | +0.040 | 0.75 / 0.75 | [emo-Pain](patterns/emo-Pain.md) |
| `emo/Pleasure_Ecstasy` | `adapter` | +0.031 | 0.65 | `adapter+steer` | +0.076 | 1.25 / 1.25 | [emo-Pleasure_Ecstasy](patterns/emo-Pleasure_Ecstasy.md) |
| `emo/Pride` | `adapter+steer` | +0.142 | 4.91 | `adapter+steer+cfg` | +0.214 | none | [emo-Pride](patterns/emo-Pride.md) |
| `emo/Relief` | **none** | — | — | `adapter+steer` | +0.061 | 1.0 / 1.0 | [emo-Relief](patterns/emo-Relief.md) |
| `emo/Sadness` | `steer+cfg` | +0.069 | 1.48 | `adapter+steer+cfg` | +0.142 | none | [emo-Sadness](patterns/emo-Sadness.md) |
| `emo/Sexual_Lust` | `adapter+cfg` | +0.150 | 2.42 | `adapter+steer+cfg` | +0.802 | 1.25 / 1.25 | [emo-Sexual_Lust](patterns/emo-Sexual_Lust.md) |
| `emo/Shame` | **none** | — | — | **none** | — | none | [emo-Shame](patterns/emo-Shame.md) |
| `emo/Sourness` | `adapter+cfg` | +0.087 | 1.27 | `adapter+cfg` | +0.087 | none | [emo-Sourness](patterns/emo-Sourness.md) |
| `emo/Teasing` | `adapter+steer` | +0.287 | 3.30 | `adapter+steer+cfg` | +0.516 | 1.5 / 1.5 | [emo-Teasing](patterns/emo-Teasing.md) |
| `emo/Thankfulness_Gratitude` | `steer` | +0.079 | 1.81 | `adapter+steer+cfg` | +0.127 | none | [emo-Thankfulness_Gratitude](patterns/emo-Thankfulness_Gratitude.md) |
| `emo/Triumph` | `steer` | +0.289 | 3.34 | `steer+cfg` | +0.712 | none | [emo-Triumph](patterns/emo-Triumph.md) |

## Delivery axes (17)

| attribute | balanced mode | Δ target | t | high-effect mode | Δ target | adapter w (safe/strong) | page |
|---|---|--:|--:|---|--:|:--|---|
| `vn/AROU_high` | `adapter+cfg` | +0.562 | 3.57 | `adapter+steer+cfg` | +2.287 | 1.5 / 1.5 | [vn-AROU_high](patterns/vn-AROU_high.md) |
| `vn/AROU_low` | `adapter+cfg` | +0.289 | 2.32 | `adapter+cfg` | +0.289 | 1.5 / 1.5 | [vn-AROU_low](patterns/vn-AROU_low.md) |
| `vn/ARSH_high` | `adapter+cfg` | +0.513 | 4.68 | `adapter+cfg` | +0.513 | 1.5 / 1.5 | [vn-ARSH_high](patterns/vn-ARSH_high.md) |
| `vn/ARSH_low` | **none** | — | — | **none** | — | 1.25 / 1.25 | [vn-ARSH_low](patterns/vn-ARSH_low.md) |
| `vn/EMPH_high` | `adapter` | +0.431 | 2.46 | `adapter` | +0.431 | 1.25 / 1.25 | [vn-EMPH_high](patterns/vn-EMPH_high.md) |
| `vn/EXPL_high` | `adapter+cfg` | +0.125 | 3.66 | `adapter+cfg` | +0.125 | 1.5 / 1.5 | [vn-EXPL_high](patterns/vn-EXPL_high.md) |
| `vn/S_ASMR_high` | `adapter+cfg` | +0.696 | 2.48 | `adapter+cfg` | +0.696 | 1.5 / 1.5 | [vn-S_ASMR_high](patterns/vn-S_ASMR_high.md) |
| `vn/S_DRAM_high` | `adapter` | +0.812 | 6.13 | `adapter+steer` | +2.831 | 1.5 / 1.5 | [vn-S_DRAM_high](patterns/vn-S_DRAM_high.md) |
| `vn/S_RANT_high` | `adapter+cfg` | +1.183 | 5.18 | `adapter+steer+cfg` | +3.112 | 1.5 / 1.5 | [vn-S_RANT_high](patterns/vn-S_RANT_high.md) |
| `vn/TENS_high` | `adapter` | +0.312 | 2.10 | `adapter+steer` | +2.152 | 1.5 / 1.5 | [vn-TENS_high](patterns/vn-TENS_high.md) |
| `vn/VALN_high` | `adapter` | +0.417 | 3.59 | `adapter` | +0.417 | 1.0 / 1.0 | [vn-VALN_high](patterns/vn-VALN_high.md) |
| `vn/VALN_low` | `adapter+cfg` | +0.082 | 0.86 | `adapter+cfg` | +0.082 | 1.25 / 1.25 | [vn-VALN_low](patterns/vn-VALN_low.md) |
| `vn/VALS_high` | `adapter+cfg` | +0.509 | 5.81 | `adapter+cfg` | +0.509 | none | [vn-VALS_high](patterns/vn-VALS_high.md) |
| `vn/VALS_low` | `adapter+cfg` | +0.001 | 0.01 | `adapter+cfg` | +0.001 | 1.25 / 1.25 | [vn-VALS_low](patterns/vn-VALS_low.md) |
| `vn/VFLX_high` | `adapter` | +0.843 | 5.82 | `adapter` | +0.843 | 1.5 / 1.5 | [vn-VFLX_high](patterns/vn-VFLX_high.md) |
| `vn/VOLT_high` | `adapter+cfg` | +0.270 | 2.89 | `adapter+steer` | +1.997 | 1.5 / 1.5 | [vn-VOLT_high](patterns/vn-VOLT_high.md) |
| `vn/VULN_high` | `adapter+cfg` | +0.171 | 1.37 | `adapter+cfg` | +0.171 | 1.25 / 1.25 | [vn-VULN_high](patterns/vn-VULN_high.md) |

## Quality axes (3)

| attribute | balanced mode | Δ target | t | high-effect mode | Δ target | adapter w (safe/strong) | page |
|---|---|--:|--:|---|--:|:--|---|
| `qual/blend_high` | `adapter+steer` | +1.847 | 2.73 | `adapter+steer` | +1.847 | none | [quality-blend](patterns/quality-blend.md) |
| `qual/esthetics_high` | `adapter` | +0.215 | 1.44 | `adapter+steer+cfg` | +0.323 | 1.5 / 1.5 | [quality-esthetics](patterns/quality-esthetics.md) |
| `qual/genuineness_high` | `adapter+steer+cfg` | +0.810 | 6.18 | `adapter+steer+cfg` | +0.819 | none | [quality-genuineness](patterns/quality-genuineness.md) |

## Cross-cutting pages

* [interactions.md](interactions.md) — the 2×2×2 factorial: which levers combine and which cancel, per family.

## What is not here yet

* **Vocal-burst classes.** ~~No operating point to write down.~~ **Superseded 2026-09-02:** see [VOCAL_BURSTS.md](VOCAL_BURSTS.md) — 12 classes now have a measured recipe (weight, prompt form, hit rate, best-of-N), and three are documented as having none. All 15 were measured in the full server stack, family-relaxed.
* **Logs and skill-impact.** `logs.md` and `skill-impact.md` in the whitepaper layout are written by the consolidation cycle, which has not run. This draft is cycle zero.
* **Listening tests.** Every number here is one model's judgement of another model's output. Nobody has listened.
