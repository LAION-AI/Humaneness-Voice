# Guidance: 3.0, measured twice

Classifier-free guidance runs the model twice per frame — once with the delivery
direction and once with it stripped — and amplifies the difference. It costs
about **1.93× wall clock** and cannot stream.

It is now **on by default at 3.0**, in the UI and in `/api/speak`.

## Is it better than nothing?

Four acting tasks, two English and two German, each rendered from **one script**
at each guidance level so the director's choices are held constant. Three seeds
per cell, every clip rated 0–5 by `gemini-3.8-flash` on pleasantness, fit to the
task, and naturalness.

**First run**, 24 clips:

| | pleasant | fit | natural | total of 15 |
|---|--:|--:|--:|--:|
| no guidance | 3.08 | 1.58 | 1.67 | 6.33 |
| **g = 3** | 3.00 | **2.92** | **2.75** | **8.67** |

Paired within (task, seed): **+2.33 points, t 2.73, p 0.006**, better in 9 of 12
pairs.

## Is 4 better than 3?

**No.** A second run added g = 4 on the same design, 36 clips:

| | pleasant | fit | natural | total |
|---|--:|--:|--:|--:|
| no guidance | 3.17 | 2.08 | 2.25 | 7.50 |
| **g = 3** | **3.50** | **2.92** | **3.00** | **9.42** |
| g = 4 | 2.92 | 2.83 | 2.42 | 8.17 |

| contrast | difference | t | p | better in |
|---|--:|--:|--:|--:|
| g3 − g1 | +1.92 | +1.52 | 0.128 | 9/12 |
| g4 − g1 | +0.67 | +0.66 | 0.509 | 8/12 |
| **g4 − g3** | **−1.25** | −1.02 | 0.308 | **4/12** |

Four is worse than three on **all three rubrics** and wins fewer than half its
pairs. Neither contrast is significant on its own, but the direction is
consistent and g3 − g1 replicates across the two runs (+2.33 and +1.92), so 3.0
is the default and 4.0 is not.

By language, from the second run: German gains far more from guidance
(g3 − g1 = **+3.50**) than English (+0.33), and both lose from going to 4
(−1.83 and −0.67).

## What it actually improves

`fit` and `natural` — the two axes the arena study found the system weakest on,
where nineteen prompt additions moved nothing. `pleasant` barely moves, which is
consistent with everything else measured here: the sound quality is set by the
model and the adapter stack, not by how the sampling is steered.

Reproduce with `eval/cfg_ab.py`.
