# Interactions between the three levers

Generated 2026-09-02 from `combination-study/stats/analysis.json` (`t2`, the 2×2×2 factorial over adapter × steering × guidance). Target is in SD units of the attribute's own metric.

## Main effects, by family

| family | adapter | t | steering | t | guidance | t | n |
|---|--:|--:|--:|--:|--:|--:|--:|
| emotion | +0.077 | 2.81 | +0.384 | 9.41 | +0.050 | 1.84 | 399 |
| delivery | +0.377 | 8.66 | +0.614 | 6.15 | +0.026 | 1.13 | 170 |
| quality | +0.399 | 3.26 | +0.006 | 0.04 | +0.032 | 0.34 | 30 |

**The best single lever flips by family.** Steering is by far the strongest on emotion and on delivery; on the quality axes it does nothing at all, and the adapter is the only lever that moves them.

## Pairwise interactions, by family

| family | adapter × steering | t | adapter × guidance | t | steering × guidance | t | cumulativity |
|---|--:|--:|--:|--:|--:|--:|--:|
| emotion | +0.038 | 1.36 | -0.031 | -1.22 | +0.277 | 7.51 | 1.52 |
| delivery | -0.164 | -3.75 | -0.125 | -3.50 | +0.144 | 2.02 | 0.97 |
| quality | +0.259 | 2.05 | -0.060 | -0.63 | -0.334 | -2.24 | 0.42 |

Three rules follow, and they are measurements rather than preferences:

1. **On an emotion, adapter and steering are additive.** The interaction is +0.038 (t 1.36) — not significant. The combination is predictable, so `adapter+steer` is the sensible default there.
2. **On a delivery axis, pick one lever.** Adapter × steering is -0.164 (t -3.75) and adapter × guidance is -0.125 (t -3.50). Both are significantly sub-additive: the two levers are doing the same job.
3. **Steering × guidance is the only real synergy, and it is a coupled package.** On emotion it is +0.277 (t 7.51) on the target — and the same term carries +0.078 of word error (t 11.80), -0.862 of genuineness (t -21.53) and -0.070 of burst realisation (t -6.73). Every damage term has a larger |t| than the gain.

## Controls

* **Base replication.** 360 paired cells, max absolute difference 0.0, 100% exact. The harness reproduces the bare model bit for bit.
* **Zero-strength path equivalence.** 510 cells, max absolute difference 0.0. Running the steering code path at α = 0 is identical to not running it, which is what makes it a control rather than a second condition.
* **Random direction, matched norm.** Pooled -0.033 — null on its own. At the combined operating point it is **+0.106 (t 2.78)** — not null. Anything shipped has to beat that floor, not zero.
