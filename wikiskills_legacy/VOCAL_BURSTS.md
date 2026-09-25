# Vocal bursts — measured recipes (2026-09-02)

Written from `$SC/out/vb_lever/vl_agg.json` and `$SC/out/vb_ext/ve_agg.json`; no number
here was typed by hand. Full studies `~/reports/burst_levers.md` (protocol §51, the
levers, 15 classes) and `~/reports/burst_ext.md` (protocol §52, the remaining 55 classes
and the listening page).

This closes the gap `index.md` names at its end: *"71 adapters exist … but the combination study
did not cover them, so there is no operating point to write down."* There is now.

**Listen:** [https://huggingface.co/spaces/laion/moss-vocal-burst-recipes](https://huggingface.co/spaces/laion/moss-vocal-burst-recipes) — 248 takes for these classes, each marked with whether the detector scored it a hit, plus weight-0 controls.

## The 31 classes with a recipe

Everything below is family-relaxed hit rate — the owner accepted a neighbouring class as
a success, and that decision is load-bearing: for several of these classes the **strict**
rate is below 0.05 while the family rate is above 0.45. The model reliably makes a sound
of the right family and does not honour the sub-label.

`N` is how many candidates to generate so at least one realises with 90 % probability.
`N cons.` uses the hit rate minus one seed-noise standard deviation (0.068), because
every recipe here is an argmax over many cells and is therefore optimistic by about that
much.  Clips for every one of these recipes: `$SC/hf_vb/index.html`.

⚠ marks a recipe that reaches its hit rate by degrading the spoken line (Parakeet WER more than 0.10 above its own no-adapter control).

| class | weight | prompt form | hit (fam) | conservative | strict | N | N cons. | from |
|---|--:|---|--:|--:|--:|--:|--:|---|
| [`vb/relief_sigh`](patterns/vb-relief_sigh.md) | 0.8 | GENERAL-Ursache | **0.75** | 0.68 | 0.00 | 2 | 3 | §51 |
| [`vb/chuckle`](patterns/vb-chuckle.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.73** | 0.66 | 0.38 | 2 | 3 | §51 |
| [`vb/contented_sigh`](patterns/vb-contented_sigh.md) | 1.5 | GENERAL-Ursache + längere Dauer | **0.68** | 0.61 | 0.68 | 3 | 3 | §51 |
| [`vb/soft_hum`](patterns/vb-soft_hum.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.68** | 0.61 | 0.30 | 3 | 3 | §51 |
| [`vb/wistful_sigh`](patterns/vb-wistful_sigh.md) | 1.0 | Grundform | **0.55** | 0.48 | 0.07 | 3 | 4 | §52 |
| [`vb/cackle`](patterns/vb-cackle.md) | 1.8 | Grundform | **0.54** | 0.47 | 0.02 | 3 | 4 | §51 |
| [`vb/clears_throat`](patterns/vb-clears_throat.md) | 1.25 | Etikett mitten im Satz | **0.48** | 0.41 | 0.00 | 4 | 5 | §51 |
| [`vb/low_mumble`](patterns/vb-low_mumble.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.48** | 0.41 | 0.45 | 4 | 5 | §51 |
| [`vb/childlike_giggle`](patterns/vb-childlike_giggle.md) | 0.8 | Grundform | **0.47** | 0.40 | 0.05 | 4 | 5 | §51 |
| [`vb/exasperated_sigh`](patterns/vb-exasperated_sigh.md) | 1.25 | Grundform | **0.45** | 0.38 | 0.05 | 4 | 5 | §52 |
| [`vb/breathy_giggle`](patterns/vb-breathy_giggle.md) | 2.0 | Grundform | **0.40** | 0.33 | 0.23 | 5 | 6 | §52 |
| [`vb/sharp_inhale`](patterns/vb-sharp_inhale.md) | 2.3 | Grundform | **0.38** | 0.32 | 0.38 | 5 | 7 | §52 |
| [`vb/resonant_hum`](patterns/vb-resonant_hum.md) | 1.5 | Grundform | **0.37** | 0.30 | 0.23 | 6 | 7 | §52 |
| [`vbr/nervous_giggle`](patterns/vb-nervous_giggle.md) | 1.5 | GENERAL-Ursache + längere Dauer | **0.33** | 0.27 | 0.00 | 6 | 8 | §52 |
| [`vb/ahem`](patterns/vb-ahem.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.33** | 0.26 | 0.33 | 6 | 8 | §52 ⚠ |
| [`vb/fearful_gasp`](patterns/vb-fearful_gasp.md) | 1.25 | GENERAL-Ursache + längere Dauer | **0.28** | 0.22 | 0.00 | 7 | 10 | §52 |
| [`vb/frustrated_groan`](patterns/vb-frustrated_groan.md) | 2.0 | GENERAL-Ursache | **0.27** | 0.20 | 0.00 | 8 | 11 | §51 |
| [`vb/deep_breath`](patterns/vb-deep_breath.md) | 2.3 | Grundform | **0.27** | 0.20 | 0.23 | 8 | 11 | §52 |
| [`vbr/guffaw`](patterns/vb-guffaw.md) | 1.0 | Grundform | **0.27** | 0.20 | 0.00 | 8 | 11 | §52 |
| [`vb/displeased_grunt`](patterns/vb-displeased_grunt.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.25** | 0.18 | 0.00 | 9 | 12 | §52 ⚠ |
| [`vb/purr`](patterns/vb-purr.md) | 1.5 | Grundform | **0.25** | 0.18 | 0.00 | 9 | 12 | §52 |
| [`vbr/snicker`](patterns/vb-snicker.md) | 1.25 | GENERAL-Ursache + längere Dauer | **0.25** | 0.18 | 0.00 | 9 | 12 | §52 |
| [`vbr/whispered_mumble`](patterns/vb-whispered_mumble.md) | 0.25 | Grundform | **0.23** | 0.17 | 0.00 | 9 | 13 | §52 |
| [`vb/cough`](patterns/vb-cough.md) | 1.0 | Grundform | **0.23** | 0.17 | 0.02 | 9 | 13 | §52 |
| [`vbr/humming`](patterns/vb-humming.md) | 0.8 | Grundform | **0.23** | 0.17 | 0.00 | 9 | 13 | §52 |
| [`vb/exhausted_groan`](patterns/vb-exhausted_groan.md) | 1.8 | Grundform | **0.22** | 0.15 | 0.22 | 10 | 15 | §51 |
| [`vbr/pain_moan`](patterns/vb-pain_moan.md) | 1.5 | GENERAL-Ursache + längere Dauer | **0.18** | 0.12 | 0.00 | 12 | 19 | §52 |
| [`vb/surprised_gasp`](patterns/vb-surprised_gasp.md) | 1.0 | längere Dauer | **0.18** | 0.11 | 0.17 | 12 | 19 | §51 |
| [`vbr/coughing`](patterns/vb-coughing.md) | 0.25 | Grundform | **0.18** | 0.12 | 0.00 | 12 | 19 | §52 ⚠ |
| [`vb/scream`](patterns/vb-scream.md) | 1.3 | längere Dauer | **0.17** | 0.10 | 0.15 | 13 | 23 | §51 |
| [`vbr/sniff`](patterns/vb-sniff.md) | 2.0 | GENERAL-Ursache + längere Dauer | **0.15** | 0.08 | 0.03 | 15 | 27 | §52 |

## The classes with no recipe

**19 of the 55 classes measured in §52 never realise at any weight and under
either prompt form — not one hit in the whole grid.**  Plus `shriek` 0.117,
`lip_smack` 0.017 and `sharp_whistle` 0.000 from §51, and 17 more classes that
produce something but stay below the 0.15 bar.  Do not offer any of them.  A caller who
asks should be told the sound is not available, not handed a silent take.

**Never realised, at any dose:**

`clicks_tongue` `convulsive_sob` `gulps` `gurgling` `hiccup` `hiccups` `hiss` `nervous_gulp` `person_whistling_to_get_attention` `quiet_sob` `smack_one_s_lips` `smacks_lips` `sobs` `soft_whistle` `spitting` `swallows` `tongue_click` `tsk` `wolf_whistle` 

**Below the bar (a recipe exists but is not worth shipping):**

`snort` (0.13) `fast_breathing` (0.13) `snorting_giggle` (0.12) `yawn` (0.10) `growl` (0.10) `effort_grunt` (0.08) `trembling_whimper` (0.08) `panting` (0.07) `pleasure_moan` (0.07) `affirmative_grunt` (0.05) `heavy_breathing` (0.05) `normal_breathing` (0.05) `slow_breathing` (0.03) `deep_breathing` (0.02) `drinking_noises` (0.02) `kissing_sounds` (0.02) `mournful_wail` (0.02) 

The pattern is families, not classes: **all fourteen mouth classes and all four whistle
classes in the bank are on one of these two lists**, none above 0.017, across §51 and §52
together.  They are not weak adapters that need a larger weight; the sound is absent from
what the model can produce, and the fix is data, not a knob.

## What actually moved the needle

* **Dose past where §43 stopped.** One more level gave +0.024 family and **+0.022 strict
  (t 2.3)** — as much strict accuracy again as the entire step from no adapter to §43's
  recommendation. Optima are now *interior* for 11 of 15 classes, typically w\*+0.5…+1.0.
  **Nine of 400 ladder cells produced no decodable audio, all at w ≥ 2.3** — that is the ceiling,
  not the score.
* **Two prompt forms, cleanly additive.** A GENERAL-line sentence naming the sound's cause
  (+0.026) plus a longer stated duration (+0.022) together give **+0.044 family, +0.030 strict
  (t +3.3)** — the only prompt result significant on the strict metric.

## What did not work, and must not be tried again

* **The burst+stop DPO adapter is a null on burst realisation**: +0.007 / +0.017 / +0.006 at its
  three recommended checkpoints, none significant. It is not harmful — WER falls slightly, DNSMOS
  rises — and step 896 buys genuineness (+0.044) on top of the burst adapter. Ship it for that if
  at all, never for bursts.
* **The two adapters are redundant, not additive.** `both − dose` = +0.001…+0.006 (|t| ≤ 0.4);
  `both − dpo` = +0.077…+0.092 (t 3.5–4.2). All of the gain is the burst adapter.
* **Neighbour-class substitution — asking for a tired groan to get a frustrated one — is null on
  family and a significant *harm* on strict** (−0.021, t −2.9). The cheapest hoped-for fix does
  not work.
* **The cue written as an action** (−0.08…−0.11): the model degrades to *silence*, miss rate +0.12.
* **Mid-clause placement** (−0.07…−0.12, miss rate +0.31…+0.37, t 8–10, **15/15 classes worse**).
  One inversion worth knowing: `clears_throat` goes 0.250 → 0.483 there.

## What it costs

**Blend is the consistent casualty** — −0.4 to −0.6 of ten for every lever that raises the hit
rate. That is a real trade, not noise. **WER is the price of length, not of dose**: §43's dose
costs +0.003, one more level +0.130, the longer cue +0.095.

**Do not use DNSMOS to choose between these recipes.** It moves less than 0.04 for every lever
here, but **+0.38 (t 16.7, 9/9)** between two script kinds that differ only in what is said. It is
measuring the speech, not the burst. Under the action-cue form blend *rose* +0.18 and DNSMOS +0.03
while the burst stopped happening altogether — an object lesson in reading the wrong metric.

## The honest caveat on the longer-duration gain

It is largely the **detector's** eyesight improving, not the model's behaviour. The delivered
burst grows only +0.054 s against a +1.0 s request; the eyesight slope times 0.054 predicts +0.020
and +0.012 was observed, leaving no residual. Across classes, Δ detected duration correlates +0.53
with Δ hit. The sound is not much longer; it is easier to recognise. Useful either way, but it
should not be described as making the model produce more burst.
