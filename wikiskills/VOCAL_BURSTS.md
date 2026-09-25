# Vocal bursts — measured recipes (2026-09-02)

Written from `$SC/out/vb_lever/vl_agg.json` and `$SC/out/vb_ext/ve_agg.json`; no number
here was typed by hand. Full studies `~/reports/burst_levers.md` (protocol §51, the
levers, 15 classes) and `~/reports/burst_ext.md` (protocol §52, the remaining 55 classes
and the listening page).

This closes the gap `index.md` names at its end: *"71 adapters exist … but the combination study
did not cover them, so there is no operating point to write down."* There is now.

**Listen:** [https://huggingface.co/spaces/laion/moss-vocal-burst-recipes](https://huggingface.co/spaces/laion/moss-vocal-burst-recipes) — 248 takes for these classes, each marked with whether the detector scored it a hit, plus weight-0 controls.

## How many bursts, and how hard — added 2026-09-07

**One burst may be driven to 1.5. Two may not.** The general ceiling
`BURST_LAM_MAX` is 1.25 and it exists to protect a line carrying two adapters:
two at 1.5 each destroyed the line in 5 of 5 seeds (median Parakeet word error
0.82), while the same script with one at 1.5 came back clean every time. A reply
that tags exactly one burst therefore gets `BURST_LAM_MAX_SOLO` = 1.5 instead.

That raise is **gated on a burst-aware ranker**, and the reason is measured.
Study `vb_opt` (protocol §71) raised the ceiling to 1.5 across the board and got
**+0.0275 on the candidate mean (t 2.31) and +0.0050 on the delivered take
(t 0.15)** — the louder burst was generated and then not chosen, because nothing
in the old reward asked whether the sound was there. With `BON_BURST_WEIGHT` the
ranker can see it, so the raise now reaches a listener. Without best-of-N the
server keeps 1.25.

The practical consequence for a director: **when in doubt write one burst, not
two.** The whole reply shares a merge budget of 2.0, so a second burst does not
add to the first — it divides what the first could have had.

## Borrowing a group's adapter (`bestmem`) — READ THE STACK NOTE

`bestmem` means: for a class with no adapter of its own, load the strongest
adapter of its burst-family group instead. It is available and `skills.py` can
resolve it, but **its sign depends on the stack it is measured in, and that is
the whole story of this entry.**

| measured in | Δ delivered vs shipped | t | n |
|---|--:|--:|--:|
| study `vb_grp`, 4 always-on adapters | **+0.0762** | +3.76 | — |
| study `vb_opt`, production stack, 7 always-on adapters | **−0.0531** | −2.50 | 320 |

Same adapters, same prompts, same metric, same seed rule — **opposite sign**.
The difference is the stack. So:

* **Do not borrow over a class that already has its own measured recipe.** That
  is exactly the substitution `vb_opt` measured, and under the production stack
  it loses (−0.0492 on the candidate mean, t −4.70).
* **Borrowing for a class with NO recipe is a different question and is not
  measured either way.** There the alternative is no burst adapter at all, and
  that arm (`w0`) is the worst measured: **−0.0938 delivered (t −3.96)**. A
  borrowed adapter is the better bet on the evidence available, and it is
  offered for that case only.

This entry replaces the earlier unqualified recommendation to use `bestmem`
generally. The rule both studies now agree on is worth stating once: **the stack
a measurement was made in is part of the measurement.**

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

<!-- vb_cls2:sec64 -->

## Nachtrag §64 (2026-09-04) — der Adapter und seine Stärke, getrennt vom Prompt

Die Tabelle oben variiert die **Prompt-Form** bei festem Adapter. Die Studie `vb_cls2` (Protokoll §64) variiert **Adapter und Skalierungsfaktor** bei fester Prompt-Form, auf 45 Klassen, und trainiert die Adapter dafür neu — teils auf DramaBox-TTS-Audio.

**Vergleichbarkeit, ausdrücklich.** Beide Zahlenmengen sind mit demselben Instrument gemessen (`reward.RewardModel` → `laion/vocal-burst-detector-v2`), und §51 baut seine Grundform so, dass sie den Trägersatz `burst_dose2/prompt_sets.json` byteweise reproduziert — genau den, den §64 benutzt. Instrument und Trägerkorpus sind also gemeinsam. **Der variierte Faktor ist es nicht.** Eine alte und eine neue Zeile vergleichen deshalb zwei *Rezepte als Ganzes*; ein Unterschied darf weder dem Adapter noch dem Prompt allein zugeschrieben werden. Die Herkunft steht in jeder Zeile.

**Ersetzungsregel.** Ein altes Rezept wird nur ersetzt, wenn das neue seine familienweite Trefferquote um mehr als eine Seed-Rausch-Standardabweichung (0.068, in §52 gemessen) schlägt. Beide Zahlen sind Argmaxe über viele Zellen und damit nach oben verzerrt; ohne die Marge hätte eine Klasse bei +0,003 gewechselt.

**Ergebnis der Zusammenführung: 14 alte Rezepte bleiben stehen, 12 werden ersetzt, 5 Klassen wurden nicht neu gemessen, 19 Klassen haben zum ersten Mal ein Rezept** (und noch keine eigene Musterseite).

### Die Zusammenführung, Klasse für Klasse

| Klasse | steht | alt: Form / w / Familie | neu: Adapter / Skript / w / Familie | neu streng | 23-Gruppen | N | Messgerät | Detektor-Boden (streng/Familie, echt) |
|---|---|---|---|--:|--:|--:|---|---|
| `exasperated_sigh` | **neu** | Grundform / 1,25 / **0,450** (§52) | `d2_matched` ⚠ / inline / 1,00 / **0,833** | 0,067 | 0,200 ⚑ | 2 | v2 + x2 | 0,226 / 0,340 ⛔ |
| `relief_sigh` | **alt** | GENERAL-Ursache / 0,80 / **0,750** (§51) | `shipped` / inline / 1,00 / **0,733** | 0,000 | 0,733 | 2 | v2 + x2 | 0,062 / 0,312 ⛔ |
| `contented_sigh` | alt (nicht neu gemessen) | GENERAL-Ursache + längere Dauer / 1,50 / **0,680** (§51) | nicht gemessen | — | — | 3 | v2 (§51/52) | — |
| `cackle` | **neu** | Grundform / 1,80 / **0,540** (§51) | `bulk` ⚠ / solo / 1,50 / **0,633** | 0,067 | 0,100 ⚑ | 3 | v2 | — |
| `nervous_giggle` | **neu** | GENERAL-Ursache + längere Dauer / 1,50 / **0,330** (§52) | `bulk` ⚠ / inline / 1,50 / **0,633** | 0,000 | 0,633 | 3 | v2 | — |
| `guffaw` | **neu** | Grundform / 1,00 / **0,270** (§52) | `bulk` ⚠ (allein) / inline / 1,00 / **0,633** | 0,100 | 0,133 ⚑ | 3 | v2 | — |
| `wistful_sigh` | **neu** | Grundform / 1,00 / **0,550** (§52) | `shipped` / inline / 1,00 / **0,633** | 0,167 | 0,633 | 3 | v2 + x2 | 0,143 / 0,393 ⛔ (n<30) |
| `childlike_giggle` | **neu** | Grundform / 0,80 / **0,470** (§51) | `bulk` ⚠ (allein) / inline / 1,00 / **0,567** | 0,350 | 0,567 | 3 | v2 | — |
| `chuckle` | **alt** | GENERAL-Ursache + längere Dauer / 2,00 / **0,730** (§51) | `shipped` / inline / 2,00 / **0,533** | 0,333 | 0,533 | 2 | v2 + x2 | 0,641 / 0,849 |
| `humming` | **neu** | Grundform / 0,80 / **0,230** (§52) | `bulk` ⚠ / solo / 1,00 / **0,533** | 0,033 | 0,533 | 4 | v2 + x2 | 0,472 / 0,611 |
| `scream` | **neu** | längere Dauer / 1,30 / **0,170** (§51) | `d2_matched` ⚠ (allein) / inline / 1,50 / **0,500** | 0,500 | 0,500 | 4 | v2 + x2 | 0,595 / 0,595 (n<30) |
| `low_mumble` | alt (nicht neu gemessen) | GENERAL-Ursache + längere Dauer / 2,00 / **0,480** (§51) | nicht gemessen | — | — | 4 | v2 (§51/52) | — |
| `soft_hum` | **alt** | GENERAL-Ursache + längere Dauer / 2,00 / **0,680** (§51) | `shipped` / solo / 1,50 / **0,467** | 0,133 | 0,467 | 3 | v2 + x2 | 0,393 / 0,571 (n<30) |
| `snicker` | **neu** | GENERAL-Ursache + längere Dauer / 1,25 / **0,250** (§52) | `bulk` ⚠ / inline / 1,00 / **0,467** | 0,000 | 0,467 | 4 | v2 | — |
| `surprised_gasp` | **neu** | längere Dauer / 1,00 / **0,180** (§51) | `bulk` ⚠ / inline / 1,50 / **0,433** | 0,433 | 0,433 | 5 | v2 | — |
| `breathy_giggle` | **alt** | Grundform / 2,00 / **0,400** (§52) | `shipped` / solo / 1,00 / **0,400** | 0,300 | 0,400 | 5 | v2 + x2 | 0,469 / 0,969 (n<30) |
| `fearful_gasp` | **neu** | GENERAL-Ursache + längere Dauer / 1,25 / **0,280** (§52) | `bulk` ⚠ (allein) / inline / 1,50 / **0,400** | 0,000 | 0,400 | 5 | v2 | — |
| `resonant_hum` | alt (nicht neu gemessen) | Grundform / 1,50 / **0,370** (§52) | nicht gemessen | — | — | 6 | v2 (§51/52) | — |
| `purr` | **neu** | Grundform / 1,50 / **0,250** (§52) | `shipped` (allein) / solo / 0,50 / **0,333** | 0,000 | 0,333 | 6 | v2 | — |
| `ahem` | alt (nicht neu gemessen) | GENERAL-Ursache + längere Dauer / 2,00 / **0,330** (§52 ⚠) | nicht gemessen | — | — | 6 | v2 (§51/52) | — |
| `deep_breath` | **alt** | Grundform / 2,30 / **0,270** (§52) | `shipped` / inline / 2,00 / **0,300** | 0,300 | 0,300 | 8 | v2 + x2 | 0,186 / 0,712 ⛔ |
| `clears_throat` | **alt** | Etikett mitten im Satz / 1,25 / **0,480** (§51) | `bulk` ⚠ / inline / 1,50 / **0,267** | 0,000 | 0,267 | 4 | v2 | — |
| `exhausted_groan` | **alt** | Grundform / 1,80 / **0,220** (§51) | `bulk` ⚠ / solo / 1,50 / **0,267** | 0,267 | 0,267 | 10 | v2 + x2 | 0,484 / 0,871 (n<30) |
| `sharp_inhale` | **alt** | Grundform / 2,30 / **0,380** (§52) | `shipped` / inline / 2,00 / **0,267** | 0,267 | 0,267 | 5 | v2 + x2 | 0,897 / 0,897 |
| `frustrated_groan` | **alt** | GENERAL-Ursache / 2,00 / **0,270** (§51) | `bulk` ⚠ / inline / 1,00 / **0,233** | 0,000 | 0,233 | 8 | v2 + x2 | 0,594 / 0,812 (n<30) |
| `growl` | neu (erstmals) | — | `bulk` ⚠ / inline / 1,50 / **0,233** | 0,000 | 0,000 ⚑ | 9 | v2 | — |
| `cough` | **alt** | Grundform / 1,00 / **0,230** (§52) | `bulk_top1` ⚠ (allein) / solo / 0,25 / **0,233** | 0,067 | 0,233 | 9 | v2 | — |
| `pain_moan` | **alt** | GENERAL-Ursache + längere Dauer / 1,50 / **0,180** (§52) | `bulk` ⚠ / inline / 1,00 / **0,233** | 0,000 | 0,000 ⚑ | 12 | v2 | — |
| `whispered_mumble` | alt (nicht neu gemessen) | Grundform / 0,25 / **0,230** (§52) | nicht gemessen | — | — | 9 | v2 (§51/52) | — |
| `sniff` | **alt** | GENERAL-Ursache + längere Dauer / 2,00 / **0,150** (§52) | `shipped` / solo / 2,00 / **0,200** | 0,000 | 0,000 ⚑ | 15 | v2 | — |
| `snort` | neu (erstmals) | — | `shipped` (allein) / solo / 1,00 / **0,183** | 0,000 | 0,000 ⚑ | 12 | v2 | — |
| `coughing` | **alt** | Grundform / 0,25 / **0,180** (§52 ⚠) | `shipped` / solo / 0,25 / **0,167** | 0,033 | 0,167 | 12 | v2 | — |
| `effort_grunt` | neu (erstmals) | — | `bulk` ⚠ (allein) / inline / 1,00 / **0,167** | 0,033 | 0,033 | 13 | v2 | — |
| `fast_breathing` | neu (erstmals) | — | `bulk` ⚠ / solo / 1,50 / **0,167** | 0,000 | 0,000 ⚑ | 13 | v2 | — |
| `mournful_wail` | neu (erstmals) | — | `bulk` ⚠ (allein) / inline / 1,50 / **0,167** | 0,033 | 0,033 | 13 | v2 | — |
| `affirmative_grunt` | neu (erstmals) | — | `gem` (allein) / inline / 1,00 / **0,133** | 0,000 | 0,000 | 17 | v2 + x2 | 0,700 / 0,767 (n<30) |
| `panting` | neu (erstmals) | — | `bulk` ⚠ / inline / 0,25 / **0,133** | 0,000 | 0,000 | 17 | v2 + x2 | 0,647 / 0,882 (n<30) |
| `pleasure_moan` | neu (erstmals) | — | `bulk` ⚠ / inline / 1,00 / **0,133** | 0,067 | 0,067 | 17 | v2 | — |
| `shriek` | neu (erstmals) | — | `bulk` ⚠ / solo / 1,00 / **0,133** | 0,033 | 0,111 | 17 | v2 | — |
| `displeased_grunt` | **alt** | GENERAL-Ursache + längere Dauer / 2,00 / **0,250** (§52 ⚠) | `shipped` (allein) / solo / 0,50 / **0,100** | 0,000 | 0,100 | 9 | v2 | — |
| `heavy_breathing` | neu (erstmals) | — | `bulk` ⚠ (allein) / inline / 1,50 / **0,100** | 0,000 | 0,000 | 22 | v2 + x2 | 0,120 / 0,640 ⛔ (n<30) |
| `yawn` | neu (erstmals) | — | `shipped` / solo / 2,00 / **0,100** | 0,100 | 0,100 | 22 | v2 + x2 | 0,460 / 0,460 |
| `deep_breathing` | neu (erstmals) | — | `bulk` ⚠ / inline / 0,50 / **0,067** | 0,033 | 0,033 | 34 | v2 | — |
| `gulps` | neu (erstmals) | — | `bulk` ⚠ (allein) / solo / 0,50 / **0,033** | 0,000 | 0,000 | 68 | v2 | — |
| `hiss` | neu (erstmals) | — | `bulk` ⚠ / solo / 1,00 / **0,033** | 0,033 | 0,033 | 68 | v2 | — |
| `normal_breathing` | neu (erstmals) | — | `shipped` / inline / 0,00 / **0,033** | 0,000 | 0,000 | 68 | v2 | — |
| `hiccup` | neu (erstmals) | — | `shipped` / inline / 0,00 / **0,000** | 0,000 | 0,000 | None | v2 | — |
| `hiccups` | neu (erstmals) | — | `shipped` / inline / 0,00 / **0,000** | 0,000 | 0,000 | None | v2 | — |
| `sobs` | neu (erstmals) | — | `shipped` / inline / 0,00 / **0,000** | 0,000 | 0,000 | None | v2 | — |
| `swallows` | neu (erstmals) | — | `shipped` / inline / 0,00 / **0,000** | 0,000 | 0,000 | None | v2 | — |

**Spalte „Messgerät“.** `v2` = `reward.RewardModel` → `laion/vocal-burst-detector-v2`, das Gerät, das die Laufzeit selbst benutzt und mit dem **jede** Zahl in beiden Tabellen gemessen ist. `+ x2` heißt zusätzlich: der Produktionsdetektor `laion/vocal-burst-detector-x2` kennt diese Klasse und hat dasselbe Audio nachgescort (16 der 45 Klassen). Keine Zeile mischt die beiden Geräte.

**Spalte „23-Gruppen“.** Dieselbe Zelle unter dem veröffentlichten 23-Gruppen-Schema (`vm_groups.py` md5 `f83e3850`, `vocal_burst_groups.json` md5 `3e774204`) statt unter der gröberen `burst_family.py` (md5 `19a0607b`). Die grobe Tabelle überschreitet Gruppengrenzen — ihr `groan` umfasst `grunt`, `sigh_neg`, `moan` und `growl`, ihr `breath` umfasst `gasp`, `breath_calm`, `breath_fast` und `sniff` — und ist deshalb **systematisch großzügiger**. ⚑ markiert eine Zeile, in der die Familienzahl um mehr als 0,15 über der Gruppenzahl liegt: dort ist der Laut ein Nachbar, kein Treffer. **30 Klassen erreichen 0,15 familienweit, aber nur 21 unter dem 23-Gruppen-Schema.**

**„Streng“ ist streng bis auf 0,37 %.** `reward._same_class` ist ein Teilstring-Test, ein `Coughing` zählt also bereits als Treffer für `cough`. Über 97.349 Ziel-Cues betrifft das 12 von 3.269 gezählten Erkennungen (cough <- Coughing, coughing <- Cough, deep_breathing <- Deep Breath), und **0** davon überschreiten eine 23-Gruppen-Grenze.

Die Spalte **neu streng** ist die beste strenge Zelle dieser Klasse — sie sitzt oft in einer *anderen* Einstellung als die familienweite daneben; welche, steht auf der jeweiligen Musterseite. **N** gilt für das Rezept, das in Spalte 2 steht.

⛔ = der Produktionsdetektor erkennt diese Klasse auf echter Sprache streng unter 30 %; eine strenge Trefferquote unterhalb dieses Werts misst das Gerät, nicht den Adapter. Für diese Klassen ist die familienweite Quote die primäre Zahl. `(n<30)` = hinter dem Recall stehen in mindestens einer Quelle weniger als 30 ausgelassene Segmente; das ist eine Angabe, keine Messung. Quelle: `laion/vocal-burst-detector-x2`, `production/per_class_recall.json`.

### Derselbe Ton, zwei Messgeräte — und was davon übrig bleibt

Die Maßstab-Auswertung hat jeden erzeugten Clip aufgehoben, also konnte der Produktionsdetektor des Schwesteragenten nachträglich über dasselbe Audio laufen: 22825 Clips aus den 16 Klassen, die er kennt. **Der Lokalisierer ist derselbe und findet dasselbe** (22379 gegen 22453 Spannen, Verhältnis 1,0033), der Vergleich ist also sauber einer des *Benennens*.

**Das zerlegt den gepoolten Gewinn, und er hält der Zerlegung nicht stand:**

* auf den 16 überprüfbaren Klassen ist der Neutrainings-Arm **negativ** — -0,0292 (t -1,28) bei w = 1,0 und -0,0667 (t -2,78) bei w = 2,0 — und **das zweite Gerät bestätigt es unabhängig** auf demselben Audio: -0,008 (t -0,31) bzw. -0,131 (t -4,33);
* der **ganze** gepoolte Gewinn sitzt in den 29 Klassen, die das zweite Gerät nicht kennt: 0,0713 (t 4,44, 69/290) — und **genau dort gibt es kein zweites Messgerät**, mit dem sich prüfen ließe, ob das ein echter Gewinn ist oder eine Eigenheit des ersten Detektors.

Die ehrliche Formulierung lautet daher: *der Gewinn liegt vollständig in den selteneren Klassen, in denen der ausgelieferte Adapter ohnehin fast nichts konnte, und er ist dort nicht unabhängig bestätigt; in den gut belegten Klassen sind sich beide Messgeräte einig, dass das Neutraining nichts bringt oder schadet.* Eine Ausnahme ist groß und auf beiden Geräten dieselbe: `scream` bei `inline`, w = 1,5 — +0,267 (t +2,75) auf `v2`, **+0,400 (t +4,13)** auf `x2`.

**Ändert sich dadurch eine Empfehlung? Gemessen, nicht behauptet.**

Auf den 16 überprüfbaren Klassen wandert die *beste Zelle* bei **14 von 16**, wenn dasselbe Audio mit `x2` statt `v2` bewertet wird. Das allein sagt wenig: ein Argmax über 24 zulässige Zellen zu je 10 Prompts wandert bei fast jeder Störung. Die entscheidende Zahl ist, **was es kostet, bei der hier empfohlenen Zelle zu bleiben**, gemessen mit dem anderen Gerät: im Mittel 0,106, im Median 0,083, im schlimmsten Fall 0,300. Umgekehrt kostet die Empfehlung von `x2`, auf `v2` bewertet, im Mittel 0,100 — **der Tausch ist symmetrisch, keines der beiden Geräte hat das bessere Rezept.** Der Median liegt knapp über einer Seed-Rausch-Standardabweichung (0,068); bei 8 der 16 Klassen liegt er darüber.

Zur Einordnung: auf `v2` liegen im Median nur 2 der ~24 zulässigen Zellen innerhalb einer Seed-Rausch-Standardabweichung der besten, auf `x2` 4. Die Spitze ist also durchaus ausgeprägt — aber *welche* Zelle die Spitze ist, ist es nicht.

**Praktische Folge.** Es wird keine Empfehlung in dieser Datei zurückgezogen: kein zweites Gerät sagt, dass ein empfohlenes Rezept schlecht *ist*, nur dass ein anderes um etwa eine Rauschbreite besser sein könnte. Aber die Zeilen sind zu lesen als **„eine gute Einstellung“, nicht als „die beste“** — und wer eine Klasse produktiv fahren will, gewinnt mehr mit Best-of-N (Spalte `N`) als mit dem Feilen am Gewicht. Was sich wirklich ändert, ist die **Begründung**: der Satz „DramaBox verallgemeinert sich“ gilt nur noch für die nicht überprüfbare Hälfte des Klassenfelds.

### Was der Neutrainings-Arm über alle 45 Klassen wirklich bringt

Gepaart über alle 45 Klassen, neu trainiert (echt + DramaBox) gegen ausgeliefert:

* **familienweit, `inline`, w = 1,0: +0.0356 (t +2.67, 93/450 Prompts besser)**; w = 1,5 +0.0274 (t +2.20). Ohne den Produktions-Stapel +0.0385 (t +3.34).
* **streng ist es ein Nullergebnis**: -0.0015 (t -0.29) bei w = 1,0.
* auf `solo` kippt es bei w = 2,0 ins Negative: -0.0506 (t -3.52).

Der Satz, der daraus folgt: **DramaBox-Audio bringt das Modell dazu, zuverlässiger einen Laut der richtigen *Familie* zu machen — nicht zuverlässiger genau den bestellten Unterlaut.** Der Gewinn sitzt bei eingebetteten Skripten und bei w = 1,0 bis 1,5; bei w = 2,0 und bei allein stehenden Bursts ist er weg.

Zwei Klassen tragen den Effekt sichtbar: `surprised_gasp` (+0,367 streng, t +3,50, 8/10 Prompts, `inline` w = 1,5) und `childlike_giggle` (+0,300, t +2,86). Drei verlieren deutlich: `deep_breath`, `sharp_inhale`, `breathy_giggle` — bei 450 Einzelzellen sind rund 23 Zufallstreffer zu erwarten, deshalb trägt die gepoolte Zeile oben die Antwort und nicht diese Liste.

**Rang 0 gegen Rang ≤ 1 ist ein Nullergebnis** (größtes |t| über alle Zellen: 1,92). Das Umschreiben des Cues auf Rang-1-Zeilen kostet nichts und bringt nichts.

**Die Dosis-Leiter** (nur `scream`/`exasperated_sigh`, DramaBox-Anteil 0 → 10 → 25 → 50 → 100 %): die Kurve springt zwischen 10 % und 25 % an und ist danach flach. 10 % kaufen nichts, 25 % kaufen bei `scream` praktisch den ganzen Effekt (0,333 streng `inline` gegen 0,100 des ausgelieferten Adapters) bei gehaltener WER-Schranke. Die Schranke bricht erst am oberen Ende: die reinen DramaBox-Arme reißen sie bei w = 2,0 (WER bis 2,24 gegen 0,20 Grundlinie). **Wer den Effekt will und den Preis begrenzen: 25 % DramaBox, w = 1,5 bis 2,0.**

`exasperated_sigh` bleibt streng in allen zehn Armen bei 0,000–0,033 — auch im ausgelieferten und im reinen Echt-Arm. Familienweit steht dieselbe Klasse bei 0,833. Der Laut kommt; er bekommt nur den Namen eines Nachbarn, und der Detektor-Boden von 22,6 % erklärt, warum die strenge Zahl das nicht zeigen kann.

### Was sich an der Auswahlregel geändert hat

**Genuineness begrenzt nichts mehr.** Ein Schrei soll nicht wie eine gefasste, natürliche Ansprache klingen; dass die Genuineness bei Burst-Klassen fällt, ist der erwartete Preis und kein Ausschlussgrund. Sie steht neben jeder Zahl, sie wählt kein Gewicht mehr aus. **Die WER-Schranke bleibt** und ist die, die tatsächlich gerissen wurde: gepaarte Δ Parakeet-WER ≤ +0,104 gegen die eigene w = 0-Zelle, auf `inline` zusätzlich absolut ≤ 0,25. Das alte `BURST_LAM = 0.25` war von der Genuineness-Schranke gedeckelt; die klassenweisen Optima liegen jetzt bei 1,0–2,0.

**Hörproben zu jeder Zeile dieser Tabelle**: `~/reports/bericht_vokale_bursts.html` — 1.905 Takes, pro Szene der Take ohne Adapter zuerst und die drei Stärken nebeneinander.

<!-- /vb_cls2:sec64 -->

<!-- vb_grp:2026-09-05 -->

## Nachtrag 2026-09-05 — geliehene Adapter, zwei Sackgassen, und eine Obergrenze für w

> Studie `vb_grp`. Alles hier ist auf dem Trägersatz der Grundform gemessen, gepaart auf
> Prompt-Ebene, Startwert allein vom Prompt-Index abhängig. Gruppen-Schema `vm_groups.py`
> md5 `f83e3850` (23 Gruppen, 117 Mitgliedsnamen).

### 1. Einen Adapter aus der Gruppe leihen — die brauchbarste Neuerung

Den Adapter des **stärksten gemessenen Mitglieds** einer Gruppe für **alle** Mitglieder zu laden
schlägt beides, was bisher da war, und kostet keine GPU-Stunde:

| Vergleich | Form | w | d | t | n |
|---|---|--:|--:|--:|--:|
| geliehen − ausgeliefert | inline | 1,50 | **+0,0762** | **+3,76** | 280 |
| geliehen − nachtrainiert | inline | 1,00 | **+0,0357** | **+2,32** | 280 |

Gemessen über die 28 Klassen der zehn zweiquelligen Gruppen, Produktionsdetektor, Gruppenebene.
Für eine Klasse, deren eigener Adapter schwach ist, deren Gruppe aber ein starkes Mitglied hat,
ist das jetzt das empfohlene Vorgehen. Welchen Adapter man leiht, steht auf jeder Klassenseite.

Klassen, für die eine geliehene Einstellung derzeit die beste gemessene Zelle ist (5):
`childlike_giggle`, `breathy_giggle`, `chuckle`, `displeased_grunt`, `heavy_breathing`.

### 2. Zwei Sackgassen, mit Zahlen, damit sie niemand noch einmal ableitet

**Gepoolte Gruppen-Adapter** — ein Adapter pro Gruppe, trainiert auf den zusammengeworfenen
Zeilen aller Mitglieder — sind **schlechter** als der Klassen-Adapter, gegen alle drei
Vergleichsmaßstäbe:

| Vergleich | Form | w | d | t |
|---|---|--:|--:|--:|
| Gruppe-voll − nachtrainiert | inline | 1,50 | −0,0500 | −2,79 |
| Gruppe-25 % − nachtrainiert | inline | 1,50 | −0,0595 | −3,16 |
| Gruppe-voll − geliehen | inline | 1,50 | −0,0869 | −4,65 |
| Gruppe-25 % − geliehen | inline | 1,50 | −0,0964 | −4,80 |

Über die besten Zellen: Mittel −0,090; besser bei 6 Klassen, schlechter bei 18, gleich bei 4.
Der alte Detektor stimmt zu (−0,098; 3 / 17 / 8). **Mehr Daten aus verwandten Klassen machen
den Adapter nicht besser, sondern schlechter.**

**Mehrere Mitglieds-Adapter bei festem Gesamtbudget kombinieren** ist ebenfalls verlässlich
schlechter als der beste einzelne (Schwesterstudie: negativ in 8 von 8 Feldern, 0 von 120
Klassen-Siegen); alle Adapter gleichzeitig auf vollem Gewicht reißt in 20 von 26 Zellen das
WER-Tor bei Trefferquote 0,000.

### 3. Obergrenze: kein Rezept auf dieser Seite nennt w = 2,0

Über dieselben 28 Klassen, inline, reißen bei w = 2,0 **alle vier** Adapter-Arme das WER-Tor:
nachtrainiert +0,167, geliehen +0,168, Gruppe-voll +0,123, Gruppe-25 % +0,109 — gegen die
Schranke +0,104. Bis w = 1,5 hält jeder Arm. Wo die beste gemessene Zelle einer Klasse früher
bei w = 2,0 lag, nennt ihre Seite die beste Zelle **unter** der Grenze und sagt, was das kostet.

### 4. Abdeckung: jede Bezeichnung, die ein Nutzer anfragen kann, hat jetzt eine Seite

Der Raum ist die Vereinigung aus angeforderten Klassen, Etiketten, die der Detektor vergeben
kann, und Mitgliedern des Gruppen-Schemas: **117 Bezeichnungen**. Vorher gab es 31 Seiten.

| Kategorie | Anzahl | was die Seite sagt |
|---|--:|---|
| trägt (≥ 0,15 auf Gruppenebene) | 24 | Rezept, Gewicht, Best-of-N |
| schwach (0 < Quote < 0,15) | 11 | warum, und ob Modell oder Detektor die Grenze ist |
| funktioniert nicht (0,000) | 10 | dass nichts geht, und was stattdessen |
| nicht einzeln gemessen | 72 | das gemessene Geschwister und dessen Ergebnis |

Eine Seite für eine Klasse, die **nicht** funktioniert, ist kein Mangel — sie ist die
nützlichste Seite im Satz, weil sie jemandem einen verlorenen Tag erspart.

Ohne funktionierendes Rezept, ausdrücklich: `growl`, `gulps`, `hiccup`, `hiccups`, `pain_moan`, `pleasure_moan`, `sniff`, `snort`, `sobs`, `swallows`.
Die Gruppe `sob` (`quiet_sob`, `convulsive_sob`, `trembling_whimper`, `whimper`, `weep`, `sob`,
`sobs`) ist der klarste Fall: semantisch stimmig, gemessen bei 0,000 auf jedem Arm und jedem
Gewicht — **die Gruppierung rettet sie nicht**.

### 5. Wo die Grenze das Messgerät ist, nicht das Modell

Eine gemessene Trefferquote kann den Recall des Detektors nicht wesentlich überschreiten.
Auf echter Sprache, streng:

| Klasse | Detektor-Boden | heißt |
|---|--:|---|
| `relief_sigh` | 0,062 | eine Zahl darüber ist nicht erreichbar |
| `heavy_breathing` | 0,120 | eine Zahl darüber ist nicht erreichbar |
| `wistful_sigh` | 0,143 | eine Zahl darüber ist nicht erreichbar |
| `deep_breath` | 0,186 | eine Zahl darüber ist nicht erreichbar |
| `exasperated_sigh` | 0,226 | eine Zahl darüber ist nicht erreichbar |

Diese fünf lesen auf **Gruppenebene** deutlich besser; genau das ist das Argument für
die Gruppen, und es ist ein Argument über das *Messen*, nicht über das Erzeugen.

<!-- /vb_grp:2026-09-05 -->

<!-- links:2026-09-05 -->

## Wo die Adapter liegen

| Satz | Repository | was drin ist |
|---|---|---|
| **neu, v2** | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2) | 105 Adapter: 45 pro Klasse, 30 pro Gruppe, dazu Ablations- und Dosis-Arme als Beleg |
| ausgeliefert | [`laion/moss-va-sft3-vocal-burst-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters) | 71 Adapter, die Rückfallvariante |

`index.json` im v2-Satz nennt zu jedem Adapter die Zeilenzahl und den synthetischen Anteil.
Jede `vb-*.md` verlinkt unten den für sie zuständigen Adapter direkt.

## Cues immer auf Englisch

**Auch wenn der gesprochene Text deutsch ist.** Nachgeprüft am Korpus, nicht behauptet: die
deutschen Trainingszeilen lauten `Das zerreißt einen einfach, weißt du? (relief sigh)` und
`… sehen sie mich vielleicht endlich. (person whistling to get attention)`. Das gilt für
Burst-Cues **und** für Regieanweisungen.

Der Burst ist eine eigene Klammer; **nie eine Zahl in eine Klammer** — eine runde Klammer mit
Zahl wird zum Burst statt zur Anweisung. Die Dauern setzt der Server. Anleitung für das
Regie-Modell: [`docs/DIRECTOR.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/DIRECTOR.md).

## Zu den Gewichten über 1,5 in der Tabelle

Zwölf Zeilen oben nennen ein Gewicht über 1,5, bis hinauf zu 2,3 — und der Nachtrag vom
5. September sagt, kein Rezept solle 2,0 nennen, weil dort über 28 Klassen **alle vier**
Adapter-Arme die WER-Schranke rissen (+0,109 bis +0,168 gegen +0,104).

**Das ist ein echter Widerspruch, und er wird hier nicht durch Löschen aufgelöst.** Die beiden
Zahlen stammen aus verschiedenen Studien: die Tabellenwerte aus einem Sweep über die *Prompt-Form*,
bei dem die WER **jeder einzelnen Zelle** geprüft wurde und bestand (`sharp_inhale` bei w = 2,3
liegt bei 0,091); der Nachtrag misst gepaart über viele Klassen auf einem anderen Trägersatz.
Die guten Rezepte auf schwächerer Evidenz zu verwerfen wäre falsch — `chuckle` bei w = 2,0 ist
mit 0,73 das beste Rezept im ganzen Satz.

**Praktisch:** wer die Einzelzelle nicht selbst nachgemessen hat, bleibt bei 1,5. Der Server
kennt dafür `MOSS_BURST_LAM_MAX`; auf 1,5 gesetzt erzwingt es die Regel des Nachtrags, in der
Vorgabe serviert er die gemessenen Werte.

<!-- /links:2026-09-05 -->
