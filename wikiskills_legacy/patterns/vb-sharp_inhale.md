# vb/sharp_inhale

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/sharp_inhale` bei Gewicht **2.3** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.383** | +0.317 (t +3.1, 8/10) |
| Trefferquote, streng | 0.383 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.315** | |
| Genuineness | 2.66 | |
| Blend | 5.03 | |
| WER (Parakeet) | 0.091 | |
| DNSMOS | 3.27 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.38 braucht es **5 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.32): **7 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.8: 0.15 · w 1.25: 0.18 · w 1.8: 0.28 · w 2.3: 0.38.

§43 hatte für diese Klasse 1.25 empfohlen und dort 0.167 gemessen.
