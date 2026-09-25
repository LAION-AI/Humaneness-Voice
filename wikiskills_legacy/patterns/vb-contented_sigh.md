# vb/contented_sigh

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/contented_sigh` bei Gewicht **1.5** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — C_gl |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.683** | +0.233 (t +2.1, 7/10) |
| Trefferquote, streng | 0.683 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.615** | |
| Genuineness | 2.55 | |
| Blend | 5.16 | |
| WER (Parakeet) | 0.126 | |
| DNSMOS | 3.32 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.68 braucht es **3 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.61): **3 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
