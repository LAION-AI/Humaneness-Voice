# vb/surprised_gasp

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/surprised_gasp` bei Gewicht **1** |
| Burst+Stop-DPO | 616 |
| Prompt-Form | **P3** — längere angegebene Dauer (+1,0 s) |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.183** | +0.117 (t +2.1, 4/10) |
| Trefferquote, streng | 0.167 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.115** | |
| Genuineness | 3.31 | |
| Blend | 5.42 | |
| WER (Parakeet) | 0.357 | |
| DNSMOS | 2.87 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.18 braucht es **12 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.11): **19 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
