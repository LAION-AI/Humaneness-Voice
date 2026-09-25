# vb/scream

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/scream` bei Gewicht **1.3** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P3** — längere angegebene Dauer (+1,0 s) |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.167** | +0.167 (t +4.7, 8/10) |
| Trefferquote, streng | 0.150 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.098** | |
| Genuineness | 2.66 | |
| Blend | 4.10 | |
| WER (Parakeet) | 0.243 | |
| DNSMOS | 2.92 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.17 braucht es **13 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.10): **23 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
