# vb/exhausted_groan

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/exhausted_groan` bei Gewicht **1.8** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.217** | +0.150 (t +1.8, 4/10) |
| Trefferquote, streng | 0.217 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.148** | |
| Genuineness | 3.08 | |
| Blend | 5.80 | |
| WER (Parakeet) | 0.470 | |
| DNSMOS | 3.06 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.22 braucht es **10 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.15): **15 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
