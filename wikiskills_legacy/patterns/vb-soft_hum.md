# vb/soft_hum

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/soft_hum` bei Gewicht **2** |
| Burst+Stop-DPO | 896 |
| Prompt-Form | **C_gl** — C_gl |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.683** | +0.467 (t +5.7, 9/10) |
| Trefferquote, streng | 0.300 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.615** | |
| Genuineness | 3.02 | |
| Blend | 4.67 | |
| WER (Parakeet) | 0.717 | |
| DNSMOS | 2.91 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.68 braucht es **3 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.61): **3 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
