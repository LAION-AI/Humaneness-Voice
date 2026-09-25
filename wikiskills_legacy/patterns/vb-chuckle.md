# vb/chuckle

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/chuckle` bei Gewicht **2** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — C_gl |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.725** | +0.392 (t +3.0, 7/10) |
| Trefferquote, streng | 0.377 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.657** | |
| Genuineness | 3.76 | |
| Blend | 3.58 | |
| WER (Parakeet) | 0.581 | |
| DNSMOS | 2.94 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.73 braucht es **2 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.66): **3 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
