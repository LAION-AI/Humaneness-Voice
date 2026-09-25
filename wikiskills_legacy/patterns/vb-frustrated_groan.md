# vb/frustrated_groan

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/frustrated_groan` bei Gewicht **2** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P1** — GENERAL-Zeile nennt Ursache + Wirkung des Lauts |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.267** | +0.250 (t +2.4, 5/10) |
| Trefferquote, streng | 0.000 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.198** | |
| Genuineness | 2.84 | |
| Blend | 5.11 | |
| WER (Parakeet) | 0.360 | |
| DNSMOS | 3.05 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.27 braucht es **8 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.20): **11 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
