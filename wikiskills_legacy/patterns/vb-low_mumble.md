# vb/low_mumble

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/low_mumble` bei Gewicht **2** |
| Burst+Stop-DPO | 896 |
| Prompt-Form | **C_gl** — C_gl |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.483** | +0.183 (t +1.9, 6/10) |
| Trefferquote, streng | 0.450 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.415** | |
| Genuineness | 3.63 | |
| Blend | 4.02 | |
| WER (Parakeet) | 0.627 | |
| DNSMOS | 3.20 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.48 braucht es **4 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.41): **5 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
