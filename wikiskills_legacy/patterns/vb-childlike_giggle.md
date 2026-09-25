# vb/childlike_giggle

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §51, Studie `~/reports/burst_levers.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/childlike_giggle` bei Gewicht **0.8** |
| Burst+Stop-DPO | 336 |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.467** | +0.250 (t +2.8, 7/10) |
| Trefferquote, streng | 0.050 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.398** | |
| Genuineness | 2.83 | |
| Blend | 3.89 | |
| WER (Parakeet) | 0.122 | |
| DNSMOS | 3.36 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.47 braucht es **4 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.40): **5 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.
