# vb/deep_breath

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/deep_breath` bei Gewicht **2.3** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.267** | +0.250 (t +4.0, 7/10) |
| Trefferquote, streng | 0.233 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.199** | |
| Genuineness | 2.45 | |
| Blend | 6.20 | |
| WER (Parakeet) | 0.157 | |
| DNSMOS | 2.74 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.27 braucht es **8 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.20): **11 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.8: 0.05 · w 1.25: 0.13 · w 1.8: 0.13 · w 2.3: 0.27.

§43 hatte für diese Klasse 1.25 empfohlen und dort 0.167 gemessen.
