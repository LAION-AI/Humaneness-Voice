# vb/wistful_sigh

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/wistful_sigh` bei Gewicht **1.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.550** | +0.117 (t +2.7, 5/10) |
| Trefferquote, streng | 0.067 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.482** | |
| Genuineness | 2.58 | |
| Blend | 5.24 | |
| WER (Parakeet) | 0.107 | |
| DNSMOS | 3.37 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.55 braucht es **3 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.48): **4 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.5: 0.42 · w 1.0: 0.55 · w 1.5: 0.45 · w 2.0: 0.30.

§43 hatte für diese Klasse 1.0 empfohlen und dort 0.633 gemessen.
