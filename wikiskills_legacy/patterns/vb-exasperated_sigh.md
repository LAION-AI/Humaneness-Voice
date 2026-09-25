# vb/exasperated_sigh

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/exasperated_sigh` bei Gewicht **1.25** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.450** | +0.133 (t +1.6, 7/10) |
| Trefferquote, streng | 0.050 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.382** | |
| Genuineness | 2.36 | |
| Blend | 5.46 | |
| WER (Parakeet) | 0.110 | |
| DNSMOS | 3.29 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.45 braucht es **4 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.38): **5 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.8: 0.42 · w 1.25: 0.45 · w 1.8: 0.37 · w 2.3: 0.38.

§43 hatte für diese Klasse 1.25 empfohlen und dort 0.467 gemessen.
