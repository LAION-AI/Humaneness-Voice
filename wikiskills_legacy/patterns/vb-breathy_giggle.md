# vb/breathy_giggle

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/breathy_giggle` bei Gewicht **2.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.400** | +0.100 (t +1.4, 4/10) |
| Trefferquote, streng | 0.233 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.332** | |
| Genuineness | 3.41 | |
| Blend | 5.21 | |
| WER (Parakeet) | 0.313 | |
| DNSMOS | 3.00 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.40 braucht es **5 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.33): **6 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.5: 0.33 · w 1.0: 0.37 · w 1.5: 0.32 · w 2.0: 0.40.

§43 hatte für diese Klasse 1.0 empfohlen und dort 0.400 gemessen.
