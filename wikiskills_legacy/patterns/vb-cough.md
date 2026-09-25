# vb/cough

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/cough` bei Gewicht **1.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.233** | +0.117 (t +2.1, 4/10) |
| Trefferquote, streng | 0.017 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.165** | |
| Genuineness | 3.54 | |
| Blend | 4.32 | |
| WER (Parakeet) | 0.337 | |
| DNSMOS | 2.88 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.23 braucht es **9 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.17): **13 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.017, familien-gelockert bei 0.233.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 1.0: 0.23 · w 1.5: 0.22 · w 2.0: 0.19 · w 2.5: 0.41.

§43 hatte für diese Klasse 1.5 empfohlen und dort 0.133 gemessen.
