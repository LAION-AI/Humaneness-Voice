# vb/fearful_gasp

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/fearful_gasp` bei Gewicht **1.25** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — GENERAL-Zeile nennt Ursache und Wirkung des Lauts, **und** die angegebene Dauer ist verlängert |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.283** | +0.233 (t +2.1, 6/10) |
| Trefferquote, streng | 0.000 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.215** | |
| Genuineness | 2.52 | |
| Blend | 4.79 | |
| WER (Parakeet) | 0.412 | |
| DNSMOS | 2.71 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.28 braucht es **7 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.22): **10 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.000, familien-gelockert bei 0.283.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 0.8: 0.17 · w 1.25: 0.22 · w 1.8: 0.17 · w 2.3: 0.04.

§43 hatte für diese Klasse 1.25 empfohlen und dort 0.296 gemessen.
