# vbr/pain_moan

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vbr/pain_moan` bei Gewicht **1.5** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — GENERAL-Zeile nennt Ursache und Wirkung des Lauts, **und** die angegebene Dauer ist verlängert |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.183** | +0.167 (t +2.1, 4/10) |
| Trefferquote, streng | 0.000 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.115** | |
| Genuineness | 2.43 | |
| Blend | 6.18 | |
| WER (Parakeet) | 0.293 | |
| DNSMOS | 2.78 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.18 braucht es **12 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.12): **19 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.000, familien-gelockert bei 0.183.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 0.5: 0.05 · w 1.0: 0.15 · w 1.5: 0.13 · w 2.0: 0.12.

§43 hatte für diese Klasse 1.0 empfohlen und dort 0.133 gemessen.
