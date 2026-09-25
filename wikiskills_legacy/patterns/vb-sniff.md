# vbr/sniff

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vbr/sniff` bei Gewicht **2.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — GENERAL-Zeile nennt Ursache und Wirkung des Lauts, **und** die angegebene Dauer ist verlängert |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.150** | +0.150 (t +2.4, 5/10) |
| Trefferquote, streng | 0.033 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.082** | |
| Genuineness | 2.93 | |
| Blend | 6.10 | |
| WER (Parakeet) | 0.237 | |
| DNSMOS | 2.93 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.15 braucht es **15 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.08): **27 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.033, familien-gelockert bei 0.150.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 0.5: 0.00 · w 1.0: 0.05 · w 1.5: 0.10 · w 2.0: 0.10.

§43 hatte für diese Klasse 1.0 empfohlen und dort 0.100 gemessen.
