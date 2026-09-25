# vbr/coughing

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vbr/coughing` bei Gewicht **0.25** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.183** | -0.050 (t -1.4, 1/10) |
| Trefferquote, streng | 0.000 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.115** | |
| Genuineness | 3.22 | |
| Blend | 6.06 | |
| WER (Parakeet) | 0.657 | |
| DNSMOS | 2.97 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.18 braucht es **12 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.12): **19 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: dieses Rezept kostet die Sprache

Bei diesem Gewicht liegt die Parakeet-WER bei 0.657 gegen 0.270 ohne Adapter.
Das Modell erkauft den Laut damit, dass es den Satz **nicht mehr spricht**. Es gibt
für diese Klasse keine Stärke, die den Laut über die Schwelle bringt, ohne die Worte
zu beschädigen. Nur einsetzen, wenn der Laut allein stehen darf.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.000, familien-gelockert bei 0.183.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 0.25: 0.18 · w 0.8: 0.12 · w 1.3: 0.08.

§43 hatte für diese Klasse 0.25 empfohlen und dort 0.167 gemessen.
