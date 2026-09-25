# vb/displeased_grunt

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/displeased_grunt` bei Gewicht **2.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — GENERAL-Zeile nennt Ursache und Wirkung des Lauts, **und** die angegebene Dauer ist verlängert |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.250** | +0.250 (t +1.0, 1/4) |
| Trefferquote, streng | 0.000 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.182** | |
| Genuineness | 2.81 | |
| Blend | 4.65 | |
| WER (Parakeet) | 1.000 | |
| DNSMOS | 2.69 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.25 braucht es **9 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.18): **12 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: dieses Rezept kostet die Sprache

Bei diesem Gewicht liegt die Parakeet-WER bei 1.000 gegen 0.283 ohne Adapter.
Das Modell erkauft den Laut damit, dass es den Satz **nicht mehr spricht**. Es gibt
für diese Klasse keine Stärke, die den Laut über die Schwelle bringt, ohne die Worte
zu beschädigen. Nur einsetzen, wenn der Laut allein stehen darf.

## Achtung: die Unterart trifft er nicht

Streng gemessen liegt diese Klasse bei 0.000, familien-gelockert bei 0.250.
Das Modell erzeugt zuverlässig einen Laut **derselben Familie**, aber nicht die
etikettierte Unterart. Wer auf genau dieser Unterart besteht, wird enttäuscht — den
gewünschten Charakter deshalb über die Emotionsanweisung steuern, nicht über das Etikett.

## Der Verlauf über die Stärke

Grundform: w 1.0: 0.12 · w 1.5: 0.23 · w 2.0: 0.17 · w 2.5: 0.19.

§43 hatte für diese Klasse 1.5 empfohlen und dort 0.208 gemessen.
