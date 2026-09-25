# vb/ahem

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **allein stehend** |
| Burst-Adapter | `vb/ahem` bei Gewicht **2.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **C_gl** — GENERAL-Zeile nennt Ursache und Wirkung des Lauts, **und** die angegebene Dauer ist verlängert |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.327** | +0.193 (t +4.3, 8/10) |
| Trefferquote, streng | 0.327 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.259** | |
| Genuineness | 3.52 | |
| Blend | 4.95 | |
| WER (Parakeet) | 0.692 | |
| DNSMOS | 2.99 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.33 braucht es **6 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.26): **8 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Achtung: dieses Rezept kostet die Sprache

Bei diesem Gewicht liegt die Parakeet-WER bei 0.692 gegen 0.347 ohne Adapter.
Das Modell erkauft den Laut damit, dass es den Satz **nicht mehr spricht**. Es gibt
für diese Klasse keine Stärke, die den Laut über die Schwelle bringt, ohne die Worte
zu beschädigen. Nur einsetzen, wenn der Laut allein stehen darf.

## Der Verlauf über die Stärke

Grundform: w 1.0: 0.13 · w 1.5: 0.12 · w 2.0: 0.19 · w 2.5: 0.22.

§43 hatte für diese Klasse 1.5 empfohlen und dort 0.233 gemessen.
