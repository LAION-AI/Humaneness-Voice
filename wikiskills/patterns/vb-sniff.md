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

<!-- vb_cls2:sec64 -->

## Nachtrag §64 (2026-09-04) — Adapter und Stärke, neu gemessen

> Studie `vb_cls2`, Bericht mit Hörproben: `~/reports/bericht_vokale_bursts.html`.
> Variiert wurden **Adapter und Skalierungsfaktor** auf dem Trägersatz der
> Grundform, gemessen mit demselben Detektor (`vocal-burst-detector-v2`) wie oben.
> Die alte Messung variiert die **Prompt-Form**. Beide Zahlen stehen auf demselben
> Instrument und demselben Trägerkorpus, aber sie variieren **verschiedene**
> Faktoren: ein Unterschied ist ein Unterschied zwischen zwei *Rezepten als
> Ganzem* und darf keinem einzelnen Faktor zugeschrieben werden.

| | |
|---|---|
| bester Adapter | der ausgelieferte Adapter |
| Skalierungsfaktor | **2,00** |
| Skriptart | **allein stehend** |
| Stapel | voller Produktions-Stapel |
| Trefferquote, Familie (grobe Tabelle) | **0,200** |
| dieselbe Zelle im **veröffentlichten 23-Gruppen-Schema** | 0,000 — bestes Rezept dort: `bulk` / inline / w = 0,00 |
| Trefferquote, streng | 0,000 — in **keiner** Zelle über null |
| Genuineness | 2,85 |
| WER (Parakeet) | 0,387, gepaart +0,087 gegen die eigene w = 0-Zelle |

**Entscheidung: die alte Prompt-Strategie bleibt stehen.** Sie misst 0,150 familienweit gegen 0,200 für das beste neue Adapter/Stärke-Paar. Ersetzt wird nur, was die alte Zahl um mehr als eine Seed-Rausch-Standardabweichung (0,068, §52) schlägt; das ist hier nicht der Fall. Das oben stehende Rezept gilt weiter.

### Wie oft erzeugen (neu gerechnet)

Es gilt weiter die Rechnung des alten Rezepts oben (15 Kandidaten, konservativ 27). Zum Vergleich: bei der neu gemessenen Quote von 0,20 wären es **11 Kandidaten**, konservativ (0,13) **17**.

Auswahl unter den Treffern nach Klang. Best-of-N bleibt der wirksamere Hebel als jede Stärkeänderung.

### Woher die Zahlen kommen

* **Messgerät:** `reward.RewardModel -> laion/vocal-burst-detector-v2` — dasselbe, das die Laufzeit selbst benutzt, und dasselbe, mit dem die Tabelle oben gemessen wurde.
* **Zweites Messgerät zur Kontrolle:** `laion/vocal-burst-detector-x2 (production, VoiceCLAP ensemble)`. Es kennt 16 der 45 Klassen; wo es diese Klasse kennt, steht sein Urteil in `VOCAL_BURSTS.md`. Keine Zeile mischt die beiden.
* **„Familie“** ist `burst_family.py` (md5 `19a0607b`), eine **gröbere** Einteilung als das veröffentlichte 23-Gruppen-Schema (`vm_groups.py` md5 `f83e3850`) — sie überschreitet dessen Grenzen, ist also systematisch großzügiger. Deshalb steht beides da.
* **„Streng“ ist streng bis auf 0,37 %:** `reward._same_class` ist ein Teilstring-Test, ein `Coughing` zählt also als Treffer für `cough`. Über 97.349 Ziel-Cues betrifft das 12 von 3.269 gezählten Erkennungen, **0** davon überschreiten eine 23-Gruppen-Grenze.

<!-- /vb_cls2:sec64 -->

<!-- vb_grp:2026-09-05 -->

## Nachtrag 2026-09-05 — Gruppen, geliehene Adapter, WER-Obergrenze

> Studie `vb_grp`. **Das bestehende Rezept oben bleibt gültig**, wenn es besser gemessen hat: die älteren Studien variieren die *Prompt-Form* und werden von einer Adapter-Messung nicht überholt. Neu ist hier, was die Gruppe beiträgt und wo die WER-Obergrenze liegt.

## Rezept

| | |
|---|---|
| Adapter | der ausgelieferte Klassen-Adapter |
| Gewicht | **0,50** |
| Skriptart | **eingebettet (inline)** |
| Best-of-N | **—** Kandidaten für 90 % (konservativ, eine Seed-Rausch-Standardabweichung abgezogen: 2302) |

Best-of-N ist hier der größere Hebel als jede Stärkeänderung.

## Was dabei herauskommt

Gemessen mit `laion/vocal-burst-detector-v2` (83 Klassen, das Instrument der Laufzeit), Gruppenebene nach dem 23-Gruppen-Schema.

| Maß | Wert |
|---|--:|
| Trefferquote, **streng** (exakt dieselbe Klasse) | **0,000** |
| Trefferquote, **Gruppe** (irgendein Mitglied) | **0,000** |
| größengleiche Zufallskontrolle | 0,008 |
| netto über der Zufallskontrolle | -0,008 |
| Genuineness (nur berichtet, kein Tor) | 2,42 |
| WER (Parakeet), absolut | 0,120 |
| WER gepaart gegen die eigene w = 0-Zelle | -0,055 (Grenze +0,104) |

Dieselbe Klasse, die drei Wege nebeneinander (jeweils die beste Zelle unter w ≤ 1,5):

| Weg | Adapter | Form / w | Gruppe | streng |
|---|---|---|--:|--:|
| eigener Adapter | `shipped` | inline / 0,50 | **0,000** | 0,000 |
| Gruppen-Adapter (gepoolt) | `grpfull` | inline / 0,50 | **0,000** | 0,000 |

## Warum die Zahl so ist

**Hier funktioniert nichts.** Kein Adapter, kein Gewicht und keine Skriptart bringt diese Klasse über 0,000 — und die Gruppierung rettet sie nicht: auch auf Gruppenebene bleibt es bei 0,000. Das ist ein Ergebnis, kein fehlender Messwert.

## Wenn es nicht geht

Nichts an diesem Etikett funktioniert. Die ehrliche Empfehlung ist, den Laut **nicht über das Burst-Etikett** anzufordern, sondern den Charakter über die Emotionsanweisung zu steuern.
 In derselben Gruppe `sniff` sind gemessen: `snort` — dort lohnt der Blick.


## Woher die Zahlen kommen

* **Messgerät:** `laion/vocal-burst-detector-v2` (83 Klassen, das Instrument der Laufzeit). Die beiden Detektoren sind **nicht** austauschbar; keine Zeile mischt sie.
* **Gruppen-Schema:** `vm_groups.py` md5 `f83e3850` (23 Gruppen, 117 Mitgliedsnamen).
* **Trägersatz:** `$SC/out/burst_dose2/prompt_sets.json`, 10 Prompts je (Skriptart, Klasse), Startwert hängt allein vom Prompt-Index ab — alle Arme ziehen dasselbe Rauschen.
* **Statistik:** die drei Stichproben eines Prompts werden zuerst gemittelt, n ist die Zahl der Prompts.
* **WER-Obergrenze:** w = 2,0 ist ausgeschlossen. Über die 28 Klassen der zehn zweiquelligen Gruppen reißen dort **alle vier** Adapter-Arme das Tor (nachtrainiert +0,167, geliehen +0,168, Gruppe-voll +0,123, Gruppe-25 % +0,109 gegen die Schranke +0,104). Bis w = 1,5 hält jeder Arm.
* Studie `vb_grp`, Zustand `~/reports/STATE_burst_group_lora.md`.

<!-- /vb_grp:2026-09-05 -->

<!-- links:2026-09-05 -->

## Woher der Adapter kommt

Die Namen in den Rezepten oben (`bestmem`, `grpfull`, `bulk_mix_full` …) sind Studien-Arme, keine Dateien. Hier stehen die tatsächlich abrufbaren Adapter.

| Rolle | Adapter | abrufen |
|---|---|---|
| **eigener Adapter, nachtrainiert** | `sniff` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/per_class/sniff) |
| **ausgeliefert** (bisheriger Satz, Rückfallvariante) | `sniff` | [`laion/moss-va-sft3-vocal-burst-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters/tree/main/adapters/sniff) |
| Gruppen-Adapter, volle Dosis — Gruppe `sniff` | `sniff` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/groups_full/sniff) |

> **Gewicht beachten.** Rang 16, Alpha 32 — die eingebaute Skalierung ist also 2,0, und das Gewicht `w` multipliziert sie. `w = 1,0` bedeutet demnach bereits Verstärkung 2,0.

## Wie der Cue geschrieben wird

**Cues stehen immer auf Englisch, auch wenn der gesprochene Text deutsch ist.** Das ist keine Stilfrage, sondern die Schreibweise der Trainingsdaten: die deutschen Zeilen im Korpus lauten `Das zerreißt einen einfach, weißt du? (relief sigh)`. Ein deutscher Cue ist außerhalb der Verteilung und verhält sich unvorhersehbar.

* Der Burst ist **eine eigene Klammer** zwischen den Wörtern — `(clearly amused) … (sniff) …`. Innerhalb einer Regieanweisung genannt entsteht **kein** Laut.

* **Nie eine Zahl in eine Klammer schreiben.** Eine runde Klammer mit Zahl hört auf, eine Anweisung zu sein, und wird zum Burst. Die Dauern rechnet der Server aus (`(label, N.N seconds)`, Vorgabe 0,28 s, zulässig 0,14–1,2).

* Eckige Klammern nur für Pausen: `[pause]`, `[long pause]`. Keine Großbuchstaben — das Modell buchstabiert sie.

Vollständige Anleitung für das Regie-Sprachmodell: [`docs/DIRECTOR.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/DIRECTOR.md) · Lademechanik: [`docs/ADAPTERS.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/ADAPTERS.md)

<!-- /links:2026-09-05 -->
