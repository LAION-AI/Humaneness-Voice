# vb/wistful_sigh

> Gemessen 2026-09-02 im vollen Server-Stapel, gepaart gegen dieselben 10 Trägersätze
> mit gleichem Seed. Protokoll §52, Studie `~/reports/burst_ext.md`.

## Rezept

| | |
|---|---|
| Skriptart | **eingebettet** |
| Burst-Adapter | `vb/wistful_sigh` bei Gewicht **1.0** |
| Burst+Stop-DPO | keiner |
| Prompt-Form | **P0** — Grundform aus §43 — Hinweis als Etikett, Standarddauer |

## Was dabei herauskommt

| Maß | Wert | gegen Grundform |
|---|--:|--:|
| Trefferquote, Familie | **0.550** | +0.117 (t +2.7, 5/10) |
| Trefferquote, streng | 0.067 | — |
| konservativ (−1 sd Seed-Rauschen) | **0.482** | |
| Genuineness | 2.58 | |
| Blend | 5.24 | |
| WER (Parakeet) | 0.107 | |
| DNSMOS | 3.37 | |

## Wie oft erzeugen

Bei einer Trefferquote von 0.55 braucht es **3 Kandidaten**, damit mit 90 %
Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0.48): **4 Kandidaten**.

Auswahl unter den Treffern nach Klang; wenn mehrere sitzen, den mit der höchsten
Genuineness nehmen. Best-of-N ist hier der wirksamere Hebel als jede Stärkeänderung.

## Der Verlauf über die Stärke

Grundform: w 0.5: 0.42 · w 1.0: 0.55 · w 1.5: 0.45 · w 2.0: 0.30.

§43 hatte für diese Klasse 1.0 empfohlen und dort 0.633 gemessen.

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
| Skalierungsfaktor | **1,00** |
| Skriptart | **eingebettet** |
| Stapel | voller Produktions-Stapel |
| Trefferquote, Familie (grobe Tabelle) | **0,633** |
| dieselbe Zelle im **veröffentlichten 23-Gruppen-Schema** | 0,633 — bestes Rezept dort: `shipped` / inline / w = 1,00 |
| Trefferquote, streng | 0,167 — aber in einer anderen Zelle: `shipped`, allein stehend, w = 2,00 |
| Genuineness | 2,63 |
| WER (Parakeet) | 0,115, gepaart +0,024 gegen die eigene w = 0-Zelle |
| Detektor-Boden (echte Sprache) | streng 0,143, Familie 0,393 — **nicht belastbar** (n < 30) |

**Die strenge Quote ist hier nach oben gedeckelt.** Der Produktionsdetektor erkennt diese Klasse auf echter Sprache nur zu 14,3 % (familienweit 39,3 %). Eine strenge Trefferquote unterhalb dieses Werts ist eine **Ablesung am Messgerät, kein Urteil über den Adapter**. Für diese Klasse ist die familienweite Quote die primäre Zahl.

**Entscheidung: das neue Rezept ersetzt das alte.** 0,633 gegen 0,550 familienweit, also mehr als eine Seed-Rausch-Standardabweichung (0,068) besser.

### Wie oft erzeugen (neu gerechnet)

Bei einer Trefferquote von 0,63 braucht es **3 Kandidaten**, damit mit 90 % Wahrscheinlichkeit mindestens einer den Laut realisiert. Konservativ gerechnet (0,57, eine Seed-Rausch-Standardabweichung abgezogen): **3 Kandidaten**.

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
| Adapter | der nachtrainierte Klassen-Adapter (`vb_cls2/loras/bulk_mix_full`) |
| Gewicht | **1,00** |
| Skriptart | **eingebettet (inline)** |
| Best-of-N | **22** Kandidaten für 90 % (konservativ, eine Seed-Rausch-Standardabweichung abgezogen: 71) |

Best-of-N ist hier der größere Hebel als jede Stärkeänderung.


⚠ **Lizenz.** Dieser Adapter ist auf DramaBox-TTS-Audio mittrainiert; die LTX-2 Community Licence ist dafür **nicht geprüft** (§61). Solange das offen ist, bleibt der ausgelieferte Adapter (`shipped`) die freigegebene Rückfallvariante.

## Was dabei herauskommt

Gemessen mit `laion/vocal-burst-detector-x2` (Produktion, VoiceCLAP-Ensemble, 16 Klassen), Gruppenebene nach dem 23-Gruppen-Schema.

| Maß | Wert |
|---|--:|
| Trefferquote, **streng** (exakt dieselbe Klasse) | **0,067** |
| Trefferquote, **Gruppe** (irgendein Mitglied) | **0,100** |
| größengleiche Zufallskontrolle | 0,091 |
| netto über der Zufallskontrolle | 0,009 |
| Genuineness (nur berichtet, kein Tor) | 1,81 |
| WER (Parakeet), absolut | 0,051 |
| WER gepaart gegen die eigene w = 0-Zelle | -0,041 (Grenze +0,104) |
| **Detektor-Boden**, echte Sprache, streng | 0,143 ⛔ *nicht belastbar, n < 30* |

Dieselbe Klasse, die drei Wege nebeneinander (jeweils die beste Zelle unter w ≤ 1,5):

| Weg | Adapter | Form / w | Gruppe | streng |
|---|---|---|--:|--:|
| eigener Adapter | `percls` | inline / 1,00 | **0,100** | 0,067 |
| geliehen vom Gruppenmitglied | `bestmem` | inline / 1,00 | **0,100** | 0,067 |
| Gruppen-Adapter (gepoolt) | `grp25` | inline / 0,50 | **0,067** | 0,067 |

Geliehen gegen eigen, gepaart bei gleicher Form und gleichem Gewicht: d = 0,0000, t = —, n = 10.

## Warum die Zahl so ist

**Die Grenze ist hier das Messgerät, nicht das Modell.** Der Produktionsdetektor erkennt diese Klasse auf echter Sprache nur zu 0,143. Eine gemessene Trefferquote wesentlich über 0,143 ist deshalb gar nicht erreichbar, wie gut der Adapter auch sein mag. Die Zahl oben ist eine Untergrenze für das, was tatsächlich erzeugt wird.

## Wenn es nicht geht

Wenn dieses Etikett nicht trifft: die Gruppe `sigh_pos` enthält `relief_sigh`. Der Adapter des stärksten gemessenen Mitglieds — `vb/wistful_sigh` — ist für die ganze Gruppe eine belegte Alternative (dort gemessen: 0,200).

## Woher die Zahlen kommen

* **Messgerät:** `laion/vocal-burst-detector-x2` (Produktion, VoiceCLAP-Ensemble, 16 Klassen). Die beiden Detektoren sind **nicht** austauschbar; keine Zeile mischt sie.
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
| **eigener Adapter, nachtrainiert** | `wistful_sigh` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/per_class/wistful_sigh) |
| eigener Adapter, nur Erstwahl-Zeilen | `wistful_sigh` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/per_class_top1/wistful_sigh) |
| **ausgeliefert** (bisheriger Satz, Rückfallvariante) | `wistful_sigh` | [`laion/moss-va-sft3-vocal-burst-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters/tree/main/adapters/wistful_sigh) |
| Gruppen-Adapter, volle Dosis — Gruppe `sigh_pos` | `sigh_pos` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/groups_full/sigh_pos) |
| Gruppen-Adapter, 25 % synthetisch — Gruppe `sigh_pos` | `sigh_pos` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/groups_dose25/sigh_pos) |

> **Gewicht beachten.** Rang 16, Alpha 32 — die eingebaute Skalierung ist also 2,0, und das Gewicht `w` multipliziert sie. `w = 1,0` bedeutet demnach bereits Verstärkung 2,0.

## Wie der Cue geschrieben wird

**Cues stehen immer auf Englisch, auch wenn der gesprochene Text deutsch ist.** Das ist keine Stilfrage, sondern die Schreibweise der Trainingsdaten: die deutschen Zeilen im Korpus lauten `Das zerreißt einen einfach, weißt du? (relief sigh)`. Ein deutscher Cue ist außerhalb der Verteilung und verhält sich unvorhersehbar.

* Der Burst ist **eine eigene Klammer** zwischen den Wörtern — `(clearly amused) … (wistful sigh) …`. Innerhalb einer Regieanweisung genannt entsteht **kein** Laut.

* **Nie eine Zahl in eine Klammer schreiben.** Eine runde Klammer mit Zahl hört auf, eine Anweisung zu sein, und wird zum Burst. Die Dauern rechnet der Server aus (`(label, N.N seconds)`, Vorgabe 0,28 s, zulässig 0,14–1,2).

* Eckige Klammern nur für Pausen: `[pause]`, `[long pause]`. Keine Großbuchstaben — das Modell buchstabiert sie.

Vollständige Anleitung für das Regie-Sprachmodell: [`docs/DIRECTOR.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/DIRECTOR.md) · Lademechanik: [`docs/ADAPTERS.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/ADAPTERS.md)

<!-- /links:2026-09-05 -->
