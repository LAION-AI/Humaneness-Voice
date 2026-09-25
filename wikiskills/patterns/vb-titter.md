<!-- vb_grp:2026-09-05 -->

# vb/titter

> Stand 5. September 2026 · Gruppe `laugh_soft` im 23-Gruppen-Schema · **nicht einzeln gemessen** · eigener Adapter vorhanden: nein
>
> Gruppen-Mitglieder: `breathy_giggle`, `childlike_giggle`, `chuckle`, `chuckling`, `giggle`, `nervous_giggle`, `snicker`, `snickering`, `snickering_giggle`, `snorting_giggle`, `titter`

## Rezept

Diese Bezeichnung wurde **nicht einzeln gemessen**. Sie gehört zur Gruppe `laugh_soft`; dort sind gemessen: `breathy_giggle`, `childlike_giggle`, `chuckle`, `nervous_giggle`, `snicker`.


Wer diesen Laut braucht, nimmt den Adapter des stärksten gemessenen Geschwisters, `vb/nervous_giggle`, mit der dort belegten Einstellung — siehe `vb-nervous_giggle.md` (dort gemessen: 0,900 auf Gruppenebene).


## Warum die Zahl so ist

Für diese Bezeichnung selbst gibt es **keine Messung**. Was unten steht, gilt für das gemessene Gruppenmitglied, nicht für dieses Etikett. Ob der Detektor diese Schreibweise überhaupt jemals vergibt, ist ebenfalls ungeprüft.

## Wenn es nicht geht

Wenn dieses Etikett nicht trifft: die Gruppe `laugh_soft` enthält `breathy_giggle`, `childlike_giggle`, `chuckle`, `nervous_giggle`, `snicker`. Der Adapter des stärksten gemessenen Mitglieds — `vb/nervous_giggle` — ist für die ganze Gruppe eine belegte Alternative (dort gemessen: 0,900).

## Woher die Zahlen kommen

* **Messgerät:** `laion/vocal-burst-detector-x2` (Produktion, VoiceCLAP-Ensemble, 16 Klassen). Die beiden Detektoren sind **nicht** austauschbar; keine Zeile mischt sie.
* **Gruppen-Schema:** `vm_groups.py` md5 `f83e3850` (23 Gruppen, 117 Mitgliedsnamen).
* **Trägersatz:** `$SC/out/burst_dose2/prompt_sets.json`, 10 Prompts je (Skriptart, Klasse), Startwert hängt allein vom Prompt-Index ab — alle Arme ziehen dasselbe Rauschen.
* **Statistik:** die drei Stichproben eines Prompts werden zuerst gemittelt, n ist die Zahl der Prompts.
* **WER-Obergrenze:** w = 2,0 ist ausgeschlossen. Über die 28 Klassen der zehn zweiquelligen Gruppen reißen dort **alle vier** Adapter-Arme das Tor (nachtrainiert +0,167, geliehen +0,168, Gruppe-voll +0,123, Gruppe-25 % +0,109 gegen die Schranke +0,104). Bis w = 1,5 hält jeder Arm.
* Studie `vb_grp`, Zustand `~/reports/STATE_burst_group_lora.md`.

<!-- links:2026-09-05 -->

## Woher der Adapter kommt

Die Namen in den Rezepten oben (`bestmem`, `grpfull`, `bulk_mix_full` …) sind Studien-Arme, keine Dateien. Hier stehen die tatsächlich abrufbaren Adapter.

| Rolle | Adapter | abrufen |
|---|---|---|
| Gruppen-Adapter, volle Dosis — Gruppe `laugh_soft` | `laugh_soft` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/groups_full/laugh_soft) |
| Gruppen-Adapter, 25 % synthetisch — Gruppe `laugh_soft` | `laugh_soft` | [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2/tree/main/groups_dose25/laugh_soft) |

> **Gewicht beachten.** Rang 16, Alpha 32 — die eingebaute Skalierung ist also 2,0, und das Gewicht `w` multipliziert sie. `w = 1,0` bedeutet demnach bereits Verstärkung 2,0.

## Wie der Cue geschrieben wird

**Cues stehen immer auf Englisch, auch wenn der gesprochene Text deutsch ist.** Das ist keine Stilfrage, sondern die Schreibweise der Trainingsdaten: die deutschen Zeilen im Korpus lauten `Das zerreißt einen einfach, weißt du? (relief sigh)`. Ein deutscher Cue ist außerhalb der Verteilung und verhält sich unvorhersehbar.

* Der Burst ist **eine eigene Klammer** zwischen den Wörtern — `(clearly amused) … (titter) …`. Innerhalb einer Regieanweisung genannt entsteht **kein** Laut.

* **Nie eine Zahl in eine Klammer schreiben.** Eine runde Klammer mit Zahl hört auf, eine Anweisung zu sein, und wird zum Burst. Die Dauern rechnet der Server aus (`(label, N.N seconds)`, Vorgabe 0,28 s, zulässig 0,14–1,2).

* Eckige Klammern nur für Pausen: `[pause]`, `[long pause]`. Keine Großbuchstaben — das Modell buchstabiert sie.

Vollständige Anleitung für das Regie-Sprachmodell: [`docs/DIRECTOR.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/DIRECTOR.md) · Lademechanik: [`docs/ADAPTERS.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/ADAPTERS.md)

<!-- /links:2026-09-05 -->
