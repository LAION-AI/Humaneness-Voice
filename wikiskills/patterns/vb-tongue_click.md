<!-- vb_grp:2026-09-05 -->

# vb/tongue_click

> Stand 5. September 2026 · Gruppe `mouth` im 23-Gruppen-Schema · **nicht einzeln gemessen** · eigener Adapter vorhanden: ja
>
> Gruppen-Mitglieder: `blowing_a_kiss`, `chewing_noises`, `click_one_s_tongue`, `clicks_tongue`, `kissing_noises`, `kissing_sounds`, `licking_sound`, `lip_smack`, `smack`, `smack_one_s_lips`, `smacks_lips`, `spitting`, `sucking_noise`, `tongue_click`, `tsk`

## Rezept

Diese Bezeichnung wurde **nicht gemessen**, und in ihrer Gruppe `mouth` ist ebenfalls keine Klasse gemessen. Es gibt dafür **kein belegtes Rezept**.


## Warum die Zahl so ist

Für diese Bezeichnung gibt es **keine Messung** — weder für sie selbst noch für ein Mitglied ihrer Gruppe. Es ist unbekannt, ob das Modell sie erzeugen kann; niemand hat es geprüft.

## Wenn es nicht geht

Diese Gruppe hat kein zweites gemessenes Mitglied, auf das man ausweichen könnte. Wenn das Etikett nicht trifft, bleibt Best-of-N oder der Verzicht auf die Anforderung.

## Woher die Zahlen kommen

* **Messgerät:** `laion/vocal-burst-detector-v2` (83 Klassen, das Instrument der Laufzeit). Die beiden Detektoren sind **nicht** austauschbar; keine Zeile mischt sie.
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
| **ausgeliefert** (bisheriger Satz, Rückfallvariante) | `tongue_click` | [`laion/moss-va-sft3-vocal-burst-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters/tree/main/adapters/tongue_click) |

> **Gewicht beachten.** Rang 16, Alpha 32 — die eingebaute Skalierung ist also 2,0, und das Gewicht `w` multipliziert sie. `w = 1,0` bedeutet demnach bereits Verstärkung 2,0.

## Wie der Cue geschrieben wird

**Cues stehen immer auf Englisch, auch wenn der gesprochene Text deutsch ist.** Das ist keine Stilfrage, sondern die Schreibweise der Trainingsdaten: die deutschen Zeilen im Korpus lauten `Das zerreißt einen einfach, weißt du? (relief sigh)`. Ein deutscher Cue ist außerhalb der Verteilung und verhält sich unvorhersehbar.

* Der Burst ist **eine eigene Klammer** zwischen den Wörtern — `(clearly amused) … (tongue click) …`. Innerhalb einer Regieanweisung genannt entsteht **kein** Laut.

* **Nie eine Zahl in eine Klammer schreiben.** Eine runde Klammer mit Zahl hört auf, eine Anweisung zu sein, und wird zum Burst. Die Dauern rechnet der Server aus (`(label, N.N seconds)`, Vorgabe 0,28 s, zulässig 0,14–1,2).

* Eckige Klammern nur für Pausen: `[pause]`, `[long pause]`. Keine Großbuchstaben — das Modell buchstabiert sie.

Vollständige Anleitung für das Regie-Sprachmodell: [`docs/DIRECTOR.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/DIRECTOR.md) · Lademechanik: [`docs/ADAPTERS.md`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/blob/main/docs/ADAPTERS.md)

<!-- /links:2026-09-05 -->
