# Plan — LoRA-Adapter im Streaming-Demo

Stand: 2026-08-12. Ziel: Der Agent wählt pro Turn nicht nur die Referenzstimme,
sondern auch die passenden Adapter, und die werden ohne spürbare Latenz aktiv.

## 1. Was es gibt, und was davon passt

| Repo | Adapter | Größe gesamt | pro Adapter |
|---|---|---|---|
| `TTS-AGI/moss-emotion-loras-v3` | 40 Emotionen | 11.0 GB | 275 MB |
| `laion/moss-voicenet-dimension-loras` | 114 (57 dims × high/low) | 31.3 GB | 275 MB |
| `laion/vocal-burst-lora-adapters` | 64 Bursts | 22.5 GB | 275–550 MB |
| `TTS-AGI/moss-character-loras-refined-public` | 120 Chars × ep1/ep2/ep3 | 99.0 GB | 275 MB |
| `TTS-AGI/moss-explicitness-loras` | 6 (a1/adult, r32/r64) | 1.8 GB | 275 MB |
| `laion/moss-sports-commentator-lora` | 6 (r32/r64 × ep1-3) | 2.5 GB | 275–550 MB |
| **Summe** | **350** | **168 GB** | |

**168 GB gegen 33 GB freie Platte.** Alles vorzuhalten geht nicht. RAM ist mit
110 GB frei die deutlich größere Ressource, aber der HF-Download läuft zwingend
über die Platte.

### Auswahl (~17 GB, lässt ~16 GB Luft)

- **Tier 1 — komplett, im RAM:** 40 Emotions-Adapter (11 GB). Das ist die Achse,
  die der Agent am häufigsten wählt (in den Messungen ~60 % der Turns).
- **Tier 2 — auf Platte, lazy:** die 12 Charaktere, die auch im Referenz-Korpus
  vorkommen (ep2, mittlerer Checkpoint), 8 häufige Bursts, Explicitness `adult_r32`,
  Sports `r32_e2` (~6 GB).
- **Tier 3 — nicht geladen, dokumentiert:** VoiceNet (31 GB), die übrigen 108
  Charaktere, die übrigen 56 Bursts. Nachladbar per `lora_bank.fetch(name)`.

## 2. Warum *nicht* gemerged wird

Naheliegend wäre `weight += B@A` vor der Generierung. Drei Probleme:

1. **Rückweg.** Nach dem Turn muss der Adapter wieder raus. Subtrahieren
   akkumuliert bf16-Fehler über hunderte Turns; eine saubere Kopie der Basis-
   gewichte kostet ~9 GB VRAM, und die Karte ist mit 20 von 24 GB schon belegt.
2. **Latenz.** rank-256 über alle attn+MLP-Projektionen sind hunderte `B@A`
   Produkte — pro Turn zweimal (rein und raus), direkt auf der Time-to-First-Audio.
3. **Kombinationen.** `burst@1.0 + emotion@0.5` müsste zweimal gemerged werden.

**Stattdessen: Laufzeit-Hooks.** Die Faktoren A und B bleiben getrennt auf der
GPU, ein `forward_pre_hook` auf jedem Ziel-`Linear` addiert
`scale * (x @ Aᵀ) @ Bᵀ` auf den Ausgang. Swap = Zeiger umbiegen, praktisch 0 ms.
Mehrfachadapter = Summe mehrerer Terme, exakt das, was ein Merge auch täte.
Kosten: rank 256 gegen hidden 2560, also grob 10–20 % mehr Rechenzeit pro Token.
Bei RTF 0.74 bleibt das unter Echtzeit.

## 3. Dosierung — aus dem Manual, nicht geraten

| Fall | λ | Quelle |
|---|---|---|
| Emotion allein | **0.35–0.75**, Standard 0.5 | "keep emotion λ moderate, 0.35–0.75" |
| Burst mitten im Satz | **0.5** | ~50 % Burst-Präsenz, ~90 % der Sprache danach erhalten |
| Burst isoliert | **0.75** | "the knee": 91 % der vollen Präsenz, besserer Blend |
| Burst + Emotion | **burst 1.0 + emotion 0.5** | Emotion muss ≤ halbe Burst-Dosis sein |
| Charakter | 0.75 | Analogie zu Burst-Knee; noch zu messen |
| Explicitness | 1.0, **ohne** Burst-Adapter | Blend 1.60 → 7.82; mit Burst-Adapter fällt er auf 5.42 |

Harte Regel aus dem Manual: **liegt ein Burst-Adapter drunter, darf die Emotion
höchstens auf der halben Burst-Dosis sitzen.** Bei Burst ≤ 0.75 unterdrückt eine
Emotion bei 0.5 den Burst sogar. Das wird im Code erzwungen, nicht dem LLM
überlassen.

Bekannte Fallstricke: `Sexual_Lust` und `Pain` werden bei Merge 1.0 schlechter.
Emotions-Adapter allein landet Bursts in nur 16.7 % der Fälle — unter dem
Baseline-Prompt (23.6 %); Bursts brauchen ihren eigenen Adapter.

## 4. Wer entscheidet

Das Tool `select_reference_voice` bekommt zwei zusätzliche Felder:

```
emotion_lora   : Adaptername oder null   (Standard: der gewählten Emotion folgen)
burst_lora     : Burst-Adaptername oder null, wenn im SCRIPT ein (burst) steht
```

Die λ-Werte setzt das LLM **nicht** — die kommen aus der Tabelle oben, abhängig
davon, welche Kombination zustande kommt. Ein Sprachmodell, das Dosierungen
raten darf, rät sie falsch, und die Werte sind ohnehin gemessen.

Zusätzlich automatisch: enthält das SCRIPT `(laughs)`, `(gasp)`, `(sighs)` o.ä.,
wird der passende Burst-Adapter auch ohne explizite Wahl gezogen — laut Manual
steigt die Landerate von 23.6 % auf 71.9 %.

## 5. Reihenfolge der Umsetzung

1. `lora_bank.py` — Download-Tiers, RAM-Cache, Hook-Anwendung, `apply(specs)` /
   `clear()`, Zustand pro Request unter demselben Lock wie die Generierung.
2. Tool-Schema + Systemprompt um die zwei Felder erweitern, λ serverseitig.
3. Burst-Autoerkennung aus dem SCRIPT.
4. Messen: `eval_consistency.py` mit/ohne Adapter, dazu Burst-Landerate und
   Speaker-Similarity — Adapter dürfen die Identität nicht zerlegen.
5. VoiceNet erst, wenn Platte frei ist.

## 6. Offene Risiken

- **Identität vs. Adapter.** Emotions-Adapter bei hoher Dosis ziehen die Stimme
  weg vom Anker. Muss gegen die Speaker-Similarity gemessen werden; im Zweifel
  λ runter statt Adapter raus.
- **VRAM.** Jeder aktive Adapter kostet ~275 MB fp16 auf der GPU. Bei 20 von
  24 GB belegt passen ~10 gleichzeitig. Adapter werden nur bei Benutzung auf die
  GPU geschoben, LRU-verdrängt.
- **Hook-Kosten unter Streaming.** Muss gegen RTF 0.74 gemessen werden; wenn es
  über 1.0 geht, ist Streaming kaputt und wir brauchen doch einen Merge-Pfad.
