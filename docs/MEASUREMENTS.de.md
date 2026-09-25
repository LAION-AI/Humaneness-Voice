# Protokoll — was beim Bauen herauskam

Chronologisch, mit Messwerten. Alles hier ist an dieser Maschine überprüft, nicht
aus der Dokumentation abgeschrieben.

---

## Umgebung

- **`sgl-omni` läuft auf dieser Kiste nicht.** Es gibt nur `manylinux_2_34`-Wheels,
  Ubuntu 20.04 hat glibc 2.31. Bliebe ein ~20 GB Container — bei 33 GB freier
  Platte und `/mnt/raid` zu 100 % voll nicht vertretbar. Der Streaming-Pfad ist
  jetzt LAION's eigene Referenzimplementierung, und die ist für diesen Zweck
  sogar besser: die sgl-omni-HTTP-API nimmt Referenzaudio als Dateipfad und würde
  es pro Turn neu enkodieren, statt den fertigen Code-Tensor aus dem RAM zu nehmen.
- **`/usr/bin/nvcc` ist CUDA 10.1** (Ubuntu-Paket), das echte Toolkit liegt in
  `/usr/local/cuda` (12.3). cmake findet ohne `-DCMAKE_CUDA_COMPILER` das falsche
  und bricht ab.
- **`pkill -f "muster"` killt die eigene Shell**, wenn das Muster im eigenen
  Kommando steht. Zweimal reingefallen (Exit 144). Entweder Klammertrick
  `[u]vicorn` — der aber nichts nützt, wenn derselbe String später im selben
  Kommando nochmal echt vorkommt — oder Kill und Start in getrennte Aufrufe.

## Modell und Prozessor

- transformers 5.15 lädt den Remote-Code ohne Anpassung. `input_ids` haben Form
  `(1, T, 13)` = 1 Textspalte + 12 RVQ-Codebooks.
- Die Meldung `text_lm_head / audio_lm_heads MISSING` beim Laden ist harmlos
  (weight tying), steht so auch in der Model Card.
- **Der Codec muss unter `torch.inference_mode()` laufen, nicht `torch.no_grad()`.**
  Die Generierung läuft unter `inference_mode` und lässt interne Puffer des Codecs
  als Inference-Tensoren zurück; ein späteres In-place-Update unter `no_grad`
  wirft dann `Inplace update to inference tensor outside InferenceMode`. Kostete
  598 von 832 Vor-Tokenisierungen, bevor es auffiel.
- `build_user_message(reference=[...])` nimmt eine **Liste**; jeder Eintrag wird
  als eigenes Audiosegment eingebettet. Damit lässt sich ein Identitäts-Anker vor
  den Delivery-Clip stapeln.

## Sprachmodell

- **Gemma-4 ist ein Reasoning-Modell.** Unangetastet verbraucht es das gesamte
  Token-Budget in `reasoning_content` und liefert **leeren** `content` zurück
  (`finish_reason: length`). Mit `chat_template_kwargs: {enable_thinking: false}`
  bzw. `--reasoning off --reasoning-budget 0`: 24 statt 114 Token für dieselbe
  Antwort.
- **Enums im JSON-Schema sieht das Modell nicht.** llama.cpp macht daraus eine
  Grammatik, die beim Sampling einschränkt — im Prompt steht sie nicht. Ergebnis:
  das Modell griff immer zu denselben zwei, drei Konditionen. Nachdem der volle
  Katalog (~810 Token, gecachter Prefix) im Systemprompt steht: **8 von 8 Turns
  unterschiedlich** und passend gewählt ("say that again but annoyed" →
  `Impatience_and_Irritability`).
- Das `reply`-Feld war reine Dopplung des `script`. Rausgeworfen, die gesprochenen
  Worte werden aus dem Skript durch Entfernen der Cues abgeleitet — spart ein
  Drittel der Output-Token.
- Gemma-4 hat **nativ einen Audio-Encoder**: `clip.has_audio_encoder = True`,
  Projektor `gemma4ua`, 128 Mel-Bins, mmproj 168 MB. ASR über
  `/v1/chat/completions` mit `input_audio`: **588 ms** für einen 10.8-s-Clip,
  Transkript wortgenau. Kein Whisper nötig.

## Latenz (RTX 3090 je Modell)

| | Mittel |
|---|---|
| LLM | 2.97 s |
| Referenz-Retrieval | **0.6 ms** warm (6.1 s kalt) |
| TTFA GPU | 0.73 s |
| TTFA Server | 3.71 s |
| Realtime-Faktor | **0.737** |

Die Time-to-First-Audio hängt fast komplett am Sprachmodell, nicht am
Stimmmodell. Das Vorhalten des Korpus im RAM nimmt das Retrieval vollständig aus
dem kritischen Pfad (832 Konditionen in 89.6 s vor-tokenisiert).

## Stimm-Konsistenz — der eigentliche Knackpunkt

Gemessen mit ECAPA (`speechbrain/spkrec-ecapa-voxceleb`) gegen
`reference/reference_target.mp3`. Schwellen laut Manual: unter 0.58 neu
generieren, unter 0.45 reparieren, unter 0.40 verwerfen.

| Konfiguration | vs. Anker | Turn-zu-Turn |
|---|---|---|
| `original` + freie GENERAL-Beschreibung | 0.292 | 0.419 |
| `voice_converted` + freie GENERAL | 0.302 | 0.366 |
| `voice_converted` + **falsche** feste Identität | 0.302 | 0.366 |
| `voice_converted` + **korrekte** feste Identität | **0.373** | 0.397 |

**Der teuerste Fehler war eine erfundene Identität.** Ich hatte den festen
GENERAL-Block als "a woman's voice in her late twenties" geschrieben. Der
Anker-Sprecher des Korpus ist laut `index.json → _speaker` aber
**"Velvet Sage Baritone", männlich, Ende 40 bis 60**. Die Textbeschreibung hat
also gegen die Referenz-Codes gearbeitet. Korrektur: +0.071.

Weiter gefunden, noch nicht zu Ende gemessen:

- **Die Sprache wurde nie durchgereicht.** `language` stand fest auf `"English"`,
  auch bei deutschen Antworten, und die Referenz wurde immer aus dem englischen
  Teil des Korpus gezogen. Genau die deutschen Turns hatten die schlechtesten
  Werte (0.194–0.317 gegen 0.406–0.661 bei englischen). Das Modell setzt jetzt
  ein `language`-Feld, Referenzauswahl und TTS folgen ihm.
- **Anker + Delivery-Clip als zwei Referenzen** ist implementiert (der Prozessor
  akzeptiert eine Liste), Messung steht aus.

Was aus dem Manual noch nicht umgesetzt ist: die eigentliche
Continuation-Methode (`mode="continuation"` mit Anker + 4 s Tail des vorherigen
Clips als Assistant-Turn). Die ist für mehrteilige Performances gedacht und wäre
der nächste Hebel, wenn Anker-als-zweite-Referenz nicht reicht. Achtung: dort
wird `reference=` ignoriert und der Text muss **kumulativ** sein.

## Prompting

Aus dem Manual übernommen und im Systemprompt erzwungen:

- Runde Klammern für Cues und Bursts, eckige nur für `[pause]`.
- **Nie Großbuchstaben in Tags** — `(SCREAM)` wird buchstabiert.
- Mindestens ~10 Wörter pro Zeile (unter 8 Wörtern passt keine 3 Sekunden rein;
  gemessen sank der Anteil zu kurzer Takes von 12.0 % auf 2.1 %).
- `[pause]` statt `...` — WER 0.122 gegen 0.238.
- Cue **vor** jedem Satz, nicht einer für die ganze Antwort.
- Nie mit einem Burst öffnen oder schließen, nie `[pause]` direkt nach einem Burst.
- Cues als Veränderung schreiben, nicht als Zustand: "composure breaking into
  sobbing" schlägt "sad".

Zwei Qualitätsfehler, die erst der Test zeigte: Das Modell **spiegelte** den Nutzer
("ich habe heute meinen Job verloren", statt zu trösten), und "whisper something
conspiratorial" landete auf `Sexual_Lust` statt auf der Flüster-Dimension
`S_WHIS`. Beides im Systemprompt geradegezogen.

## LoRA-Adapter

- **168 GB veröffentlichte Adapter gegen 33 GB freie Platte.** Geladen: 40
  Emotionen (11 GB), 12 Charaktere aus dem Referenzkorpus (ep2), 7 Bursts.
  `TTS-AGI/moss-explicitness-loras` ist **gated** (401) und braucht einen
  HF-Token. VoiceNet (31 GB) und die übrigen 108 Charaktere sind ausgelassen.
- Die Adapter sind **rank 32, alpha 64** (Skalierung 2.0), nicht rank 256 wie das
  Basismodell. 268 Zielmodule je Adapter: q/k/v/o, gate/up/down über 36 Layer,
  dazu die 12 `audio_lm_heads` und der lokale Transformer.
- **Laufzeit-Hooks waren die falsche Wahl.** Elegant, kein Merge nötig, sofort
  umschaltbar — aber der Streaming-Loop läuft batch-1, Token für Token, und 268
  Module × 2 Zusatz-Matmuls sind 536 extra Kernel-Launches **pro Token**. Der
  Launch-Overhead frisst die rank-32-Ersparnis komplett auf:

  | | RTF | TTFA GPU |
  |---|---|---|
  | ohne Adapter | 0.737 | 730 ms |
  | Adapter als Forward-Hooks | **1.288** | 1290 ms |
  | Adapter gemerged | **0.764** | 840 ms |

  Über 1.0 ist die Generierung langsamer als die Wiedergabe, das Streaming
  läuft leer. Gemerged kostet der Adapter nur ~110 ms einmalig.
- Rückweg: Delta auf der GPU zurückrechnen und subtrahieren (kein PCIe-Verkehr).
  Weil bf16 dabei rundet, liegt eine unveränderte Kopie der berührten Gewichte im
  Host-RAM und wird alle 25 Turns exakt zurückgeschrieben. Ein reiner
  RAM-Restore pro Turn wäre ein 9-GB-PCIe-Transfer gewesen.
- **Dosierung kommt aus dem Manual, nicht vom Sprachmodell.** Emotion 0.5,
  Burst inline 0.5, Burst solo 0.75, Charakter 0.75 — und die harte Regel, dass
  eine Emotion unter einem Burst-Adapter höchstens auf dessen halber Dosis sitzt.
  Ein LLM, das Dosierungen raten darf, rät sie falsch, und die Werte sind gemessen.
- Der Burst-Adapter wird **automatisch** aus den Cues im SCRIPT gezogen
  (`(an exasperated sigh)` → `exasperated_sigh`). Laut Manual steigt die
  Landerate von 23.6 % auf 71.9 %. Erste Fassung wählte nur nach Familie und
  landete bei "mocking exhale" auf `contented_sigh`; jetzt entscheidet zusätzlich
  die Überschneidung der Qualifizierer.
- **Kosten für die Identität:** mit Adaptern 0.481 gegen Anker statt 0.522 ohne,
  Turn-zu-Turn 0.443 statt 0.445. Also grob 0.04 Similarity für die Emotion — im
  Bereich der Seed-Streuung, aber im Auge zu behalten.

## Tempo, zweite Referenz, gestapelte Adapter

- **Optionale Schema-Felder werden nie befüllt.** `voice2` und `style` standen nicht unter
  `required` — die Grammatik übersprang sie ausnahmslos, in jedem einzelnen Turn.
  Seit beide Pflicht sind (mit `"none"` bzw. leerer Liste als legalem Wert),
  kommen zweite Referenz und Stil-Adapter tatsächlich vor: „Gutenachtgeschichte"
  ergab `S_STRY → S_WHIS` mit `WARM high` + `TEMP low` bei 0.75× Tempo, ohne dass
  nach dem Tempo gefragt wurde.
- **Die Tempo-Vorbereitung konkurrierte mit der Generierung um die GPU** und trieb
  den RTF von 0.74 auf **über 3**. Der Hintergrundlauf teilt sich jetzt das
  Generierungs-Lock des TTS-Engines. Damit dauert das Vorbereiten länger, aber die
  Latenz einer laufenden Antwort bleibt unberührt.
- Vier gleichzeitig gemergte Adapter (Emotion/VoiceNet × mehrere) kosten
  RTF 0.77 gegen 0.74 ohne — der Merge skaliert also gutmütig.
- Tempo-Varianten: 832 Konditionen × 5 Geschwindigkeiten = **4160 Takes**,
  tonhöhenerhaltend über `audiostretchy` (nicht `audio-stretchy`, das Paket gibt
  es nicht). `ratio` ist ein Dauer-Faktor, also der Kehrwert der Geschwindigkeit.
  Die gestreckten Wavs liegen auf `/mnt/nvme` — auf `/` hätten 3 GB das
  Dateisystem gesprengt.

## Was das Sprachmodell falsch macht, trotz Prompt

Zwei Fehler wiederholten sich so zuverlässig, dass sie serverseitig abgefangen
werden mussten statt nur verboten zu sein:

- **Konditionsnamen als Regieanweisung.** `(ga_pain_scream)` ist ein
  Datenbankschlüssel und wäre vorgelesen worden. Cues mit Unterstrichen fliegen
  jetzt raus.
- **Großbuchstaben.** `AAAAAAAGH!` wird von diesem Modell buchstabiert
  („ay-ay-gee-aitch"). Jede Folge ab drei Großbuchstaben wird kleingeschrieben;
  kurze Akronyme wie „OK" und „AI" bleiben stehen.
- Dazu die Manual-Regel, dass `[pause]` direkt nach einem Burst diesen abschneidet —
  ebenfalls im Sanitizer.

Die deutsche Grammatik ist weiterhin nicht durchgehend sauber (etwa
„tiefstenseele" statt „tiefsten Seele"). Der Prompt verlangt jetzt ausdrücklich
korrekten Kasus, Genus und Verbstellung. Eine schwächer quantisierte Gemma-Variante
wäre der nächste Hebel, kostet aber Plattenplatz, den es gerade nicht gibt.

## ASR

Gemma-4 macht die Spracherkennung selbst, kein Whisper daneben:

| Eingabe | Zeit |
|---|---|
| deutsch, webm/opus (Browser-Format), 13 s | 1120 ms |
| englisch, wav, 10.8 s | 574 ms |
| dasselbe über den Cloudflare-Tunnel | 699 ms |

Der Browser liefert webm/opus, das der Multimodal-Loader nicht lesen kann —
ffmpeg normalisiert serverseitig auf 16 kHz mono wav (~70 ms).

## Audio

- Der Pegel schwankt stark je Referenzstimme: ein Orc-Take kam auf RMS 0.011,
  sein eigener Referenzclip liegt bei 0.0588. Bewusst **nicht** normalisiert —
  das würde Flüstern und Schreien einebnen, also genau das, wofür das Modell da
  ist. Stattdessen ein Gain-Regler in der Oberfläche (Standard 2×).
- Ausgabe ist echtes 48 kHz Stereo-fähiges Material; für die Demo auf Mono
  gemischt.

## SFT3, und Retrieval statt dekodierter Codes

### Modellwechsel
`laion/…-voice-acting-v2-sft3` ersetzt den SFT+DPO-Checkpoint. Runde 3 hat die
Inline-Regieanweisungen zurücktrainiert, die Runde 2 verloren hatte — Word Error
Rate auf anweisungstragenden Prompts 0.447 → 0.099, Burst-Trefferquote 0.516 →
0.666, alle Clips innerhalb 0.5 s der angeforderten Länge. Diese Demo schreibt in
*jeden* Satz eine Anweisung, insofern betrifft das alles.

Dazu die dafür trainierten Adapter: 10 Voice-LoRAs (`sft3_voice`, Gewicht 1.0,
der trainierte Wert) und 40 Emotion-LoRAs (`sft3_emotion`, Gewicht 1.5 aus dem
publizierten Sweep). Die Rangungleichheit (DPO 64, diese 16), an der PEFTs
`add_weighted_adapter` scheitert, ist hier kein Thema: die LoRA-Bank merged
deltabasiert mit Gain, was dem Aktivieren mit Skalierung entspricht.

Die alten Emotion-v3-Adapter wurden gegen die *ungetunten* v2-Gewichte trainiert
und sind auf sft3 off-distribution; sie werden ersetzt, wenn Retrieval eine
Emotion liefert.

### Referenzen: Top-3 statt Top-1
`fetch_profile_refs3.py` zieht die drei besten Takes pro Kondition und Profil —
25 260 Takes (10 × 842 × 3, 3.5 GB) statt vorher 8 420. Der beste Take behält die
schlichte gid, die Verfolger heißen `gid#2`/`gid#3`.

**Die Scores mussten nicht berechnet werden.** Die Parquet-Metadaten des Korpus
führen bereits `voiceclap` (768-d, L2-normalisiert), `genuineness`, `blend`,
`voicenet`, `emonet` und `spk_emb` pro Take mit. Gegen lokale Neuberechnung
verifiziert: Cosinus 0.98 im Mittel, Minimum 0.92 — dasselbe Modell, der Rest ist
mp3-Verlust.

### Was am Retrieval funktioniert, und was nicht
Gemessen an 18–32 Regie-Prosa-Queries mit bekanntem Ziel, 40 Emotionen
(Zufall 0.025) bzw. 57 VoiceNet-Dimensionen (Zufall 0.018):

| Achse | top-1 |
|---|---|
| Text → Audio, einzelner Clip | 0.071 |
| Text → Audio, Condition-Zentroid, roh | 0.22 |
| Text → Audio, Condition-Zentroid, zentriert | 0.44 |
| Text → Emotions-Textanker, 6 Templates, zentriert | **0.61** |
| VoiceNet-Dimensionen, jede Variante | ≤ 0.08 |

Vier Dinge, die das erklären und die Bauform bestimmt haben:

1. **Der Einzelclip-Wert 0.071 ist kein Fehler.** Er ist exakt die `emonet top1
   = 0.0721` der Model Card. Das Modell wird korrekt benutzt; es ist einfach so
   genau. Erst das Mitteln der Takes einer Kondition zum Zentroid macht die
   Audioseite brauchbar — deshalb wird nirgends gegen einen einzelnen Clip
   gematcht.
2. **Anisotropie.** Rohe Cosinus liegen in einem schmalen Band um 0.9, und
   einzelne Konditionen (`undead`, `hiss`, `crow`) sind Nachbarn von allem.
   Mittelwert-Zentrierung beider Seiten hebt 0.35 → 0.61 bzw. 0.22 → 0.44. Eine
   Einzelquery kann nicht gegen ihren eigenen Batch zentriert werden, also wird
   sie gegen das Ankermittel zentriert — die Anordnung, unter der 0.61 gemessen
   wurde.
3. **Der Textturm ist `all-MiniLM-L6-v2`,** trainiert auf kurzen Sätzen im
   `__moss_short__`-Schema. Die langen GENERAL-Specs sind weit außerhalb dieser
   Verteilung; mit ihnen fällt Emotion auf 0.12.
4. **VoiceNet-Dimensionen trägt der Textturm nicht.** Alles kollabiert auf
   `TEMP`. Die Voice-Nuancen kommen deshalb weiter aus den expliziten Codes und
   dem Basis-Stil, nicht aus dem Retrieval.

### Die Aufteilung, die daraus folgt
GENERAL beschreibt, *wer* spricht, und ändert sich zwischen Turns kaum; die
runden Klammern sagen, was *jetzt* zu tun ist. Konkateniert überstimmt die
Identitäts-Boilerplate die Cues um ein Mehrfaches. Gemessen:

* Emotion aus den **Cues allein**: 0.61 — mit GENERAL davor nur 0.39.
* Clip aus **GENERAL + Cues**: 0.28 — mit Cues allein nur 0.11, weil die
  Audio-Zentroide stark Sprecheridentität kodieren, die GENERAL mitträgt.

Beide Achsen werden also getrennt gespeist. Da die Emotionsachse mehr als
doppelt so stark ist, führt sie die Clipwahl über einen Bonus für Konditionen
der Gewinner-Emotion; die Audioähnlichkeit wählt dann Level und Take darin.
Sweep: 0.0 → 0.17, 0.15 → 0.39, **0.3 → 0.61**, darüber flach. 0.3 ist der Knick.

### Zwei Fehler, die dabei sichtbar wurden
* **Der Director meldet die Sprache falsch.** Ein vollständig auf Deutsch
  geschriebener Turn kam mit `language: English` zurück und zog prompt einen
  englischen Referenzclip — genau die Konstellation, die früher den englischen
  Akzent erzeugt hat. Die Sprache wird jetzt an den Wörtern selbst entschieden,
  und sie ist ein *harter* Filter auf den Clip, keine Präferenz.
* **Deutsche Cues ergeben falsche Emotionen,** nicht bloß schwächere: „Stimme
  brüchig vor zurückgehaltener Trauer" kam als `Teasing` zurück. Bei nicht
  englischen Cues wird deshalb auf das englische Emotionslabel zurückgegriffen,
  das der Director ohnehin aus festem Vokabular nennt. Danach: `Sadness` 0.80
  und ein deutscher Clip.

### Schalter
„Emotion nuances" (beide Demos, standardmäßig an) steuert nur den Emotions-
Adapter. Aus bleibt: sft3-Basis + abgerufener Referenzclip, sonst nichts.

---

# Nachtrag 5.–6. September 2026

Ausführlich, mit Reproduktionsskript zu jedem Punkt, steht das in
[`SEPTEMBER.md`](SEPTEMBER.md). Hier die Kurzfassung auf Deutsch.

**Die wichtigste Erkenntnis ist methodisch:** ein ausgeschriebenes Beispiel
bewegt diesen Regisseur, eine Regel nicht. Dieselbe Anweisung brachte null
Wirkung als Regel unter dreißig anderen, ~1 Pause pro Antwort mit einem
Beispiel, und 1–3 mit einem Beispielpaar pro Gefühl. Das gilt auch umgekehrt:
neunzehn Prompt-Zusätze aus zwei Suchläufen (2.949 bewertete Clips) haben
**nichts** bewegt, was einen gepaarten Vergleich übersteht — und die zwei, die
im ersten Lauf vielversprechend aussahen, kippten im zweiten das Vorzeichen.

**Vier echte Fehler gefunden und behoben.** Die Bereinigung löschte jede Pause
hinter einer runden Klammer, also auch hinter einer Regieanweisung — genau dort,
wo die erste Stille eines Satzes steht. Die Innenpausen-Prüfung hielt eine Pause
am Satzanfang für eine im Satz und schaltete damit den Boden ab. Der Renderer
verwarf jede vom Regisseur geschriebene Satzdauer. Und die Budget-Skalierung für
Burst-Adapter wurde danach wieder überschrieben.

**Zwei Ursachen für unverständliches Audio**, beide gemessen: im Englischen die
**Summe** der Burst-Adapter (zwei à 1.5 zerstören die Zeile in 5 von 5 Seeds;
Budget jetzt 2.0 statt 3.0, Einzelgewicht 1.25 statt 1.5, und über Budget wird
skaliert statt verworfen). Im Deutschen die **Sprache der Klammern** — deutsche
Regieanweisungen liegen außerhalb der Verteilung des Korpus, Median-WER 0.267
gegen 0.000 bei identischem deutschen Wortlaut mit englischen Cues. Beides in
[`BABBLE.md`](BABBLE.md).

**Guidance 3.0** ist jetzt Standard, zweimal gemessen: +2.33 und +1.92 Punkte
von 15 gegen keine Guidance, und **4.0 ist schlechter als 3.0** (−1.25, auf
allen drei Rubriken, gewinnt 4 von 12 Paaren). Deutsch profitiert deutlich
stärker als Englisch. Details in [`GUIDANCE.md`](GUIDANCE.md).

**SIDON** hebt als einziger gemessener Hebel die Klangqualität (+0.43 von 5,
p < 0.001) und lässt Natürlichkeit und Passung unberührt. Es lief einen Tag als
Standard und ist wieder **aus**: bei Schreien und lauten Bursts verzerrt es, und
die 21 Messclips enthielten kein Schreien.

**Das Tempo folgt jetzt dem Gefühl.** Der Regisseur setzt `speed` für die Szene
und schreibt Satzdauern; die Obergrenze von 1.5× ist gemessen (ab 2× kommt in
einer von vier Aufnahmen erfundener Text zurück, ab 2.5× in den meisten).

**Offen geblieben:** Füllwörter erscheinen nur in etwa jeder vierten bis fünften
Antwort — bewusst nicht erzwungen, weil eingefügte Stille umkehrbar ist und
eingefügte Wörter nicht. Der VoiceNet-Track liegt bei 5.93 von 15 gegen 8.88 der
Emotions-Aufgaben, was auf die Adapter zeigt, nicht auf den Prompt.
