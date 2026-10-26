# Robby: fast, contextual bridge phrases
## Decision-model routing while the full answer is generated
Design proposal • 26 October 2026 • English technical plan

**Goal.** Give an immediate, appropriate sign that Robby heard the user, without replacing a useful answer with filler. A small decision model selects an allowed, pre-recorded phrase and an optional expression. The main language model generates the actual answer concurrently. This document proposes a new bridge controller; it does not claim that it is already deployed.

**Recommended starting point.** Use a small, curated German phrase bank for the current German-default demo, with an equivalent English bank. Generate audio in advance using Robby's selected voice. Keep decoded audio in a bounded RAM cache. Do not put another TTS job ahead of the real answer. Silence is always a valid decision.

## Parallel execution, one playback queue
1. During listening, the decision model may classify the developing request speculatively. It must not play a phrase or commit a turn while the person is still speaking.
2. The existing speech-pause and speaker-focus controller accepts a turn. Start the main LLM immediately. In parallel, confirm the bridge category from the final transcript and current conversation.
3. If a safe, cached phrase is available and the real answer is not ready, enqueue one bridge with a small matching gesture. Otherwise skip it. Use one turn ID and one audio scheduler for both parts.
4. The first complete LLM sentence goes directly to streaming TTS while the bridge plays. At the phrase boundary, play the first answer segment without overlap or an artificial pause. Subsequent sentences continue normally.
5. An accepted interruption cancels the bridge, queued answer, TTS work and obsolete LLM results together. A later result with an old turn ID can never resume playback.

**Context consistency.** Tell the LLM which bridge is committed for playback through structured controller metadata. Do not insert private stage notes into the assistant's spoken text. If the bridge is skipped or interrupted, record that fact separately. The full answer should not repeat an acknowledgment, pretend to have performed a lookup, or assume unheard words reached the user.

## Latency: measure useful speech as well as first sound
The current demo's default speech-pause threshold is **700 ms**. A six-case local Laya check measured roughly **268–281 ms** per HTTP decision request. The browser currently reserves approximately **450 ms** for initial streamed playback. These are separate measurements/settings, not an end-to-end bridge benchmark, and overlapping stages must not be added indiscriminately.

The unchanged pipeline therefore cannot promise a response a few hundred milliseconds after the last spoken word. The pilot should overlap speculative classification with the pause and separately test a shorter startup reserve for fully cached audio. A proposed target is **under 300 ms from an accepted turn to a cached audible reaction**, with last-user-word-to-sound reported separately. It is a target, not a result.

---page---
# A taxonomy that permits silence
Choose the conversational function first, then a phrase variant. The German examples below illustrate the current demo language; they are suggestions for a reviewed bank, not universal responses.

| User turn / category | Possible reaction | Gesture / guard |
| Greeting or “Wie geht's dir?” | “Hallo, schön dich zu sehen!” / “Mir geht's gut — und dir?” | Small smile. Often a complete social reply, not a thinking bridge. |
| Reflective or philosophical question | “Da steckt einiges drin.” | Thoughtful gaze; no exaggerated profundity. |
| Explanation / how-to | “Gehen wir das Schritt für Schritt durch.” | Small nod. Skip if the answer is short. |
| Several constraints or a complex task | “Okay, ich behalte die Punkte im Blick.” | Attentive pose; still fulfill every constraint. |
| News, weather or factual lookup | “Ich schaue nach.” | Only after an actual tool job starts; never claim a search from intent alone. |
| Story, poem or playful request | Usually no bridge; optionally “Eine kleine Geschichte? Sehr gern.” | Warm expression. Begin the requested work immediately if ready. |
| Personal difficulty / emotional disclosure | “Ich höre dir zu.” | Gentle attentive pose. No joke, diagnosis or asserted emotion. |
| Correction / misunderstanding | “Danke für die Korrektur.” | Brief nod; apply the correction in the substantive answer. |
| Thanks or goodbye | A short “Gern!” / “Bis bald!” may finish the turn. | Cached direct reply only when no question or task remains. |
| Stop, urgent request, exact-word answer | No bridge. Execute stop or produce the required response. | Cancellation has priority; do not delay with an apology. |
| Background speech, uncertain addressee | No bridge. | Preserve speaker focus; no acknowledgment of a television. |

**Classification is contextual.** “Danke, aber das beantwortet meine Frage nicht” is a repair request, not a completed thank-you exchange. “Kannst du einen Witz erzählen?” needs a complete joke, not a social acknowledgment. “Wie geht's dir?” normally does not warrant “Let me think.” If the taxonomy does not fit, choose silence and let the LLM answer.

**Acoustic information is optional evidence.** Humaneness Ears delivery captions may suggest a gentler response, but are uncertain estimates. The person's own words and the conversational context take precedence. Never select a phrase that states a diagnosis, mind-reads an emotion or judges honesty from voice scores. Missing captions must not block the controller.

**Humor and personality.** Use a restrained playful variant only in an established playful context. Avoid jokes during distress, corrections, interruptions or urgent requests. Favor a small smile or nod over another spoken filler. Respect reduced-motion settings and the user's animation controls.

---page---
# Controller, cache and failure handling
## Small structured decision, bounded choices
Proposed response fields: **action** = silence / bridge / cached_direct / stop; **category**; **phrase_id**; **gesture_id**; and an uncertainty indicator. IDs must come from the approved catalog. The controller validates the decision against the active turn, speaker, language, tool state and user settings. Do not interpret an arbitrary generated confidence number as a calibrated probability.

The current Laya integration handles turn-taking decisions. Routing among bridge categories is an additional task that needs labeled examples and held-out evaluation. Start in shadow mode: log choices on synthetic tests without playing them. If quality is insufficient, use a smaller set of high-precision categories rather than forcing every request into a phrase.

## Avoid repetitive or empty conversation
- At most one bridge per turn. Never chain filler while waiting for a failing tool.
- Proposed initial limits: no identical phrase within the previous five turns and at least a 20-second bridge cooldown. Tune these with users; keep them configurable.
- Offer an independent “Quick spoken reactions” toggle and a restrained frequency setting. Disabling bridges preserves normal ASR, speaker memory, tools and answer streaming.
- If the first useful answer segment is ready, skip the bridge. If a bridge is already audible, join the answer at its end; do not overlap voices or cut a word to save a few milliseconds.
- For a cached direct social response, cancel unnecessary LLM work only after the controller verifies that no factual question, correction or task remains.

## Cache design
Key each entry by text, language, voice/reference fingerprint, model/checkpoint revision, delivery settings and audio format. Precompute gesture timing against the actual cached waveform. A voice or checkpoint switch invalidates incompatible entries. Build the cache outside active conversations and validate every generated phrase for intelligibility and duration.

Example storage assumption: 30 phrases × 3 variants × 1 second × 24,000 mono samples × 2 bytes is **4.32 MB** of PCM16. Actual durations and multiple voices increase this linearly. Keep only the active voice's frequently used entries in RAM. Record manifest hashes and model/settings provenance; never put microphone recordings or enrolled voiceprints into a public cache.

**No-cache path.** Skip the bridge by default. On-demand TTS is an experimental fallback only if it uses spare capacity and measurably beats the first useful answer. Otherwise it can make the response slower by competing with the answer's TTS.

**Failure rules.** A missing cache entry, late classifier, model error, language mismatch or uncertain speaker produces no bridge. Tool failure receives an honest final explanation, not repeated “I'm checking.” Cancellation and disconnect clear scheduled audio and expressive actions. Bound queues, memory and deadlines; never replay a bridge from an earlier turn.

---page---
# Reproducible pilot and rollout
## Three conditions on identical inputs
Compare **A: no bridge**, **B: cached bridge**, and **C: on-demand bridge TTS**. Keep voice, checkpoint, transcript, prompt and network conditions fixed where possible. Randomize condition order. Separate warm-cache and cold-start results. Use unseen prompts and speakers, not just the examples that selected thresholds.

Start with approximately 200 labeled turns covering the taxonomy, including ambiguous and negative cases. Then replay at least 50 multi-turn synthetic conversations through the real browser/microphone transport, ASR, speaker gate, decision model, LLM, TTS and playback queue. These are proposed sample sizes; expand them until confidence intervals are useful.

| Measure | What to report |
| Timing | Last user word → turn commit; commit → first audible bridge; commit → first useful answer; total task duration; p50 / p95. |
| Appropriateness | Blinded human ratings of fit, warmth, repetitiveness and whether the bridge changes or ignores the request. |
| Answer quality | Complete poem/joke/instructions, correct tool use, preserved conversation thread, no duplicate acknowledgment. |
| Interruptibility | Stop during bridge, at bridge/answer handoff, during answer; cancellation delay and stale-audio replay count. |
| Speaker safety | Unknown speaker, television, overlapping speakers and familiar voice with changed emotion; false acknowledgments and false interruptions. |
| Resource cost | Cache hit rate, extra LLM/TTS work, CPU/GPU use, audio underruns and memory growth. |

## Required adversarial cases
Test “Danke, das reicht” versus “Danke, das reicht nicht”; user speech resuming during the pause; a new sentence arriving during the bridge; repeated requests; a tool that stalls or fails; an exact-word answer; a long poem; conflicting voice and lexical emotion; language switching; noise/music/TV; two speakers; microphone echo; disconnect/reconnect; an old response arriving after cancellation; and all options disabled.

Extend the existing **VOICE_ASSISTANT_TEST_PROTOCOL.md** and browser fixture runner instead of starting a separate untracked demo. Freeze input audio hashes, model revisions, random seeds and controller settings. Retain every failed run and raw event timeline. Distinguish digitally injected audio from a physical microphone/loudspeaker test. Verify the selected phrase's actual playback, not merely a decision log.

## Gates for deployment
1. Implement the queue/cancellation contract and a five-category cache behind a disabled feature flag. Test offline and in the browser.
2. Run shadow classification, then the controlled A/B/C experiment. Suggested acceptance targets: at least 95% appropriate bridges in blinded review, no obsolete audio replay, and no material regression in task completion or interruption latency. Report uncertainty; these are targets, not measured guarantees.
3. Conduct consented physical-room trials with speakers, television and echo. Enable only categories supported by evidence, then expose the independent UI toggle.

**Deliverables.** Versioned phrase manifest and generator, classifier prompt/schema, cache manager, browser playback receipts, test corpus manifest, latency report and rollback switch. Existing code and reproducible protocols are in https://github.com/LAION-AI/Humaneness-Voice/tree/main/robby . This plan is deliberately separate from the deployed acoustic-caption integration.
