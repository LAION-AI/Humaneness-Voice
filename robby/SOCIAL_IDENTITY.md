# Social identity, role grounding and curious listening

**Current override:** Camera auto-start and gaze following now default ON with browser permission, while Face ID and face-VAD remain OFF. Hey Robby mode defaults ON and automatic sound reactions default OFF; see [WAKE_MODE.md](WAKE_MODE.md). The proactive defaults described below are historical.

This update fixes a reproduced **correct acoustic identity / wrong spoken name**
failure and reduces repeated onboarding after changes in intonation. It adds
inspectable voice profiles without replacing ASR, interruption handling, camera
tracking, speaker verification, tools, or the optional factual wiki.

## What failed, and which model said it?

The user's exact historical quote, “Du bist Christoph, wie du dir selbst genannt
hast”, was not retained in private transcript logs. Its exact model cannot be
proven retrospectively. The current main model is **qwen/qwen3.8-27b through
OpenRouter / Cerebras**; the optional local reaction is off by default.

A real HTTPS microphone replay reproduced a more serious error: after Selma had
introduced herself, ECAPA and the endpoint event correctly identified Christoph,
but Qwen/Cerebras answered **“Du bist Selma.”** A separate role-correction probe
produced “Ach du Scherzkeks!”, dismissing a correction. The failed receipts are
preserved; a passing short prompt test was insufficient evidence on its own.

Changes:

* The current user is explicitly distinguished from Robby in the system prompt
  and the final user-message header. User “ich” means the user; assistant “ich”
  means Robby. Explicit name corrections take priority over earlier replies.
* A narrow direct question such as “Und wer bin ich?” or “Wie heiße ich?” reads
  the **named current speaker** from session identity instead of asking an LLM
  to choose a name from conversation history. It says “Du heißt Christoph.”
  Weak continuity says “Ich glaube, du bist Christoph. Stimmt das?” No lookup is
  made when the person is unnamed. General memory questions remain generative.
* Those exact name lookups skip the optional filler reaction and still use real
  TTS. The UI labels them **Session identity · name lookup · local controller**.
  Ordinary replies show their actual model/provider metadata, including a local
  reaction model if one was used.
* One attempted extra system message immediately before the user was rejected
  by this provider with HTTP 400. The final implementation uses a plain user
  header and one initial system message; it does not retry against other paid
  providers or silently change the selected model.

This does not guarantee perfect German grammar in arbitrary generated replies.

## Voice identity and prototypes

The final turn uses the existing **192-dimensional SpeechBrain ECAPA** embedding.
Open, wake-free speaker matching now defaults to **0.32**, replacing a separate
0.38 default that had survived alongside the older visible threshold. The UI
slider remains adjustable. Existing nondefault choices are preserved.

A recent named speaker may be retained below that threshold only when it is the
last matched person, seen less than 60 seconds ago, cosine is at least
`max(0.24, threshold - 0.08)`, and the lead over another person is at least 0.08.
This is marked uncertain and **never trains a new prototype**. An explicit
different name and a convincing different voice are not collapsed into that
speaker. Thresholds are operating heuristics, not calibrated probabilities.

Each person can have eight acoustic prototypes. Explicit “I am also Christoph”
corrections can add another variant to the same declared name. Strong automatic
variants require cosine at least 0.55 and a 0.12 lead over another speaker; older
high-confidence/adaptation rules remain. The original anchor is preserved.
Naming prompts have a 90-second per-person and 45-second session cooldown, plus
the existing turn-count guard, so several uncertain fragments do not each ask
the same question. Session-only identities expire after the configured timeout
(15 minutes by default); persistence is separate.

## Vocal impressions and facts

**Include uncertain vocal age / gender and timbre impressions is ON by default**.
There is a one-time default migration, after which turning it off is remembered.
The existing frozen Humaneness Ears Medium BF16 worker runs on GPU 1 alongside
ASR. It supplies AGEV, GEND, REGS, BRGT, DARC, R_CHST, R_THRT, FULL, ROUG and WARM.
These cover perceived vocal age/gender, register, brightness, articulation
darkness, resonance, fullness, roughness and warmth.

Up to 12 accepted-turn observations are retained per person. The UI shows
approximate median-based descriptions and observation counts; structured context
also carries means and spread. Disabling impressions stops collection and
excludes these summaries from subsequent response context. An in-flight result
requested with the old enabled setting is discarded. Existing stored observations
are retained until profiles are forgotten; disabling use is not deletion.

These are **how a voice may sound**, not actual age, gender identity, health,
personality or proof of who someone is. They are not used to assign names or
change speaker-match thresholds. The LLM is told to be tentative and tactful,
not volunteer demographic judgments, and accept the person's correction.
No embedding vector or face image enters the LLM prompt.

Confirmed user facts, relationships and interests remain in the separate,
optional [wiki memory](PERSISTENT_MEMORY.md). It is still off by default and
retrieves a bounded relevant subset. Acoustic impressions are not silently
promoted to factual wiki entries.

## Session profiles and optional persistence

The lower **People Robby knows · voice profiles** panel shows names, prototype
counts and uncertain traits. No introduction or wake phrase is required to talk.

“Remember these voice profiles on this browser, with everyone's permission” is
**OFF by default**. After explicit confirmation it requests at most eight named
profiles and saves their names, ECAPA prototypes and trait observations in this
browser's localStorage. These are biometric clues, not secure authentication.
Browser storage is unencrypted and should only be used on a suitable device.
Profile contents are sent to the existing server to restore a session, never to
the language-model provider. Stable random profile IDs support reconnects.

New people stay session-only until permission is renewed; the existing consent
does not silently enroll every future visitor into durable storage. Turning the
checkbox off removes this voice-profile store. **Forget all face + voice profiles**
clears both biometric UI stores and current server identity; the separately
enabled factual wiki is unchanged and has its own reset control.

## Camera recognition

Camera auto-start, webcam Face ID and face-VAD gating remain **OFF by default**,
as previously requested. Position following remains available when the camera
is enabled. To use optional multimodal recognition, enable the camera and
**Recognize people by webcam Face ID**. Face VAD does not need to block speech.

The old optional visual-binding code was not connected to the newer sentence
pipeline. It now binds an unambiguous, time-aligned speaking face to the current
named voice, and can corroborate a weak acoustic match with a known face.
Contradictory strong audio evidence or an explicit different name wins. Multiple
ambiguous faces do not force an identity. Existing MediaPipe/face-api CPU workers
are reused; there is no additional server GPU face model.

Browser tests use fake camera access, synthetic landmark motion and public
NASA/skimage imagery. They establish that model inference and integration run;
they do **not** establish recognition accuracy on the user's physical webcam.

## Curious background listening

Prominent sound sensing and questions about unfamiliar audible voices are on by
default, while respecting the user's saved off setting and requests to ignore
background speakers. Sounds retain the 8-dB default and the existing parallel
EfficientAT/Whisper path described in [PROACTIVE_ENVIRONMENT.md](PROACTIVE_ENVIRONMENT.md).

Open-mode speech rejected as quiet background can now invite one cautious
question after **two audible observations within 25 seconds**. Each must contain
at least one second, RMS at least 0.003 and estimated prominence at least 8 dB;
a known-voice match of 0.5 or greater is not treated as an unfamiliar background
voice. Questions are separated by at least 120 seconds, wait for an idle slot,
and honor playback guards and the user's permission to ask. The ignored voice
is not enrolled and focus is not transferred. “It may be a TV, recording or the
same person speaking differently” is explicitly part of the model context.

A repeated browser run correctly routed the event and played audio, but Qwen
only said “Hallo Christoph! Schön, dass du da bist!”. That is a retained semantic
failure, not a successful curious question. The latest controller request now
states the actual background-voice question at the newest conversational edge,
so it does not resemble a request to repeat the previous introduction. A separate
five-generation provider probe and a question-content check supplement the
browser routing/audio assertions. Prompting alone still failed once in five
probes. The optional voice question is therefore buffered before TTS and checked
for a relevant question. Any greeting-only draft is discarded; one bounded
regeneration may repair it, otherwise the optional comment is skipped. Only the
first relevant question is spoken. Normal foreground sentence streaming does
not use this guard or acquire its extra wait. These remain small development
checks, not general semantic understanding guarantees.

These gates intentionally avoid chatting over every quiet TV fragment. They
cannot distinguish all addressed human speech from recordings or overlapping
voices. The caption and main answer may still be wrong; Robby should ask, not
assert. They do not add a blocking model call to normal foreground ASR.

## Reproduce the checks

Start the normal stack, including the updated `voice_caption_worker.py`, and
verify `/api/state` reports the loaded source hash. Use the same configured
OpenRouter account for API tests. Keep credentials outside this checkout.

```bash
python run_unit_checks.py
python check_social_dialogue.py
python check_social_curiosity_prompt.py
python generate_adversarial_fixtures.py
python generate_social_fixtures.py
python browser_social_profiles.py --url https://YOUR-ROBBY-HOST
python browser_social_curiosity.py --url https://YOUR-ROBBY-HOST
python browser_multimodal_checks.py --url https://YOUR-ROBBY-HOST/desk-robot/
```

Use the existing Python environment and Playwright/Chromium installation. The
fixture generator can reuse the existing manifest/cache; do not regenerate
everything merely to replay a test. Browser audio suites must run sequentially
because the live ASR worker has one session slot. Test traffic is real authored
synthetic PCM delivered in 20-ms packets through the browser, with real ASR,
ECAPA, Ears, the selected dialogue backend, TTS and browser playback. Camera
tests do not acquire the ASR slot. No private microphone recording or voiceprint
array is published in receipts.

The linked English [report](static/social-identity-report.html) inventories the
passed and failed receipts, exact test scope, measured timings and limitations.
The [test protocol](VOICE_ASSISTANT_TEST_PROTOCOL.md) includes the regression
matrix. Test passes on a few synthetic voices are not a calibrated real-world
false-match rate or a guarantee of perfect conversation.
