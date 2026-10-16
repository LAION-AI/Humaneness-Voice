# Current prompting, references and ranking settings

Use [the exact system prompt](LUNA_SYSTEM_PROMPT.md) as the source of truth.
These are practical recipes, not a claim that the older Small studies found
the global optimum for newly released RL checkpoints.

## Preserve identity; direct the performance

Use one clean, distinct reference of the intended speaker, ideally 5–15 s,
without music or another speaker. Keep it fixed when comparing emotion or
settings. Do not describe a new age, gender, accent or timbre on every turn:
that can fight the reference. When reference audio is enabled, the current
director starts GENERAL and CAPTION with:

> Use the same speaker as the reference audio. Preserve that speaker's voice identity.

Then add a concise English performance direction: emotion, intensity and
containment, and audible pace/emphasis/pause changes. Strong emotion is allowed;
“contained” means a strong feeling held back, not necessarily a weak emotion.
If the user explicitly requests another characteristic, preserve that instruction
as a deliberate treatment rather than silently inventing a biography.

Catalog reference retrieval is **off by default**. If enabled, VoiceCLAP searches
within the selected speaker's profile for a fitting performance; it does not
change to a different speaker. The shared text tower expects English direction
descriptions, even for German speech. Unsupported/unavailable retrieval falls
back with a report. A custom upload remains the reference until removed; it is
not dynamically replaced by the director. No-reference mode supplies no hidden
voice. The TTS reference cap defaults to 15 s and supports up to 20 s using
`SMALL_REFERENCE_SECONDS`; the optional converter has a separate 10-s target cap.

## GENERAL/SCRIPT: current demo default

```text
GENERAL: Use the same speaker as the reference audio. Preserve that speaker's voice identity. Contained anger, becoming quieter and more deliberate rather than louder.
SCRIPT:
(intensely angry, held back) [3.2 seconds duration] You knew [0.4 seconds pause] what this meant to me.
(quieter now, the anger still contained) [2.5 seconds duration] And you did it anyway.
```

The plain text sent to the model is the actual spoken words, separately from
the instruction containing GENERAL/SCRIPT. Do not accidentally put tags or
duplicate narration into the literal text channel. Parentheses describe a
delivery change; square brackets request timing. Pauses inside a sentence can
sound like a person thinking; place them at motivated turns, not randomly.
Durations are requests, not guaranteed exact timings. Avoid stretching two
words into an implausible duration just to fill a budget.

Vocal sounds have their own timed bracket, for example `(chuckle, 0.3 seconds)`;
do not hide them inside a cue such as “amused, with a chuckle.” Numeric round
brackets can be parsed as a requested sound rather than an ordinary delivery
cue. Prefer one motivated reaction, usually between spoken words. A `[pause]`
is silence; an ellipsis trails off. Only add hesitation **words** to invented
chat dialogue, never to a fixed transcript in exact-text mode.

## CAPTION/TRANSCRIPT and other supported formats

```text
CAPTION: Use the same speaker as the reference audio. Preserve that speaker's voice identity. Close, restrained fear, fast breath and one hesitant pause.
TRANSCRIPT: I thought I heard someone outside. Please stay here.
```

CAPTION describes how the audio should sound; TRANSCRIPT is only spoken words.
The UI also supports tags and transcript-only input. Transcript-only has no
acting-description channel, so there is no distinct style condition for CFG
to amplify. Reference audio is supplied separately, not by writing a filename
into GENERAL. See [`prompting.py`](../playground/runtime/prompting.py) for
parsing, neutral prompts, identity preservation, timing budgets and tag handling.

An acting **brief** asks the director to create/interpret a scene, not echo the
assignment. “Comfort me after a difficult day” should yield comforting dialogue.
Choose **Perform exact text** for words that must not change. Without an explicit
style request, GENERAL and CAPTION include “Gentle, natural and spontaneous
delivery.” Explicit anger, whispering or other acting requests take precedence.

## Sampling and selection

Current starting point: temperature **0.70**, CFG **2.0**, automatic duration,
one streaming take and 800-ms codec chunks. The temperature follows the RL
release recommendation. CFG 2.0 and this prompt surface are demo choices,
not the exact no-CFG published RL baseline. Use CFG 1.0 as a useful control.
Temperature 0 is true argmax; Best-of-N at 0 need not produce distinct candidates.
Higher CFG or longer references do not guarantee better identity or emotion.

Single-take streaming is causal codec output with buffered ASR boundary/fade
checks; browser playback can lag codec chunks. Completed audio receives final
forced alignment. Requested edge bursts/pauses are protected. Alignment failure
is reported rather than used to pretend the model spoke correctly.

For Best-of-N, candidate batch size is automatic, `min(N, model cap)`, not the
single-take field's default 1. Caps: Small 8; previous Large 4; Large RL 2.
Larger N uses parallel microbatches and reports OOM retries if needed. A
full-utterance winner cannot be known before all candidates have been judged.

## Exact reward choices

Let `C` be the existing fixed-normalized VoiceCLAP performance-text score,
`E` the chosen Ears normalized metric, and `G = max(0, 1 − prefix_WER)`.
The current Ears ranking path uses the prefix scorer's word-alignment gate.
For final completed takes it explicitly sets `final=True`, so `prefix_WER`
equals full-transcript WER; partial search uses the prefix-aware surrogate.
Do not interchange partial-prefix and final full-transcript scores silently.

- **Default `ears_mix`:** `R = ((1 − w) × C + w × E) × G`, with `w = 0.5`,
  Ears Medium and `audiobox:CE`. CE is normalized using fixed native bounds
  1–10, not the min/max of the current candidates.
- **`ears_only`:** sort by the selected **raw** Ears regression score. No hidden
  VoiceCLAP, speaker or WER term. Alternatives include Production Quality and
  DNSMOS P.808 MOS; these are distilled estimates, not human ratings.
- **Legacy speaker reward:** mix performance fit with ECAPA cosine or a fixed-
  scale VoiceNet timbre-vector proxy. The legacy speaker slider starts at
  60% speaker / 40% prompt, but this term is **not active inside Ears ranking**.
- **Adaptive/manual dimensions:** choose 1–4 VoiceNet/EIV+ dimensions, objectives
  (`maximize`, `minimize`, `target`) and weights. Native ranges/endpoint meanings
  and fixed normalization are in `reward_catalog.json`. Different emotions
  are not interchangeable probabilities. Automatic planning uses the LLM;
  manual targets do not.
- **Optional burst reward:** evaluate location and optionally type of a requested
  laugh, gasp, sigh, etc., with timestamps, tolerance and one-to-one matching.
  It is separate from global Burst Blend. Extra sounds can be penalized.

Ears and adaptive/burst mixtures are deliberately exclusive in request
validation: select legacy ranking for adaptive/speaker/burst combinations.
Inspect `ears_reward.py`, `speaker_reward.py`, `adaptive_reward.py`,
`burst_reward.py` and `search_scoring.py` for exact implementation. Default
ranking still penalizes speech errors. If the priority is purely enjoyment,
select quality-only explicitly and inspect the transcript yourself.

The RL model's **training** reward (including its Orange speaker judge) is a
different protocol from this web app's **inference-time** ranking. Do not claim
that the app reproduces the training judge just because it loads the RL weights.
All learned scores can reward artifacts; listen to samples and report failures.
