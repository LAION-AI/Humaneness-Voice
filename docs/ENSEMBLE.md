# The ensemble protocol

How the whole thing fits together: which models are loaded, where each asset
comes from, what runs on which card, and what happens between a user message
arriving and audio leaving.

Every value here is read out of the code. Where a number exists because
something was measured, the measurement is given. The companion pages are
`docs/DIRECTOR.md` (what the director is told), `docs/LEVERS.md` (the three
levers in depth), `docs/ADAPTERS.md` (the adapter families) and `docs/SKILLS.md`
(the measured knowledge layer).

---

## 1. The shape of it

Nine model families run in one FastAPI process plus one llama.cpp process:

| role | model | where from | device |
|---|---|---|---|
| director (default) | `gpt-5.6-luna` | hosted, `https://api.hyprlab.io` | — |
| director (local alternative) | `gemma-4-12B-it-qat-UD-Q4_K_XL.gguf` | `unsloth/gemma-4-12B-it-qat-GGUF` | physical GPU 0 |
| voice acting | `laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3` | HF | `cuda:0` |
| audio codec | `OpenMOSS-Team/MOSS-Audio-Tokenizer-v2` | HF | `cuda:0` |
| retrieval | `laion/voiceclap-commercial` (text tower) | HF | `cuda:0` |
| steering vectors | `p3_vectors_server.npz` | local, `/mnt/nvme/.../steering` | `cuda:0` |
| adapters | 71 burst + 17 delivery + emotion/voice/quality/DPO | local + HF | `cuda:0`, LRU |
| speech recognition | `nvidia/parakeet-tdt-0.6b-v3` | HF | `cuda:1` |
| forced alignment | `Qwen/Qwen3-ForcedAligner-0.6B-hf` | HF | `cuda:1` |
| user-voice scoring | BUD-E-Whisper + `laion/Empathic-Insight-Voice-Small` + VoiceNet heads | local | `cuda:1` |
| voice conversion (optional) | Chatterbox VC | local package | `cuda:1` |
| speaker similarity | `speechbrain/spkrec-ecapa-voxceleb` | HF | CPU |

**The card numbering is inverted and this trips everyone.** `run.sh` launches the
app with `CUDA_VISIBLE_DEVICES="$MOSS_TTS_GPU,$MOSS_LLM_GPU"` = `"1,0"`, so
inside the app process **`cuda:0` is physical GPU 1** (the voice card, which gets
the whole voice stack) and **`cuda:1` is physical GPU 0** (shared with the local
llama.cpp server, and home to the small models). `config.py:21–24` states it.

**Ports.** llama.cpp on `127.0.0.1:8790`, not externally reachable. The app on
`0.0.0.0:8792`, which is. The hosted director's API key is read from a file
outside the repo (`$MOSS_LUNA_KEY_FILE` or `$HYPRLAB_API_KEY`) and never leaves the
server — the browser only ever sends the string `"luna"`.

---

## 2. The base model

`laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3` — a full-parameter
SFT + DPO tuning of the v2 base. Same architecture, tokenizer and prompt format,
so it swaps in directly.

SFT round 3 is the round that trained the **inline delivery directions** back in,
and that is the whole reason this demo uses it: word error on direction-carrying
prompts **0.447 → 0.099**, vocal-burst hit rate **0.516 → 0.666**, every clip
within 0.5 s of the requested length. This demo writes a direction into every
single sentence.

Structure:

| property | value | where |
|---|--:|---|
| sample rate | 48 000 Hz | read from `proc.model_config.sampling_rate` |
| frame rate | 12.5 Hz | `config.FRAME_RATE` |
| samples per frame | 3840 | derived, `sr / FRAME_RATE` |
| RVQ codebooks per frame | 12 | `cfg.n_vq` |
| dtype | `torch.bfloat16` on GPU | `tts_engine.py` |
| attention | `sdpa` | flash-attn 2 is incompatible with this remote-code attention |

The 12 channels are sampled **autoregressively within a frame** — channel c+1's
logits depend on the token sampled for channel c. This matters for guidance, see
§6.

`samples_per_frame` is derived from the frame rate rather than from decoded
length on purpose: deriving it from decoded length lets rounding drift walk the
splice point off the frame grid, which is audible as a stutter well into a long
reply.

Streaming decode uses a sliding window with `CTX_FRAMES` = 160 (≈13 s of left
context, so most replies never slide at all), `HOLDBACK_S` = 0.08 and
`CROSSFADE_S` = 0.06 — widened from 30 ms, which left an audible edge on long
takes.

---

## 3. The request flow

`POST /api/turn`. The response is a stream of
`[1 byte tag][4 byte big-endian length][payload]`, tag 0 = UTF-8 JSON event,
tag 1 = raw PCM int16 LE mono at 48 kHz.

**0 — Parse.** Brain, prompt style (prose or codes), skills flag, persona brief,
voice profile.

**1 — Director.** `LLMAgent.turn` POSTs to an OpenAI-compatible
`/v1/chat/completions` with a strict JSON schema. Thinking is disabled
(`reasoning_effort="none"` hosted, an explicit off for llama.cpp). History depth
8 turns local, 40 hosted. Then `_clean` assembles three things that matter later:

- `general` — fixed identity head + this turn's `delivery` + `BASE_REGISTER` +
  `CONTINUITY` + a studio tail;
- `general_unc` — **the same block with the director's clause removed**. This is
  what makes guidance possible at all;
- `perform` — `{mode, dimension, strength}`, unknown words falling back to `auto`
  and `moderate`.

**2 — Reference selection.** Language is re-derived from the words rather than
trusted from the director's declaration. `VoiceBank.select` picks a clip.

**3 — Retrieval, which overrides step 2.** See §5. Returns a clip *and* an
emotion; the clip replaces the one from step 2.

**4 — Reference stack.** Three layers, in this order, and the order is the point:

```
anchor (who this is)  →  tails of the last 2 turns (how they were just speaking)  →  this moment's condition clip (what to do now)
```

Tails are 50 frames (4 s) from each of the last 2 turns, not whole clips.
Chaining whole clips was measured to collapse speaker similarity from 0.777 to
0.280 by clip four.

**5 — Adapter planning (lever 1).** In assembly order: quality → preference-DPO →
delivery axes → **interference adjustment** → general DPO → emotion from
retrieval → burst adapters with per-class weights from the skills layer → UI
slider overrides. Detail in §7.

**6 — Lever planning (levers 2 and 3).** `levers.plan`. This *must* run after
step 5, because it needs to know which delivery adapters are already active and
it may **remove** one from the stack.

**7 — The `llm` event is emitted**, before any audio, carrying the plan and its
reasons.

**8 — Timed script.** `timed_script.render` computes every duration, pause and
burst length, and the frame budget. See `docs/DIRECTOR.md` §3 — this is the stage
the director must not do itself.

**9 — Generation**, on a producer thread, under a single GPU lock so a turn can
never run under another turn's adapters. Inside: merge adapters → build inputs
(twice, if guided) → attach steering hooks → sample frames → sliding-window codec
decode → clear adapters → extract tail codes.

**10 — Per-chunk post-processing**, strictly in this order: **voice conversion
first, then alignment**, so the aligner sees the converted audio.

**11 — End.** Flush the alignment guard, flush VC, store tail codes into the
session, and score speaker similarity **last**, after the audio has already been
streamed, so it never delays playback.

---

## 4. The prompt that reaches the voice model

Two fields, assembled in `tts_engine.stream_utterance`:

```
GENERAL: <one compact line>; reads as <emotion>; <N.N>s, <EN|DE>.
SCRIPT:
[0.3 seconds pause] (cue) [4.2 seconds duration] the words. (chuckle, 0.3 seconds) …
```

`Text:` carries the tagged script **byte-identically**, and `Tokens:` is the
frame count. `GENERAL` is folded to one line by `timed_script.general_line`,
because the format's own example is one line and the standing prose this demo had
accumulated — register, continuity, genuineness, room — was four clauses the
model had to wade through before reaching the part that changes between turns.

When guidance runs, a second prompt is built: `general_unc` plus
`timed_script.neutralise(tagged)`, which removes every round bracket *without* a
number and keeps every one *with* a number. Durations, pauses, burst tags, words
and the token budget are byte-identical, so the two branches differ in affect and
in nothing else. The neutralised prompt is in distribution: 20 % of the CFG-DPO
corpus had its instruction words removed, and 15–30 % of every supervised round
rendered scripts without directions.

---

## 5. Retrieval

**Index:** `<RETRIEVAL_DIR>/index.npz` + `index.json` — audio centroids per acting
condition, text anchors per emotion, and a by-voice index so each profile is
retrieved against its own recordings.

**Embedding:** `laion/voiceclap-commercial`, text tower only, fp32, on `cuda:0`,
`max_length=128`.

Three design facts, each measured:

- **Nothing is matched against a single clip.** Classifying one clip scores
  0.071, which is precisely the emonet top-1 on this model's own card. Centroids
  and anchors are used instead.
- **Both sides are mean-centred first.** Raw cosines in this space sit in a narrow
  band around 0.9 and a few conditions are near-neighbours of everything.
  Centring moves top-1 from **0.35 to 0.61** against the text anchors and from
  **0.22 to 0.44** against the audio centroids.
- **The query is split in two, and the halves are used differently.** The clip is
  matched on the whole direction; the **emotion is matched on the round-bracket
  cues alone**. A delighted line whose cues read "sharp inhale, delighted laugh,
  voice bursting upward" came back as `Jealousy_and_Envy` because the GENERAL
  text swamped it.

Scoring is `clip_similarity + emotion_bonus + level_penalty`, with
`RETRIEVAL_EMO_BONUS` = 0.5 and `RETRIEVAL_LEVEL_PENALTY` = 0.05. The bonus was
swept: 0.1 → 0.33, 0.3 → 0.50, 0.5 → 0.61, flat above.

**Language is a hard filter, not a preference.** Conditioning a German turn on an
English clip put an audible English accent on the output earlier in this project,
and a soft penalty was not enough.

Retrieval feeds three things forward: the reference clip (as pre-tokenised
codes), the **emotion adapter**, and the `reads as` clause of the GENERAL line.

---

## 6. The three levers

| lever | what it does | implemented in | default |
|---|---|---|---|
| 1. adapter merge | folds LoRA deltas into the weights | `lora_bank.py` | **always on** |
| 2. steering | adds a difference-of-means direction to the hidden state | `steer_engine.py` | available, not default |
| 3. guidance | classifier-free guidance on the delivery condition | `tts_engine._stream_frames_cfg` | available, not default |

**The shipped default is `GEN_MODE = "adapter"`.** `auto` was rolled back after a
listening report: it picked steering on nearly every turn and a human heard
artefacts and an off timbre even though the scoring models liked it. That is the
documented blind spot of the evidence — the steering study states plainly that no
listening test was run on any of its results, and every figure in it is one model
judging another model's output. Both other levers are one environment variable
away, and a director that names a non-`auto` mode explicitly still gets it (see
`docs/DIRECTOR.md` §6).

### Lever 1 — merge

Deltas are folded in place, and pristine CPU copies are restored every
`RESYNC_EVERY` = 25 merges to stop bf16 drift accumulating. Adapters live in CPU
RAM as fp16 and are pushed to the card on demand, LRU-bounded at 8 on GPU and 64
on CPU — the host cache was once unbounded, and with 65 GB of adapters on disk a
long session grew it until the kernel killed the process.

**Tied weights are never merged.** `audio_lm_heads.N` and `audio_embeddings.N`
are weight-tied; merging into one would corrupt the other. They run as forward
hooks instead, and the code asserts rather than trusting the regex — ties are
discovered by `data_ptr()` grouping.

### Lever 2 — steering

`h ← h + α · (v/‖v‖) · ‖h‖`, applied to the **last position only** of each
forward pass, at taps chosen per attribute.

Accumulation is in **fp32 deliberately**: a steering direction is a difference of
means, one to two orders of magnitude smaller than the means themselves, and
doing the accumulation in bf16 quantises it to a few significant bits.

`STEER_ALPHA` = 0.10 is the free setting — emotion percentile 0.4354 → 0.5840
with word error *falling*. Everything breaks above 0.3. Two ceilings guard it:
0.15 per single component (half the measured break point) and 0.25 per layer
after summation, and **exceeding the second refuses the composition rather than
trimming it**.

### Lever 3 — guidance

`logit = unc + g · (cond − unc)`, with g = 1 the exact "off" value and the
control. Below 1 guidance actively hurts (−0.0370 at g = 0.5, t −2.56).

Three structural facts:

- Branches interleave at **channel** level, not frame level, so both are
  conditioned on the same channel prefix.
- The **continue/end decision is taken on the conditional branch alone** and is
  not guided.
- **It does not stream.** The caller passes an effectively infinite chunk size, so
  exactly one decode happens at the end. It costs **1.93×**.

### How `auto` resolves

Emotion → `adapter+steer`. Delivery → `adapter`. Quality → `adapter`. **`auto`
never spends guidance**, because it costs 1.93× and the first two levers reach
the band on their own for the great majority of attributes.

The reasoning is measured per family:

- **Emotion:** the two are cleanly additive (interaction +0.038, t 1.36), and
  steering is five times the adapter's effect there (+0.384 t 9.4 against +0.077
  t 2.8).
- **Delivery:** adapter and steering are significantly **sub**-additive (−0.164,
  t −3.7), so pick one rather than stack them.
- **Quality:** steering does not move it at all (+0.006, t 0.0). The adapter is
  the only lever that moves the quality axes (+0.399, t 6.0).

Hard refusals regardless of mode: steering on quality axes, and steering on the
`_low` delivery tails — the vector table holds the high-minus-low difference and
the two tails are orthogonal (median cos −0.0004).

**A missing asset degrades loudly.** Every drop records a reason into the plan
payload, which is emitted to the client. The principle is stated three times in
the codebase: *a dial that reads a value while the thing it names is off is worse
than no dial.*

---

## 7. The adapters

Full taxonomy in `docs/DIRECTOR.md` §5 and `docs/ADAPTERS.md`. The protocol-level
summary:

**Always merged:** `sft3_dpo:p2` at 1.0, the voice adapter at 0.25, three quality
adapters at 0.25 / 0.5 / 0.5, and `sft3_qdpo:quality_dpo` at 1.5.

**Per turn, chosen by the system:** one emotion adapter at 1.0, picked by
retrieval rather than by the director.

**Per turn, chosen by the director:** at most one delivery axis, at one of
`0.5, 0.75, 1.0, 1.25, 1.5`.

**Per turn, implied by the script:** one burst adapter per burst written, at the
weight the wiki measured for that class (0.25–2.3), falling back to a flat 0.25
(0.5 standing alone) for a class with no measured recipe.

**Doses never come from the language model.** The director picks *which*
condition; the doses are the measured values, applied server-side so a
hallucinated number cannot get through. The same principle governs steering
strengths, which arrive as the words `gentle` / `moderate` / `strong` and are
mapped to a fixed table.

Two adapter families are **parked** as off-distribution on SFT3: the v3 emotion
set and the 57-dimension VoiceNet set, both trained against the untuned v2
weights. The character adapters are in the same position but are offered as a
switch rather than removed.

One measured interference rule: `esthetics_high` × `S_RANT_high` cancels outright
(+0.196…+0.317 and +0.464 alone; −0.012 together), so the aesthetics adapter is
scaled to 0.0 when ranting is active and to 0.5 against `S_DRAM_high`.

---

## 8. The alignment stage

**What it is for.** The model spends the duration it is given. It does not
overrun and it does not stop early — it *fills*. So when a take improvises, the
filler sits **inside** the requested duration rather than past it, and no
clock-based cut can find it.

**Why alignment and not recognition.** We already know which words were supposed
to be spoken. An aligner cannot hallucinate a word that was never in the script,
which a transcriber can.

**Model:** `Qwen/Qwen3-ForcedAligner-0.6B-hf`, bf16, on `cuda:1` — 1.84 GB
loaded, which does not fit beside the voice model on a 24 GB card. Fallback is an
MMS Wav2Vec2 CTC ONNX aligner (31 tokens, 20 ms frames at 16 kHz), fp32 on GPU
and int8 on CPU. Qwen is first **because it is Apache-2.0 and the MMS aligner is
CC BY-NC 4.0** — the only non-commercial component in the stack.

It aligns the generated audio against the *plain words of the script that was
requested* (the third return of `timed_script.render`), and produces
`(word, start, end, score)`. Two edits follow: fade in so full level is reached
exactly at the first word's onset, and fade out after the last word.

**Two protections, both from real failures:**

- A scripted **closing burst is protected** by reading the burst tag's own
  declared length and allowing `ALIGN_BURST_SLACK` = 2.0 × that. A line ending on
  `(breathy giggle)` had 1.03 s removed, which read as a successful trim on every
  metric while actually deleting the giggle the director wrote.
- In the streaming variant the **lead-in decision is taken once**, from a longer
  first stretch (1.3 s, first 3 words). Onsets run to 1.3 s; deciding the lead-in
  at 0.5 s asked the aligner to fit the whole script into half a second, which
  fails — so the fade silently never fired in the stream while firing 28 times in
  36 takes offline.

**Two gates against wrong cuts:** nothing is cut below `ALIGN_MIN_SCORE` = 0.35,
and the tail is not looked for until `ALIGN_TAIL_AFTER` = 0.6 of the requested
duration exists, because forced alignment always places every target somewhere
and against a prefix will happily report the last word finished while half the
line is still unspoken. The stream tail cut additionally requires **all** words
found.

A stream cannot retract what it has emitted, so `StreamGuard` holds a lookahead
buffer and re-aligns everything generated so far every 0.5 s.

---

## 9. The scoring stage

There are three scoring systems, and **none of them gates generation.**

**1. `VoiceScorer` scores the *user's microphone audio*, not the model's output.**
BUD-E-Whisper encoder (fp32) → 40 emotion heads + 4 attribute heads (fp16), plus
VoiceCLAP-commercial (fp32) → 57 VoiceNet regression heads. The heads are fp16
deliberately: each is dominated by one 64 × 1,152,000 projection, 295 MB in fp32,
so all of them together would be 16 GB instead of 5.9 GB.

VoiceNet dimensions are ranked by **z-score against a speech baseline**, not raw
value — half the dimensions sit near zero for every clip, because nobody is a
newsreader.

The result is fed **forward into the director's prompt**, appended to the user
message as `[heard in their voice: … — respond to how they sound, never mention
this note]`. It is an input, not a gate.

**2. `SpeakerSim`** runs ECAPA cosine on the finished take, on CPU, after the
audio has already been streamed. The thresholds in its docstring — regenerate
below 0.58, repair below 0.45, reject below 0.40 — are **the upstream manual's
policy, not implemented behaviour**. Nothing in the server acts on them; their
only consumer is a colour in the browser.

**3. Retrieval scoring**, §5.

**There is no best-of-N and no rejection sampling. The server generates exactly
one take per turn.** This is worth stating plainly because the knowledge layer is
full of best-of-N advice: the wiki's `N` column says how many candidates a burst
class needs for a 90 % chance of realising, and records that best-of-N is a more
effective lever than any weight change. **That advice is not implemented here.**
The two multi-take endpoints — `/api/cfg_sweep` and `/api/say_batch` — are A/B
harnesses for a human to choose between, not automatic selection.

The only per-generation gating is on **length**, not quality:
`MIN_FRAME_FRACTION` = 0.55 refuses the end token below 55 % of the requested
duration, and `STOP_BIAS` = 2.0 nats leans against stopping.

---

## 10. The knowledge layer

`wikiskills/` is a generated corpus: attribute pages, a machine-readable
`coefficients.json` that the lever planner reads, and `VOCAL_BURSTS.md`, which is
the part that changes what the director writes. It is **parsed, never
transcribed**, for the same reason it is generated rather than typed: a copy
stops agreeing with its source the moment either moves.

Three consumers:

- **`levers.py`** reads `coefficients.json` to resolve `auto` per attribute. With
  no row for an attribute, `auto` refuses to make a claim and falls back to
  `adapter`.
- **`skills.py`** reads `VOCAL_BURSTS.md` for which bursts to offer the director
  (`SKILLS_MIN_HIT` = 0.15) and at which per-class merge weight.
- **`timed_script.py`** reads `patterns/vb-*.md` for which round brackets count as
  sounds rather than as instructions.

The third is new, and closes a gap that had been silently costing recipes: the
vocabulary was a hard-coded list of 22 while the wiki carried 117 pages, so **9
of the 36 classes the server actually offered were re-read as delivery
directions** and produced no sound. `tests/test_burst_vocabulary.py` now asserts
the two cannot diverge again.

---

## 11. Where the assets come from

| asset | location |
|---|---|
| adapters | `/mnt/nvme/moss-15-v2-assets/loras/{sft3_burst,sft3_voicenet,sft3_quality,sft3_qdpo,sft3_dpo,sft3_voice,sft3_emotion,profiles}` |
| character / speaker / sports adapters | HF snapshots under `~/.cache/huggingface/hub` |
| reference corpus | `/mnt/nvme/moss-15-v2-assets/refs2` (and `refs3` for top-3 takes) |
| decoded wav cache | `/mnt/nvme/moss-15-v2-assets/ref_wav_cache` |
| pre-tokenised codes | `/mnt/nvme/moss-15-v2-assets/code_cache` (int16 on disk, int64 in RAM) |
| retrieval index | `/mnt/nvme/moss-15-v2-assets/retrieval` |
| steering pack | `/mnt/nvme/moss-15-v2-assets/steering/p3_vectors_server.npz` (~5 MB, distilled from a 112 MB research file) |
| wikiskills | `/mnt/nvme/moss-15-v2-assets/wikiskills` |
| aligner (MMS fallback) | `/mnt/nvme/moss-15-v2-assets/aligner` |
| scoring models | `/mnt/nvme/empathic-insights-voice-small`, `/mnt/nvme/moss-15-v2-assets/{bude-whisper,voicenet-pred}` |

Caches sit on `/mnt/nvme` rather than the root filesystem: five tempo variants of
832 clips is ~3.4 GB and `/` is down to single-digit gigabytes.

**The recommended burst adapters are not published.** The Hugging Face line for
`laion/vocal-burst-lora-adapters` is commented out in `config.LORA_ROOTS`, and the
arm names the wiki uses — `bestmem`, `grpfull`, `bulk_mix_full`, `d2_matched` —
are scratch paths, not artefacts. A clone of this repo on a machine without
`/mnt/nvme` gets the character, speaker and sports adapters from HF and nothing
else.

---

## 12. Boot order, and why it is what it is

1. TTS engine and codec on `cuda:0`.
2. `VoiceBank`, borrowing the TTS processor **and its GPU lock**, so background
   pre-tokenising never collides with a live turn — sharing the card without the
   lock pushed the measured realtime factor from 0.74 to over 3.
3. The aligner, **after** the TTS model, so `onnxruntime` finds cuDNN and cuBLAS
   already pulled in by torch and takes the CUDA provider instead of silently
   falling back to CPU, which is 20× slower and cannot keep a stream fed. The
   `onnxruntime-gpu==1.22.0` pin belongs to the same problem: 1.29 requires CUDA
   13.
4. Retriever, LoRA bank, wiki, steering pack.
5. The side models — ASR, VC, scorer, sim — **serially in one thread**, because
   `transformers` mutates torch's global default dtype while building a bf16
   model.
6. Corpus pre-tokenisation and adapter preloading, in the background.

---

## 13. Known gaps

Recorded here because they are load-bearing and currently open.

1. **The burst adapters recommended by the wiki are not published**, so the
   recipes cannot be reproduced outside this machine. §11.
2. **No best-of-N.** The knowledge layer's central operational advice — generate
   N candidates, keep the one that realises — has no implementation in the
   server. §9.
3. **A weight ceiling the wiki argues for and does not apply.** The 2026-09-05
   addendum measures all four adapter arms breaking the WER gate at w = 2.0 and
   states that no recipe should name it; the §51/52 table above it still carries
   ten recipes at 1.8–2.3, and that table is what `skills.py` parses and serves.
   `BURST_LAM_MAX` exists as a one-variable switch (default 2.3, i.e. no change);

> **Note added on the demo side, 6 September 2026.** The weights this page
> assumes are not the ones this server runs. `BURST_LAM_MAX` is **1.25**
> here, under a **total** budget of 2.0 across a turn, because two burst
> adapters at 1.5 each destroyed the line in 5 of 5 seeds under the full
> shipped stack — it is the sum that breaks it, not the single weight.
> See [`BABBLE.md`](BABBLE.md).

   setting it to 1.5 enforces the addendum.
4. **Three prompt rules are unenforced**: a number inside a round bracket, the
   ten-word minimum, and opening on a burst. The first is the dangerous one — it
   silently converts a delivery cue into a vocal burst. `docs/DIRECTOR.md` §7.
5. **The similarity thresholds are documentation, not behaviour.** §9.
6. **`SFT3_VOICE_LAM` is dead config.** It is defined at 1.0 and read by nothing;
   the weight actually applied is `PROFILE_LORA_LAM` = 0.25.
