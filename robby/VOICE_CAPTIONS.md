# Uncertain voice-delivery context for Robby

Robby now sends **what was said** and a small, separate hint about **how it sounded** to the response model. Humaneness Ears Medium runs on a GPU in parallel with the pause-final ASR pass. A deterministic procedural captioner selects up to two delivery dimensions, one emotional cue, and tempo. It does not run another ASR or a generative caption model.

The text is explicitly uncertain. The response prompt says that acoustic predictions are not facts about feelings, intent, honesty, health or identity; background speech, noise, acting and short clips may mislead them. The user's own account takes priority. The model may respond gently when the context supports it, without announcing diagnoses, quoting scores, or adding a disclaimer to every answer. This does not establish that an emotion estimate is accurate.

## Components and provenance

| Component | Pin / role |
| --- | --- |
| [Humaneness Ears Base/Medium](https://huggingface.co/laion/humaneness-ears-base-medium) | Medium only; `818506970d93c809a3295a0a02c40dee8ff2bfc3`; encoder plus multitask heads |
| Medium checkpoint SHA-256 | `00f30de0f7d656d2a1e86bdc629673ff8b576532e1263b02bcebdfc2763e54f1` |
| [Procedural Voice Captions](https://github.com/LAION-AI/procedural-voice-captions) | `7ebd4a9b7da96df989af8fc6ad0ce34f4e8aa3ed`; pinned Medium backend, speech-reference normalization, caption rules |
| Caption normalization | `baseline_stats_humaneness_medium_speech.json`; the upstream measured speech reference; z-scores are deviations, not confidence probabilities |
| Runtime | Four CPU threads for frontend/dispatch; GPU BF16 autocast for the encoder; no text decoder |

Attribute LAION — Procedural Voice Captions / Humaneness Ears Medium integration, and retain its [license notice](https://github.com/LAION-AI/procedural-voice-captions/blob/7ebd4a9b7da96df989af8fc6ad0ce34f4e8aa3ed/LICENSE-HUMANENESS-MEDIUM.md). The integration reuses its pinned rules and changes selection to an allowlist of prosodic dimensions and ordinary emotional cues. It excludes demographic, timbre, perceived honesty, content-enjoyment and recording-quality judgments from the conversational hint. Upstream weights and normalization are not retrained or relicensed.

## Scheduling and isolation

```text
Neural speech pause -> snapshot this turn's audio
                       |-> selected final ASR on GPU 0 -----|
                       |-> Medium + caption on GPU 1 -------|-> speaker admission
                                                          |-> use ready hint or skip
                                                          |-> LLM -> sentence TTS
```

The live two-GPU deployment uses physical `cuda:0` for ASR and `cuda:1` for captions. The caption worker is an independent loopback-only process. TTS already uses the machine too; two GPUs do not imply dedicated hardware. A one-GPU machine can set `ROBO_CAPTION_DEVICE=cuda:0`, but contention and latency must be remeasured there. CUDA ordinals are relative to that process's `CUDA_VISIBLE_DEVICES`.

`SentenceSession._maybe_probe()` starts the caption request before awaiting final transcription. The future travels with that same turn, not a global “last emotion.” `completed_hint()` consumes only an already-finished result after speaker admission. An unfinished result is cancelled and omitted, with **zero additional awaited caption delay** before the LLM. A fresh turn, resumed speech, explicit handover, disabled checkbox, typed input or cancellation cannot reuse an old hint. Existing speaker gates, TYGER and Laya do not take emotion scores as authority.

Only the last **eight seconds of the current turn** are analyzed. That scope is stated in the metadata for longer turns. Audio shorter than 0.5 seconds, silence, nonfinite input, busy worker or a failed/late job produces a transcript-only reply. The sidecar accepts one job at a time and rejects competing requests rather than building an audio backlog. Its HTTP deadline defaults to 350 ms. There is no microphone file cache and no published private recording or voiceprint.

The checkbox **Use uncertain tone-of-voice hints** is on by default, independently persisted, and appears under **Speech pause & interruption options**. The UI displays the current hint, uncertainty, window length and timing. Typed messages have no inferred tone. In OpenRouter mode the short caption is text sent to the chosen provider alongside the transcript; raw audio remains on this server.

## Reproduce

Install the existing Robby requirements and GPU PyTorch build first. The bootstrap fetches only the pinned caption sources and approximately 389 MB of Medium inference weights, not optimizer checkpoints or another ASR model:

```bash
python bootstrap_voice_captions.py
# Separate process; ensure both GPUs are visible, or choose its relative device.
ROBO_CAPTION_DEVICE=cuda:1 python voice_caption_worker.py
# Existing compatible snapshot: set HUMANENESS_EARS_DIR to its directory.
# Alternative shared HF cache: set ROBO_CAPTION_CACHE before bootstrapping.

python run_unit_checks.py
python check_voice_captions.py
python check_voice_caption_prompt.py
uv run --no-project --with playwright --python /path/to/python \
  python browser_voice_caption_check.py --url https://YOUR-TUNNEL
```

The owned stack launcher also supports `--with-voice-captions`. Do not start a second sidecar on port 8802. Its state endpoint is `http://127.0.0.1:8802/state`. Set `ROBO_CAPTION_URL`, `ROBO_CAPTION_DEVICE`, `ROBO_CAPTION_THREADS` and `ROBO_CAPTION_TIMEOUT_S` for another local deployment. The browser test needs a free native ASR slot and the pinned public v2 test fixtures described in `VOICE_ASSISTANT_TEST_PROTOCOL.md`.

## Measurements, 26 October 2026

Warm HTTP wall-time medians, three repeats per duration/mode, on the two RTX 3090 machine. The timing probe repeats one synthetic recording to control input duration; these are not independent emotion-quality examples. Parallel time is the wall time until both HTTP requests finish, not a sum of component times.

| Audio duration | Parakeet alone | GPU caption alone | Both in parallel |
| --- | ---: | ---: | ---: |
| 1 s | 42.6 ms | 20.1 ms | 44.2 ms |
| 2 s | 43.9 ms | 20.5 ms | 49.6 ms |
| 4 s | 62.9 ms | 21.6 ms | 62.4 ms |
| 8 s | 85.4 ms | 26.1 ms | 89.4 ms |

The earlier CPU probe took about 59–325 ms for the same duration range. GPU is now the live default. Raw GPU results: `static/voice-caption-timing.json`; the retained CPU receipt is `static/voice-caption-cpu-before.json`.

The real HTTPS Chromium test passed default selection, one-time migration of old defaults, subsequent explicit ASR preference retention, microphone-transport enrollment, Parakeet final transcripts, actual GPU hints, response audio, caption-off behavior and persistence. Two accepted turn segments had available GPU captions: 39.39/32.35 ms inference, 52.06/44.84 ms HTTP, **0 ms extra awaited caption delay**. The disabled turn still produced an answer. See `static/voice-caption-browser.json` and its exact fixture/source hashes. The test injects synthetic PCM, not a physical microphone.

All **465 offline checks** pass. A real Qwen/Cerebras smoke request supplied a deliberately conflicting anger hint and the explicit user statement “I am not angry”; the response followed the joke request without insisting on anger. Its opening wording was awkward. This single example does not establish empathy or calibrated emotion recognition. Raw response: `static/voice-caption-model.json`.

## Separate known limitation: acoustic Stop reliability

Do not confuse these integration passes with the harder interruption suite. The new completion-Stop run recognized some closing words but rejected their speaker evidence; all four positive noise/phase cases missed cancellation. The negated control passed. The synthetic “Halt die Klappe” take also produced a poor clean Parakeet transcript (“Halte gemeins.”), so its input fidelity needs review. The subsequent spin check timed out. Preserve these failures; neither vocabulary fixes nor acoustic captions establish universal noisy-room interruptibility. See `static/completion-stop-browser.json` and the full test protocol.

## Bridge-phrase proposal

The separate [four-page English plan](https://github.com/LAION-AI/Humaneness-Voice/releases/download/robby-caption-bridges-v1/Robby_Decision_Model_Bridge_Phrases_Plan.pdf) covers cached contextual reactions, a taxonomy, simultaneous main-answer generation, expressions, cancellation and A/B/C evaluation. It is a design proposal, not an implemented bridge controller. Source: `BRIDGE_PHRASES_PLAN.md`; rebuild with `uv run --no-project --with reportlab python build_bridge_phrase_pdf.py` (DejaVu fonts required).
