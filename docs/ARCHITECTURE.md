# How Humaneness Voice is assembled

An ordinary TTS call asks a model to read a string. Humaneness Voice treats a scene as a performance problem: it decides *how* each sentence should be spoken, gives the speech model a compatible prompt and reference, then can listen to several takes before returning one. The four functional parts are deliberately separate so a new director, instrument or judge can be evaluated without renaming the whole agent.

## Actor's mind: direction

[`../llm_agent.py`](../llm_agent.py) holds the director prompt and provider interface. The director receives the scene or chat turn, a stable speaker description and the available performance controls. It emits words, a `GENERAL` voice/delivery description and a script with local cues. It can ask for a laugh, sigh or other vocal reaction only from classes advertised by the loaded burst catalog. [`../skills.py`](../skills.py) reads the checked-in [`../wikiskills/`](../wikiskills/) so the director sees measured class-specific recipes, limitations and conflict rules; `wikiskills_legacy/` is preserved only for older experiments. `setup/build_system_prompts.py` regenerates [`SYSTEM_PROMPTS.md`](SYSTEM_PROMPTS.md) from the actual code.

The director is a language model, not an audio generator. The default is a local Gemma GGUF via llama.cpp. A hosted OpenAI-compatible endpoint may be substituted, with the resulting model ID, prompt and version recorded if outputs are compared scientifically. The user text is not a substitute for a fixed spoken script in a controlled evaluation: [`../timed_script.py`](../timed_script.py) separates spoken words from inline directions and computes the duration sum.

## Actor's instrument: generation

[`../tts_engine.py`](../tts_engine.py) loads Humaneness Voice Base and the MOSS audio codec, then streams audio frames. [`../voice_bank.py`](../voice_bank.py) holds selected reference recordings and pre-tokenizes them; [`../retrieval.py`](../retrieval.py) maps a direction to a reference condition using VoiceCLAP. [`../voice_profiles.py`](../voice_profiles.py) pairs a voice identity LoRA with an anchor from the *same* profile. This pairing matters: a mismatched anchor and identity adapter fight one another.

[`../lora_bank.py`](../lora_bank.py) loads multiple compatible LoRAs at distinct strengths. In the default profile, preference p2, the selected voice, quality DPO and the three quality axes are present; an emotion, delivery direction and vocal reaction are conditional. [`MODELS_AND_ADAPTERS.md`](MODELS_AND_ADAPTERS.md) records which repository and runtime key supplies each role. LoRA weights are not merged permanently into a redistributed checkpoint. The instrument also supports timing, pauses, CFG and optional steering, but the latter research controls are not silently credited as part of every output.

The prompt sent to the base has fixed fields inside one `<user_inst>` block. `SCRIPT:` contains spoken segments, inline acting cues and timed reactions. `Text` repeats the script exactly, and the `Tokens` field is the sum of its duration tags at 12.5 frames per second. The director does not do this arithmetic itself. See [`PROMPTING.md`](PROMPTING.md) for a worked block and the distinction between `(laugh)` as a sound and a parenthesized delivery cue.

## Actor's ears: selection and diagnostics

[`../bestofn.py`](../bestofn.py) can generate and rank candidates. The ranker combines within-candidate-set normalized genuineness, vocal-burst blending and VoiceCLAP audio/text fit; it adds a vocal-reaction detector term only when requested and applies an ASR-based intelligibility gate. An absolute model score is not a calibrated probability that a human will prefer the take. Its job is to pick among takes of the *same* scene; independent human comparisons are still needed to test the choice. [`../score_engine.py`](../score_engine.py) adds optional emotion and VoiceNet diagnostics for the Studio interface. These analyzers do not change the base weights.

The streaming one-take route can emit PCM before a line finishes. Best-of-N must first finish candidates and score them, so it has a different latency and may use more GPU memory. The browser's `/api/turn` and the headless `/api/speak` expose both modes. [`SERVER.md`](SERVER.md) documents their payloads, and [`BEST_OF_N.md`](BEST_OF_N.md) documents the reward and its observed trade-offs.

## Actor's tools and applications

`app.py` is the FastAPI application; `index.html` is the main browser demo and `studio.html` adds microphone-side diagnostics. Optional tools include [`align_engine.py`](../align_engine.py) for boundary trimming, [`sidon_restore.py`](../sidon_restore.py) for restoration, `vc_engine.py` for voice conversion, and guidance/steering controls. `run.sh` starts only the local director plus app by default; optional services are separate. None of these optional tools are needed to understand the base-plus-adapter identity of the release.

## Artifact boundaries

1. GitHub carries the code, prompts, skills, manifest and lightweight tests; it does not carry model weights, voice audio or a packaged 4.55B checkpoint.
2. Hugging Face carries individually versioned models and reference data; [`../manifests/agent-default-v1.json`](../manifests/agent-default-v1.json) pins the source revisions and keeps their old IDs visible for provenance.
3. Generated local caches, reference selections, retrieval indexes and API keys stay outside Git. `setup/fetch_all.py` writes `.env.generated` with non-secret pinned paths; `.env.local` is for machine-specific overrides.
4. The Voice Acting Arena paper evaluates finished takes under a comparison protocol. A live server response is not automatically the paper's frozen pass@10 treatment: scene selection, director prompts, model versions and ranking settings must match.
