# Models, adapters and serving compatibility

Humaneness Voice is an agent assembled from several separately licensed artifacts. [`../manifests/agent-default-v1.json`](../manifests/agent-default-v1.json) records their full Hugging Face revisions. The table below is a navigation map, not an assertion that every upstream recording has the same rights as its annotation or model card. The base is a complete fine-tuned checkpoint. LoRAs are *deltas trained against it* and cannot be treated as independent models or interchanged freely with the older untuned MOSS v2 line.

## Humaneness Voice Base and required audio path

| Role | Repository | Declared card licence | Notes |
| --- | --- | --- | --- |
| Humaneness Voice Base | [`laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3`](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3) | CC BY 4.0 | Historical SFT3 ID; this is the full checkpoint *without* LoRAs. The model card describes the 398,282-row final expressive round, timing format, inference and limitations. |
| Architecture lineage | [`OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5`](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5) | See upstream card | Original MOSS family reference. The immediate predecessor to the released fine-tune is described on the Base card; do not substitute v1.5 weights for the Humaneness Voice Base when loading the adapters below. |
| Codec | [`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) | Apache 2.0 | 12-codebook audio tokenizer at 12.5 frames/s; required for decode. |
| Reference data | [`TTS-AGI/moss-voice-profile-references`](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references) | Apache 2.0 card label | Selected recorded/synthetic reference clips, anchors and metadata. Setup extracts one voice's winning takes rather than downloading the full repository. Check recording provenance and identity rights separately. |

The runtime keys and folders still say `sft3_*` because they identify adapter *compatibility*. The public component names are Humaneness Voice Base, voice identity, preference, quality, emotion, delivery and vocal-reaction adapters. `MOSS_BASE_FAMILY=v2` is only for a deliberate legacy configuration; with a future renamed Base repo, the SFT3-compatible behavior remains the default.

## Compatible adapter families

| Public role | Pinned repository | Installed runtime key and default | Card licence |
| --- | --- | --- | --- |
| Preference DPO p2 | [`laion/moss-va-sft3-dpo-lora-p2`](https://huggingface.co/laion/moss-va-sft3-dpo-lora-p2) | `sft3_dpo:p2` at **1.0**. The Hub repo stores adapter files at its root; `fetch_all.py` places them under `loras/sft3_dpo/p2/`. | CC BY 4.0 |
| Voice identity | [`laion/moss-va-sft3-voice-loras`](https://huggingface.co/laion/moss-va-sft3-voice-loras) | `sft3_voice:<profile>` at **1.0**, paired with the *same* profile's reference anchor. Ten tested pilot IDs are in the manifest; one is selected per turn. | CC BY 4.0 |
| Quality DPO | [`laion/moss-va-sft3-quality-dpo-lora`](https://huggingface.co/laion/moss-va-sft3-quality-dpo-lora) | `sft3_qdpo:quality_dpo` at **1.5**. The repository root has the same safetensors SHA as `checkpoints/step376`; step1504 is a different research checkpoint, not the default. | Apache 2.0 |
| Quality axes | [`laion/moss-va-sft3-quality-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-quality-lora-adapters) | `sft3_quality:genuineness_high` **0.25**, `blend_high` **0.5**, `esthetics_high` **0.5**. These are generation LoRAs, not the separate scoring heads. | Apache 2.0 |
| Emotion | [`laion/moss-va-sft3-emotion-loras`](https://huggingface.co/laion/moss-va-sft3-emotion-loras) | One `sft3_emotion:<emotion>` at **1.0** when selected by retrieval. The full installer fetches 40; the minimal installer fetches Relief only. | CC BY 4.0 |
| Vocal delivery | [`laion/moss-va-sft3-voicenet-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-voicenet-lora-adapters) | At most one `sft3_voicenet:<axis>` by default. Available doses are 0.5, 0.75, 1.0, 1.25 and 1.5; no single dose is imposed on every prompt. | Apache 2.0 |
| Vocal reaction | [`laion/moss-va-sft3-vocal-burst-lora-adapters`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters) | `burst:<class>` only for requested, available reactions. WikiSkills choose per-class doses, capped at 1.25 for one adapter and 2.0 combined under the shipped recipe. | Apache 2.0 |

The LoRA loader in [`../lora_bank.py`](../lora_bank.py) discovers each named child directory containing `adapter_model.safetensors` and `adapter_config.json`. It applies adapter deltas separately, so rank-64 preference and rank-16 voice/emotion LoRAs can coexist. Most module deltas are temporarily merged for streaming speed; weight-tied audio heads use hooks to avoid corrupting the shared embedding weights. The adapter state is cleared between turns. The public `setup/fetch_all.py` arranges the root-level DPO files and the `adapters/` trees so those discovery rules resolve on a fresh machine. Older server documents describe historical doses; [`../config.py`](../config.py) and the manifest are authoritative for this snapshot.

## Mind, ears and optional tools

| Role | Repository | Used for |
| --- | --- | --- |
| Local director | [`unsloth/gemma-4-12B-it-qat-GGUF`](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) | A local LLM through llama.cpp writes text and per-sentence acting cues. It never synthesizes audio. Other supported backends require their own credentials and versioning. Verify the upstream model terms independently. |
| Retrieval embedding | [`laion/voiceclap-commercial`](https://huggingface.co/laion/voiceclap-commercial) | Audio/text embeddings for reference selection and instruction-fit contribution in candidate ranking. |
| Genuineness head | [`laion/voiceclap-commercial-genuineness`](https://huggingface.co/laion/voiceclap-commercial-genuineness) | Scores perceived naturalness over VoiceCLAP embeddings. A small predictor, **not** a TTS LoRA. |
| Vocal-burst blend head | [`laion/voiceclap-commercial-vocalburst-blend`](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) | Scores how a laugh, sob, gasp or similar event joins the neighboring speech. A predictor, **not** a TTS LoRA. |
| ASR | [`nvidia/parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | Transcription and word-error term in the ranker; optional microphone input in Studio. |
| Aligner | [`Qwen/Qwen3-ForcedAligner-0.6B-hf`](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | Optional speech-boundary trimming. |
| Restoration | [`sarulab-speech/sidon-v0.1`](https://huggingface.co/sarulab-speech/sidon-v0.1) | Optional post-processing, off by default; not recommended for every reaction. |
| Studio emotion analysis | [`laion/Empathic-Insight-Voice-Small`](https://huggingface.co/laion/Empathic-Insight-Voice-Small) | Optional microphone-side diagnostic, not part of the TTS weights. |
| Studio voice attributes | [`laion/voicenet-dimension-predictors-commercial`](https://huggingface.co/laion/voicenet-dimension-predictors-commercial) | Optional attribute diagnostics, not a generation adapter. |

The two small VoiceCLAP heads are used for data filtering and best-of-N selection. Their reported model-label validations do not establish complete human validity. The listener-based Voice Acting Arena remains the external evaluation. [`BEST_OF_N.md`](BEST_OF_N.md) explains the within-set normalization, instruction-fit and ASR terms; requested vocal reactions may add an event-detector term. `/api/speak` defaults to ten candidates, while ordinary chat streaming does not use best-of-N unless requested.

## Research and compatibility boundary

- [`laion/moss-va-sft3-vocal-burst-lora-adapters-v2`](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2) contains alternate class, group, dose and ablation arms. Its card currently has no declared licence metadata. The recommended installer does **not** fetch it or claim a default benefit from it.
- [`laion/moss-va-sft3-burst-stop-dpo-lora`](https://huggingface.co/laion/moss-va-sft3-burst-stop-dpo-lora) is experimental and has default weight **0**. The quality-DPO repository's step1504 is also not the default.
- Older [`laion/moss-voice-profile-loras-500`](https://huggingface.co/laion/moss-voice-profile-loras-500), [`TTS-AGI/moss-character-loras-refined-public`](https://huggingface.co/TTS-AGI/moss-character-loras-refined-public) and v2 VoiceNet adapters target an earlier checkpoint. They are not interchangeable with the Humaneness Voice Base and are not installed by the recommended setup.
- The repository retains historical research notes and evaluation scripts. A figure measured under a different adapter stack, guidance value, voice or checkpoint should not be relabelled as a current default result.

To check the effective local stack, use `python setup/check_release.py` and `GET /api/adapters` after startup. A missing adapter is not silently part of the system merely because its repository appears in this table. Each Hub item has its own licence and provenance; the [`../LICENSE`](../LICENSE) file applies to this code and documentation, not to every asset linked here.
