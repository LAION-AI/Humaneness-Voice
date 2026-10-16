# Checkpoints, bases, adapters and dependencies

This page describes the October 2026 Playground. The older SFT3 system has a
separate [asset guide](MODELS_AND_ADAPTERS.md) and
[pinned manifest](../manifests/agent-default-v1.json). Never transfer an adapter
between these families merely because both contain a Qwen/MOSS decoder.

## Six selectable current model options

| UI/API ID | Source | Loaded weights and role |
| --- | --- | --- |
| `small_rl` — **default** | [Small 1.1 Alpha RL r128 v1](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1) | Bundled final SFT5 step 6,109 **plus** fixed r128 RL residual adapter; 2,434 optimizer updates |
| `large_rl` | [Large 1.1 Alpha RL r128 v1](https://huggingface.co/laion/Humaneness-Voice-Large-1.1-Alpha-RL-r128-v1) | Bundled final SFT5 step 3,055 **plus** fixed r128 RL residual adapter; 2,370 optimizer updates |
| `small_acting` | [Humaneness Voice Small](https://huggingface.co/laion/Humaneness-Voice-Small) | Complete `checkpoints/HVS-Small-2026-10-12/model_bf16.pt`; stable API ID retained after its former Acting weights were replaced |
| `large_acting` | [Large Acting](https://huggingface.co/laion/Humaneness-Voice-Large-Acting) | Complete `checkpoints/step-00000432/model.safetensors` |
| `small_alpha` | [Small 1.1 Alpha](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha) | Opt-in interim `intermediate-checkpoints/step-00003775/model.safetensors`, not the final base used by Small RL |
| `large_alpha` | [Large 1.1 Alpha](https://huggingface.co/laion/Humaneness-Voice-Large-1.1-Alpha) | Opt-in interim `intermediate-checkpoints/step-00002377/model.safetensors`, not the final base used by Large RL |

All exact runtime commits, checkpoint SHA-256 values, limits and adapter hashes:
[`playground-2026-10-16.json`](../manifests/playground-2026-10-16.json), derived
from [`acting_models.py`](../playground/runtime/acting_models.py).
The [file inventory](../manifests/checkpoint-inventory-2026-10-16.json) links
individual weights at immutable commits, including other checkpoints retained
inside each relevant repository. Files with optimizer/training names are not
automatically inference weights. A latest Hub inventory is not the runtime pin.

## RL bundles: what to download and how to load

Each RL repository supplies its matching full base under `base/`,
`adapter_config.json`, `adapter.safetensors`, inference code and documentation.
Keep the complete bundle structure, including base processor/config assets.

| Bundle | Bundle revision | Exact upstream base revision / checkpoint |
| --- | --- | --- |
| Small RL | `182bc700cac29b2d69295a33398b13e1ca7fba69` | [Small Alpha](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha/tree/b92b1f86e17344b33bf12f255767f589b785acee), `checkpoints/step-00006109/model.safetensors` |
| Large RL | `7f050dc0e5c1b2415258c357c1edc27e1452cf07` | [Large Alpha](https://huggingface.co/laion/Humaneness-Voice-Large-1.1-Alpha/tree/a055f498b06988531134fa21f06a24998fb93f19), `checkpoints/step-00003055/model.safetensors` |

Use only the paired base/hash. Round base floating weights to BF16, then upcast
the rounded base to FP32; retain separate FP32 adapter residuals; disable TF32
and policy autocast. The custom format is `humaneness-moss-attention-lora-v1`,
rank/alpha 128, residual scale 1; both semantic attention and local attention
projections are adapted. It is **not** ordinary `PeftModel.from_pretrained()`.
[`rl_adapter.py`](../playground/runtime/rl_adapter.py) validates every key,
shape and parameter count; [`acting_engine.py`](../playground/runtime/acting_engine.py)
implements base loading and the model-specific precision path.

Do not merge the residual into a BF16 base to save memory and call that the
published RL model. The current verified boundary is single take and parallel
Best-of-N; [chunk-search precision is not verified](PLAYGROUND_EXPERIMENTS.md#current-rl-validation-boundary).

## Earlier Small weights and LoRAs — available, not active in the current UI

| Source | What it contains / compatibility |
| --- | --- |
| [Humaneness Voice Small](https://huggingface.co/laion/Humaneness-Voice-Small/tree/main/checkpoints) | Original S1–S10 ladder, reference full-FT releases and later full exports. Follow the exact checkpoint path and associated inference version, not the word “Small” alone. Earlier studies used S3 or S3-Ref15-FullFT, not current RL. |
| [Small Acting](https://huggingface.co/laion/Humaneness-Voice-Small-Acting) | Separate earlier full Acting model. Its previous UI identifier is not evidence the UI still loads these weights. |
| [Rank-1 LoRAs](https://huggingface.co/laion/Humaneness-Voice-Small-Rank1-LoRAs) | Earlier Small speaker/emotion/style bank; packed custom loading. Not the new RL adapter and not compatible by assumption with Acting/Alpha/RL. |
| [DPO LoRAs](https://huggingface.co/laion/Humaneness-Voice-Small-DPO-LoRAs) | Earlier Small preference adapters, including rank-16 variants. They are **disabled/absent from the current UI**; do not add one to an RL base as a “quality default.” |
| [Ref15 r128 LoRA](https://huggingface.co/laion/Humaneness-Voice-Small-Ref15-LoRA-r128) | Earlier reference-conditioned r128 adapter. This is a different artifact from RL r128; the common rank does not imply a common base. |
| [500 voice profiles](https://huggingface.co/laion/moss-voice-profile-loras-500) | Reference/voice-profile provenance for catalog research; not 500 validated steering vectors for the new RL checkpoints. |
| [Original 4.55B SFT3 base](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3) | Base for the separate September agent and its SFT3 DPO, identity, emotion, delivery, quality and burst LoRAs. See [complete historical mapping](MODELS_AND_ADAPTERS.md). |

The original Small S3 adapter loader checks base SHA-256
`60079723bf797a81107def1e6fc479e5353b3cc51f3780f98ea65a9f5c109f97`.
Honor adapter configuration/base declarations and loader checks; do not disable
them to force a combination. Later fully fine-tuned/merged exports are not that
base merely because parameter names match. Historical mixtures were experiments,
not a general compatibility guarantee. Follow each adapter's pinned README for
its own training/inference code and resume checkpoint, if supplied.

## Other models in the current system

| Component | Published asset | When needed |
| --- | --- | --- |
| Audio codec | [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) | Always for reference encoding and audio decoding; 12 codebooks, 12.5 frames/s, 48 kHz |
| Prompt/audio fit and retrieval | [VoiceCLAP Commercial](https://huggingface.co/laion/voiceclap-commercial) | Shared audio/text tower; default ranking and optional same-speaker reference retrieval |
| Quality | [Humaneness Ears Base/Medium](https://huggingface.co/laion/humaneness-ears-base-medium) | Default Content Enjoyment ranking; optional Production Quality, MOS and other heads. “Medium” here uses Whisper Small, not Whisper Medium. |
| ASR + word timestamps | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | WER gate, microphone and buffered live speech-boundary checks |
| Completed-take forced alignment | [Qwen3 ForcedAligner 0.6B HF](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | Default final fades/cutoff. Do not substitute the historical non-commercial MMS fallback without checking terms. |
| Optional speaker embedding | [ECAPA VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | Explicit legacy speaker-similarity reward |
| Optional timbre/style predictors | [VoiceNet dimension predictors](https://huggingface.co/laion/voicenet-dimension-predictors-commercial) | Timbre-vector speaker proxy or 1–4 selected reward dimensions; uses VoiceCLAP embeddings |
| Optional emotion experts | [Empathic Insight Voice Plus](https://huggingface.co/laion/Empathic-Insight-Voice-Plus) + [BUD-E Whisper](https://huggingface.co/laion/BUD-E-Whisper) | Adaptive/manual emotion dimensions; intensity estimates, not class probabilities |
| Legacy diagnostic quality heads | [Genuineness](https://huggingface.co/laion/voiceclap-commercial-genuineness), [Burst Blend](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) | Current initializer loads these even when Ears is the selected reward; share the VoiceCLAP tower |
| Optional burst position/type | [Vocalburst locator](https://huggingface.co/laion/vocalburst-locator), [commercial detector](https://huggingface.co/laion/vocal-burst-detector-commercial) | Position-aware requested-burst checks; an overall blend score does not prove realization at the requested place |
| Optional voice conversion | [Humaneness Voice Converter](https://huggingface.co/laion/Humaneness-Voice-Converter) + [LAION Chatterbox VC code](https://github.com/LAION-AI/chatterbox-voice-conversion) | Current full merged 40k converter, off by default; applies before final ranking |
| Earlier VC research | [GROW CE/Speaker 24k](https://huggingface.co/laion/chatterbox-s3gen-vc-grow-ce-speaker), [GROW parent](https://huggingface.co/laion/chatterbox-s3gen-vc-grow), [vanilla Chatterbox](https://huggingface.co/ResembleAI/chatterbox) | Historical ablations/bootstrap; not silent fallbacks for the current full converter |
| Reference data | [Acted profile corpus](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references), [500 profiles](https://huggingface.co/laion/moss-voice-profile-loras-500) | Only selected consenting speakers/clips are needed; source audio is not bundled in Git |

Pinned Ears, ECAPA, emotion/dimension, burst and converter revisions live in the
source/config and current manifest/inventory. Some inherited loaders still
resolve codec, ASR, VoiceCLAP or aligner without an explicit revision argument;
the older manifest gives published reference pins, but is **not proof of the
exact model object resident in a running process**. Record resolved paths/hashes
and pin those loaders for a new controlled experiment. Licenses and reference
consent must be checked per asset; “commercial” in a name is not a legal clearance.
