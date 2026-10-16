# Shipped defaults

> Historical September SFT3 guide. For the October Small/Large RL Playground,
> start with [the current guide](PLAYGROUND_PROMPTING.md) and the repository README.

This page describes the **Humaneness Voice agent-default-v1** release. The source of truth is [`../config.py`](../config.py); [`../manifests/agent-default-v1.json`](../manifests/agent-default-v1.json) pins the external artifacts. `python setup/check_release.py` checks the main values against the code. Environment variables and request fields can override them, so record the effective configuration of a particular take.

| Setting | Shipped value | Meaning |
| --- | --- | --- |
| `MOSS_DEFAULT_BRAIN` | `local` | llama.cpp director, no hosted key required |
| `MOSS_DATA_DIR` | `~/.cache/humaneness-voice` | portable asset root; `HUMANENESS_DATA_DIR` is the preferred alias |
| `MOSS_TTS_REPO` | pinned local Base snapshot after setup | historical HF ID is in the manifest |
| `MOSS_CODEC_REPO` | pinned local codec snapshot after setup | separate tokenizer |
| `MOSS_DEFAULT_PROFILE` | `emolia_c1699` | one voice ID, requiring a matching LoRA and anchor |
| `MOSS_SFT3_DPO_LORA`, `MOSS_SFT3_DPO_LAM` | `sft3_dpo:p2`, `1.0` | preference p2 |
| `MOSS_PROFILE_LORA_LAM` | `1.0` | selected speaker identity adapter |
| `MOSS_LAM_QDPO` | `1.5` | quality-DPO step376 |
| `MOSS_LAM_GENUINE`, `MOSS_LAM_BLEND`, `MOSS_LAM_ESTH` | `0.25`, `0.5`, `0.5` | generation quality-axis LoRAs |
| `MOSS_SFT3_EMOTION_LAM` | `1.0` | one retrieved emotion, if available |
| `MOSS_SFT3_VN_MAX` | `1` | no more than one vocal-delivery adapter per turn |
| `MOSS_BURST_SET` | `recipe` | class-specific WikiSkill choice, not a fixed adapter for every event |
| `MOSS_BURST_LAM_BUDGET`, `MOSS_BURST_LAM_MAX` | `2.0`, `1.25` | combined and per-adapter vocal-reaction limits |
| `MOSS_BON` and `MOSS_BON_N` | off; `8` if switched on for chat | streaming chat avoids selection latency by default |
| `MOSS_SPEAK_BEST_OF`, `MOSS_SPEAK_GUIDANCE` | `10`, `3.0` | headless `/api/speak` defaults; can be overridden in the request |
| `MOSS_ALIGN` | on | Qwen forced alignment where the asset is installed |
| `MOSS_SIDON` | off | restoration is opt-in; it can alter intense reactions |

The old `sft3_*` strings are adapter keys, not a second system name. The default full installer creates their expected folders. The research-only burst-v2 and burst-stop DPO alternatives are not default serving assets. A missing optional scorer or adapter must be reported from the live state rather than silently described as active. `docs/ADAPTERS.md` contains the mechanical explanation and historical ablations; this page and the manifest govern the published setup.
