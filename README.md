# Humaneness Voice

Humaneness Voice is an open voice-acting agent, not a single text-to-speech checkpoint. A language-model **director** interprets a scene and writes local performance cues; a speech **instrument** renders the words with a chosen voice, references and compatible LoRA adapters; perceptual **ears** can compare candidate takes. The included web app is one way to run that pipeline.

This repository is a reviewed, standalone snapshot of the [Humaneness Voice Demo Server](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server) at commit [`aa299f1`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/commit/aa299f18879b4985136375ccb0c8d2d533995932). The original remains available. This repository contains code, director prompts, WikiSkills, tests and an exact asset manifest; model weights and reference audio remain in their respective Hugging Face repositories. The historical `sft3_*` names in Python and adapter keys identify training-lineage compatibility. They do not name a different serving system.

## What happens to one scene

```text
scene or chat message
    → director LLM: fixed words, sentence-level acting cues, timing and reactions
    → retrieval: matching reference take and emotion/delivery choices
    → Humaneness Voice Base + one voice LoRA + compatible acting/quality LoRAs
    → streamed take, or N candidates scored and ranked by the perceptual ears
```

The director uses the checked-in [`wikiskills/`](wikiskills/) to turn measured acting recipes into a prompt. The instrument is the full-parameter [Humaneness Voice Base](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3), historically called the MOSS SFT3 checkpoint, with the [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2). Its starting point was [MOSS-TTS-Local-Transformer-v1.5](https://huggingface.co/OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5), followed by the documented voice-acting adaptation. The ears use [VoiceCLAP Commercial](https://huggingface.co/laion/voiceclap-commercial), the separate [genuineness](https://huggingface.co/laion/voiceclap-commercial-genuineness) and [vocal-burst blend](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) heads, audio/text fit and ASR. These heads help select takes; they are not a substitute for human judgments of acting.

## Reproduce a local run

The original server was developed for Linux, Python 3.12, CUDA 12.4 and two 24 GB GPUs. A smaller configuration may be possible but is not measured here. Install `ffmpeg`, a CUDA-compatible PyTorch build, and [llama.cpp](https://github.com/ggml-org/llama.cpp) with `llama-server` for the optional local director. You need substantial free disk space for the base model, LoRAs, director and temporary reference-data shards; the full upstream reference dataset is **not** downloaded by the setup below.

```bash
git clone https://github.com/LAION-AI/Humaneness-Voice.git
cd Humaneness-Voice
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt

# Pinned base, codec, the serving LoRAs, scorers and local director weights.
.venv/bin/python setup/fetch_all.py --profile emolia_c1699 --director
# Extract only the chosen pilot voice's best reference take per condition.
.venv/bin/python setup/fetch_profile_refs.py emolia_c1699
# Build its VoiceCLAP retrieval index.
.venv/bin/python setup/build_retrieval_index.py emolia_c1699

export LLAMA_SERVER=/absolute/path/to/llama-server
./run.sh both
```

Open <http://127.0.0.1:8792>. In another terminal, request a single take first:

```bash
curl -fsS http://127.0.0.1:8792/api/speak \
  -H 'content-type: application/json' \
  -d '{"text":"Tell her I came back because I could not leave it unfinished.",
       "best_of":1,"sidon":false}' -o take.mp3
```

For candidate selection, change `"best_of":1` to `"best_of":10`. That requires the scorer and ASR assets installed by the full `fetch_all.py` command; it is slower and returns the highest-ranked take. The plain `/api/speak` default is ten candidates, whereas streaming `/api/turn` uses one take unless explicitly configured. `./run.sh app` starts only the speech app and expects a reachable director endpoint. `./run.sh stop` stops only processes started by this repository's `both` mode. See [installation and troubleshooting](docs/INSTALL.md) for the individual steps, data paths and checks, and [HTTP examples](docs/SERVER.md) for each endpoint.

The installer writes a gitignored `.env.generated` containing only local paths to pinned downloaded snapshots. Put personal overrides or hosted-provider credentials in `.env.local`, also gitignored. `run.sh` sources both. The preferred asset root is `HUMANENESS_DATA_DIR` (default `~/.cache/humaneness-voice`); `MOSS_DATA_DIR` and other historical `MOSS_*` environment names remain supported for existing deployments. No API key is needed for the local director.

## The serving stack

The labels below refer to the *role* of each asset. The linked repository IDs retain their original names for traceable provenance; nothing has been silently renamed or relicensed.

| Role | Asset and current default |
| --- | --- |
| Humaneness Voice Base | [full SFT3 checkpoint](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3), without the LoRA stack |
| Codec | [MOSS Audio Tokenizer v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2), 12 RVQ codebooks at 12.5 frames/s |
| Preference | [DPO p2](https://huggingface.co/laion/moss-va-sft3-dpo-lora-p2), `sft3_dpo:p2` at 1.0 |
| Identity | [SFT3 voice LoRAs](https://huggingface.co/laion/moss-va-sft3-voice-loras), one selected pilot voice at 1.0, paired with its [reference recording](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references) |
| General quality | [quality DPO step376](https://huggingface.co/laion/moss-va-sft3-quality-dpo-lora) at 1.5; the separate [quality-axis LoRAs](https://huggingface.co/laion/moss-va-sft3-quality-lora-adapters) use genuineness 0.25, blend 0.5 and aesthetics 0.5 |
| Acting direction | One retrieved [emotion LoRA](https://huggingface.co/laion/moss-va-sft3-emotion-loras) at 1.0; up to one [vocal-delivery LoRA](https://huggingface.co/laion/moss-va-sft3-voicenet-lora-adapters) as directed |
| Vocal reactions | Requested [burst LoRAs](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters), with class-specific WikiSkill recipes and a 2.0 combined dose budget |
| Perception and ranking | [VoiceCLAP Commercial](https://huggingface.co/laion/voiceclap-commercial), [genuineness](https://huggingface.co/laion/voiceclap-commercial-genuineness), [blend](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend), and [Parakeet-TDT-v3 ASR](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |

The complete, pinned list, including optional aligner, restoration, research-only adapters and licence declarations, is in [`manifests/agent-default-v1.json`](manifests/agent-default-v1.json) and [the asset guide](docs/MODELS_AND_ADAPTERS.md). The installer downloads only the selected voice profile by default; `--all-profiles` selects the ten tested pilot IDs, not every voice in the larger upstream LoRA repository. `--minimal` installs a deliberately reduced stack and does **not** reproduce the default agent or the pass@10 configuration.

## Where to read next

| Question | Document |
| --- | --- |
| How do I install, check missing assets or bring my own model path? | [Installation](docs/INSTALL.md) |
| Which Hub repository does each adapter key load, and at what weight? | [Models and adapters](docs/MODELS_AND_ADAPTERS.md) |
| What do Mind, Instrument, Tools and Ears do? | [Architecture](docs/ARCHITECTURE.md) |
| What does the director send to the speech model? | [Prompt format](docs/PROMPTING.md), [exact system prompts](docs/SYSTEM_PROMPTS.md) |
| What is the server's HTTP interface? | [Server API and examples](docs/SERVER.md), [acting-task cookbook](docs/VOICE_ACTING_TASKS.md) |
| How is best-of-N computed? | [Candidate ranking](docs/BEST_OF_N.md), [burst reward](docs/BURST_REWARD.md) |
| Which claims are from the original experiment and which remain to validate? | [Provenance and limits](docs/PROVENANCE.md), [historical measurements](docs/EXPERIMENTS.md) |

Run `python setup/check_release.py` for manifest checks, `python setup/check_docs.py` for prompt/document consistency, and `pip install -r requirements-dev.txt && python -m pytest -q tests` for the lightweight tests. A GPU/audio end-to-end test requires the downloaded weights, recordings and suitable hardware; this repository publication does not by itself establish that such a test ran on every target machine.

## Licence, attribution and use

This repository's code and documentation retain their existing [CC BY 4.0](LICENSE) attribution. The Humaneness Voice Base card declares CC BY 4.0; the original MOSS tokenizer card declares Apache 2.0; adapter and perception repositories have **their own** declarations. Some experimental assets have missing licence metadata, and reference-audio rights must be checked separately. Consequently the entire downloaded stack must not be described as a single Apache-licensed package. Consult each pinned card and [the provenance guide](docs/PROVENANCE.md) before redistributing weights, recordings or generated voices. Use a speaker's identity only with appropriate consent.

This public repository intentionally names its source and contributors. It is **not** an anonymized ICLR review artifact; a separate review copy must be audited for URLs, history, metadata and attribution obligations.
