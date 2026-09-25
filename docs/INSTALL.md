# Install and run Humaneness Voice

This guide describes the public agent stack in [`../manifests/agent-default-v1.json`](../manifests/agent-default-v1.json). The base model is a full checkpoint; the voice, preference, emotion, delivery, quality and reaction LoRAs are separate. A download of the base model alone will not give the performance controls shown in the demo.

## Prerequisites

- Linux with Python 3.12, `ffmpeg` on `PATH`, a CUDA driver compatible with the PyTorch wheels below, and sufficient disk for several large model snapshots. The source deployment used two 24 GB GPUs; other memory profiles are not validated here.
- An up-to-date Git client and access to the listed public Hugging Face repositories. Network access is needed once for installation, unless the pinned snapshots already exist locally.
- For a local, key-free director, compile [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA support and locate its `llama-server` binary. A hosted OpenAI-compatible director can be used instead through the settings in `config.py`; that is an optional external service.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install torch==2.6.0 torchaudio==2.6.0 torchvision==0.21.0 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -r requirements.txt
```

The version list is inherited from the source server's September 2026 environment. A package resolver can vary by platform; if the CUDA wheel index changes, install a compatible PyTorch trio first, then run the requirements command. We do not bundle the llama.cpp binary or the model files in Git.

## Fetch a usable agent stack

```bash
.venv/bin/python setup/fetch_all.py --profile emolia_c1699 --director
```

`fetch_all.py` pins the base and codec snapshots, installs the selected identity adapter, all 40 emotion adapters, three quality axes, the two distinct preference adapters (p2 and quality-DPO), 17 delivery axes, the shipped vocal-reaction adapters, VoiceCLAP and its two scoring heads, ASR and the forced aligner. `--director` adds the local GGUF and projector; omit it when you already run a compatible local director or choose a hosted one. This is a large download. The script prints every asset it fetched or failed to fetch and exits nonzero on failure. A safe inspection without downloading is:

```bash
.venv/bin/python setup/fetch_all.py --check --profile emolia_c1699 --director
```

For an exploratory reduced install, `--minimal` downloads the base, codec, p2 adapter, one chosen voice and the Relief emotion adapter. It omits the perceptual ranker and most expressive tools. Do not report this as the full Humaneness Voice agent. `--all-profiles` installs the ten pilot identity LoRAs named in the manifest; it does not download all 500 voice-profile adapters.

The installer writes `.env.generated` with local paths to the pinned base and codec snapshots. The file contains no token and is gitignored. `run.sh` reads it automatically. For a different storage volume, set `HUMANENESS_DATA_DIR` before fetching and keep that variable set for all later commands:

```bash
export HUMANENESS_DATA_DIR=/your/large/storage/humaneness-voice
```

## Add reference speech without downloading the whole dataset

The [voice-profile reference dataset](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references) is much larger than this agent needs. For one voice, the extractor downloads five temporary WebDataset shards, keeps the top-ranked take per condition, plus that profile's anchor recording, voice card and corpus index. Temporary shard files are removed after extraction; allow roughly one shard of peak extra disk. The first installation requires network time.

```bash
.venv/bin/python setup/fetch_profile_refs.py emolia_c1699
.venv/bin/python setup/build_retrieval_index.py emolia_c1699
```

Repeat the voice ID in both commands for every installed voice you intend to serve, or pass several IDs in one invocation. For each profile, the reference extractor writes `refs/index_profile_<voice>.json` and `refs/audio/profiles/<voice>/`. The index builder writes `retrieval/index.npz` and `retrieval/index.json`. An optional `setup/fetch_profile_refs3.py <voice>` retains the top three takes per condition and needs more storage; the one-take path above is enough to start. Voice-reference audio is a separate dataset with its own rights and consent questions; do not mistake its Hub card for a blanket right to clone a real person's voice.

## Start and test

```bash
export LLAMA_SERVER=/absolute/path/to/llama-server
./run.sh both
```

The browser UI is on `http://127.0.0.1:8792`. The director listens only on loopback port 8790. `run.sh both` starts the director and speech app, writes local logs under `logs/`, and does not start optional restoration or the heavyweight separate burst-detector service. To inspect failures, read `logs/llm.log` and `logs/app.log`. If you already run a director, set `MOSS_LLM_BASE` and use `./run.sh app`. `./run.sh stop` only stops processes this repository's `both` mode recorded.

```bash
curl -fsS http://127.0.0.1:8792/api/speak -H 'content-type: application/json' \
  -d '{"text":"I thought you would never come back.","best_of":1,"sidon":false}' \
  -o test.mp3
```

Use `"best_of":10` to exercise candidate ranking, but expect a larger latency and make sure `setup/fetch_all.py --check` passes for the scorers and ASR. A score is only a selection aid. `/api/state` lists visible profiles and readiness, and `/api/adapters` lists loaded adapter keys. See [SERVER.md](SERVER.md) for the remaining fields and direct timed-script calls.

## Data layout and overrides

By default the scripts use `~/.cache/humaneness-voice/`:

```text
loras/sft3_dpo/p2/                preference adapter
loras/sft3_voice/<profile>/      speaker identity
loras/sft3_emotion/<emotion>/    affect
loras/sft3_qdpo/quality_dpo/     quality DPO step376
loras/sft3_quality/adapters/     three quality axes
loras/sft3_voicenet/adapters/    delivery axes
loras/burst/adapters/            vocal reactions
refs/                             selected recordings and anchors
retrieval/                        VoiceCLAP nearest-neighbour index
scorers/genuineness/, blend/      two perceptual heads
director/                         optional local GGUF
```

The old `sft3_*` directory and adapter-key prefixes are retained because the runtime uses them as compatibility identifiers. They are not alternate product names. The preferred public name is Humaneness Voice. For existing deployments, `MOSS_DATA_DIR`, `MOSS_ASSETS`, `MOSS_REF_DIR`, `MOSS_RETRIEVAL_DIR`, `MOSS_PROFILE_LORAS`, `MOSS_LLM_BASE`, `MOSS_DEFAULT_BRAIN` and the other `MOSS_*` settings in [`../config.py`](../config.py) continue to work. Put overrides in `.env.local`, never in committed code or logs.

## Diagnostics

```bash
.venv/bin/python setup/check_release.py
.venv/bin/python setup/check_docs.py
.venv/bin/python -m pip install -r requirements-dev.txt
.venv/bin/python -m pytest -q tests
```

`check_release.py` verifies manifest and configuration consistency without fetching weights. After installation, run `python setup/check_release.py --assets` to check the expected local files and adapter layout. `check_docs.py` checks prompt/document invariants and can optionally probe Hub links with `--urls`. The small unit tests do not replace a CUDA audio run. If a LoRA is not listed by `/api/adapters`, first check that its `adapter_model.safetensors` and `adapter_config.json` are inside the exact directory shown above. If a voice is not listed by `/api/state`, it needs both its identity LoRA and its matching anchor under `refs/pilot/<voice>/`.
