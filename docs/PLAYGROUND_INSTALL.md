# Running the current speech models and full Playground

There are **three different launch paths**. Choose one deliberately:

| Goal | Entry point | What it reproduces |
| --- | --- | --- |
| Synthesize with a released RL model | Model bundle `infer.py` | Speech model/base/adapter inference, without a director, scorers, web UI or experimental search |
| Run the October web system | `playground/runtime/small_app.py` | The captured integration after provisioning its additional assets/configuration |
| Run the September SFT3 agent | Root `run.sh` and `setup/fetch_all.py` | Older base, tools, LoRA stack and local director; **not** the RL Playground |

## 1. Standalone model: the smallest reproducible path

Download a complete **pinned** bundle, not only its `adapter.safetensors`:

```bash
python -m pip install huggingface-hub
hf download laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1 \
  --revision 182bc700cac29b2d69295a33398b13e1ca7fba69 \
  --local-dir weights/small-rl
python -m pip install -r weights/small-rl/requirements.txt
python weights/small-rl/infer.py --adapter-dir weights/small-rl \
  --language en --text 'We finally made it home.' \
  --prompt 'GENERAL: Gentle, natural and spontaneous delivery, warmly relieved.
SCRIPT:
(warmly relieved) We finally made it home.' \
  --reference-wav /path/to/consented-clean-speaker.wav \
  --frames 75 --seed 777 --output example.wav
```

This follows the published CLI structure. For Large, use
`laion/Humaneness-Voice-Large-1.1-Alpha-RL-r128-v1` at revision
`7f050dc0e5c1b2415258c357c1edc27e1452cf07` and a separate output directory.
Read that bundle's arguments/requirements before running. The released model
card reports GH200 / PyTorch 2.9.1+CUDA 12.8 / Transformers 5.14.1 canaries;
the captured Playground host instead used PyTorch 2.6.0+cu124 / Transformers
5.15.0. Do not call these the same benchmark environment.

The bundle already includes its exact TTS base. It still needs the upstream
MOSS-v2 codec. For offline use, provision the codec ahead of time and pass
`--codec-path /path/to/codec`. No LLM, Parakeet, VoiceCLAP, Ears or Orange judge
is required merely to synthesize. RL **training rewards are not inference
dependencies**. Use clean single-speaker references, preferably 5–15 seconds.
Named alias mode was not trained by this RL run.

## 2. Full October Playground integration

The checked-in snapshot makes the source, prompt, weights and protocols
inspectable. It is **not a newly validated clean-machine installer**. In
particular, reference audio, metadata/indexes, scorer packages and optional VC
source are not obtained simply by installing Python requirements. The original
host GPU/HTTPS smoke tests do not prove a fresh clone works without these steps.

### Environment and hardware

Use Linux, Python 3.12, FFmpeg with `libmp3lame`, a compatible NVIDIA driver and
CUDA PyTorch/torchaudio/torchvision wheels. Inspect the observed versions in
[`environment-2026-10-16.json`](../playground/environment-2026-10-16.json).
The older root requirements omit some current app packages such as
`python-multipart` and are not a complete October lockfile.
For the captured environment, install the CUDA 12.4 PyTorch 2.6.0 /
torchaudio 2.6.0 / torchvision 0.21.0 wheels first, then the root requirements
and the additional observed `pydantic==2.13.4`, `python-multipart==0.0.32`,
`httpx==0.28.1`, `librosa==0.11.0`, `einops==0.8.2` and `omegaconf==2.3.1`.
Check the full observed package JSON rather than assuming these examples form
a platform-independent dependency lock. Optional conversion additionally needs
the LAION `chatterbox_vc` wrapper and its upstream `chatterbox` package, following
[that repository's install instructions](https://github.com/LAION-AI/chatterbox-voice-conversion).
Its source revision was not captured as an installed distribution; do not
claim that this is a fully pinned VC software environment.

The live host has two RTX 3090 24-GB GPUs and 125 GiB RAM; the tested app runs
on GPU 1, leaving the unrelated service on GPU 0 alone. A local LLM needs its
own compute/memory or CPU placement. Small RL and Large RL were exercised with
their FP32 policy contracts, not quantized estimates. Large RL's parallel cap
is 2; ASR/Ears/alignment move to CPU while it is selected. More than one model's
base is prefetched into CPU RAM, and disk caches can exceed tens of GB. Plan
capacity for every enabled snapshot and auxiliary, not just the default base.

On the original host, new RL downloads live in `/dev/shm/humaneness-rl/hub`
because both disks were almost full. This is **tmpfs**, consumes RAM, and loses
files at reboot. For a new installation explicitly set `HUMANENESS_RL_CACHE`
to a persistent location with enough space. No deletion of previous caches is
required or recommended by this guide.

### Required assets and hidden startup requirements

Provision the exact sources in [the model/dependency guide](PLAYGROUND_MODELS.md):

1. Pinned selected TTS bundle/full checkpoint and its bundled Python/config/
   tokenizer assets; the registry verifies base and RL adapter hashes.
2. MOSS-v2 codec, Parakeet, shared VoiceCLAP, Ears Base and Medium, and the two
   Genuineness/Blend scorer packages. The initializer currently loads these
   scorers even for one take; selecting “single” is not a scorer-free install.
3. The Qwen forced aligner for default final fades. Failures are reported and
   can leave a take unchanged; inspect alignment availability rather than
   assuming a checkbox proves alignment ran.
4. A VoiceCLAP retrieval index (`index.json`, `index.npz`). The inherited
   `Scorers` initializer currently requires a valid index even if dynamic
   reference selection is off or a custom reference is intended.
5. For catalog presets: the selected voice-profile metadata/anchors and the
   exact normalized fixed reference files plus selection manifest. The current
   profile-discovery compatibility code also expects a legacy profile adapter
   file on disk to enumerate a profile; it does **not** apply that adapter to RL.
   Custom uploads/no-reference mode do not clone an unavailable Mira preset.

The root portable helpers are useful **asset preparation tools**, not a new
model launcher: `setup/fetch_scorers.py` downloads scorer source; the pinned
`setup/fetch_profile_refs.py` and `setup/build_retrieval_index.py` prepare selected
pilot references/indexes. Their output directories/layout must match the
October configuration. They do not recreate the later refs3 shortlist automatically.
Avoid the old default `fetch_all.py` if you do not want its additional SFT3
weights/LoRAs; read the helper's manifest and flags first.

The published compatibility module adds one path-only environment override:
`HUMANENESS_SCORER_DIR`, containing `vc_genuineness/` and `vc_blend/`. Place the
respective published head package in each, including its `voiceclap_commercial/`
subfolder. This replaces the source host's hardcoded package prefix without
changing predictor math. Other machine defaults remain disclosed in source.

### Fixed references and reproduction limits

The original six recommended presets and four remaining voices are recorded
in `playground/experiments/fixed-references/selection.json`, with source clip
keys, lengths, CE estimates and normalized-wave hashes. Some clips are under
15 seconds; the cap does not imply every input has length 15 s. Private/source
recordings are deliberately not in Git. Obtain the exact referenced data with
appropriate rights; normalize it using the captured `small_app.normalize()`
path and verify hashes. A different clip from the same speaker is a different
experimental condition, not an exact reproduction. The published 500-profile
anchor and the demo's later selected fixed clip can have different hashes.

`select_fixed_references.py` depends on the original refs3 metadata and runs
Ears scoring. It is a deliberately new selection, not a guaranteed recreation
without the original inputs. For your own application use a consented custom
upload, or deliberately rebuild a catalog manifest. Do not claim the original
Mira condition when substituting another file. Missing source data is a real
reproducibility limitation, not something the installer silently invents.

### Configure and start

Review [`env.example`](../playground/runtime/env.example) and set real paths/
GPU indices. `MOSS_*` variables configure inherited parent modules; `SMALL_*`
and `ACTING_DEVICE` configure the new app. Merely changing `SMALL_DEVICE` does
not change `ActingEngine`'s device. Do not put secrets in committed files.

After provisioning the assets/reference manifest above:

```bash
cd playground/runtime
source env.example                 # review values before sourcing
# Hosted director: export SMALL_LUNA_API_KEY securely, or use Manual mode.
python -m uvicorn small_app:app --host 127.0.0.1 --port 8792
```

Wait for `GET /api/state` to report `ready: true`, no error, and the expected
`model_id`/checkpoint hashes. First-use compilation may take minutes. Open
`http://127.0.0.1:8792`; choose a provisioned profile, no voice or your own
upload. The model-selection endpoint serializes changes; do not run a second
GPU server over the same weights to switch models.

Use an authenticated reverse proxy for deployment. For a temporary test,
`cloudflared tunnel --url http://127.0.0.1:8792` can provide HTTPS; a random
TryCloudflare URL changes and is not authentication. Do not publish a private
director/LLM endpoint or API key. The historical machine-specific `run.sh`,
`service.py`, `acting_deploy.py` and GPU-waiter scripts retain local process/
interpreter assumptions; they are not the portable entry point above.

### Validation, not assumptions

The original complete 359-test run needs its fixtures/data. Fresh-clone CPU
checks without source audio can run, for example:

```bash
cd playground/runtime
python -m pytest -q tests/test_rl_adapter.py tests/test_checkpoint_ram.py
```

The upstream model CLIs, source manifest and these lightweight checks do not
prove complete GPU deployment. To repeat the actual end-to-end canaries, first
provision the original reference condition and a tunnel, then review
`rl_live_check.py` and `experimental_live_check.py`. They generate audio and
switch the running model; run them only against your own idle server. The
older canary retains historical expected defaults and must be updated deliberately.
See [published evidence](../playground/evidence/) and [API examples](PLAYGROUND_API.md).
