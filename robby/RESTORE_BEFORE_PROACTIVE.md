# Restore the working Robby baseline before environmental awareness

This checkpoint preserves the memory-informed two-stage speech demo **before**
adding automatic startup or proactive environmental sound observation.

- Immutable code: `7e5e58d02a5a172b5da341d47f0a2eabfaf18d36` (merged PR #17).
- Annotated Git tag with this restoration guide: `robby-before-proactive-2026-10-27`.
- Machine-readable inventory: `restore-before-proactive-2026-10-27.json`.
- Verified baseline: 501 Robby unit checks, three KV/admission checks, real GPU
  batching, actual HTTPS playback/cancellation and zero-history wiki recall.
  See `TWO_STAGE_SPEECH.md` and `static/two-stage-report.html`, including failures.

## Recover the source without overwriting later work

```bash
git clone https://github.com/LAION-AI/Humaneness-Voice.git
cd Humaneness-Voice
git fetch origin --tags
git switch --detach robby-before-proactive-2026-10-27
python robby/verify_restore_checkpoint.py
```

For an existing checkout with newer work, use an isolated worktree instead:

```bash
git worktree add ../robby-restored robby-before-proactive-2026-10-27
```

The verifier checks the baseline's tracked Python/JavaScript/CSS byte hashes,
without loading a GPU model, contacting a provider or reading private data.
The baseline code commit remains usable independently of the later guide commit.

## Preserve assets and private state separately

Git contains code, prompts, configuration examples, model revisions/hashes,
test programs, authored test evidence and reports. It does **not** contain model
weights, source reference audio, downloaded retrieval indexes, API credentials,
visitor recordings, browser-local preferences or private wiki workspaces.
Do not erase the existing asset directories during rollback.

For the original host, keep the existing `/mnt/nvme/moss-15-v2/humaneness-small`
assets, `/mnt/nvme/robby-asr`, `/mnt/nvme/robby-bolt`, TYGER caches and the
speaker-student/ECAPA caches. The original TTS RL cache may reside in `/dev/shm`;
that is volatile across reboot. Re-download its pinned bundle to persistent
storage as described in `../docs/PLAYGROUND_INSTALL.md` when necessary.

If retaining personal knowledge across a machine migration is desired, back up
the configured `ROBO_MEMORY_DIR` privately (default `~/.local/share/robby-memory`)
and the matching browser's memory capability/settings. Protect that backup; do
not commit it. Without both, restore starts with a fresh private memory space.
Audio voice enrollment is session-local and is not recovered by wiki files.
Restore provider credentials independently using `credential_store.py` and
`llm_session.py` instructions; private encrypted settings and their encryption
key must stay together. A fresh host requires its own valid provider key.

## Recreate the model dependencies

Use Linux, Python 3.12, FFmpeg, Chrome, an NVIDIA driver compatible with the
recorded CUDA PyTorch wheels, and `requirements.txt`. The JSON records installed
versions and hashes; this is an observed environment, not a fresh universal lock.
Follow `../docs/PLAYGROUND_INSTALL.md` for the separate TTS runtime, public
model bundles, scorer packages, fixed reference manifest and retrieval assets.
Its disclosed asset prerequisites remain real requirements. A clean-machine
full installation has not been certified merely by creating this checkpoint.

From `robby/`:

```bash
python bootstrap.py
python bootstrap_voice_captions.py
```

These fetch pinned TYGER source/bundle and Humaneness Ears/procedural caption
rules. ASR alternatives are pinned in `asr_catalog.py`; the selected final
recognizer is Parakeet v3. Student and ECAPA hashes are in the inventory; see
the README's released-model setup and set `ROBO_SPEAKER_MODEL_DIR` and
`ROBO_ECAPA_ROOT` to the provisioned assets. The selected speech model is Small
1.1 Alpha RL at revision `182bc700cac29b2d69295a33398b13e1ca7fba69`.
Local Bolt is `nvidia/NVIDIA-Nemotron-3.5-Bolt-E4B-4GB`, revision
`1db54d5b12533ff1bcaa8e42cf4c667988a1f1be`, `Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf`.
The main conversational provider is Qwen/Cerebras through OpenRouter. A remote
provider is not frozen by this Git tag and may change or become unavailable.

## Start in dependency order

Only stop owned, idle demo processes. Never reset a GPU or kill unrelated
services as part of restoration. Existing `manage.py` and `manage_laya.py`
validate their own recorded processes; historical absolute-path launch scripts
must be reviewed before use on another machine.

1. Provision the TTS assets and fixed Mira reference. From `playground/runtime`,
   start `python -m uvicorn small_app:app --host 127.0.0.1 --port 8792` with
   `ACTING_DEVICE=cuda:1`, `HVS_RL_PRECISION=fp32_merged`,
   `HVS_SEMANTIC_CUDA_GRAPH=0` and the reviewed `env.example` asset paths.
   Wait for `/api/state` to show `ready`, the pinned Small RL model and
   `continuous_tts: true`. Startup compiles/warms the two-stream shapes.
2. Start the existing Laya decision worker with `python manage_laya.py start`.
   Follow its README bootstrap instructions if its pinned source is absent.
3. Start the local llama.cpp Bolt service, ASR bank, voice-caption worker,
   Robby server and FastAPI facade with `launch_bolt_stack.py`. Example:

   ```bash
   ROBO_ASR_MODEL=parakeet_v3 ROBO_SPEECH_PAUSE_MS=700 \
   ROBO_CAPTION_DEVICE=cuda:1 ROBO_REPLY_LANGUAGE=de \
   python launch_bolt_stack.py --llama-server /path/to/llama-server \
     --models /path/to/bolt-model-directory --gpu 0 \
     --with-robo --with-asr-models --with-voice-captions
   ```

   Supply the documented ECAPA/student/source/cache environment paths. Reuse
   already-running workers instead of requesting occupied ports. TTS is not
   started by this launcher. Full startup requires the independently configured
   OpenRouter credential; no key belongs on the command line or in Git.
4. Open the facade at `http://127.0.0.1:8798/desk-robot/`. For HTTPS use the
   existing reverse proxy or `cloudflared tunnel --url http://127.0.0.1:8798`.
   A new temporary tunnel has a new URL; an old trycloudflare address cannot be
   restored from source code alone.

The baseline's two-stage path uses TTS temperature 0.8, CFG 2.0, one weight copy,
two active streams, 160-ms first / 400-ms following blocks and a 220-ms browser
reserve. Voice is Mira (`emolia_c1699`), camera auto-start and persistent wiki
memory are off, quick replies are on. Browser-local saved preferences override
defaults; use a fresh browser profile when comparing the baseline.

## Verify after restoration

```bash
python run_unit_checks.py
python browser_two_stage_check.py --url https://YOUR-DEMO
python browser_two_stage_memory.py --url https://YOUR-DEMO
```

Run the browser tests only against an idle owned demo. They create authored
test conversations; the memory test uses/reset its own workspace. Inspect
`/api/bolt/state`, worker health and real audio output as well as readiness.
Compare the source/model hashes with the inventory, not only the page's title.
The published cold-start and occasional SCRIPT-validation limitations remain
documented; a successful restore does not imply a new reliability guarantee.
