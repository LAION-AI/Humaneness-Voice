# Every model this server loads, with its URL

> **Release note:** This is the source server's September 2026 dependency audit.
> Its old path and downloader statements below do not describe the new portable
> setup. Use the [pinned manifest](../manifests/agent-default-v1.json),
> [asset guide](MODELS_AND_ADAPTERS.md) and `setup/fetch_all.py` for a fresh
> Humaneness Voice installation. This page remains for historical context.

Checked against the Hub on 7 September 2026: **every repository below resolves.**
Nothing this server needs exists only on the machine it was built on. The paths
under `/mnt/nvme/…` that appear in `config.py` are caches — `config._root()`
falls back to the Hub copy whenever the local directory is absent, and prints
the repository name and `setup/fetch_all.py` when neither is there.

```bash
python setup/fetch_all.py --check     # what is present, downloads nothing
python setup/fetch_all.py             # fetch everything
python setup/fetch_all.py --minimal   # only what the server cannot start without
```

## Adapters

`config.LORA_ROOTS` maps each root to a directory; `config.HUB_FOR_ROOT` maps
the same root to the repository it came from.

| root | adapters | repository |
|---|--:|---|
| `sft3_dpo` | 2 | [laion/moss-va-sft3-dpo-lora-p2](https://huggingface.co/laion/moss-va-sft3-dpo-lora-p2) |
| `sft3_voice` | 10 | [laion/moss-va-sft3-voice-loras](https://huggingface.co/laion/moss-va-sft3-voice-loras) |
| `sft3_emotion` | 40 | [laion/moss-va-sft3-emotion-loras](https://huggingface.co/laion/moss-va-sft3-emotion-loras) |
| `sft3_voicenet` | 17 | [laion/moss-va-sft3-voicenet-lora-adapters](https://huggingface.co/laion/moss-va-sft3-voicenet-lora-adapters) |
| `sft3_quality` | 3 | [laion/moss-va-sft3-quality-lora-adapters](https://huggingface.co/laion/moss-va-sft3-quality-lora-adapters) |
| `sft3_qdpo` | 3 | [laion/moss-va-sft3-quality-dpo-lora](https://huggingface.co/laion/moss-va-sft3-quality-dpo-lora) + […-burst-stop-dpo-lora](https://huggingface.co/laion/moss-va-sft3-burst-stop-dpo-lora) |
| `burst` | 71 | [laion/moss-va-sft3-vocal-burst-lora-adapters](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters) |
| `burst_v2` | 45 | [laion/moss-va-sft3-vocal-burst-lora-adapters-v2](https://huggingface.co/laion/moss-va-sft3-vocal-burst-lora-adapters-v2) `/per_class` |
| `burst_v2_top1` | 12 | the same repo, `/per_class_top1` |
| `burst_grp` | 20 | the same repo, `/groups_full` |
| `burst_grp25` | 10 | the same repo, `/groups_dose25` |
| `burst_abl` | 12 | the same repo, `/ablation` |
| `burst_dose` | 6 | the same repo, `/dose` |
| `character` | 12 | [TTS-AGI/moss-character-loras-refined-public](https://huggingface.co/TTS-AGI/moss-character-loras-refined-public) |
| `profile` | 10 | [laion/moss-voice-profile-loras-500](https://huggingface.co/laion/moss-voice-profile-loras-500) — inactive on SFT3 |
| `speaker` | 1 | [TTS-AGI/moss-voice-lora-velvet-sage-baritone](https://huggingface.co/TTS-AGI/moss-voice-lora-velvet-sage-baritone) |
| `sports` | 0 here | [laion/moss-sports-commentator-lora](https://huggingface.co/laion/moss-sports-commentator-lora) — exists, not installed on this box |

The six `burst_*` roots are six arms of **one** 105-adapter release, split by
subdirectory. Counts verified against the repository file list.

## Everything else

| what | required | repository |
|---|:-:|---|
| speech model, 4.55B | ✅ | [laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3) |
| audio codec | ✅ | [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) |
| reference recordings | ✅ | [TTS-AGI/moss-voice-profile-references](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references) |
| retrieval + reward towers | | [laion/voiceclap-commercial](https://huggingface.co/laion/voiceclap-commercial) |
| genuineness head | | [laion/voiceclap-commercial-genuineness](https://huggingface.co/laion/voiceclap-commercial-genuineness) |
| burst-blend head | | [laion/voiceclap-commercial-vocalburst-blend](https://huggingface.co/laion/voiceclap-commercial-vocalburst-blend) |
| speech recognition | | [nvidia/parakeet-tdt-0.6b-v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) |
| forced aligner, Apache-2.0 | | [Qwen/Qwen3-ForcedAligner-0.6B-hf](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) |
| restoration, MIT, off by default | | [sarulab-speech/sidon-v0.1](https://huggingface.co/sarulab-speech/sidon-v0.1) |
| local director | | [unsloth/gemma-4-12B-it-qat-GGUF](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF) |

## Not on the Hub

Two things, both of which ship **inside this repository**:

* **`wikiskills/`** — the burst recipes, the never-realise list and the
  interaction tables that `skills.py` parses. It is a directory here, not a
  model repository.

The listening evidence for those recipes **is** on the Hub, and an earlier
version of this page said otherwise:
[laion/moss-vocal-burst-recipes](https://huggingface.co/spaces/laion/moss-vocal-burst-recipes)
is a **Space** — 248 takes across the recipe classes, each marked with whether
the detector scored it a hit, plus weight-0 controls. Resolved 7 September 2026:
`api/spaces/…` 200, `api/models/…` 401, `api/datasets/…` 401. A checker that
looks only in the model namespace reports it dead; `SKILLS.md` cited it
correctly all along.
* **`wikiskills_legacy/`** — the same tree as it stood before the 2026-09-04
  revision, kept so an older result can be traced to the table that produced it.

Two more are **built**, not downloaded:

* the retrieval index — `python setup/build_retrieval_index.py`
* the voice profiles — `python setup/fetch_profile_refs3.py`

---

## The stack a measurement was made in is part of the measurement

This matters for anyone comparing numbers with us, and it is the reason two
groups can honestly disagree about a ceiling.

**Our burst ceiling is 1.25 and a single-adapter ladder gives 1.5.** Both are
right. The difference is what else is merged. Every take here already carries,
before a burst adapter is even considered:

| adapter | weight |
|---|--:|
| `sft3_dpo:p2` | 1.0 |
| `sft3_voice:<profile>` | 1.0 |
| `sft3_quality:genuineness_high` | 0.25 |
| `sft3_quality:blend_high` | 0.5 |
| `sft3_quality:esthetics_high` | 0.5 |
| `sft3_qdpo:quality_dpo` | 1.5 |
| `sft3_emotion:<retrieved>` | 1.0 |
| `sft3_voicenet:<axis>`, when the director picks one | up to 1.5 |

That is 5.75 to 7.25 of merged weight before the first burst. Measured under it,
**two burst adapters at 1.5 each destroy the line in 5 of 5 seeds** (median word
error 0.82), while one at 1.5 breaks none — it is the **sum** that breaks a line
here, which is exactly what a single-adapter ladder cannot see. Hence
`BURST_LAM_BUDGET = 2.0` and `BURST_LAM_MAX = 1.25`, with the weights scaled
rather than dropped when a turn wants more. The full table is in
[`BABBLE.md`](BABBLE.md) and the harness is `eval/sweetspot.py`.

**If you are reproducing this, fix the stack first.** `eval/why_babble.py` sets
it explicitly at the top; the constant is called `BASE`. Measuring the burst
adapters alone answers a different question, and both answers are worth having
as long as nobody mistakes one for the other.

## The burst detector

Two more repositories, needed only when `./run.sh burst` is used:

| what | size | repository |
|---|--:|---|
| burst detector, 5-head ensemble + the reference scorer code | 40 MB | [laion/vocal-burst-detector-x2](https://huggingface.co/laion/vocal-burst-detector-x2) |
| its encoder | 18.15 GB | [laion/voiceclap-large-v2](https://huggingface.co/laion/voiceclap-large-v2) |

**The encoder is only needed for the optional large-v2 path.** By default the
burst term is scored in-process on the `commercial/` heads, which take the
768-d embedding of `laion/voiceclap-commercial` — already loaded for retrieval —
so nothing extra is downloaded and the local language model keeps its card.
`laion/voiceclap-large-v2` wins 12 of 17 classes and loses on Scream; it costs
5.7 GB in 4-bit and the local director. See [`BURST_REWARD.md`](BURST_REWARD.md).
