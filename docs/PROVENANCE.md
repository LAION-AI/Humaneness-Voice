# Provenance, release boundary and known limits

## What this repository is

The initial code, UI and WikiSkills came from the public [Humaneness Voice Demo Server](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server) at commit [`aa299f18879b4985136375ccb0c8d2d533995932`](https://github.com/LAION-AI/Humaneness-Voice-Demo-Server/commit/aa299f18879b4985136375ccb0c8d2d533995932). This is a new release snapshot with its own history, not a rename or deletion of that repository. The release adds a system-level README, pinned asset manifest, portable installer/path defaults, a corrected LoRA directory layout, local-run instructions and checks. It preserves the implementation and prompt/skill files so earlier experiments remain traceable. The historical `sft3_*` adapter keys are implementation identifiers; the public system name is **Humaneness Voice**.

The base checkpoint is [here](https://huggingface.co/laion/moss-tts-local-transformer-4.55b-voice-acting-v2-sft3). It is a complete fine-tuned MOSS-family speech model, not a LoRA. Its card describes the final 398,282-example expressive SFT round, measured duration and speech-error outcomes, and the older MOSS architecture. The released reference corpus, voice LoRAs, DPO adapters, acting LoRAs, perception heads and director weights are separate artifacts. This GitHub repository does not republish their weights or assert that their original cards are already renamed.

## What a reproduction needs to record

For a generated take, retain: this Git revision; `manifests/agent-default-v1.json`; the resolved Hugging Face asset revisions; selected speaker ID and reference clip; director model and prompt revision; the exact script and spoken text; ordered adapter keys and weights; generation seed, temperature, guidance and frame budget; optional alignment/restoration; and every candidate score if best-of-N was used. Without these, "Humaneness Voice" names a family of configurations rather than one reproducible treatment. `GET /api/state` and `GET /api/adapters` expose runtime availability; the manifest states intended sources. A listed asset is not evidence that it loaded successfully on a given machine.

`wikiskills/` is checked-in inference material. Its generator scripts refer to source statistics from separate combination, dose and steering studies; those raw study inputs are not in this Git release. An external researcher can run the shipped recipes, but cannot reproduce every historical study solely from this repository. The technical paper and historical documents in `docs/` describe those studies and their conditions.

## Data and licence boundaries

The repository's [`LICENSE`](../LICENSE) covers its code and documentation under CC BY 4.0. The Base and some adapters declare CC BY 4.0; other adapters and the codec declare Apache 2.0; restoration declares MIT; the optional burst-v2 research repository currently lacks a declared card licence. A mixture of these files is **not** automatically an Apache-licensed whole. Always check the pinned cards before copying or redistributing weights.

The [voice-profile reference dataset](https://huggingface.co/datasets/TTS-AGI/moss-voice-profile-references) supplies the selected anchors and conditioned recordings. The server pairs identity adapters and reference audio by profile ID. A repository licence label does not alone settle a real speaker's consent, publicity rights or the terms of every upstream recording. Only use a voice with the necessary permissions. The broader [annotated expressive-speech releases](https://huggingface.co/datasets/laion/laion-tts-annotated-v1) and [German–English scaling ladder](https://huggingface.co/datasets/laion/tts-scaling-ladder-de-en) support future training work; they are **not** a claim that this checkpoint trained on every released hour. Data views overlap, and source-specific rights remain relevant.

## Scientific and engineering limits

- The genuineness and vocal-burst-blend predictors are small heads on frozen VoiceCLAP audio embeddings. Their model-labelled training and validation do not establish human validity for every accent, language, scene or acoustic condition. Ranking by them can reward source artifacts. A listener-based Arena is an external check, not a replacement for continued scorer validation.
- Best-of-N is a within-scene selection procedure. Its `N`, reward weights and available tools must be stated; `N=10` at `/api/speak` does not establish that every server call duplicates the paper's Arena pass@10 configuration. Candidate selection improves some observed preferences but is not a proof of state-of-the-art acting.
- The original server used two 24 GB GPUs. We test syntax, data layout and lightweight invariants here; an end-to-end audio run requires downloading the pinned weights and suitable hardware. Different GPU, driver and library combinations may need adaptation.
- The optional `eval/` scripts and older prose notes contain historical machine-specific assumptions. The supported new-install path is `README.md` → `docs/INSTALL.md` → the pinned manifest and setup scripts, not every research harness.
- The public repository is authored under LAION-AI and cannot be used as a blind-review URL. A separate review artifact must be checked for history, repository/Hub links, metadata and licence obligations before submission.
