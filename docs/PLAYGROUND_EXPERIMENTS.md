# Experiment and evidence map

These were exploratory, evolving experiments on **different checkpoints and
protocols**. Read each design/config/seed set before comparing results. An
improvement to an ECAPA/CE/emotion proxy is not evidence of human-perceived
acting quality or a speaker-disjoint benchmark. Earlier recommendations were
not all rerun on the current RL weights.

The [`playground/experiments/`](../playground/experiments/) directories retain
selected machine-readable designs and summaries; the
[dated development log](../playground/DEVELOPMENT_LOG.md) contains the original
conditions, caveats, commands and intermediate settings. Research scripts are
in [`playground/runtime/`](../playground/runtime/). They may need original data,
vectors or host paths and can allocate GPUs; review them before running.

| Question / experiment | Artifacts and what can be concluded |
| --- | --- |
| Single streaming, offline and batch throughput | [Permanent HTML](https://laion-humaneness-voice-small-studies.static.hf.space/performance/index.html), `performance/` JSON, `throughput_benchmark.py`, `performance_validation.py`. CUDA timing includes its stated scope, not browser/LLM time. Original Small S3/codec conditions, not current RL throughput. |
| Emotion × CFG × temperature | [Permanent 660-take pilot HTML](https://laion-humaneness-voice-small-studies.static.hf.space/study/index.html), `study/design.json`, `summary.json`, `emotion_study.py`. Earlier concise CAPTION/TRANSCRIPT, T=1.0/CFG=1.5 recommendations are historical; they do not establish an RL optimum. |
| Chunk winner vs top-2 search | [Permanent report and code](https://laion-humaneness-voice-small-studies.static.hf.space/search/index.html), `search_benchmark.py`, `search_validation.py`, `search_emotion_check.py`. Prefix-aware ASR gate and shared VoiceCLAP; mixed quality results. Top-2 can delay playback until a common acoustic prefix exists. |
| Adaptive VoiceNet/EIV+ and vocal-burst realization | `adaptive_reward_validation.py`, `burst_reward_validation.py`, `realization_cpu_test.py`, UI tests and `realization-check.html` when hosted. Burst Blend alone does not verify requested event timing/type; optional locator/detector does a separate noisy check. |
| Reference prompting, gender/timbre cues and DPO | `speaker-study/` design/summary, `speaker_study.py`: 240 takes, including 10-seed conditions. Distinguish conditioning vs held-out anchor similarity and transcript errors; reference identity cues are not a guaranteed cloning fix. |
| New full-FT base: identity, quality and reliability | `full-ft-speaker-study/` pilot, diagnostic, follow-up and validation summaries; `full_ft_speaker_study.py`. Follow-up's WER-balanced winner was `auto_cfg1`; later user priorities deliberately favored stronger acting/CE. Do not flatten different objectives into one universal winner. |
| Neutral/engaging fixed reference shortlist | `fixed-references/selection.json`, `select_fixed_references.py`. Ten metadata-shortlisted candidates scored with Ears; six promoted, not an exhaustive corpus scan or human listening study. Current Mira default supersedes the earlier selection-file default. |
| Mira absolute means, semantic vectors, multi-layer and reference/no-reference contrasts | `mira-activation-study/` summaries/design, `mira_activation_study.py`, `mira_vector_diagnostics.py`. Some interventions hurt WER, Production Quality or MOS even when an identity proxy rose. Absolute average, high-minus-low and ref-minus-no-ref are different estimators. |
| Catalog high-minus-low speaker directions | `catalogue-steering-study/` design/summary; `catalogue_steering_study.py`. Layer-16 contrast and strength were studied on earlier Small, with held-out prompts/seeds. Not proof of a pure speaker-identity direction or compatibility with new weights. |
| More seeds and top 10% / 5% / 2% / 1% tails | `seed-scaling-study/`, `strict-tail-study/` designs and summaries; corresponding study/analysis scripts. These compare selection size and quality tradeoffs; strict tails can have tiny effective sample size and winner-selection bias. Report screened and held-out confirmation results separately. |
| 500-profile target collection, layers 14–18 | `catalogue-target-study/design.json`, `catalogue_target_study.py`, `catalogue_target_analysis.py`. Planned 100-sample/top-2% multi-prompt training targets and speaker-disjoint splits; **a design file is not proof all 500 profiles finished**. No trained universal embedding→direction MLP is released/claimed here. |
| Voice conversion: streaming, batching and BF16 flow | `vc-performance/` JSON, `vc_performance_study.py`, `vc_bf16_study.py`, `vc_hardware_study.py`, `vc_quality_check.py`. Source/context windows and seams matter. Flow-only throughput is not whole-pipeline throughput, and old vanilla/GROW results do not automatically measure the newer 40k converter. |
| Large Acting optimizations and interim Alpha | `acting_speed_check.py`, `acting-performance.html`, `evidence/experimental-live.json`. Four earlier model options and actual parallel N were tested. Keep BF16 Acting/Alpha numbers separate from the later FP32 RL path. |
| Current RL smoke canaries | `evidence/rl-live.json`, `evidence/rl-ui.json`, `rl_live_check.py`: both bundled base+adapter models, streamed audio, parallel Best-of-N, MP3, nonempty ASR, Ears availability and applied fades on the working host. Smoke tests, not a new multi-emotion study. |

## Listening report locations

The first three reports above are permanently hosted in the
[HF study Space](https://huggingface.co/spaces/laion/humaneness-voice-small-studies).
It is a **static publication**, not a GPU inference server. Its old source
folder and recommendations do not supersede the new Git snapshot.

Additional working-host HTML routes, when the temporary tunnel is up:

- [Speaker prompting](https://fuji-formal-while-plugin.trycloudflare.com/speaker-study/)
- [Full-FT speaker study](https://fuji-formal-while-plugin.trycloudflare.com/full-ft-speaker-study/)
- [Mira activation study](https://fuji-formal-while-plugin.trycloudflare.com/mira-activation-study/)
- [Catalog steering and strict-tail sections](https://fuji-formal-while-plugin.trycloudflare.com/catalogue-steering-study/)
- [Voice-conversion throughput](https://fuji-formal-while-plugin.trycloudflare.com/vc-performance/)
- [Large Acting speed](https://fuji-formal-while-plugin.trycloudflare.com/acting-performance/)
- [Realization checks](https://fuji-formal-while-plugin.trycloudflare.com/realization-check.html)

These URLs are **ephemeral**; a stored JSON result is not a promise its audio
URL still resolves. The current Git update does not republish source recordings
or blindly copy HTML that might embed them. Replaying an experiment with a
new consented reference is useful, but it is not an exact original-condition rerun.

## Current RL validation boundary

The tested October RL deployment preserved FP32 policy/residual computation
for single take and `generate_many()`/Best-of-N, and kept the codec BF16. The
source audit for this documentation found the separate `ChunkSearch.generate()`
still wraps its policy loop in a **hardcoded BF16 autocast context**. Therefore
the presence of search controls does **not** establish that chunk search follows
the RL release precision contract. Do not call that combination verified or
publish its quality as a faithful RL result. This repository update documents
the captured state; it does not silently alter the running server. A future
fix requires a precision-aware search context and separate canaries.

Legacy DPO, Ref15, identity LoRAs and hidden-state steering are disabled/rejected
for Acting/Alpha/RL in the current UI/API. Layer indices/vectors are tied to
their source model, reference, feature selection, prompt set and coefficient.
No historical layer-16 bank should be automatically applied to a new checkpoint.

The last saved 500-profile collection progress is dated **2026-10-12 05:14 UTC**:
294,470 / 500,000 takes and 294 / 500 complete voices. The pipeline recorded a
worker failure at that time; this is not an all-500 completed bank. The saved
progress does not establish whether an external job resumed later. This docs
update neither restarts that collection nor allocates another GPU for it.

### Actual RL smoke timings, not steady-state promises

| Model / request | Actual batches | Aggregate device RTF | Request wall time |
| --- | --- | ---: | ---: |
| Small RL single streaming | `[1]` | 0.593 | 4.11 s |
| Small RL Best-of-3 | `[3]` | 0.199 | 4.09 s |
| Large RL single streaming, first switch | `[1]` | 1.188 | 176.13 s, including cold switch/compilation |
| Large RL Best-of-3 | `[2,1]` | 0.577 | 19.72 s |

These short same-prompt canaries include first-use graph capture; model-only
RTF excludes CPU ASR/scoring/alignment and network. Large RL CPU offload adds
substantial buffered-stream and final-result latency. They were not obtained
on a 4090/5090/H100, nor extrapolated to those GPUs. Read the recorded request,
performance scope and actual candidates before comparing numbers.

## Reproducible experiment record

Retain this Git revision; source/environment manifests; exact full base and
adapter hashes; codec/scorer revisions; input mode and full director request/
response/system prompt; reference source, normalized waveform hash and used
seconds; fixed vs dynamic selection; text and actual packed prompt; every seed;
temperature/top-k/top-p/CFG/frame cap; batch/microbatch and OOM retries; precision/
TF32/compilation; VC checkpoint/precision/context; raw vs aligned audio;
normalization/reward definitions; failures and all candidate scores, not just
the chosen take. Use held-out prompts/seeds/speakers for validation and listen
blind where possible. Proxy selection and source-data reuse can inflate apparent gains.
