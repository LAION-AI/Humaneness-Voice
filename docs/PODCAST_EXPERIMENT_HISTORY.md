# Earlier Podcast experiments and measurements

Historical record. For current defaults use [Podcast Generation Pipeline](PODCAST_GENERATION_PIPELINE.md). Earlier styles, rewards and hardware conditions differ; labels below preserve those distinctions.

## Measurements and interpretation

Single RTX 3090, BF16. The earlier fresh 50/50 experiment used the subsequently rejected per-sentence emotion prompts; these figures measure throughput, not approval of that style.

| Fresh Best-of-4 run | Final audio | Generation + ranking | Useful RTF |
|---|---:|---:|---:|
| German | 236 s | 179.13 s | 0.760 |
| English | 212 s | 174.23 s | 0.820 |

RTF = generation/ranking / final edited duration, not all candidate audio. Combined production including archival and cleanup was 380.65 s; generation/ranking accounted for 353.36 s. Cold setup, tests and timing probes took 356.52 s. Final audit load 30.10 s, audit 45.10 s, parallel CPU export 49.01 s; full experiment to exports 894.12 s. Do not call these request-to-first-audio latencies.

Shared prefill measured 0.273 vs 1.125 s for one group and 0.544 vs 2.110 s for two groups, with exact equality. A warm 104-frame probe took 3.572 s for one segment and 5.238 s for two (2.619 s/segment). The latter is an approximate workload comparison: two different scripts and different RNG batch shapes. An earlier run shared the card with a training job; speed differences cannot all be assigned to code.

The **earlier approved restrained rerank** reused 160 saved candidates, generated no new TTS, and changed 15/20 German and 12/20 English winners. Final German / English means: genuineness 3.967 / 3.822 out of 6; ECAPA cosine 0.655 / 0.401; turn WER 1.01% / 1.79%. Reranking took 32.18 s; do not report that as synthesis throughput. These are model metrics, not human quality ratings.

## Next efficiency experiments

1. Already implemented: overlap bounded CPU candidate archival with the next GPU batch; preserve all takes and propagate write errors before declaring completion. The earlier run spent roughly 27 s outside measured generation/ranking, including archival and cleanup; that is a ceiling, not a promised speedup.
2. Reuse a warmed worker, cached reference embeddings and tested graph shapes across episodes. Report cold start separately.
3. Explicitly reuse a previously measured batch size instead of repeating timing sweeps every episode, while retaining correctness checks after runtime changes.
4. Bucket comparable turn lengths to reduce completed-row padding, but record batch assignment because it changes RNG histories.
5. Benchmark EOS/continuation behavior before shortening frame caps. Never accelerate by silently clipping words or dropping candidates.

## Current standard: sparse spontaneous delivery

`--prompt-style spontaneous` retains every spoken word and original moderate turn direction, adding a short relaxed/gentle/spontaneous cue. Alternating turns get at most one additional cue at an existing sentence boundary. It does not repeat a list of emotions before every sentence. The listener-approved `spontaneous` style is now the default; `legacy` reproduces the earlier restrained variant.

```bash
python speaker_run.py --model "$MODEL" --device cuda:0 --n 4 --groups 2 \
  --prompt-style spontaneous --output outputs/spontaneous/generation
python speaker_finish.py --runs outputs/spontaneous/generation \
  --output outputs/spontaneous --device cuda:0
python verify_artifacts.py outputs/spontaneous
```

The explicit `--groups 2` reuses the previously measured batch size and skips repeated timing sweeps. It retains the real-model codec, generation-equivalence and prefill checks. First production batches can include cold graph/compilation costs; those are not hidden. If comparing throughput, keep this warm-up policy identical.

Candidate FLAC archival now runs on one bounded CPU thread alongside the next GPU batch. No candidates are discarded; write failures propagate, and a `result.json` is published only after its files exist. `archive-timing.json` distinguishes CPU work from the remaining blocking wait. Resume keeps stable batch grouping and replaces the same batch timing record rather than double-counting it.

The listener approved these samples and requested this prompt style as the default. Model-score changes alone still do not establish general superiority.


## New spontaneous variant: measured outcome, 8 November 2026

The portable package generated and scored all **160 candidates** (40 turns × 4), passed exact GPU codec/token/PCM and prefill checks, and exported both episodes. The eleven CPU tests cover prefix selection, CFG state copying, prompt word preservation, exact reward and incomplete archival receipts.

| Strict 50/50 selection | Final duration | Generation + ranking | Useful RTF | Genuineness / 6 | ECAPA cosine | Mean turn WER |
|---|---:|---:|---:|---:|---:|---:|
| German | 219.38 s | 179.98 s | 0.821 | 3.943 | 0.640 | 1.98% |
| English | 193.30 s | 195.83 s | 1.013 | 3.880 | 0.407 | 5.50% |

**Important observed failure:** the highest-reward take for English turn 6 ends after its first sentence (4 s instead of the complete turn). Its reward is 0.500111; the next candidate scores 0.499803. The strict experiment is kept unchanged and explicitly flagged. A separately labeled content-repaired listening copy can use a complete alternative, but that adds a content eligibility check and must not be reported as unconstrained 50/50 argmax. This illustrates why genuineness and speaker similarity alone cannot certify a complete script. The sparse spontaneous style is now the listener-approved default. The earlier restrained variant remains available for comparison.

Across both episodes, generation/ranking took 375.80 s: policy token prediction 324.85 s, codec decoding 24.85 s, scoring including preprocessing 21.25 s, and approximately 4.86 s in remaining generation overhead. Ears forward work (4.35 s) and ECAPA work (3.08 s) are **included** in scoring, not extra. Archived audio writing consumed 24.87 CPU seconds but blocked production for only 1.08 s; almost all writing overlapped GPU work. Total production including final archival/cleanup was 380.10 s. Process start to both MP3 exports was 645.59 s, excluding service restoration and the additional listening repair.

Do not attribute the entire startup reduction from the earlier 894.12 s experiment to code: this run reused an existing compiler cache, skipped repeated speed probes, used different prompts and inherited 512-position semantic buckets rather than the earlier 256. Overall production time was about the same as the earlier 380.65 s run; the archival overlap itself is measured, but a controlled aggregate speedup has not been established.

The largest remaining opportunity is **long-tail candidate computation**: 37 of 160 takes reached the 25.6 s cap and 19 of 20 batches waited for at least one capped take. Only 2,726.64 s of candidate audio were produced versus 4,096 s if all rows ran to the cap, but a batched policy can still spend time on already finished rows. Next, test conservative text-completion checks for lagging candidates, length-aware grouping, or active-row compaction with verified CFG/codec cache copying. Require complete-script checks and keep the original outputs for comparison; simply shortening the cap risks losing real speech. Keeping a warmed dedicated worker is the lowest-risk way to avoid repeated loading and compilation.

Machine-readable measurements: [spontaneous-2026-11-08.json](../podcast_pipeline/benchmarks/spontaneous-2026-11-08.json).


The additional English listening copy is **201.22 s** long. It replaces only turn 6 with existing candidate 1 (raw reward 0.499669, ASR WER 3.33%, plausible forced alignment). Candidate 3 was also checked first in reward order. Nothing was regenerated; the strict episode and its scores remain unchanged. `repair_listening.py OUTPUT --turn 6 --language en --asr-url http://127.0.0.1:8801/transcribe/parakeet_v3` reproduces this explicit repair with the recorded loopback ASR service; the default generation/audit commands never apply this content gate. Its HTTP ASR contract is mono 16 kHz float32 bytes in, JSON containing `text` out. CPU alignment uses FP32 to avoid slow BF16 emulation on older CPUs.
