# Podcast generation pipeline — reproducible Best-of-N

This recipe uses the **Podcast full fine-tune**, not the general Large 1.7 checkpoint. It generates complete independent speaker turns, samples N takes per turn in parallel, ranks them, and assembles the winners into a two-host episode. Voice conversion is disabled. German and English use the same two reference recordings.

## Best practice so far: restrained authentic delivery

On 8 November 2026 the experiment owner preferred the **original restrained prompt**, reranked with the 50/50 reward below, over a later experiment that repeated emotion and authenticity directions before every sentence. This is a listening preference for these episodes, not a blinded study or a universal optimum. Keep this approved recipe as the default; new spontaneous-delivery experiments must use separate output directories.

Exact GENERAL:

> A warm, emotionally thoughtful two-host podcast conversation. Very authentic, very genuine. Natural, intimate, spontaneous spoken delivery with clear words, subtle emotion, relaxed conversational pacing and brief meaningful pauses. Be sincere, without an exaggerated announcer voice or forced laughter. Perform only the current speaker turn and speak exactly the SCRIPT. Keep the same speaker identity as the reference audio.

Use **one moderate parenthesized direction per turn**, e.g. `(warm, intimate welcome; gently curious)`, followed by the literal transcript. Do not repeatedly demand strong emotions, forced laughter, or exaggerated acting. “Very pleasant to listen to” is deliberately absent. The experimental per-sentence style remains available only for reproducing the rejected comparison.

Temperature **1.0**, CFG **2.5**, top-p **0.95**, top-k **50**, BF16, free timing (`tokens=None`), 320 semantic frames / 25.6 s maximum per turn. Split longer scripts into shorter turns. No DPO, additional LoRAs, or voice conversion. These settings apply to this Podcast checkpoint, not all Humaneness models.

## Exact reward and quality audit

```
G = clip(native Humaneness Ears genuineness / 6, 0, 1)
S = clip(cosine(ECAPA(candidate), ECAPA(correct reference)), 0, 1)
reward = 0.5 * G + 0.5 * S
```

There is **no Content Enjoyment term and no WER multiplier or WER selection gate**. Fixed scales avoid unstable per-batch min/max normalization. Exact ties select the lowest candidate index. Score complete raw candidates; audit the edited winner separately. ECAPA is an approximate identity cue, not a guarantee of successful cloning. A high reward does not guarantee correct words.

After selection, Qwen forced alignment supplies approximate word boundaries; keep 80 ms before the first word and 150 ms after the last, with small fades outside the aligned words. Uncertain alignments are kept untrimmed and flagged. Reassembly uses a 400 ms **word-to-word** gap, not a 400 ms file-to-file gap. Final Parakeet WER is diagnostic only; never silently replace the reward winner. Export mono 48 kHz / 192 kbps MP3 plus lossless source, subtitles, per-turn scores, review flags, and listening HTML. No internal pause compaction.

## Weights and references

| Component | Published source | Pinned revision |
|---|---|---|
| Podcast policy, full FT step 9874, 4.13B | [laion/Humanness-Voice-Podcast](https://huggingface.co/laion/Humanness-Voice-Podcast) | `a1f8946bd654db1c19b6c476ed6b4acde384e882` |
| MOSS v2 codec | [OpenMOSS-Team/MOSS-Audio-Tokenizer-v2](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) | `f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169` |
| Genuineness judge | [Humaneness Ears Medium](https://huggingface.co/laion/humaneness-ears-base-medium) | `818506970d93c809a3295a0a02c40dee8ff2bfc3` |
| Speaker similarity, 192D | [SpeechBrain ECAPA VoxCeleb](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb) | `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286` |
| Final transcript diagnostic | [Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) | `541d1f99c6b0c3cd0b11a95167540bb8edefd82b` |
| Final alignment | [Qwen3 ForcedAligner](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B-hf) | `c07281df297b9905d24a508279258cccf987a064` |

Both languages condition S1 on `de_s1.mp3` and S2 on `de_s2.mp3`, from the Podcast repository's [reference release](https://huggingface.co/laion/Humanness-Voice-Podcast/tree/a1f8946bd654db1c19b6c476ed6b4acde384e882/examples/references). Recordings are approximately 14.52 / 13.08 s; cached codec conditioning contains 181 / 163 frames (14.48 / 13.04 s). `references.json` contains SHA-256 checks. Do not confuse fictional host labels with real identities. Source recordings retain original terms; this package downloads them rather than relicensing or republishing them.

## Run on another machine

The [standalone code](../podcast_pipeline/) includes the optimized policy/codec runtime, scoring, example scripts, alignment and reassembly. It does not import the Robby server, depend on private absolute paths, or require API keys. NVIDIA CUDA with BF16 support is required for this optimized path. Tested environment: RTX 3090 24 GB, Python 3.12, PyTorch 2.6.0+cu124, Transformers 5.15.0. Install system `ffmpeg` and `libsndfile` first. Use an empty GPU; unrelated workloads can cause OOM or change timings. The policy alone occupies about 8.26 GB; codec, judges, KV caches and graph capture need additional memory.

```bash
cd podcast_pipeline
python3.12 -m venv .venv
source .venv/bin/activate
pip install torch==2.6.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt
# Optionally set HF_HOME and PODCAST_CACHE on a drive with sufficient free space.
python prepare.py
MODEL=$(python -c 'import json; print(json.load(open("assets.local.json"))["model"])')
python -m unittest test_contracts test_speaker_recipe
python speaker_run.py --model "$MODEL" --device cuda:0 --n 4 \
  --prompt-style legacy --output outputs/approved/generation
python speaker_finish.py --runs outputs/approved/generation \
  --output outputs/approved --device cuda:0
# Open outputs/approved/index.html.
```

`--n 1` is also supported. N is takes **per segment**; the automatic pilot chooses one or two independent segments per policy batch, at most 16 simultaneous candidates. With two segments and N=4 the policy handles eight candidate histories, plus their CFG branches. The sample scripts have 20 turns per language. `--limit 2` is a useful first smoke test. Re-run exactly the same configuration to resume completed turns; use a fresh directory when prompts/references/settings change. A live local Parakeet service is optional via the finisher's `--asr-url`; the default audit loads it locally after generation releases the policy.

Code includes checks that reused prompt prefill has identical hidden states and valid KV entries, that batched generation reproduces the prior one-segment runner, and that codec encoder offloading preserves decoding. If prefill equivalence fails, reuse is disabled. CUDA/Triton compilation is cold-start work and can take minutes. Same seeds do not promise identical outputs across hardware, libraries or batch grouping. Preserve all recorded scripts, source hashes, seeds and candidate scores.

The archived causal chunk search lives in `generation.py`; the default command above is **offline complete-turn selection**, not real-time Best-of-N streaming. Its legacy search reward differs; do not present it as the 50/50 experiment.

## Measurements and interpretation

Single RTX 3090, BF16. The earlier fresh 50/50 experiment used the subsequently rejected per-sentence emotion prompts; these figures measure throughput, not approval of that style.

| Fresh Best-of-4 run | Final audio | Generation + ranking | Useful RTF |
|---|---:|---:|---:|
| German | 236 s | 179.13 s | 0.760 |
| English | 212 s | 174.23 s | 0.820 |

RTF = generation/ranking / final edited duration, not all candidate audio. Combined production including archival and cleanup was 380.65 s; generation/ranking accounted for 353.36 s. Cold setup, tests and timing probes took 356.52 s. Final audit load 30.10 s, audit 45.10 s, parallel CPU export 49.01 s; full experiment to exports 894.12 s. Do not call these request-to-first-audio latencies.

Shared prefill measured 0.273 vs 1.125 s for one group and 0.544 vs 2.110 s for two groups, with exact equality. A warm 104-frame probe took 3.572 s for one segment and 5.238 s for two (2.619 s/segment). The latter is an approximate workload comparison: two different scripts and different RNG batch shapes. An earlier run shared the card with a training job; speed differences cannot all be assigned to code.

The **approved restrained rerank** reused 160 saved candidates, generated no new TTS, and changed 15/20 German and 12/20 English winners. Final German / English means: genuineness 3.967 / 3.822 out of 6; ECAPA cosine 0.655 / 0.401; turn WER 1.01% / 1.79%. Reranking took 32.18 s; do not report that as synthesis throughput. These are model metrics, not human quality ratings.

## Next efficiency experiments

1. Overlap bounded CPU candidate archival with the next GPU batch; preserve all takes and propagate write errors before declaring completion. The earlier run spent roughly 27 s outside measured generation/ranking, including archival and cleanup; that is a ceiling, not a promised speedup.
2. Reuse a warmed worker, cached reference embeddings and tested graph shapes across episodes. Report cold start separately.
3. Explicitly reuse a previously measured batch size instead of repeating timing sweeps every episode, while retaining correctness checks after runtime changes.
4. Bucket comparable turn lengths to reduce completed-row padding, but record batch assignment because it changes RNG histories.
5. Benchmark EOS/continuation behavior before shortening frame caps. Never accelerate by silently clipping words or dropping candidates.

## Separate spontaneous-delivery experiment

`--prompt-style spontaneous` retains every spoken word and original moderate turn direction, adding a short relaxed/gentle/spontaneous cue. Alternating turns get at most one additional cue at an existing sentence boundary. It does not repeat a list of emotions before every sentence. The approved `legacy` style remains the default.

```bash
python speaker_run.py --model "$MODEL" --device cuda:0 --n 4 --groups 2 \
  --prompt-style spontaneous --output outputs/spontaneous/generation
python speaker_finish.py --runs outputs/spontaneous/generation \
  --output outputs/spontaneous --device cuda:0
python verify_artifacts.py outputs/spontaneous
```

The explicit `--groups 2` reuses the previously measured batch size and skips repeated timing sweeps. It retains the real-model codec, generation-equivalence and prefill checks. First production batches can include cold graph/compilation costs; those are not hidden. If comparing throughput, keep this warm-up policy identical.

Candidate FLAC archival now runs on one bounded CPU thread alongside the next GPU batch. No candidates are discarded; write failures propagate, and a `result.json` is published only after its files exist. `archive-timing.json` distinguishes CPU work from the remaining blocking wait. Resume keeps stable batch grouping and replaces the same batch timing record rather than double-counting it.

This experiment is pending the listener’s preference decision. Model-score changes alone do not establish that the acting sounds better.


## New spontaneous variant: measured outcome, 8 November 2026

The portable package generated and scored all **160 candidates** (40 turns × 4), passed exact GPU codec/token/PCM and prefill checks, and exported both episodes. The eleven CPU tests cover prefix selection, CFG state copying, prompt word preservation, exact reward and incomplete archival receipts.

| Strict 50/50 selection | Final duration | Generation + ranking | Useful RTF | Genuineness / 6 | ECAPA cosine | Mean turn WER |
|---|---:|---:|---:|---:|---:|---:|
| German | 219.38 s | 179.98 s | 0.821 | 3.943 | 0.640 | 1.98% |
| English | 193.30 s | 195.83 s | 1.013 | 3.880 | 0.407 | 5.50% |

**Important observed failure:** the highest-reward take for English turn 6 ends after its first sentence (4 s instead of the complete turn). Its reward is 0.500111; the next candidate scores 0.499803. The strict experiment is kept unchanged and explicitly flagged. A separately labeled content-repaired listening copy can use a complete alternative, but that adds a content eligibility check and must not be reported as unconstrained 50/50 argmax. This illustrates why genuineness and speaker similarity alone cannot certify a complete script. The approved restrained baseline remains the default.

Across both episodes, generation/ranking took 375.80 s: policy token prediction 324.85 s, codec decoding 24.85 s, scoring including preprocessing 21.25 s, and approximately 4.86 s in remaining generation overhead. Ears forward work (4.35 s) and ECAPA work (3.08 s) are **included** in scoring, not extra. Archived audio writing consumed 24.87 CPU seconds but blocked production for only 1.08 s; almost all writing overlapped GPU work. Total production including final archival/cleanup was 380.10 s. Process start to both MP3 exports was 645.59 s, excluding service restoration and the additional listening repair.

Do not attribute the entire startup reduction from the earlier 894.12 s experiment to code: this run reused an existing compiler cache, skipped repeated speed probes, used different prompts and inherited 512-position semantic buckets rather than the earlier 256. Overall production time was about the same as the earlier 380.65 s run; the archival overlap itself is measured, but a controlled aggregate speedup has not been established.

The largest remaining opportunity is **long-tail candidate computation**: 37 of 160 takes reached the 25.6 s cap and 19 of 20 batches waited for at least one capped take. Only 2,726.64 s of candidate audio were produced versus 4,096 s if all rows ran to the cap, but a batched policy can still spend time on already finished rows. Next, test conservative text-completion checks for lagging candidates, length-aware grouping, or active-row compaction with verified CFG/codec cache copying. Require complete-script checks and keep the original outputs for comparison; simply shortening the cap risks losing real speech. Keeping a warmed dedicated worker is the lowest-risk way to avoid repeated loading and compilation.

Machine-readable measurements: [spontaneous-2026-11-08.json](../podcast_pipeline/benchmarks/spontaneous-2026-11-08.json).


The additional English listening copy is **201.22 s** long. It replaces only turn 6 with existing candidate 1 (raw reward 0.499669, ASR WER 3.33%, plausible forced alignment). Candidate 3 was also checked first in reward order. Nothing was regenerated; the strict episode and its scores remain unchanged. `repair_listening.py OUTPUT --turn 6 --language en --asr-url http://127.0.0.1:8801/transcribe/parakeet_v3` reproduces this explicit repair with the recorded loopback ASR service; the default generation/audit commands never apply this content gate. Its HTTP ASR contract is mono 16 kHz float32 bytes in, JSON containing `text` out. CPU alignment uses FP32 to avoid slow BF16 emulation on older CPUs.
