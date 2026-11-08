# Podcast generation pipeline — reproducible Best-of-N

This recipe uses the **Podcast full fine-tune**, not the general Large 1.7 checkpoint. It generates complete independent speaker turns, samples N takes per turn in parallel, ranks them, and assembles the winners into a two-host episode. Voice conversion is disabled. German and English use the same two reference recordings.

## Current best practice: sparse spontaneous, relaxed and gentle delivery

On 8 November 2026 the listener approved the **sparse spontaneous/relaxed/gentle inline variant** and requested it as the standard. `--prompt-style spontaneous` is now the default. The earlier restrained `legacy` prompt remains available, while the exaggerated per-sentence emotion experiment is retained only for reproduction. This is a listener preference, not a universal or blinded-study optimum.

Read the [exact prompting guide and LLM-writing template](../podcast_pipeline/PROMPTING.md). An API-based or local LLM is recommended for writing the script and sparse cues: hand-authoring every turn and direction is unnecessarily laborious. The TTS itself does not write the dialogue and needs no LLM API key.

Exact GENERAL:

> A warm, emotionally thoughtful two-host podcast conversation. Very authentic, very genuine. Natural, intimate, spontaneous spoken delivery with clear words, subtle emotion, relaxed conversational pacing and brief meaningful pauses. Be sincere, without an exaggerated announcer voice or forced laughter. Perform only the current speaker turn and speak exactly the SCRIPT. Keep the same speaker identity as the reference audio.

Use **one moderate parenthesized direction plus a short spontaneous/gentle cue per turn**, and at most one extra cue at an existing sentence boundary, e.g. `(warm, intimate welcome; gently curious)`, followed by the literal transcript. Do not repeatedly demand strong emotions, forced laughter, or exaggerated acting. “Very pleasant to listen to” is deliberately absent. The experimental per-sentence style remains available only for reproducing the rejected comparison.

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
  --prompt-style spontaneous --output outputs/approved/generation
python speaker_finish.py --runs outputs/approved/generation \
  --output outputs/approved --device cuda:0
# Open outputs/approved/index.html.
```

`--n 1` keeps its only take and skips online scoring; final audit still measures its quality. `--n 2` uses the same 50/50 selection as N=4. N is takes **per segment**; the automatic pilot chooses one or two independent segments per policy batch, at most 16 simultaneous candidates. With two segments and N=4 the policy handles eight candidate histories, plus their CFG branches. The sample scripts have 20 turns per language. Supply custom JSON with `--script` (repeatable) and `--references`; see the prompting guide. `produce.py` runs generation plus audit/export in one command. `--limit 2` is a useful first smoke test. Re-run exactly the same configuration to resume completed turns; use a fresh directory when prompts/references/settings change. A live local Parakeet service is optional via the finisher's `--asr-url`; the default audit loads it locally after generation releases the policy.

Code includes checks that reused prompt prefill has identical hidden states and valid KV entries, that batched generation reproduces the prior one-segment runner, and that codec encoder offloading preserves decoding. If prefill equivalence fails, reuse is disabled. CUDA/Triton compilation is cold-start work and can take minutes. Same seeds do not promise identical outputs across hardware, libraries or batch grouping. Preserve all recorded scripts, source hashes, seeds and candidate scores.

The archived causal chunk search lives in `generation.py`; the default command above is **offline complete-turn selection**, not real-time Best-of-N streaming. Its legacy search reward differs; do not present it as the 50/50 experiment.

## Measured approved Best-of-4 baseline

Hardware: one RTX 3090 24 GB on a shared host; BF16, two turns × four takes per batch, 512-position semantic buckets, 26-frame internal waveform blocks, existing compiler cache. Both languages use the same fixed reference pair and approved sparse spontaneous directions.

| Language | Final audio | Generation + ranking | Useful RTF ↓ | Genuineness / 6 ↑ | ECAPA cosine ↑ | Mean turn WER ↓ |
|---|---:|---:|---:|---:|---:|---:|
| German | 219.38 s | 179.98 s | 0.821 | 3.943 | 0.640 | 1.98% |
| English, strict selection | 193.30 s | 195.83 s | 1.013 | 3.880 | 0.407 | 5.50% |

Useful RTF divides generation/ranking time by **final edited audio duration**, not total candidate audio. It excludes loading, correctness probes, final audit and export. Combined production including archival/cleanup took 380.10 s; process start to MP3 exports took 645.59 s, excluding service restoration and the additional listening repair.

Across both episodes: policy prediction 324.85 s, waveform decoding 24.85 s, ranking/preprocessing 21.25 s, remaining generation overhead 4.86 s. Ears forward work 4.35 s and ECAPA 3.08 s are included in ranking. Archival used 24.87 CPU seconds but blocked GPU production for just 1.08 s. These are measured stages, not promises for another machine.

**Content failure retained:** strict English turn 6 stopped after the first sentence. A separate complete listening copy replaces only that turn with existing candidate 1 (reward 0.499669; diagnostic WER 3.33%) and lasts 201.22 s. No new synthesis was used. This explicitly adds a content eligibility check; it is not the original strict 50/50 result. The default pipeline keeps strict winners and flags problems instead of silently repairing them. The listener approved the sparse style including that complete listening copy.

See [baseline machine-readable results](../podcast_pipeline/benchmarks/spontaneous-2026-11-08.json) and [historical experiments](PODCAST_EXPERIMENT_HISTORY.md). Metrics are fallible model estimates, not blinded human ratings.

## Implemented speed optimizations

- BF16 policy, reusable semantic and local-decoder CUDA graphs, fast token sampling and tested fused codec operations.
- Multiple independent turns in one batch. Histories remain separate; candidate selection is local to each turn.
- Identical candidate prompts reuse prefill only after hidden-state and valid-KV equality checks. A two-group check measured 2.009 s normally versus 0.518 s with reuse.
- Cache reference acoustic codes and speaker embeddings. Keep the unused codec encoder in CPU RAM after reference preparation.
- Batch the Ears/ECAPA judges. Skip online ranking for N=1; reserve Parakeet for final audit.
- Archive every candidate with one bounded CPU writer while the next GPU batch runs. Propagate write failures before publishing a completed-turn receipt.
- Reuse an explicitly selected `--groups` value, a warmed worker and compiler caches instead of repeating setup and timing sweeps for every episode.

`--groups 2 --n 4`, `--groups 4 --n 2`, and `--groups 8 --n 1` each maintain eight candidates per policy batch, plus CFG branches. These layouts change RNG trajectories; N1/N2 are not necessarily subsets of an N4 run. Resume requires unchanged settings and stable batch grouping.

### When VRAM is shared or limited

`low_memory_compare.py` separates **policy token generation → release policy → waveform decoding and ranking**. It saves acoustic tokens for every candidate, retains the same token sampling loop, checks shared/full prefill equality and tests optional fused decoding. It requires prepared reference codes. This is an offline comparison runner, not a streaming service and not a general low-memory implementation for arbitrary recordings.

```bash
python low_memory_compare.py --model "$MODEL" --device cuda:0 --output outputs/n-comparison
for N in 2 1; do
  python speaker_finish.py --runs "outputs/n-comparison/n$N/generation" \
    --output "outputs/n-comparison/n$N" --device cuda:0
  python verify_artifacts.py "outputs/n-comparison/n$N"
done
python compare_report.py --n4 outputs/spontaneous --n2 outputs/n-comparison/n2 \
  --n1 outputs/n-comparison/n1 --output outputs/n-comparison
```

Stage sums exclude shared loading and inter-phase waiting; the parent receipt records total generation-job time. Do not label their sum request latency. An integrated attempt ran out of memory when another user's training job appeared; no unrelated job was stopped. The phased comparison uses the same shared GPU under that extra load. A fair speed ablation should repeat N=1/2/4 with identical warm-up and an otherwise idle GPU.

## Next speed improvements to measure

Policy prediction dominates. In N4, 37/160 candidates hit the 25.6-s cap and 19/20 batches waited for a capped candidate. Length-aware grouping and safe finished-row compaction may reduce wasted work, but require verified CFG/cache remapping and complete-script checks. Do not simply reduce the ceiling or remove words to improve RTF. A persistent dedicated worker is the lowest-risk way to avoid loading and compilation latency.

Reassembly and scoring are much smaller costs. Optimizing the codec alone cannot remove the policy bottleneck. Keep truthful per-stage, per-episode and end-to-end timings; preserve all candidate files so selection experiments need no fresh synthesis.

## Fresh N=1 and N=2 comparison with the approved prompts

Both episodes were freshly generated at N=2 and N=1 after publishing the new defaults. Same literal scripts, cues, reference pair, temperature 1.0 and CFG 2.5. N1 keeps its sole take and skips online ranking. N2 uses the same 50/50 reward as N4; there is no WER gate or hidden retry.

| Language | N | Final audio | Generate + rank | Useful RTF ↓ | Genuineness / 6 ↑ | ECAPA cosine ↑ | Mean turn WER ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| DE | 1 | 220.75 s | 55.24 s | 0.250 | 3.916 | 0.553 | 1.21% |
| DE | 2 | 222.26 s | 115.80 s | 0.521 | 3.897 | 0.630 | 2.56% |
| DE | 4 | 219.38 s | 179.98 s | 0.821 | 3.943 | 0.640 | 1.98% |
| EN | 1 | 199.30 s | 68.56 s | 0.344 | 3.863 | 0.352 | 2.12% |
| EN | 2 | 199.54 s | 122.77 s | 0.615 | 3.904 | 0.383 | 2.93% |
| EN | 4 | 193.30 s | 195.83 s | 1.013 | 3.880 | 0.407 | 5.50% |

In this sample, **N=2 is a useful speed/identity compromise**: German speaker cosine rises from 0.553 at N1 to 0.630 at N2, close to N4's 0.640. English rises from 0.352 to 0.383; N4 is 0.407. Genuineness stays roughly 3.9/6. N1 is faster but less consistent in speaker identity. These are twenty turns per language, not evidence of a universal optimum. The old English N4 row includes its documented incomplete turn; the repaired listening copy is separate.

All 80 new selected turns passed the configured review checks (per-turn WER threshold 15% and usable forced alignment); none was regenerated or replaced. This is not a guarantee of perfect speech. Both candidate counts passed artifact validation, exact prompt/reference routing, strict selection and PCM equality checks. The token/full-prefill checks and optional fused decoding were bit-identical on their tested inputs. Fourteen CPU tests pass.

**Timing scope:** N2 production across both languages totals 238.57 s; N1 totals 123.80 s; old N4 totals 375.80 s. Token prediction accounts for 209.57 / 113.22 / 324.85 s respectively. N2 codec/ranking take 18.18/10.07 s; N1 codec takes 9.91 s and online ranking zero. Loading, correctness probes, final audit and export are excluded from these production sums. The complete successful N2+N1 experiment, from process start to all four MP3s, took **660.70 s**. The failed earlier OOM attempt and live-service restoration are excluded.

N2/N1 used phased execution under additional unrelated GPU training; N4 used integrated execution with a different background load. The other training ended before final auditing. Do not claim these are controlled percentage speedups from N alone. N2 uses four turns per batch and N1 eight (final partial batches smaller), maintaining up to eight simultaneous candidates. Different groups/seeds mean these are fresh runs, not nested subsets of the N4 pool.

See [aggregate results](../podcast_pipeline/benchmarks/n-comparison-2026-11-08.json) and [all per-turn candidate scores, seeds and audits](../podcast_pipeline/benchmarks/n-comparison-2026-11-08-turns.json). Recreate the listening HTML with `compare_report.py`; the six strict episodes and additional labeled N4 English listening copy are independently selectable.
