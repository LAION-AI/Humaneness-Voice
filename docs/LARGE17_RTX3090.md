# Large 1.7 in Robby: RTX 3090 streaming and batching

The shared runtime can select `large17` (Humaneness Voice Large 1.7) or the
existing Small model without changing Robby's conversation, avatar or ASR.
Robby's **Voice synthesis** selector is near the top of the page. Small remains
the default; the browser remembers an explicit selection.

For the matched **Small versus Large** throughput table and separately labelled
Jetson Thor results, see [TTS performance](TTS_PERFORMANCE.md). The same report
is published [beside the Large 1.7 weights](https://huggingface.co/laion/Humaneness-Voice-Large-1.7/blob/main/ROBBY_RTX3090.md).

## Weights and precision

- Repository: https://huggingface.co/laion/Humaneness-Voice-Large-1.7
- Pinned revision: `f008a7a685122b8d0e45f2c44f1912c93e29f352`.
- Load the root merged model **once**; do not also attach its training adapter.
- 4,129,990,144 parameters. Distribution shards occupy about 16.52 GB (FP32).
  The serving copy uses BF16 (about 8.26 GB of model weights), plus the codec,
  KV caches, CUDA graphs and working memory.
- `acting_models.py` pins the index and all three shard SHA-256 hashes.
  `checkpoint_ram.py` verifies and converts one shard at a time into private
  CPU RAM. It never retains a second complete FP32 working copy.
- `HVS_LARGE17_CACHE` can place these weights on a different disk. The portable
  default is the configured Hugging Face cache. No weights are committed here.

Keep at least 5.04 seconds of clean single-speaker reference speech. The
waveform is cropped **before** encoding to 14.96 seconds (63–187 codec frames).
The demo uses its fixed 10.26-second Mira reference. Short references fail with
an explicit error. No old 2.96-second crop is reintroduced.

## Runtime and switching

`large17_engine.py` loads the release's own configuration, tokenizer and
processor. It shares the existing MOSS Audio Tokenizer v2 with Small, but never
reuses another model's prompt processor. Verified CPU weight copies remain
available for subsequent switches; only the selected TTS model is on the GPU.
Optional scoring/aligner modules can move to CPU for GPU headroom.

On this BF16 deployment, Parakeet tail checks and forced alignment stay on GPU
to avoid pauses between sentences; optional Ears ranking moves to CPU. The older
FP32 Large RL model keeps its separate, more aggressive offload policy.

The existing TTS API accepts `model: "large17"` in `/api/turn` and
`/api/models/select`. This is a server-wide GPU selection. The Robby frontend
uses a short `POST /api/tts/models` request followed by `GET` status polling,
so cold compilation does not hold a Cloudflare request open. Active replies
reject a switch; newly arriving turns wait for an ongoing switch. The first
load and graph capture are not counted as warmed inference latency.

The selected model is carried into sentence TTS, explicit interjections and
latency receipts. Cached interjection audio is keyed by model as well as voice,
language and text. Optional Small continuous two-stage reactions retain their
existing path; Large uses ordered single-sentence streaming. No ASR, LLM,
facial-expression or turn-taking model is replaced.

## Speed options

Use the existing RTX 3090
[`rtx3090-speed.env` recipe](https://github.com/christophschuhmann/cartoon-face-generator/blob/80d00b6be7d8ea2cc592c3c5f6f18ca7920274e7/robby/voice-assistant/optimizations/rtx3090-speed.env)
(download it from the Robby repository before sourcing it): BF16 weights, cached
references, batched CFG branches, semantic static KV caches and CUDA graphs,
compiled semantic continuation, fast top-k/top-p sampling, Triton local decoder,
causal codec graphs/fusion and growing chunks. `HVS_LOCAL_COMPILE=0` avoids the
known PyTorch 2.6 scalar-capture problem; the tested Triton implementation is
selected by `HVS_LOCAL_TRITON=1`.

Single-candidate streaming emits 3, 6, 12 and then 20 codec frames per block
(240, 480, 960 and 1600 ms). The local Triton implementation and codec CUDA
graphs are specialized for batch 1. Larger candidate batches use the real
batched original local decoder and causal codec, not a loop of independent
full-utterance generations.

The additional `HVS_LARGE17_LAYERWISE=1` experiment compiles one tensor-only
semantic transformer block and reuses it across 36 layers inside a complete
CUDA graph. It bounds the compiler's problem size without quantizing weights.
The benchmark retains the original monolithic compilation comparison and
teacher-forced state checks; do not infer numerical identity from equal seeds.
The tested Robby worker enables it and warms 512- and 1024-position single-stream
KV capacities during model selection. Other batch/length shapes may still need
their own cold capture. Single-stream warm speed is similar, not demonstrably
faster, than monolithic compilation; the main advantage is bounded compilation.

## Measured resident inference

One RTX 3090, temperature 0.8, CFG 2, three warm seeds, synthetic fixed Mira
reference. First PCM means the local callback, not browser playback. RTF for
batches divides by the sum of candidate audio durations.

| Path | Batch | First PCM | Complete batch | Aggregate RTF | Audio seconds / compute second |
| --- | ---: | ---: | ---: | ---: | ---: |
| Original Small optimization package, streaming | 1 | 0.403 s | 1.410 s | 0.275 | 3.64 |
| Original Small optimization package, offline | 1 | — | 1.782 s | 0.350 | 2.86 |
| Layerwise compilation, streaming | 1 | 0.433 s | 1.491 s | 0.290 | 3.45 |
| Layerwise compilation, offline | 8 | — | 3.679 s | 0.089 | 11.28 |
| Layerwise compilation, streaming | 8 | 1.309 s | 5.624 s | 0.136 | 7.33 |
| Layerwise compilation, offline | 16 | — | 5.475 s | 0.065 | 15.45 |
| Layerwise compilation, streaming | 16 | 2.048 s | 7.483 s | 0.090 | 11.16 |

The fastest tested configuration reduces CFG to 1: batch 16 offline reaches
RTF 0.0429 (23.3 audio seconds/second); streaming reaches 0.0709 (14.1 audio
seconds/second) with 1.453 s first PCM. This changes guidance and is not a claim
of equal acting quality. Robby's CFG stays at 2. The live cap remains 8 for
longer prompts and the colocated finishing models; 16 is a short-fixture probe.
These are finite batches, not a sustained multi-user admission benchmark.

A CUDA-event profile attributes approximately 760 ms to semantic continuation,
254 ms to semantic prefill, 135 ms to acoustic-token prediction and 69 ms to
PCM decoding for one short CFG-2 stream. The **semantic backbone** dominates
this optimized Large run. Instrumented spans are explanatory, not substituted
for the uninstrumented timings above.

Teacher-forced state cosine similarity versus the native implementation stays
above 0.9986 across the checked states, including two CFG branches. BF16 states
are not bit-identical (relative RMSE up to about 4.5%). All nine retained optimized
short/long clips have zero normalized WER under the separate Parakeet canary.
This is not a blinded perceptual-quality or speaker-similarity study.

## Real HTTP streaming

With the real reference and finishing path enabled, median first PCM is
**0.474 s for Large** and **0.403 s for Small**, each over three warm repeats.
The complete short request takes 3.241 s and 2.642 s respectively, including
ASR tail checks, final alignment and output handling. With alignment disabled,
Large reaches first PCM in 0.470 s and completes in 1.538 s. Alignment remains
enabled in Robby. See `api-check.json` for all receipts.

The first Large selection after startup took 132.1 s including loading and
warming; returning to Small took 81.4 s in this run. These are model-swap costs,
not per-utterance TTFA. The later browser switch to Large used the retained CPU
weights and completed in about 16 s. Small still rebuilds its monolithic graphs
on a swap; the later browser return took 60.1 s. All four browser turns passed,
including Large speech with a spin action and the return to Small.

## Reproduce the measurements

Use a compatible local installation of the playground runtime. The measured
host uses PyTorch 2.6.0+cu124, Triton 3.2.0, Transformers 5.15.0 and RTX 3090
GPUs with driver 550.54.15. The benchmark uses one GPU; it does not sum the two
cards' throughput. Other assistant workers remained resident.

```bash
export HVS_RUNTIME=/absolute/path/to/Humaneness-Voice/playground/runtime
export HVS_LARGE17_CACHE=/a/drive/with/at/least/18GB/free/hf-large17
export ACTING_DEVICE=cuda:1
export HVS_LARGE17_LAYERWISE=1
export TMPDIR=/a/drive/with/free/space/tmp
export TORCHINDUCTOR_CACHE_DIR=/a/drive/with/free/space/inductor
set -a
source /path/to/rtx3090-speed.env
set +a
# Stop the TTS worker before starting a second GPU copy; leave unrelated jobs alone.
python playground/benchmarks/large17/benchmark.py
```

The runtime's fixed reference assets must already be installed (see the
playground setup guide). Scripts write beside themselves; keep source and
result files together, or copy the small benchmark directory into a writable
experiment directory. No microphone recordings are used.

`benchmark.py` retains cold and warm rows separately, uses three measured seeds,
tests CFG 1/2/2.5, streaming and offline generation, and candidate batches up to
16. The actual safe deployed cap follows the measured memory limit. It also
includes a longer sentence. `checks.py` compares native and optimized semantic
states under identical forced acoustic tokens and records component GPU event
spans. `layerwise_check.py` separately measures the smaller compilation unit.
`api_check.py` measures real HTTP PCM onset with and without final alignment.
`build_report.py` recreates the English HTML and 96-kbps mono MP3 listening copies
from the local receipts and generated WAV files (requires ffmpeg). Set
`HVS_REPORT_DEST` for the destination directory. These MP3 copies are not used
for the GPU timing or ASR checks.
`browser_check.py` verifies the real HTTPS model selector, actual selected model
receipts, audio delivery, a spin command and returning to Small.

RTF is generation plus codec time divided by **total generated audio seconds**.
For N candidates this is aggregate throughput, not one listener's latency.
On-device results exclude ASR, LLM, alignment, ranking, network, MP3 and reference
encoding. HTTP/browser receipts are reported separately. Peak allocated VRAM
is the measured process allocator, not all GPU memory. An OOM is a failed case.
A complete best-of-N winner still requires candidate scoring after generation.
The batch callback exposes all candidate audio blocks; it does not promise a
final best-of-N winner before the candidates finish.

The initial short-caption smoke test reached 0.139–0.142 s first PCM once warm;
its first uncached call took 393.8 s. It is retained separately in
`initial-smoke.json`: its prompt is shorter than the standardized structured
benchmark, so 0.14 s is not the headline latency for arbitrary Robby replies.
The first long-sentence sample after changing batch shapes also includes
recapture. Its later two repetitions reach about 0.40 s and RTF 0.24; do not
pool the initial recapture into a warmed-only estimate without labelling it.

The legacy eager comparison is the serving runtime's old frame loop, not a
benchmark of the release's standalone CLI. Its fixed 20-frame streaming blocks
also differ from the optimized onset ramp. Initial CFG=1 eager rows briefly
overlapped an ASR smoke check and are excluded from precise speedup claims.
Use the matched CFG=2 operating point and the retained raw receipts.

## Validation and limits

Duration/cache and zero-temperature sampler unit tests run without GPU model
weights. Model switching tests cover overlapping requests, invalid IDs,
failed swaps and turn unblocking. ASR transcripts, finite-waveform checks and
teacher-forced state agreement are correctness canaries, not a blinded quality
study. Listen to the saved synthetic samples before making a naturalness or
speaker-similarity claim. The timing study is not a guarantee for different
GPUs, concurrent workloads, reference lengths or sentence lengths.

## Next optimization targets

1. Capture the causal codec for multiple batch rows. Currently only batch 1
   has persistent codec CUDA graphs; larger batches pay extra launch overhead.
   Validate per-row stop masks, state reset and varying tail lengths first.
2. Test compiled/bucketed semantic prefill and reuse only truly identical
   prompt-prefix KV states. Prefill is a material fraction of first-audio time;
   reference-code caching alone does not eliminate it.
3. Explore weight-only quantization of the semantic backbone separately from
   the acoustic decoder. The RTX 3090 has no native FP8 tensor-core path; a
   lower-precision label alone is not evidence of a speedup. Require measured
   quality, actual kernels and end-to-end timings before deployment.
4. Measure longer utterances, multiple reference durations and sustained
   admission/load. This finite-batch study does not establish fairness or
   deadline guarantees for a multi-user server.
