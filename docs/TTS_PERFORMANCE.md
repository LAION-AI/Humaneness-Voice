# Humaneness Voice: measured Small and Large inference performance

Updated 7 November 2026. These are measured results, not estimates for untested
GPUs. The optimized **Small RL** and **Large 1.7** paths are available in the
[TTS runtime](https://github.com/LAION-AI/Humaneness-Voice/tree/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/playground/runtime).
The [Robby integration](https://github.com/christophschuhmann/cartoon-face-generator/blob/main/robby/voice-assistant/LARGE17.md)
lets users switch between them; Small remains the default.

## Small versus Large on the same RTX 3090

One NVIDIA RTX 3090 with 24 GiB VRAM, BF16, CFG 2, temperature 0.8, one streaming
candidate, the same structured German prompt and fixed 10.26-second synthetic
Mira reference. Each entry is the median of three warm repeats; the initial
warm-up is excluded. The machine has two GPUs, but **TTS uses only one**. Other
assistant models remained resident. Final alignment was disabled for this
comparison; the table below separately reports the aligned production path.

| Checkpoint | GPU | First PCM over local HTTP | On-device RTF | Audio seconds / compute second | Complete HTTP request |
| --- | --- | ---: | ---: | ---: | ---: |
| Small 1.1 Alpha RL r128 v1 | RTX 3090, 24 GiB | 0.394 s | 0.1669 | 5.99 | 0.974 s |
| Large 1.7, layerwise compilation | RTX 3090, 24 GiB | 0.470 s | 0.2873 | 3.48 | 1.538 s |

**RTF = device generation and codec time / generated audio duration.** Lower is
better. Throughput is its reciprocal; higher is better. The device event span
includes prefill, autoregressive decoding, PCM decoding and launch gaps. It
excludes reference encoding, prompt packing, ASR, LLM, alignment, ranking, voice
conversion, CPU audio copies, MP3 encoding and network transport. HTTP first PCM
includes server preparation and local transport, but not browser buffering or
physical speaker latency. First PCM can contain initial silence.

The generated durations were 5.36–5.44 seconds for Small and 5.04 seconds for
Large. Equal seed labels do not imply identical sampled tokens across models.
This is a matched speed comparison, not a finding that their quality is equal.
Three repeats are a small warm performance sample, not a tail-latency estimate.

With the normal ASR tail checks and final alignment enabled:

| Checkpoint | First PCM over local HTTP | Complete request, including finishing |
| --- | ---: | ---: |
| Small RL | 0.403 s | 2.642 s |
| Large 1.7 | 0.474 s | 3.241 s |

The audio starts before finishing completes. Do not divide these complete
request times by duration and label the result an on-device RTF.
[Raw HTTP receipts](https://github.com/LAION-AI/Humaneness-Voice/blob/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/playground/benchmarks/large17/api-check.json)
record the timing scopes, actual checkpoint, first blocks and generated duration.

## Large 1.7: transferring the Small optimizations, then batching

The first two rows apply the existing Small optimization package to **Large
1.7**, with monolithic semantic compilation. The remaining rows use the
additional bounded layerwise compiler. All rows use CFG 2 and the same short
fixture; all are medians of three warm runs. Here first PCM is a **local Python
callback**, not HTTP. These are a separate benchmark from the server table.

| Large 1.7 execution path | Candidates | First PCM | Complete batch | Aggregate device RTF | Audio seconds / compute second |
| --- | ---: | ---: | ---: | ---: | ---: |
| Small optimization package, streaming | 1 | 0.403 s | 1.410 s | 0.2747 | 3.64 |
| Small optimization package, offline | 1 | — | 1.782 s | 0.3497 | 2.86 |
| Layerwise compiler, streaming | 1 | 0.433 s | 1.491 s | 0.2897 | 3.45 |
| Layerwise compiler, offline | 1 | — | 1.862 s | 0.3651 | 2.74 |
| Layerwise compiler, streaming | 8 | 1.309 s | 5.624 s | 0.1365 | 7.33 |
| Layerwise compiler, offline | 8 | — | 3.679 s | 0.0886 | 11.28 |
| Layerwise compiler, streaming | 16 | 2.048 s | 7.483 s | 0.0896 | 11.16 |
| Layerwise compiler, offline | 16 | — | 5.475 s | 0.0647 | 15.45 |

For a batch, the denominator is the **sum of every candidate's generated audio
seconds**. This is not the latency of one listener, nor evidence that 16
independent conversations can meet real-time deadlines. The live Large cap is
**8** to retain headroom for longer prompts and colocated finishing models.
Batch 16 is an explicit short-fixture probe. A best-of-N winner additionally
requires candidate scoring after generation; scoring is excluded here.

The fastest tested Large configuration reduces guidance to CFG 1: batch 16
offline reaches **RTF 0.0429 (23.3 audio seconds/s)**; streaming reaches
**RTF 0.0709 (14.1 audio seconds/s)** with 1.453 s first PCM. CFG 1 changes the
conditioning and is not an equal-quality substitute. Robby keeps CFG 2.

Sources: [original-package measurements](https://github.com/LAION-AI/Humaneness-Voice/blob/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/playground/benchmarks/large17/benchmark.json)
and [layerwise/batch measurements](https://github.com/LAION-AI/Humaneness-Voice/blob/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/playground/benchmarks/large17/layerwise.json).
No matching Small batch sweep was made in this study; no Small batch figures
are inferred from the Large results.

## What was optimized

| Optimization | Small RL | Large 1.7 |
| --- | --- | --- |
| Resident BF16 policy | Merge the checkpoint's own fixed RL adapter once | Load the already merged root release once; do not attach its training adapter again |
| Reference reuse | Cache encoded reference codes by content | Same cache; crop waveform before encoding; 5.04–14.96 s supported reference |
| Semantic prediction | Static KV cache, batched CFG branches, CUDA graphs and Inductor compilation | Same mechanisms using the release's own model, configuration and processor |
| Acoustic token prediction | Fast top-k/top-p sampling and Triton local decoder at batch 1 | Same batch-1 acceleration; true batched original local decoder for larger batches |
| PCM codec | Causal codec CUDA graphs and fused kernels at batch 1 | Same codec; larger batches use the batched causal codec without batch-1 graph specialization |
| First audio | Growing 3 → 6 → 12 → 20-frame blocks | Same 240 → 480 → 960 → 1600 ms block ramp |
| Additional compiler option | Original monolithic path retained | `HVS_LARGE17_LAYERWISE=1`: one tensor-only block reused across 36 layers inside a complete continuation CUDA graph |

Layerwise compilation bounds compilation complexity; it did **not** improve the
measured single-stream warm speed over monolithic compilation. Changing batch
or length buckets can still require cold capture. On PyTorch 2.6, use
`HVS_LOCAL_COMPILE=0` and `HVS_LOCAL_TRITON=1`: the compiled local sampler's
scalar capture failed on this software stack.

A Large CUDA-event profile measured approximately 254 ms semantic prefill,
760 ms semantic continuation, 135 ms acoustic-token prediction and 69 ms PCM
decoding. The semantic backbone dominates this optimized short stream.
Instrumented spans explain costs; the headline results use uninstrumented runs.

## Earlier Small results on Jetson Thor: separate measurement conditions

The original optimization guide reports **NVIDIA Jetson Thor**, not RTX 3090
or T5000. It used Blackwell `sm_110`, 122 GB unified memory, CUDA 13.0, driver
580, PyTorch 2.14.1+cu130 and Transformers 5.15. Its prompts and serving tests
differ from the matched RTX 3090 study, so these rows are **not a GPU ranking**.

| Small RL metric on Jetson Thor | Reported result | Scope |
| --- | ---: | --- |
| TTS request to first audio | 0.16 s, previously 2.3–2.9 s | Warm live-server timing |
| Complete 4.4-second sentence | 1.5 s | Full live request; approximately 2.93 audio seconds/s, derived from the rounded reported times |
| GPU time per 80-ms frame | Approximately 21–28 ms | Frame-level timing, not a full-utterance benchmark |
| Typed Robby request to first sound | Median 0.66 s | Six turns, includes LLM and playback preparation |
| End of speech to first browser sound | Median 1.46–1.58 s | Twelve runs, includes ASR, LLM, TTS and browser buffering |

The derived 2.93 audio seconds/s includes request overhead; it is **not** the
device throughput column in the RTX 3090 table. No Large 1.7 measurement on Thor
is available in this study. Earlier Small RTX 3090 short-prompt onset results
(about 0.08 s) likewise use a different fixture and must not be substituted
into the matched Small/Large table.
See the [original Thor guide](https://github.com/LAION-AI/Humaneness-Voice/blob/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/INFERENCE_OPTIMIZATIONS.md)
and [earlier Small RTX 3090 study](https://github.com/christophschuhmann/cartoon-face-generator/blob/80d00b6be7d8ea2cc592c3c5f6f18ca7920274e7/robby/voice-assistant/RTX3090_OPTIMIZATIONS.md).

## Reproduce and interpret the evidence

The matched RTX 3090 host used driver 550.54.15, Python 3.12, PyTorch
2.6.0+cu124, Triton 3.2.0 and Transformers 5.15.0. Pinned model revisions:

- [Small RL](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1/tree/182bc700cac29b2d69295a33398b13e1ca7fba69): `182bc700cac29b2d69295a33398b13e1ca7fba69`.
- [Large 1.7](https://huggingface.co/laion/Humaneness-Voice-Large-1.7/tree/1c708b7ea106e097c44c48a58a194991c6749b4a): `1c708b7ea106e097c44c48a58a194991c6749b4a`.

Later documentation commits do not change which weights were benchmarked.
Large has 4.13 billion parameters; its distribution shards occupy about
16.52 GB, while the BF16 policy alone occupies about 8.26 GB. Codec weights,
graphs, KV caches and working buffers require additional memory.

Use the [TTS runtime setup](https://github.com/LAION-AI/Humaneness-Voice/blob/main/playground/README.md),
[tested environment flags](https://github.com/christophschuhmann/cartoon-face-generator/blob/80d00b6be7d8ea2cc592c3c5f6f18ca7920274e7/robby/voice-assistant/optimizations/rtx3090-speed.env)
and [Large benchmark instructions](https://github.com/LAION-AI/Humaneness-Voice/blob/main/docs/LARGE17_RTX3090.md#reproduce-the-measurements).
Set `HVS_LARGE17_LAYERWISE=1` for the deployed Large compiler option. These
environment flags apply to the shared serving runtime, **not automatically to
the Hugging Face release's standalone `infer.py`**.

The pinned [benchmark directory](https://github.com/LAION-AI/Humaneness-Voice/tree/478e74220a6b76fd7a7baa6443ed6bd44cc5559d/playground/benchmarks/large17)
contains scripts, cold/warm rows, numerical checks and ASR receipts. To recover
the matched Small/Large table without loading models, run this standard-library
Python snippet from the Humaneness-Voice repository root:

```python
import json
from statistics import median

with open("playground/benchmarks/large17/api-check.json") as f:
    measurements = json.load(f)
for model in ("small_rl", "large17"):
    rows = [r for r in measurements if r.get("kind") == "generation"
            and r["model"] == model and not r["warmup"] and not r["align"]]
    assert len(rows) == 3
    rtf = median(r["result"]["performance"]["aggregate_rtf"] for r in rows)
    print(model, "first PCM:", round(median(r["chunks"][0]["at"] for r in rows), 3),
          "RTF:", round(rtf, 4), "audio s/s:", round(1 / rtf, 2),
          "request s:", round(median(r["wall_seconds"] for r in rows), 3))
```

The Large study retained 32 complete conditions and 402 measured candidates.
Teacher-forced semantic state cosine similarity remained above 0.9986, with
relative RMSE up to about 4.5%; BF16 outputs are not bit-identical. Nine optimized
short/long clips had zero normalized Parakeet WER. These are correctness and
intelligibility canaries, **not** blinded naturalness, emotion or speaker-cloning
quality measurements. Synthetic listening clips and browser evidence are in
the [Robby report](https://github.com/christophschuhmann/cartoon-face-generator/blob/main/robby/voice-assistant/static/large17-speed.html).

Cold load/capture is excluded from warm figures. The first Large selection took
132 s; a later switch using retained CPU weights took 16 s. Returning to Small
took 60–81 s while rebuilding its graphs. A new shape or concurrent lab workload
can change latency substantially. The portable CLI and legacy serving eager
loop are different paths; no optimized-versus-native-CLI speedup is claimed.

Next experiments should target batched codec graphs, semantic prefill and safe
prefix reuse, followed by long-utterance and sustained-load tests. Weight-only
quantization requires separate quality and actual-kernel timing checks. RTX
3090 has no native FP8 tensor-core path; no FP8, FP4, H100 or RTX 5090 speedup
is claimed by these BF16 measurements.
