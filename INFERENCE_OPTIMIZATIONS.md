# Inference optimizations: fast streaming speech on small GPUs

This guide explains how we made the Humaneness Voice **Small RL** text-to-speech model
(and the Robby voice assistant built on it) start speaking much sooner. The numbers
in this guide come from an **NVIDIA Jetson Thor**, but most of the ideas apply to any
GPU where the model is small and the batch size is 1.

It is written for people who are new to ML inference. Every step says what the
problem was, how we found it, what we changed, how to switch it on, and what it costs.

The same guide exists in both repositories:
[LAION-AI/Humaneness-Voice](https://github.com/LAION-AI/Humaneness-Voice/blob/main/INFERENCE_OPTIMIZATIONS.md) (TTS server)
and [christophschuhmann/cartoon-face-generator](https://github.com/christophschuhmann/cartoon-face-generator/blob/main/robby/voice-assistant/INFERENCE_OPTIMIZATIONS.md) (Robby).

## Results

| Measurement | Start | After all steps |
|---|---|---|
| TTS: request → first audio (live server) | 2.3–2.9 s | **0.18 s** |
| TTS: GPU time per 80 ms audio frame | ~63 ms (slower than real time) | **27.6 ms** (2.9× faster than real time) |
| TTS: reference voice preparation per sentence | 646 ms | **0.9 ms** |
| Robby: user request → first sound (text test, median of 6 turns) | 4.2 s | **0.66 s** |
| Gaps during playback (simulated listener) | – | 0.0 s |

Test system: Jetson Thor (Blackwell GPU sm_110, 122 GB unified memory, CUDA 13.0,
driver 580), Python 3.12, PyTorch 2.14.1+cu130, Transformers 5.15. Model:
[`laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1`](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1)
at revision `182bc700`, codec [`OpenMOSS-Team/MOSS-Audio-Tokenizer-v2`](https://huggingface.co/OpenMOSS-Team/MOSS-Audio-Tokenizer-v2) at `f6e20e5`.

## Contents

1. [Words you need](#1-words-you-need)
2. [The most important lesson: find out what limits you](#2-the-most-important-lesson-find-out-what-limits-you)
3. [Quick start: switch everything on](#3-quick-start-switch-everything-on)
4. [The TTS optimizations, one by one](#4-the-tts-optimizations-one-by-one)
5. [Things that did not help](#5-things-that-did-not-help)
6. [The language model side (Robby)](#6-the-language-model-side-robby)
7. [How to measure](#7-how-to-measure)
8. [Environment variable reference](#8-environment-variable-reference)
9. [Ideas not tried yet](#9-ideas-not-tried-yet)
10. [Links](#10-links)

## 1. Words you need

| Word | Meaning |
|---|---|
| **Frame** | The TTS makes audio in small steps. One frame = 80 ms of sound (12.5 frames per second). |
| **Codebook / codes** | Each frame is described by 12 numbers, picked one after another, from rough to fine detail. |
| **Backbone** | The main transformer of the TTS (a Qwen3-0.6B-sized model). Runs once per frame. |
| **Local transformer** | A small helper network that picks the 12 codes of a frame one by one. |
| **Codec** | A separate network (MOSS Audio Tokenizer v2, 2.1 B parameters) that turns codes back into a 48 kHz waveform. |
| **Chunk** | A group of frames that is decoded and sent to the listener together. |
| **GPU kernel** | One small job the GPU runs, such as one matrix multiplication or one addition. A model step uses hundreds of them. |
| **Kernel overhead** | The fixed cost of starting a kernel (a few microseconds), however small the job is. |
| **CUDA graph** | A recording of a sequence of kernels that the GPU can replay in one go. It removes the CPU's launch cost. See the [PyTorch docs](https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs). |
| **Kernel fusion** | Merging several small kernels into one. `torch.compile` (Inductor) does this automatically. |
| **RTF** | Real-time factor = compute time ÷ audio length. Below 1.0 the model is faster than real time. |
| **CFG** | Classifier-free guidance: every step is computed twice (normal and neutral) for more expressive speech. |
| **Time to first audio (TTFA)** | How long the listener waits before hearing anything. This is what feels "fast". |

## 2. The most important lesson: find out what limits you

A GPU can be slow for two very different reasons:

* **Memory-bound:** the GPU spends its time *reading weights* from memory. Fix: use
  fewer bytes per weight (BF16, FP8, INT4 quantization).
* **Latency-bound (launch-bound):** the GPU spends its time *starting* hundreds of tiny
  jobs, each of which finishes almost instantly. Fix: fewer, bigger jobs (CUDA graphs,
  kernel fusion). Quantization does **not** help here.

Large models with batch size 1 are usually memory-bound. Small models like this TTS are
usually latency-bound, especially on embedded GPUs with fewer streaming multiprocessors
and a slow CPU.

**How we found out.** We timed one backbone step (13 ms, even inside a CUDA graph) and
compared it with a micro-benchmark that only runs the same matrix multiplications
(`F.linear` on random weights of the same shapes, batch 2):

```python
# 28 layers of q/k/v/o + MLP projections, hidden 1024, intermediate 3072, BF16, batch 2
eager: 4.82 ms/step, 0.88 GB weights, 183 GB/s
graph: 4.48 ms/step, 197 GB/s
```

Reading all weights takes only ~4.5 ms. The other ~8.5 ms were the many small
operations between the matrix multiplications (RMSNorm, rotary embeddings, residual
additions, attention with masks), each paying a few microseconds of fixed GPU cost.
The same was true for the 12 codebook steps and for the codec.

We also profiled where the time per frame went (synchronizing the GPU around each part):

| Part | Before | After |
|---|---|---|
| Backbone step | 15.1 ms | 10.1 ms |
| 12 codebook steps (local transformer + sampling) | 17.9 ms | 15.7 ms |
| Codec decode (per frame, spread over chunks) | ~9 ms | ~2.4 ms |
| Python / sync overhead between frames | ≈ 0 | ≈ 0 |

**Measure first.** Our first guess was "~10 ms Python overhead per frame". A run with the
codec stubbed out showed 25.2 ms per frame, which is exactly backbone + codebooks. So
there was no Python overhead to remove; the missing time was the codec. That changed our plan.

## 3. Quick start: switch everything on

All TTS optimizations are opt-in environment variables, except the reference-cache bug
fix (step 4.6), which is always on. Add this to the environment of `small_app.py`
(for example at the end of `playground/runtime/env.example` or your own env file):

```bash
# precision (step 4.1)
export HVS_RL_PRECISION=bf16_merged
# early audio release from live alignment (step 4.2)
export SMALL_LIVE_FAST_START=1
export SMALL_LIVE_FAST_TAIL_S=0.5
export SMALL_LIVE_FAST_TAIL_WORDS=3
export SMALL_LIVE_SCORE_FRACTION=0.5
# backbone CUDA graph + kernel fusion (step 4.3)
export HVS_SEMANTIC_CUDA_GRAPH=1
export HVS_SEMANTIC_COMPILE=1
export HVS_SEMANTIC_BUCKET=512
# faster sampling and compiled codebook step (steps 4.4, 4.5)
export HVS_FAST_SAMPLE=1
export HVS_LOCAL_COMPILE=1
# codec CUDA graphs (step 4.7)
export HVS_CODEC_GRAPH=1
# growing chunks 3 -> 6 -> 12 -> 20 frames (step 4.8)
export HVS_CHUNK_RAMP=1
export HVS_RAMP_START=3
# optional, unified-memory hosts: don't prefetch the other five checkpoints into RAM
export HUMANENESS_PRELOAD_EXPERIMENTAL=0
```

Then start the server as usual:

```bash
cd playground/runtime
python -m uvicorn small_app:app --host 127.0.0.1 --port 8792
```

**Warm up after every start.** `torch.compile` compiles one version per prompt-length
bucket, and the codec graphs are recorded on first use (~6 s). If a real user triggers
this, their first sentence is slow. Send a few requests with short, medium and long
sentences, twice, before you call the server ready. The `live_latency_benchmark.py`
request body (see section 7) is a good template.

Expected startup: about 75–140 s on the Thor, plus about 30–60 s of warm-up requests.

## 4. The TTS optimizations, one by one

Files are in `playground/runtime/`. "Gain" is what we measured on the Thor.

### 4.1 BF16 merged weights instead of strict FP32

* **Problem.** The released RL checkpoint runs, by default, in `fp32_strict`: FP32 base
  weights plus a separate FP32 adapter (two extra matrix multiplications per projection),
  with TF32 disabled. On a GPU with modest memory bandwidth that is slow; the TTS was
  slower than real time (RTF ~1.0).
* **Change.** `HVS_RL_PRECISION=bf16_merged` (already implemented in `rl_precision.py`)
  adds the adapter into the base weights once at load time, in FP32, then casts to BF16.
* **Gain.** First audio 2.28 s → 1.50 s; RTF ~1.0 → 0.69.
* **Cost.** This is no longer the official reference arithmetic. Sampled tokens can
  differ. We did not hear a difference, and ASR transcripts stayed word-accurate, but we
  did not run a systematic quality study.

### 4.2 Release live audio early (alignment hold-back)

* **Problem.** For live streaming, `alignment.LiveCutoff` held back 1.3 s of finished
  audio at the start of each sentence (to detect and fade a lead-in breath) and 1.0 s
  afterwards (so it can cut babble after the last word). It also ran Parakeet ASR over
  the whole sentence so far after every chunk. Result: a 1.6 s chunk was computed, but
  only 0.6 s of it was released.
* **Change.** `SMALL_LIVE_FAST_START=1`:
  * skips the lead-in scan and applies a 15 ms fade-in instead (no click);
  * holds back audio only when Parakeet has heard all but the last
    `SMALL_LIVE_FAST_TAIL_WORDS` (3) target words, and then only
    `SMALL_LIVE_FAST_TAIL_S` (0.5 s);
  * skips the Parakeet check while less than `SMALL_LIVE_SCORE_FRACTION` (50 %) of the
    expected sentence length has been generated (`small_app.py`).

  The end-of-sentence cut-off (the important part) still works.
* **Gain.** Removes up to 1.3 s from the first audio; together with 4.1: 2.28 s → ~1.3 s.
* **Cost.** Lead-in breaths are no longer faded.

```python
# alignment.py, LiveCutoff.feed (simplified)
keep_back = 0 if self.done else int(48000 * ((FAST_TAIL_S if self.fast else 1.) if self.lead_done else 1.3))
if self.fast and not self.done and len(self.words) - len(recognized) > FAST_TAIL_WORDS:
    keep_back = 0   # far from the end: release everything immediately
```

### 4.3 Backbone: CUDA graph plus kernel fusion

* **Problem.** The backbone step ran through Hugging Face code with `torch.compile(dynamic=True)`
  and no CUDA graph: 15 ms per step, of which only ~4.5 ms is weight reading.
* **Change.** Two existing flags plus one new one:
  * `HVS_SEMANTIC_CUDA_GRAPH=1` uses `semantic_graph.py`: a static KV cache and one CUDA
    graph per decode step. Alone this gave only ~4 % (13 ms), which showed us that the
    GPU-side cost of many tiny kernels is the problem, not the CPU launch cost.
  * `HVS_SEMANTIC_COMPILE=1` (new) compiles the decode function with
    `torch.compile(mode='max-autotune-no-cudagraphs')` **before** the graph is recorded.
    Inductor fuses norms, rotary embeddings and residual additions into fewer kernels.
  * `HVS_SEMANTIC_BUCKET=512` (new) rounds the KV-cache capacity up to multiples of 512
    instead of 128. Each capacity is a different shape and needs its own compilation, so
    coarser buckets mean fewer compilations during live use.
* **Gain.** Backbone 15.1 → 10.1 ms per step.
* **Cost.** The first request of each new bucket compiles for up to ~45 s (we saw 50 s
  for a long sentence on a cold server). Warm up after start (section 3).

```python
# semantic_graph.py, SemanticBank.__init__
if os.environ.get('HVS_SEMANTIC_COMPILE','0')=='1':
    self._decode=torch.compile(self._decode,dynamic=False,mode='max-autotune-no-cudagraphs')
```

### 4.4 Faster sampling (same probabilities)

* **Problem.** For each of the 12 codes per frame (×2 for CFG), the released sampler
  masks everything except the top 50 logits, then **sorts all 1,024 logits** for top-p,
  and scatters the result back. Sort and scatter are expensive small kernels.
* **Change.** `HVS_FAST_SAMPLE=1`: `torch.topk(50)` already returns the 50 best values
  sorted, so top-p can be applied to those 50 directly, and we sample among them.
* **Is it the same?** Yes, in distribution: tokens outside the top-k have probability 0
  in both versions, and top-p over the sorted top-k is exactly what the original computes.
  Only the random number stream differs, so individual samples change, as with a different seed.
  (In case of exact ties at the 50th value, the original may keep a few more tokens.)

```python
def fast_sample(mdl, logits, temperature, top_k, top_p):
    values, indices = torch.topk(logits.float() / float(temperature), int(top_k), dim=-1)
    if top_p is not None and 0.0 < float(top_p) < 1.0:
        drop = torch.softmax(values, dim=-1).cumsum(dim=-1) > float(top_p)
        drop = torch.cat((torch.zeros_like(drop[..., :1]), drop[..., :-1]), dim=-1)
        values = values.masked_fill(drop, -torch.inf)
    return indices.gather(-1, torch.multinomial(torch.softmax(values, dim=-1), 1)).squeeze(-1)
```

* **Gain.** Small alone; together with 4.3 and 4.5: RTF 0.51 → 0.44.

### 4.5 Compile the codebook step

* **Change.** `HVS_LOCAL_COMPILE=1` wraps `FastGenerator.local_frame` in
  `torch.compile(dynamic=False, fullgraph=True)`. The existing per-shape CUDA graph then
  records the fused kernels.
* **Gain.** 17.9 → 15.7 ms per frame for the 12 codebook steps.
* **Note.** This is now the largest remaining block. A hand-written Triton kernel for
  all 12 steps would be the next big win (section 9).

### 4.6 Cache the encoded reference voice by content (bug fix, always on)

* **Problem.** Larger than we expected. For every sentence, `small_app.py` copies the
  fixed reference clip to a **new file** in its audio store. `Engine.reference()` cached
  encoded voices **by path**, so it never got a cache hit. The 10-second clip went through
  the codec again for every sentence (646 ms on the Thor), and every copy stayed in the
  cache (a slow memory leak). On top of that, an MP3 for the Playground's player was made
  with an ffmpeg subprocess before generation could start.
* **Change.**
  * `engine.py`: the cache key is the SHA-256 of the file's content; at most 16 voices are
    kept (least recently used is dropped); `forget_reference(path)` only removes the
    per-request alias.
  * `small_app.py`: with `SMALL_LIVE_FAST_START=1` the MP3 is made in the background.
    The Playground's play button for that reference can be briefly unavailable.
* **Gain.** Reference preparation 646 ms → 0.9 ms per sentence.

```python
# engine.py, Engine.reference (simplified)
key = hashlib.sha256(Path(path).read_bytes()).hexdigest()
if key not in self.refs:
    ...encode once...
self.refs[key] = self.refs.pop(key)          # most recently used last
while len(self.refs) > 16:
    old = next(iter(self.refs)); self.refs.pop(old); self.reference_info.pop(old, None)
```

### 4.7 Codec decoding as CUDA graphs (idea from fast-moss-tokenizer)

* **Problem.** Decoding one chunk with the codec took **90–125 ms, almost independent of
  the chunk size** (5 frames cost about as much as 20). The codec was about a third of all
  GPU time. Same disease: hundreds of small kernels.
* **Idea.** [kadirnar/fast-moss-tokenizer](https://github.com/kadirnar/fast-moss-tokenizer)
  speeds up the same codec (same revision `f6e20e5`) with CUDA graphs and fused kernels.
  Its code only runs on an RTX 5070 Ti with PyTorch 2.8 / CUDA 12.8 (it checks this and
  refuses otherwise), so we could not install it. Its central trick is portable, though:
  the codec's streaming attention normally **replaces** its KV-cache tensors after every
  chunk, and a CUDA graph can't follow moving memory. The fix: after the original update,
  copy the new values back into the **old** tensors, so the addresses never change.

```python
def _stable_cache_update(module, state, keys, values, positions, *current):
    type(module)._update_streaming_cache(module, state, keys, values, positions, *current)
    keys.copy_(state.cached_keys)
    values.copy_(state.cached_values)
    positions.copy_(state.cached_positions)
    state.cached_keys, state.cached_values, state.cached_positions = keys, values, positions
```

* **What we built** (`performance_core.CodecGraphs`, `HVS_CODEC_GRAPH=1`):
  1. Every request used to open a new streaming session, with new memory and therefore
     invalid graphs. Instead, one **persistent** batch-1 streaming state is created for the
     **decoder modules only**. The MOSS codec allocates its bounded KV cache at full context
     length from the start, so all shapes are fixed.
  2. On first use, one CUDA graph is recorded for **every chunk size from 1 to 20 frames**
     (5.6 s once, shared memory pool).
  3. Each request attaches the persistent state, **resets** it (positions −1, keys zero,
     which is exactly a fresh state), replays graphs, clones the outputs, and detaches.
  4. The encoder never enters streaming mode, so encoding a new reference voice still
     works normally. Batch sizes other than 1 use the original code path.
* **Gain.** Codec 92 → 35 ms for a 5-frame chunk; 609 → 368 ms for a whole sentence;
  GPU time per frame 34.3 → 27.6 ms (−20 %); first audio in the engine 283 → 202 ms.
* **Correctness.** Waveforms were **bit-identical** with and without graphs in every test
  (`inference_speed_check.py`).
* **Not ported yet.** fast-moss's fused Triton/CUDA kernels (quantizer, rotary,
  residual) and cached weight casts. They would make each graph faster still, but need
  recompiling for CUDA 13 / sm_110 and proof of identical output.

### 4.8 Growing chunks (small first chunk without gaps)

* **Problem.** A small first chunk makes the first sound earlier, but the next chunk must
  arrive before the small one has finished playing. With 20-frame chunks after a 5-frame
  first chunk, we measured 0.3–0.8 s of silence.
* **Change.** `HVS_CHUNK_RAMP=1`: chunk sizes double, starting at `HVS_RAMP_START`
  frames, up to `chunk_frames`: **3 → 6 → 12 → 20 → 20 …** (a request's own
  `first_chunk_frames` overrides the start). Because generation is now ~2.9× faster than
  real time, each chunk is ready before the previous one has played.
* **Why this only works now.** With the codec at ~100 ms per chunk, many small chunks
  cost too much. After step 4.7 the codec costs ~35 ms per chunk, and the ramp became cheap.
* **Gain (live server).** First audio: start 2 → 0.16 s, start 3 → 0.18 s, start 5 → 0.23 s,
  all with 0.0 s of gaps on an otherwise idle GPU. We chose 3 for a safety margin when the
  GPU is shared.

```python
# performance_core.FastGenerator.stream_frames
block = p.get('first_chunk_frames', chunk_frames) if last == 0 else chunk_frames
if self.chunk_ramp:
    block = min(chunk_frames, max(1, p.get('first_chunk_frames') or RAMP_START) * 2 ** len(yield_sizes))
```

## 5. Things that did not help

| Idea | Result | Why |
|---|---|---|
| FP8 / INT8 / INT4 weight quantization of the TTS | not pursued | Latency-bound, not memory-bound: weights take ~4.5 ms of a 10 ms step. |
| One CUDA graph for the whole frame (backbone + codebooks) | ≈ 0 expected | Measured: no Python/sync overhead between frames (25.2 ms = 10 + 15.7). |
| CUDA graph for the backbone without kernel fusion | −4 % only | GPU-side cost per tiny kernel remains; fusion was needed (4.3). |
| Small first chunk with 20-frame follow-ups (before 4.7/4.8) | 0.3–0.8 s gaps | Next chunk too slow; solved by growing chunks. |
| Smaller chunks everywhere (10 frames), before 4.7 | slower overall | Codec cost per chunk was ~100 ms regardless of size. |
| `first_chunk_frames` of 3 or 8 via the API | rejected | The API only accepts 2, 5, 10, 20; the ramp start is an env var instead. |
| Checking power mode | already optimal | `nvpmodel -q` showed MAXN, GPU clock at maximum. |

## 6. The language model side (Robby)

Robby (`robby/voice-assistant` in cartoon-face-generator) streams sentences from a
language model to the TTS. Its first-sentence latency matters as much as the TTS.

### 6.1 Local Gemma 4 E4B in NVFP4 with llama.cpp

* Model: [`WhiskyAKM/Gemma-4-E4B-it-qat-NVFP4-GGUF`](https://huggingface.co/WhiskyAKM/Gemma-4-E4B-it-qat-NVFP4-GGUF)
  (`gemma-4-E4B-it-qat-nvfp4.gguf`, revision `db5e898`), built from Google's QAT checkpoint.
  llama.cpp runs NVFP4 natively on Blackwell; `llama-quantize` cannot create NVFP4, so a
  ready-made GGUF was used. About 41 tokens/s decode on the Thor.
* Build llama.cpp with `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=110` (Thor).
  `cmake` can come from `pip install cmake ninja` if you don't have root.
* **Use `--swa-full`.** Gemma uses sliding-window attention. Without the full SWA cache,
  llama.cpp can only reuse the KV cache at checkpoints, and every turn recomputed most of
  the prompt.

```bash
llama-server -m gemma-4-E4B-it-qat-nvfp4.gguf --alias gemma-4-e4b-nvfp4 \
  --host 127.0.0.1 --port 8799 -ngl 999 -c 16384 -np 2 -fa on \
  -b 2048 -ub 512 -t 8 --swa-full --reasoning off --jinja --no-webui
# Robby: ROBO_BOLT_BASE=http://127.0.0.1:8799 ROBO_BOLT_MODEL=gemma-4-e4b-nvfp4
#        ROBO_BOLT_LABEL='Gemma 4 E4B NVFP4 · local GPU'
```

### 6.2 Keep the prompt prefix identical across turns

* **Problem.** `bolt_director.py` appended per-turn data (speaker metadata, environment,
  memory, reply language) to the **system message, before the conversation history**. The
  prompt changed near its start every turn, so the server's prefix cache was useless:
  up to 1,900 prompt tokens recomputed per turn.
* **Change** (local backends only; `ROBO_LOCAL_STATIC_PREFIX=1`, the default):
  the system message stays byte-identical. Lines of the per-turn context that were
  identical to the previous turn stay in the system message ("sticky"); only the changing
  tail moves into the newest user message, clearly marked as context.
* **Gain.** From turn 3 on, about 200 new prompt tokens per turn instead of 500–1,900;
  Gemma's first sentence in ~0.40–0.48 s.

### 6.3 Remote model

With an OpenRouter key installed (`python credential_store.py`), Robby defaults to
Qwen 3.8 27B on Cerebras; first sentence typically ~0.4 s, sometimes up to ~3 s
because of the network and provider. The local Gemma stays selectable.
`ROBO_HIDE_GEMMA12=1` hides the Gemma 12B entry on hosts that don't serve port 18812.

### 6.4 Shared GPU

On a single-GPU device the language model, ASR and TTS compete. In the live pipeline the
TTS needed ~1.8 s for its first chunk while local Gemma was still generating the rest of
the reply, versus 1.3–1.5 s alone (before steps 4.3–4.8). Using a remote language model,
or keeping the local one idle while the TTS works, avoids that.

## 7. How to measure

Always run audio benchmarks **one at a time**: a second process on the same GPU
(another benchmark, a browser session with live ASR) makes every number worse and
creates gaps that don't exist otherwise.

* `playground/runtime/live_latency_benchmark.py` sends Robby's exact streaming request to
  the running server and reports time to first PCM, total time, audio length and **gaps**
  (silence a listener would hear if playback started at the first chunk):

  ```bash
  python playground/runtime/live_latency_benchmark.py --first-chunk 0 2 5
  ```

* `playground/runtime/inference_speed_check.py` loads its own engine copy and reports
  GPU ms per frame and engine-side first audio with and without codec graphs. It checks
  that waveforms are bit-identical and that fresh copies of the reference clip hit the cache.

* To split time between backbone, codebook steps and codec, wrap
  `engine.model.transformer.forward`, `engine.fast.frame` and
  `performance_core.StreamingCodec.decode` with a function that calls
  `torch.cuda.synchronize()` before and after and records `time.perf_counter()`. The
  synchronization slows things down a little, so use it only for the breakdown.

* To check for Python overhead, stub out the codec
  (`StreamingCodec.decode = lambda self, codes: (zeros, lengths)`) and compare ms per frame
  with the sum of the parts.

## 8. Environment variable reference

| Variable | Default | Effect | Section |
|---|---|---|---|
| `HVS_RL_PRECISION` | `fp32_strict` | `bf16_merged` for speed | 4.1 |
| `SMALL_LIVE_FAST_START` | `0` | early live audio release, background MP3 | 4.2, 4.6 |
| `SMALL_LIVE_FAST_TAIL_S` | `1.0` | hold-back near the sentence end | 4.2 |
| `SMALL_LIVE_FAST_TAIL_WORDS` | `5` | how many final words trigger the hold-back | 4.2 |
| `SMALL_LIVE_SCORE_FRACTION` | `0.5` | skip Parakeet checks before this share of the sentence | 4.2 |
| `HVS_SEMANTIC_CUDA_GRAPH` | `0` | backbone static KV cache + CUDA graph | 4.3 |
| `HVS_SEMANTIC_COMPILE` | `0` | Inductor fusion before graph capture | 4.3 |
| `HVS_SEMANTIC_BUCKET` | `128` | KV capacity rounding (fewer compilations) | 4.3 |
| `HVS_FAST_SAMPLE` | `0` | top-k-first sampling | 4.4 |
| `HVS_LOCAL_COMPILE` | `0` | compiled codebook step | 4.5 |
| `HVS_CODEC_GRAPH` | `0` | codec CUDA graphs, persistent decoder state | 4.7 |
| `HVS_CHUNK_RAMP` | `0` | growing chunks | 4.8 |
| `HVS_RAMP_START` | `5` | first chunk size of the ramp | 4.8 |
| `HUMANENESS_PRELOAD_EXPERIMENTAL` | `1` | `0` skips RAM prefetch of other checkpoints | 3 |
| `ROBO_LOCAL_STATIC_PREFIX` (Robby) | `1` | stable prompt prefix for local LLMs | 6.2 |
| `ROBO_BOLT_MODEL`, `ROBO_BOLT_LABEL` (Robby) | Nemotron | local model name and UI label | 6.1 |
| `ROBO_HIDE_GEMMA12` (Robby) | `0` | hide the Gemma 12B menu entry | 6.3 |

## 9. Ideas not tried yet

| Idea | Expected effect |
|---|---|
| Synthesize the next sentence in parallel (batch 2 costs almost the same as 1 when latency-bound) | shorter pauses between sentences |
| One Triton kernel for all 12 codebook steps | 15.7 → ~4 ms per frame |
| Port fast-moss's fused codec kernels | codec another −30 to −60 % |
| Triton kernels for the backbone (as in [MOSS-TTS-optimized](https://github.com/kadirnar/MOSS-TTS-optimized)) | backbone 10 → ~5 ms |
| Start TTS on half sentences (at a comma) | −0.2 to −0.5 s end to end; intonation risk |
| Pre-synthesized openers ("Hallo!", "Gerne!") | ~0 s first sound when used; may sound canned |

## 10. Links

* Humaneness Voice: <https://github.com/LAION-AI/Humaneness-Voice>
* Robby voice assistant: <https://github.com/christophschuhmann/cartoon-face-generator/tree/main/robby/voice-assistant>
* fast-moss-tokenizer (codec CUDA graph idea): <https://github.com/kadirnar/fast-moss-tokenizer>
* MOSS-TTS-optimized (Triton kernels for MOSS-TTS v1.5 8B): <https://github.com/kadirnar/MOSS-TTS-optimized>
* PyTorch CUDA graphs: <https://pytorch.org/docs/stable/notes/cuda.html#cuda-graphs>
* `torch.compile`: <https://pytorch.org/docs/stable/generated/torch.compile.html>
* llama.cpp: <https://github.com/ggml-org/llama.cpp>
* Gemma 4 E4B NVFP4 GGUF: <https://huggingface.co/WhiskyAKM/Gemma-4-E4B-it-qat-NVFP4-GGUF>
* PyTorch wheels for CUDA 13 (aarch64 works on Thor): <https://download.pytorch.org/whl/cu130>
