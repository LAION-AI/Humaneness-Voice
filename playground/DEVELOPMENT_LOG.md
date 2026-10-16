# Humaneness Voice Playground

## Current default: Small 1.1 Alpha RL r128 v1 (2026-10-15)

The top model selector now includes [Small RL r128 v1](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha-RL-r128-v1)
as the startup/API/UI default and [Large RL r128 v1](https://huggingface.co/laion/Humaneness-Voice-Large-1.1-Alpha-RL-r128-v1)
as an additional option. All four previous checkpoints remain selectable;
the dated sections below document their earlier defaults, not the current one.

These releases are **exact final SFT5 bases plus fixed custom RL adapters**,
not merged checkpoints, PEFT adapters, or the old optional DPO adapters.
Small uses SFT5 step 6,109 and 2,434 RL updates; Large uses step 3,055 and
2,370 RL updates. `acting_models.py` pins immutable bundle revisions and checks
SHA-256 for both files. `rl_adapter.py` requires complete semantic/local
attention coverage, exact keys/shapes/counts, rank/alpha 128 and residual scale 1.
No adapters are silently merged, substituted, or offered as legacy checkboxes.

The published arithmetic contract is preserved: round the verified base to
BF16, upcast to **FP32**, use separate FP32 residual adapters, disable policy
autocast and TF32. The audio codec remains BF16 under its own context. The
bundled helpers/processor/decoder source files match the previous Alpha runtime
byte-for-byte. Temperature defaults to the RL release recommendation **0.70**;
audio top-p 0.95/top-k 50 stay unchanged. Demo CFG 2.0, GENERAL/SCRIPT prompting,
streaming, ranking and fades are existing adjustable demo extensions, **not an
exact reproduction of the release's baseline generation protocol**. Existing
earlier speaker/temperature studies do not establish an RL quality optimum.
Mira, a fixed <=15-second reference, one streaming take, 800-ms codec chunks,
voice conversion off, and experimental steering off remain the defaults.

Only one TTS model is on GPU 1. Small RL permits parallel microbatches up to 8;
Large RL uses a conservative cap of 2 because FP32 weights occupy much more
VRAM. Best-of-N automatically uses that cap, even if the single-take field is 1.
For Large RL, Parakeet ASR, Ears and final forced alignment move to CPU; they
return to their configured GPUs on switching back. This preserves the policy
precision and avoids using the other GPU occupied by an unrelated image
service. CPU scoring/alignment can add considerable end-to-end latency.

Both disks on this host are nearly full. New bundles use the existing tmpfs
at `/dev/shm/humaneness-rl/hub` (override `HUMANENESS_RL_CACHE` for persistent
storage). It needs approximately 20 GB; **a host reboot loses this download
cache**, so startup re-downloads pinned files. Normal app restarts reuse it.
The independent CPU model cache retains BF16-rounded bases; adapters are loaded
separately as FP32. No old weights, study artifacts or user data were deleted.

`rl_live_check.py` exercises both bundles over the existing HTTPS tunnel,
single streaming, parallel Best-of-N, MP3 delivery and Small RL restoration.
Its records are saved to `test-results/rl-live.json` when run.

Verification on this host: **359 CPU tests passed**, two Node-version checks
skipped. Chrome verified six options, the Small RL default, temperature 0.70,
desktop/mobile layout without overflow and no JavaScript errors
(`test-results/rl-ui.json`). Actual HTTPS/GPU canaries completed for both
models with MP3 delivery, nonempty ASR, available Ears predictions and applied
forced-alignment fades. Small Best-of-3 executed `[3]`; Large Best-of-3 executed
`[2, 1]`, without OOM retries. The four short smoke requests recorded aggregate
on-device RTF 0.593 / 0.199 (Small streaming / Best-of-3), and 1.188 / 0.577
(Large streaming / Best-of-3), **including first-use graph capture**. These are
smoke timings, not a quality study or steady-state benchmark. Large's first
request took 176 s including model switch and cold compilation; its following
Best-of-3 took 19.7 s end-to-end with CPU auxiliaries. Buffered streaming also
pays CPU ASR latency; it is not a real-time playback guarantee. The server was
returned to ready/idle Small RL, and the existing HTTPS tunnel was preserved.

## Optional 1.1 Alpha snapshots and parallel Best-of-N (2026-10-13)

Two additional, explicitly **experimental** options are available. The stable
**Small · 12 Oct 2026** remains the startup default, and **Large Acting step 432**
remains selectable. The Alpha releases are unfinished full-parameter training
snapshots, not external LoRAs or replacements for those established models:

| Option | Frozen published snapshot | SHA-256 |
|---|---|---|
| [Small 1.1 Alpha](https://huggingface.co/laion/Humaneness-Voice-Small-1.1-Alpha) | `intermediate-checkpoints/step-00003775/model.safetensors`, revision `aed9e955dae4591922b159561bd995fd9b288ece`, step 3,775 / 6,109 | `ab7229198123135e0aa5964f155794fa6b6ef9bdfadd98c4356705f35b05b5a5` |
| [Large 1.1 Alpha](https://huggingface.co/laion/Humaneness-Voice-Large-1.1-Alpha) | `intermediate-checkpoints/step-00002377/model.safetensors`, revision `ababe387f10f663057dad9ddb4652422200acab3`, step 2,377 / 3,055 | `d63bfeb2951e05cd3a82fb1ddf80ff2634fdc7519ac1520374e7bdfec17c2cb5` |

Selection follows the published `intermediate-checkpoints/latest.json` at these
immutable commits, backed by the complete LFS file/hash and READY step proof.
The live demo never hot-updates weights during a request or silently promotes an
Alpha to default. Future training uploads need a deliberate pinned-registry
update. Quality can improve or regress; completed generation is not evidence of
general speaker/emotion gains. The current Playground still exposes DE/EN; the
release's extra-language/alias paths are not added or claimed validated here.

`checkpoint_ram.py` prefetches verified Alpha tensor dictionaries into private
**CPU RAM** after default startup. Floating weights are materialized as BF16,
as in the existing GPU runtime; integer buffers retain their dtype. These CPU
copies stay resident when a fresh inference module is moved to VRAM. Only one
TTS model resides on GPU 1 at once, with its own KV caches/graphs. Prefetch uses
no CUDA model or sampling RNG. The model selector shows preload status and
temporarily disables a still-loading Alpha option. A failed CPU download cannot
evict the currently working GPU model. `/api/state` and `/api/models/catalog`
report actual RAM bytes, revisions, hashes and preparation status.

The 16.5-GB FP32 Large Alpha file did not fit on the nearly full NVMe, so its
download cache uses the separate system filesystem at
`~/.cache/humaneness-alpha/hub` (override `HUMANENESS_EXPERIMENTAL_LARGE_CACHE`).
No existing models, study artifacts or user data are removed. CPU BF16 copies
need approximately 1.43 GB for Small Alpha and 8.26 GB for Large Alpha, plus
CPU copies of any established models subsequently selected.

**Best-of-N and All N now generate candidates in parallel GPU batches by
default, even when the single-take batch field is 1.** Effective candidate batch
is `min(N, model.max_batch_size)`: Small/stable or Alpha up to 8; Large/stable or
Alpha up to 4. N above that cap is split into parallel microbatches, not one
independent model call per take. CUDA OOM retries a smaller batch and is reported
in performance diagnostics. The single/search batch field is disabled for
Best-of-N; real requested/effective batch sizes are reported in the SSE model
event, and actual executed batch sizes in `performance.batches`. At T=0 sampling
is deterministic, so multiple candidates are not guaranteed to differ.

Run `experimental_live_check.py` for the actual HTTPS load, single-stream and
parallel-N canary; records are in `test-results/experimental-live.json`. CPU
tests cover cache ownership/materialization, concurrent prefetch, failure
visibility, unchanged defaults, and N=3 becoming a real batch of 3 for all models.

Verification completed: **355 CPU tests passed**, two Node-version-dependent
checks skipped; actual Chrome/HTTPS verified all four selectable models, ready
RAM options, mobile/desktop layout, and an N=3 Best-of-N request through the UI.
Eight actual GPU cases passed: stable Small N=3; Small Alpha streaming, N=3 and
N=8; Large Alpha streaming and N=3; stable Large N=3; stable Small restored.
The actual executed batches were `[3]` and `[8]`, with no OOM retry, not serial
single-candidate calls. First Large Alpha activation took 64 s including cold
warm-up/compilation; its actual cached-weight model construction/VRAM transfer
logged 0.7 s. This distinction is shown in UI guidance; RAM prefetch does not
promise zero compilation latency. Both Alpha CPU caches remain ready and the
loaded/default model at handoff is stable Small. The four CPU cache entries
after testing occupy 19.76 GB, experimental entries 9.69 GB. A final Hub recheck
confirmed no newer weight steps/hashes despite updated README commits.

## Current Small default: 12 October merged release (2026-10-13)

The default model is now **Small · 12 Oct 2026**, the full BF16 export from
[Humaneness Voice Small](https://huggingface.co/laion/Humaneness-Voice-Small/blob/main/checkpoints/HVS-Small-2026-10-12/model_bf16.pt),
pinned to revision `3365015e53329ae3ac3a4bb3d541ed760418797c`.
SHA-256: `45f1de4bcc62bef038afc812ac8681b3e6507d9d97e4c18d27518f5a5a346bd4`.
The previous Small Acting step-1,725 weights are no longer offered in this UI.
The stable API id `small_acting` now resolves to this explicit new release;
every model/turn event reports its actual checkpoint, revision and hash.
The original **Large Acting step 432** remains available via the model selector
at the top of the voice panel. Only one model is resident at a time. Reloading
the page reflects the currently loaded model; switching is serialized against
generation and releases old model/CUDA graphs while preserving the shared codec.

The new file is a complete ordinary PyTorch tensor state dictionary, loaded with
`weights_only=True`, memory mapping, and **strict=True** after hash and release
metadata verification. It is not an external adapter: the release's Rank-64
training delta is already merged into the full weights. No DPO, Ref15 or other
external LoRA is loaded. The old adapter launch buttons, presets and overlay are
removed from the UI; their implementation files/weights remain on disk for future
work. Both models reject adapter requests in the API, and UI requests always
send an empty adapter selection, even if an old browser preference is present.

Other defaults are unchanged: Mira, GPU batch 1, one take, streaming with 800-ms
chunks, temperature 1, CFG 2, fixed reference up to 15 seconds, gentle/natural/
spontaneous fallback delivery, word-boundary fades on, and optional Humaneness
Voice Converter 40k **off**. Small allows GPU microbatches/live-search width up to
8; Large up to 4. The existing assets/helpers and processor match the pinned
Large release byte-for-byte; actual Small/Large/Small swapping is tested separately.
Older throughput/quality reports below describe their original checkpoints, not
a quality evaluation of the 12 October release.

The service was down while the old quick tunnel was still running, causing 502
origin-connection failures. `acting_deploy.py --start` starts an absent Playground
and refuses an occupied port; `--pid PID` still safely restarts an identified
idle instance. `restart_tunnel.py --pid PID` replaces an identified quick tunnel,
waits for a valid HTTPS catalog response before retiring the old process, and
records the new URL in `logs/tunnel-url.txt`. Quick URLs change on restart; the
previous `switch-surprised-clock-sparc` address is no longer the live endpoint.
The latest live URL is **https://fuji-formal-while-plugin.trycloudflare.com/**.
No permanent uptime guarantee is implied by an account-less quick tunnel.

Validation: **348 CPU tests passed**, two Node-version-dependent legacy checks
skipped. Actual Chrome over the new HTTPS URL generated new-Small English and
German streams, switched to Large and generated a stream, then switched back to
Small for an excited performance. Every canary delivered MP3 chunks before the
final result and reported no external adapters/steering. Small hash/stage were
asserted from real generation events; final loaded/default model is Small.
The four short canaries had ASR WER 0 (not a general quality claim); Small
TTS/codec RTF was approximately 0.30–0.32, Large approximately 0.39. Warm swaps
took around 6–7 s to Small and 17 s to Large. A delayed status-poll regression
was explicitly injected during switching: it could not revert the pending
selection or enable generation. Results: `test-results/small-oct12-live.json`;
desktop/mobile screenshots: `test-results/small-oct12-1440.png` and
`test-results/small-oct12-390.png`.

## Large speed update and optional Humaneness Converter (2026-10-11)

The current English-language Playground defaults to **voice conversion off**.
Enabling it uses the full, merged **40,000-step**
[Humaneness Voice Converter](https://huggingface.co/laion/Humaneness-Voice-Converter),
revision `952ef56c945763be50432738e31a32c30b359619`, `model.safetensors`,
SHA-256 `45bbf058ea2a82f1840dc6161f01be929217d28843003f5354336c81b11ea3dd`.
This replaces the previous vanilla + parent + GROW24k adapter stack, without
loading pickle, optimizer state, a built-in target, or a silent vanilla fallback.
All full-model keys load strictly. The release's frozen Ears Medium conditioner
assets are byte-checked against the already loaded, pinned Ears predictor before
sharing it. Source delivery supplies dynamic axes; the selected reference supplies
14 static/timbre axes. Converted output is aligned, scored and delivered in
streaming and offline modes. Streaming is overlapping context-window conversion
with a 40-ms crossfade, **not native causal flow decoding**. TTS uses up to 15 s
of reference; the converter retains its own upstream 10-s acoustic-reference cap.

New default actor direction, when no other delivery is requested:
**"Gentle, natural and spontaneous delivery."** It goes in CAPTION and GENERAL,
after the reference-identity instruction when enabled, never into spoken words.
Explicit emotions, shouting, whispering and other acting directions take priority.
Exact transcripts and manual prompts remain unchanged. UI language does not
force the language of generated speech; German and English remain available.

Large's repeated semantic-frame decode is now dynamically compiled alongside
the existing local CUDA graphs; arbitrary prompt prefills stay eager to avoid
prompt-length compilation stalls. Attention-mask views use contiguous layouts.
Set `ACTING_COMPILE_GLOBAL=0` for the original eager semantic decoder.
Startup warms single-stream inference before declaring ready; new batch shapes
can still incur first-use compilation. The final Qwen aligner runs on GPU 1
instead of CPU (`SMALL_ALIGN_DEVICE=cpu` remains an override). GPU 0's research
worker remains untouched. No TTS weights, temperature, CFG, flow steps or frame
rate are altered for these optimizations. Compiled BF16 kernels are not promised
bit-identical to eager inference, so matching seeds can produce slightly
different sampled audio.

The first controlled RTX **3090 24-GB** A/B (three repeats, same 19-word English
prompt, Mira reference, seed 65431, T=1, CFG=2, batch 1, 800-ms streaming, VC off)
measured **4.700 → 3.010 GPU seconds**, **RTF 0.632 → 0.405**, about **1.56×**
TTS/codec throughput. Moving final alignment to GPU reduced its measured
**2.73 seconds to roughly 0.04–0.06 seconds**; API wall time fell from
**8.73 seconds to about 4.07 seconds**. These scopes are distinct: CUDA events
exclude Luna, reference encoding, ASR callbacks, alignment, scoring, VC, MP3 and
network; wall time includes the local API stream and evaluation. Same-case WER
was 0 before and after; Ears CE was 6.46/6.56 and PQ 8.01/8.08. These are a
one-prompt smoke check and model proxies, not proof of general quality gains.
Batch-4 canary completed without OOM, aggregate TTS/codec RTF about **0.113**.
The final decode-only implementation repeated the controlled single stream at
**2.917 GPU seconds, RTF 0.392, 4.16 seconds API wall time** (three repeats),
approximately **1.61×** baseline TTS/codec throughput. First MP3 arrived after
1.07 seconds on average; this includes buffered fades. CE/PQ were 6.54/8.15,
WER 0. Previously unseen short, long and German prompts completed without the
earlier full-prefill compilation stalls, RTF 0.381–0.411. A first-use final
batch-4 test measured 0.143 aggregate RTF, including new-shape graph work.
Raw records: `test-results/large-speed-before.json`,
`test-results/large-speed-compiled.json`, `test-results/acting-speed-final.json`,
`test-results/large-speed-decode-only.json`. An English summary is served at
`/acting-performance/` and linked from the UI's metrics section.
Run `acting_speed_check.py` against an idle Playground to reproduce the canaries;
new shapes include cold compilation and must not be confused with warm timings.

## Acting releases (2026-10-11): current live defaults

The model selector now offers the complete, hash-verified
[Large Acting](https://huggingface.co/laion/Humaneness-Voice-Large-Acting)
checkpoint (step **432**, revision `d64c8d833495ed4a18d5e7abbe468d9e2fc4fa4e`)
and [Small Acting](https://huggingface.co/laion/Humaneness-Voice-Small-Acting)
(step **1,725**, revision `95d0bff7da24729d5a7c9c9d88986d52ed8c939c`).
Current defaults: **Large Acting, Mira, one take, GPU batch 1, streaming,
800-ms codec chunks, GENERAL/SCRIPT, temperature 1.0, CFG 2.0**, and up to
15 seconds of reference audio. Humaneness Converter 40k is optional and off by
default. Fades/timestamp cutoff remain enabled; they buffer
audio, so 800-ms generation chunks do not promise 800-ms end-to-end latency.

The new Acting packing matches the public inference example: literal spoken
words alone in `Text`, performance directions in `Instruction`, real reference
codes in the reference channel. Accelerate's `base.` checkpoint prefix is
stripped; only the published omitted tied heads are allowed, and re-tied.
Old DPO/Ref15 adapters and old layer-forensics vectors are **incompatible and
disabled**, not silently merged. Optional adaptive/burst rewards remain off.
Prior Small-study recommendations are historical, not validation of Acting.

Only one TTS checkpoint is resident. `/api/models/select` swaps it while holding
the same busy guard as generation; every turn also supplies its intended model.
The shared codec is retained, old CUDA graphs/model memory released. Large has
a conservative maximum GPU microbatch / live-search width of **4**, Small **8**;
ordinary Best-of-N can produce up to 12 takes over several microbatches.
CUDA OOM during ordinary generation retries a smaller microbatch and reports it.
Model switching is unavailable during another generation. GPU 0's long-running
target study is untouched; TTS/scorers/VC and final Qwen alignment use GPU 1.

Reference selection defaults to a **fixed** profile clip. The new unchecked
VoiceCLAP checkbox selects a performance-matched clip from the **same** profile,
never another identity. Selection is reported with the actual playable clip;
unsupported direction language/missing clips fall back explicitly to the fixed
clip. Custom uploads are unaffected and continue to use exactly the uploaded
reference until removed. These are model-based reference matches, not verified
cloning/emotion guarantees. Prompting does not instruct a new age/gender/timbre.

Reproduction: `acting_models.py` prefetches both pinned files and verifies their
SHA-256; `acting_live_test.py` exercises actual HTTPS streaming, checkpoint swaps,
dynamic-reference selection and batches. These are smoke checks, not a quality
study. See `test-results/acting-live.json` for measured timings and any batch
fallbacks. Existing reports and the frozen old study/vector banks are preserved.

The initial, pre-optimization RTX **3090** HTTPS smoke test passed all seven cases: Large default
with GROW VC streamed 7 MP3 chunks before final result; Large offline microbatches
2 and 4, dynamic Mira retrieval, Small streaming, Small offline microbatch 8,
and switching back to Large. No batch OOM fallback occurred. One short prompt
gave Large single-stream TTS+codec RTF **0.67** (0.63 on the repeat), Large
offline batch-4 aggregate RTF **0.16**, and Small offline batch-8 aggregate RTF
**0.077**. These CUDA-event timings exclude VC/scoring/alignment/network and
are not an end-to-end latency or general throughput benchmark. Default VC-on
case wall time was 13.6 seconds including first VC use, evaluation and downloads
of the completed audio; no claim of real-time end-to-end playback is made.

## Historical Small-only implementation and studies

Separate demo based on the existing MOSS UI and ranking system. The original
server files are not modified. Model: LAION / Humaneness Voice Small,
**S3-Ref15-FullFT, step 26,774**, pinned release
`88faa078581276a16d69172b8a89fee509266b7b` (CC BY 4.0).
The study-backed starting defaults are **CAPTION / exact TRANSCRIPT**, audio
temperature **1.0** and CFG **2.0**, with one fixed reference per voice and
automatic duration. Temperature 0 remains available. The UI includes a
button restoring these settings; explicit structured/manual choices still work.
These are cautious pilot defaults, not a human-validated universal optimum.

The 2026-10-10 Small-only Playground default was **Best-of-N, N=8, without streaming**. Restoring the
recommended settings selects the same offline configuration; streaming and
live search remain manually selectable.

The Playground's experimental speaker difference vector is **off by default**
(including the restore-recommended-settings button). Its checkbox remains
available for manual opt-in; voice conversion keeps its independent default.

Historical HTTP-study reproduction explicitly sets `speaker_steering: false`
in `speaker_study.py` and `full_ft_speaker_study.py` and verifies it was not
applied. Omitting this field still requests the historical experimental HTTP default,
so old clients must explicitly opt out when reproducing an unsteered protocol.
No frozen study artifacts or past measurements were rewritten. Standalone
`emotion_study.py` creates its own model with dormant steering state and does
not inherit the HTTP request default or load catalog vectors.

## Historical voice conversion: GROW 24k Speaker-Mix (2026-10-10)

The Playground now defaults to the recommended **step 24,000 / `speaker_mix`**
adapter from [laion/chatterbox-s3gen-vc-grow-ce-speaker](https://huggingface.co/laion/chatterbox-s3gen-vc-grow-ce-speaker),
revision `ad1161233f38f95496008f014d960e6607e6aa0c`, checkpoint SHA-256
`974b45fd47a49df79ea545c50ddde3330c95657222c02d6cd339c962028f95af`.
This is not a standalone replacement for `s3gen.safetensors`: the pinned upstream
base is loaded, then the verified supervised 300-hour adapter from
`laion/chatterbox-s3gen-vc-grow@ede42a5aa9ded910457272fde36b11b8eec7f749`,
then the 24k continuation. `vc_grow.py` installs all 224 rank-128 attention
LoRAs and the 99-score projector with strict key/shape checks. Only the
hash-verified `.pt` uses restricted `torch.load(weights_only=True)`; no
optimizer restoration or downloaded Python execution occurs. Failure does
not silently fall back to Vanilla.

The checkbox is on for a selected profile or usable custom reference, and remains
user-disableable. Selecting no reference automatically turns it off. The HTTP
default is also on; `voice_mode=none` without an explicit VC flag turns it off,
while explicitly requesting VC without a reference is rejected. Historical
HTTP studies must explicitly send `voice_conversion:false`, as well as their
existing steering/ranking opt-outs, to reproduce their original protocol.

Live audio has no original corpus annotation, so the existing pinned **Humaneness
Ears Medium** predicts the same named 99 native targets (40 emotions, 57
VoiceNet dimensions, genuineness and burst blend). Denormalization uses the
release mean/std, then the preparation scales: first 98 axes /6, blend /10.
The 14 published static/timbre axes come from the target reference prefix;
other axes come from each actual source waveform/window. Target scores are
cached only for fixed references, alongside the existing reference cache.
This is a disclosed **distilled-score approximation** to the corpus-prepared
inference recipe, not fabricated zeros, acting-prompt scores or the exact
original teacher annotations. Audio source scoring adds inference work.

The existing same-length microbatching, per-candidate diffusion noise,
ten flow steps, BF16 flow / FP32 comparison, watermark, context streaming,
40-ms seam handling and **post-VC evaluation** remain. Each microbatch gets
its own source-style row and the reference CAMP embedding is expanded before
the style projector; conditioning is cleared after every inference call.
References remain up to 10 s for VC and up to 15 s for TTS. Short streaming
windows are outside the release's 3–8 s training-source range, and the
windowed wrapper is not native causal decoding. The old throughput tables
below concern **Vanilla**, not this adapter and its score prediction stage.
The release itself reports mixed quality/identity trade-offs; this change
does not establish improvement on every voice or emotion.

Functional GPU canary: `python grow_vc_gpu_test.py`; artifacts are
`test-results/grow24k-gpu.json`. This checks real batch-2, window-streaming
and FP32 conversion, not perceptual quality. The simultaneous 500-profile
layer-target collection now runs on **one research GPU (cuda:0)**; GPU 1
remains available for the live scorer/VC pipeline. Its immutable no-VC study
design, samples and targets do not inherit this UI default.

## Historical Chatterbox VC batching and fast-flow update (2026-10-09)

At this historical measurement, VC was opt-in and used Vanilla. The UI offers fast BF16 flow (selected
by default) and an explicit FP32 comparison. Only the flow estimator is BF16;
tokenizer, conditioning encoder and vocoder stay FP32. Original FP32 weights
are retained intact; ten original flow steps and watermark remain. Up to eight
equal-length windows share each GPU microbatch, with individual noise seeds;
unlike lengths are grouped separately. Fixed-reference conditioning caches hold
at most four entries; custom upload targets are not added to that shared cache.

Single-take and live-search conversion is chunked with 1.6 s context and 40 ms
held crossfade. Every candidate is converted BEFORE ranking; only committed
converted PCM is delivered. Best-of-N still cannot deliver a chosen winner until
all takes have been scored. Top-2 beams may wait for identical converted prefixes.

[English VC benchmark and listenable numerical comparisons](https://switch-surprised-clock-sparc.trycloudflare.com/vc-performance/).
On RTX 3090, eight 1.6 s windows: legacy serial 2.714 s, FP32 microbatch8 2.004 s,
BF16 flow microbatch8 0.702 s. With 1.6 s context + 1.6 s new audio: legacy 3.622 s,
FP32 batch8 2.425 s, BF16 flow 0.943 s. Sixteen such streams need 1.886 s in BF16,
so VC alone does NOT meet a 1.6 s interval there. Eight short-clip proxy checks:
ECAPA means serial 0.5741 / FP32 0.5745 / BF16 0.5715; CE 5.435 / 5.437 / 5.437,
PQ 7.094 / 7.095 / 7.091. Small pilot, not broad identity/emotion preservation
proof or human MOS; batch waveforms are not bit-exact. Aggregate processed-audio
RTF is not winner-stream RTF, and excludes TTS/scoring/MP3/network. Two replicas
had only a small FP32 gain; production uses one model with batched flow. TF32
was tested but is not selected; dynamic compilation failed and is not deployed.
Real HTTPS checks passed for vector ON/OFF and Mira/Nora/Selma switching, VC
single-live BF16/FP32, converted Best-of8, N4 winner-stream and N4 top-2 beam.
Live paths delivered playable mono 96 kbit/s MP3 chunks before the final result;
converted rewards saw converted PCM. All five VC cases had WER 0 in this one
script/seed smoke, but three non-VC takes omitted half the transcript (WER .50).
This is integration evidence, not universal quality or a broad reliability study.
The first new TTS shape still incurred ~39 s cold compilation; first VC use loads
its weights. Stable API/UI checks: 307 passed, 2 skipped; real browser checks
passed all ten voices, OFF/ON, precision selector, five generation modes and
320/390 px mobile widths, no JavaScript errors.

Reproduce: `python vc_performance_study.py`, `python vc_hardware_study.py`,
`python vc_bf16_study.py`, `python vc_quality_check.py`,
`python vc_performance_report.py`, `python catalogue_vc_live_test.py`.

## 500-profile five-layer training-target collection (2026-10-09)

`catalogue_target_study.py` builds candidate high-minus-low directions for all
500 synthetic profiles in
[laion/moss-voice-profile-loras-500](https://huggingface.co/laion/moss-voice-profile-loras-500),
pinned to `ec77b7b476bd31ceb9c0c0581c9aef680d62d6f5`. It downloads reference
WAVs and names only; none of those old-model LoRA weights is used. The ten demo
profiles are processed first with their existing fixed reference clips.

The prospectively fixed primary recipe is **100 seeds for each of ten cells**
(five emotions × English/German), hence 1,000 generated takes per voice and
500,000 total. Each cell keeps exactly its top **two** ECAPA cosines and its
bottom **25**. Average actual-frame predictor activations per take, average the
two high and 25 low take vectors, subtract, then equally average the ten cell
differences. Both pools use the same reference. This is **not** a
reference-minus-unconditional contrast. Five zero-based Qwen block outputs,
**14, 15, 16, 17, 18**, are captured in the same generation pass. The same
selected take IDs are used for every layer. A nested 50-seed Top2% target
(one high minus 13 low per cell) is retained as a sensitivity comparison.

Use the verified Full-FT base only, CFG 2.0, temperature 1.0 and up to 15 seconds
of reference. No DPO, pre-existing steering, voice conversion, alignment/fade,
best-of-N output selection or quality retries. The global transformer remains
eager for scoped capture hooks; the local acoustic decoder uses CUDA graphs.
A GPU preflight compares seeded uninstrumented and five-layer-instrumented
runs and requires identical acoustic codes/audio and exact emitted-frame
capture counts before collection starts. Mixed 4/4/2 batches give each
identical cell 100 distinct actual batch seeds; batch RNG is shared by its
different-condition companions, not independently seeded per row.

Each completed voice exports `catalogue-target-study/targets/<id>/n100-top2.npz`:

- `speaker_embedding`: normalized 192D pinned ECAPA embedding of the reference
  prefix actually used; the input for a future speaker-to-direction predictor.
- `delta`: float32 `[5, 1024]`, plus `layer_14__contrast` through
  `layer_18__contrast`; these are training targets, **not optimized layer weights**.
- `high_mean`, `low_mean` and `cell_deltas[10, 5, 1024]`, with layer indices.
- `metadata.json`: reference/checkpoint/scorer hashes, selected take IDs,
  per-cell scores, vector norms and dataset split. `n50-top2.npz` is secondary.

`training-index.json` lists completed input/target pairs. An 80/10/10
deterministic hash split groups exact duplicate reference content. This is
not proof that distinct references cannot represent the same underlying voice;
speaker aliases must be audited before training. One reference per voice is
also insufficient to establish cross-reference generalization. No MLP is
trained by this collection.

All takes retain mono 96-kbps MP3, ECAPA scores, normalized output embeddings,
frame counts and the five layer means. Scoring uses uncompressed PCM; large
generated WAV archives are avoided. Atomic batch-directory commits make the
job resumable without dropping or replacing seeds. A failure is recorded and
stops the owned research workers; resume uses the same declared seeds, not
additional attempts selected for success. A 20-GiB free-disk floor prevents
filling the volume. Research processes never stop/reconfigure the live TTS
server or replace its deployed steering bank.

Run from this directory using the existing TTS Python environment:

```bash
python catalogue_target_study.py prepare
python catalogue_target_study.py pipeline
python catalogue_target_study.py report --publish
```

Keep the long-running pipeline in a supervisor or detached process. Its own
`processes.json`, `progress.json`, per-voice `state.json` and worker logs provide
progress; it owns only its research workers (one GPU by default as of October 10;
`--devices cuda:0`). The existing HTTPS catalogue
report includes a `#training-targets` progress section without restarting the
live server. The reported totals include completed collection takes only,
not the separate GPU preflight generations.

**Important:** these are ECAPA-selected candidate training targets, not
validated improvements for 500 voices. Content Enjoyment, MOS, WER and emotion
are not scored/gated in this collection. Top speaker similarity is not top
quality; fresh held-out quality/listening checks are required before replacing
any live voice vector. The successful Mira result does not establish that this
recipe is optimal for every profile.

## Mira activation-steering exploration (2026-10-09)

Update: catalogue steering is now selectable for all ten existing Playground
profiles, default ON (experimental), layer 16 / raw alpha 0.75. The original
Mira direction is preserved exactly. Custom uploads, no-reference requests and
active LoRAs bypass it; hashes bind every vector to its fixed reference and base
checkpoint. Turn it OFF for a direct comparison. Failed quality screens remain
selectable with warnings, not advertised as improvements.

[Catalogue study: 1,510 takes and 410 embedded evaluation MP3s](https://switch-surprised-clock-sparc.trycloudflare.com/catalogue-steering-study/).
Collection uses 100 takes per profile plus 100 common no-reference controls;
each identical prompt has ten distinct actual seeds, with no speaker/timbre
description. Quality contrasts and unfiltered reference-effect contrasts are
separate estimators. Each new candidate has ten fresh evaluation pairs. Mira,
Nora and Osku clear the exploratory quality/speaker safeguards; other profiles
have uncertain or negative effects. No comparison multiplicity correction or
human listening panel; no MLP or new three-layer coefficient grid was run.
Rebuild with `python catalogue_steering_report.py`.

[English report with every embedded MP3](https://switch-surprised-clock-sparc.trycloudflare.com/mira-activation-study/)
and local artifacts in `mira-activation-study/`: **932 retained generations**, zero
technical generation failures, Full FT step 26,774 only, fixed 10.26-second Mira
reference, CFG 2 / T=1, no DPO, conversion, trimming, best-of selection or retries.
Three research agents independently handled block steering, semantic steering and
study design/audit. One research TTS copy per GPU shared a local-only scorer.

The best **tested and held-out** candidate is **global block 16 (zero-based),
high-minus-low contrast, raw alpha 0.75**, added to the final prediction position
in both CFG branches. On 20 fresh paired cases: ECAPA **0.376 → 0.451**, Content
Enjoyment **6.097 → 6.252**, WER **6.75% → 2.00%**, independent-file anchor cosine
**0.155 → 0.204**. Speaker gain is positive in English and German; the pooled
paired and whole-batch-bootstrap intervals stay above zero. Automated quality
safeguards pass. Emotion/aesthetics/genuineness means do not collapse, but their
wide intervals do not establish an improvement. Live steering was disabled for
that original study; the later catalogue deployment above supersedes this.
This is one voice/reference, not an optimal universal setting or human validation.

Collection: 100 bilingual, five-emotion outputs. States are averaged only for
positions whose prediction emits an acoustic frame, including the final prefill
predictor if it emits frame zero, excluding all other prompt/reference positions,
EOS and finished tails. Saved points: 28 global blocks (1024D), normalized global
semantic (1024D) and its acoustic-seed projection (2560D); this is not a search of
every internal block of the local acoustic-token decoder. Primary vectors average
samples equally within emotion then balance the five emotions. Strict ECAPA >0.50
selects 23 samples, unfortunately **22 German / one English**. Raw means mix generic
model activity, phonetics, language and speaker; they are not isolated identity
vectors. Semantic global/local means are almost linearly equivalent. Secondary
>0.25 language×emotion-balanced high-minus-all vectors remain separately labelled.

Search: 290 all-block coarse takes; 90 raw-semantic takes; 192 raw-block dose and
two/three-block refinement takes; 40 primary semantic-contrast takes; 56 balanced
diagnostics; 64 block-contrast takes; 100 held-out takes. A prospectively frozen
validation amendment used four candidates plus baseline rather than three, to
test distinct mechanisms and the original raw-mean proposal. Raw block 17 and
both semantic candidates failed to establish a held-out gain. Combination alpha
is a **total** dose divided across sites, not that dose at every site. Refinement
used eight blocks and omitted fear; final validation includes all five emotions.
Paired confidence intervals are exploratory, not multiple-testing corrected;
the secondary anchor is a different same-corpus-label file, not verified ground truth.

`vectors/selected.npz` retains all primary means/contrasts, and individual prediction
means/counts are under `activations/`. Every generation has PCM, mono 96-kbps MP3,
prompt, actual batch seed/companions, scores, checkpoint hash and intervention.
Collection's original unmasked-batch style scores are preserved; a logged correction
encoded each waveform separately without altering ECAPA, vector selection or audio.
Rows share batch RNG; supplementary bootstrap resamples whole recorded batches.

Reproduction entry points: `mira_activation_study.py design`, `collection`, `vectors`,
then `run --plan mira-activation-study/<frozen-plan>.json --device cuda:0
--score-device cuda:1 --score-url http://127.0.0.1:8794 --batch-size 4`.
The shared scorer is `python -m uvicorn mira_activation_scorer:app --host 127.0.0.1
--port 8794`. Plans and existing complete batches resume without quality resampling;
partial batches stop for explicit recovery. Alternate banks are ROOT-relative via
`--vectors vector-diagnostics/balanced025-selected.npz` or per-condition `vector_file`.
`python mira_activation_report.py` rebuilds HTML and public JSON. Research hooks are
in `activation_steering.py` / `semantic_steering.py`; production generation is untouched.

## Current playground defaults (Full-FT update, 2026-10-09)

**Best-of-N with three complete takes**, incremental codec streaming with
**1600-ms PCM chunks**, **no DPO or Ref15 LoRA**, CFG **2.0** and
temperature **1.0**. Selma remains the default voice; reference context remains
up to **15 seconds**, with fixed same-speaker clips. Ref15 stays off.
Single-take live streaming, optional DPOs, Ref15, combinations and both chunk-search modes
remain explicitly selectable. Search alternatives retain N12 / 30 seconds.
The recommendation button restores the sampling, prompt-format, generation and
adapter defaults and automatic duration without removing the user's reference
upload or changing their selected voice.
The current ranking default is now **50% normalized VoiceCLAP prompt similarity
+ 50% normalized Humaneness Ears Medium Content Enjoyment**, followed by the
existing bounded WER gate. The old ECAPA 40/60 mix is available under Previous
ranking, but is not secretly included in the new quality mixture.
This request changes prompting/sampling/reference defaults, not ranking weights:
the existing WER gate is still active in mixed ranking, not in Ears-only sorting.
Adapter preferences use a new `small.adapters.v4` key; old default selections
are not migrated. Subsequent deliberate base/adapter selections persist.

The released full-fine-tuned BF16 checkpoint is
`checkpoints/S3-Ref15-FullFT/model_bf16.pt`, SHA-256
`e1b2c4613545d1db21c6f60a5ab2c27b5320aa05c177717ae268b1ff870d406e`.
Startup verifies this hash, release metadata and strict architecture loading.
It is a full base checkpoint, not the Ref15 rank-128 adapter. The old S3-trained
DPOs and Ref15 adapter remain explicit opt-ins on this same architecture, but
their addition to Full FT is **experimental**, not a validated training match.
Switching adapters off exactly restores the loaded Full-FT base, never old S3.
`SMALL_STAGE=S3` retains the original S3 checkpoint for deliberate comparisons;
the other explicit S1–S10 stages remain available. Full FT was trained with
reference audio up to 14.96 seconds; the demo keeps its 15-second reference cap.
Without a reference, this new model's behavior is not validated by the release.
See the [release documentation](https://huggingface.co/laion/Humaneness-Voice-Small/blob/main/docs/S3_REF15_FULL_FT.md).
Existing frozen studies describe old S3, not this Full-FT checkpoint; the speaker
study runner requires an S3 server to avoid silently mixing model conditions.

Full-FT update verification: **189 unit/controller/in-memory tests pass**;
real Chromium layout/controls pass at 320–1440 px, including all 64 live-search
N/chunk combinations and the Best-of-3/no-adapter recommendation reset.
`full_ft_live_browser_test.py` passes four real HTTPS/GPU cases: default
Luna + Best-of-3 without adapters, optional rank-16 DPO, disabling it again,
and Best-of-3 with a private 15-second custom reference (187–188 codec frames).
Both Ears variants are ready; the checkpoint hash/step, active adapter reports,
cross-base warning, ranking winner, private mono 96-kbps MP3s and upload cleanup
are checked. The selected takes in these four smokes had WER 0, but this is
**not a quality study or evidence of general improvement over old S3**.
Artifacts: `test-results/full-ft-live.json`, `inprocess-tests.json` and
`ui-layout.json`; no frozen study results were relabeled as Full-FT results.
The optional single-take live mode also passes with Full FT and no DPO:
seven mono MP3 chunks, first playable audio before the final result, WER 0
in this smoke (`test-results/playground-single-live.json`).

Luna now prioritizes a concise Caption plus the exact Transcript: explicitly use
the same speaker as the reference, then describe emotion through performance,
pace, emphasis and pauses, without inventing demographic or habitual timbre hints.
Clear rules, context and examples are separated, following the
[official OpenAI prompting guidance](https://developers.openai.com/api/docs/guides/prompt-engineering);
the existing Luna provider/model and six-key JSON contract are unchanged.
The server preserves the exact same reference-identity instruction in the neutral
CFG branch while neutralizing the emotional delivery. This CFG change is an
**untested quality hypothesis**, not a result measured in the frozen speaker study.
That study explicitly sends `identity_cfg: false` to preserve its original CFG
pair on reruns; ordinary playground/API requests default to the new policy.
Manual conditional prompts stay literal; the identity instruction is kept in their
neutral branch only when the manual prompt explicitly contains that instruction.
Transcript-only keeps identical CFG branches. No-reference turns add no identity
instruction. Style retrieval and scoring exclude the identity boilerplate.

Single-take streaming now sends `audio_chunk` SSE events with private mono
96-kbps MP3 URLs before final alignment/reward scoring. It reuses the live search
buffer (one-second tail lookahead) and Parakeet word timestamps for fade/cutoff,
then final Qwen alignment for the saved take and held tail. ASR callback/CPU-copy
time is excluded from segmented CUDA-event TTS/codec timing; model wall time
still includes those waits. Streaming playback cannot retract already played
audio if final alignment differs. Best-of-N / All N still wait for all takes;
their streaming switch only controls the incremental codec path.
This is an integration change, not a new fidelity/emotion or throughput study.

## Full-FT identity / quality / reliability study (2026-10-09)

The separate [English listening report](https://switch-surprised-clock-sparc.trycloudflare.com/full-ft-speaker-study/)
contains **300 embedded mono 96-kbps MP3 takes**, summary tables, per-voice results,
paired bootstrap intervals, all prompts and ASR hypotheses. Only step 26,774 is
generated: **no DPO and no other LoRA**. The frozen old S3 study is untouched.
There are 120 pilot cases (six settings), 40 new-text/seed validation cases,
80 exploratory codec/prompt/duration checks and 60 further fresh-text/seed
automatic-duration cases. Each condition has ten mixed-task trials per speaker,
five emotions × two languages, not ten repeats per individual emotion.

Practical recommendation on these references: **T 1.0, automatic duration,
concise identity Caption, CFG 1.5**. In the final follow-up, CFG 1.5 has mean
WER 1.25%, 85% exact text, fixed-reference ECAPA cosine 0.219 and Ears CE 5.96.
CFG 1.0 raises cosine to 0.260 but has WER 5.30% and lower emotion intensity;
CFG 2.0 gives cosine 0.248 / WER 6.53% with no clear quality benefit.
The predeclared equal-importance harmonic utility selects CFG 1.0; the practical
reliability-first recommendation deliberately uses a different preference.
No live defaults were changed while conducting the study. The subsequent
user-requested emotion/consistency-first update below changes CFG to 2.0.

This does **not** establish reliable high-fidelity cloning. On the two fixed
clips, CFG 1.5 scores 0.383 for Idris and 0.055 for Selma. Corpus reference-to-anchor
controls are weak, and human listening was not scored. The pilot and its first
validation used a fixed 6.5s hint; they must not be confused with the actual
automatic-duration default. Turning off codec streaming barely changes identity.
The tested verbose parenthesized SCRIPT cues are often spoken aloud, making
that recipe unsuitable. A protocol correction, saved before aggregate analysis,
keeps the long-reference identity target fixed in the 5s-reference comparison;
the actual conditioning-reference score is still separately recorded. Idris's
chosen source only supplies 4.7s, so that length comparison mainly tests Selma.

Instrumentation checks reproduce stored ECAPA scores to <3e-8. New-checkpoint
greedy 40-frame prefixes match between fast eager and CUDA graphs, but differ
from the separately evaluated legacy CFG path. Identity is weak in both prefix
paths; these checks do not establish full distributional decoder parity or
isolate the cause of low cloning. All quality/emotion/speaker scores are proxies;
no improvement over old S3, different voices, or live Best-of-3 is established.
Reproducible runner: `full_ft_speaker_study.py`; backend diagnostic:
`full_ft_backend_check.py`; HTTPS/filter/playback/privacy verification:
`full_ft_study_browser_test.py`. The study's source reference files are never public routes.

## Consistency / emotion / enjoyment defaults and six fixed voices (2026-10-09)

When WER is not the priority, our **tested starting compromise** is CFG **2.0**,
temperature **1.0**, **automatic duration**, the new Full-FT base **without DPO**,
and a concise English **CAPTION / exact TRANSCRIPT**. In the final automatic-duration
follow-up (20 mixed-task takes per CFG), output-to-output ECAPA cosine at CFG2
was 0.456 for Idris / 0.304 for Selma, versus 0.344 / 0.271 at CFG1.5.
Mean requested-emotion EIV+ intensity was 1.797 vs 1.841, and Ears CE 5.915 vs
5.965. CFG1 had the highest mean reference cosine (0.260 vs 0.248 at CFG2),
but weaker emotional intensity. Consistency across takes is not identical to
absolute reference cloning. These are small model-proxy comparisons, not a
universal optimum or a human assessment of strong cloning.

Keep **one clean fixed reference** across emotions and languages; do not retrieve
a different emotional clip for every request. The cap remains 15s, not a promise
that 15s beats a shorter clean sample. Avoid padding/looping. Give only performance
directions, not guessed age/gender/accent/resonance/timbre. For example:

```text
CAPTION: Use the same speaker as the reference audio. Preserve that speaker's voice identity. Intense anger, forceful emphasis, a brief hesitation, then a softer ending.
TRANSCRIPT: "I trusted you. Tell me what really happened."
```

The neutral CFG branch keeps the same identity instruction. Caption mode leaves
all bracket directions out of the spoken transcript; verbose parenthesized
SCRIPT directions can be spoken aloud. Luna's acting brief now explicitly keeps
the fixed reference identity while letting the requested performance change.

We shortlisted **ten existing clips**, one per profile, using existing naturalness,
recording quality, clarity, conversational style and low arousal/ranting/drama/
whispering proxy thresholds. No exhaustive audio-model scan or new TTS study.
Each shortlisted clip was normalized, capped and evaluated once with **Humaneness
Ears Medium** on the actual conditioning PCM. Keep Selma as the requested default,
then promote the five highest-CE other voices. All six clips are immutable
sample/hash-pinned in `fixed-references/selection.json`; runtime never re-ranks
them. The other four voices remain in a secondary menu and also use fixed clips.
Curated corpus MP3 previews are public; custom uploads and study reference files
remain private. The voice menu shows length/CE and lets users audition the clip.

| Fixed voice | Used reference seconds | Ears CE proxy / 10 |
| --- | ---: | ---: |
| Selma (default) | 10.92 | 5.840 |
| Idris | 8.18 | 6.642 |
| Anton | 10.03 | 6.444 |
| Robin | 10.86 | 6.324 |
| Mira | 10.26 | 6.253 |
| Osku | 11.90 | 6.205 |

Idris and Robin use their original profile anchors: no longer neutral candidate
passed the conservative filters. Idris is the deliberate shorter exception.
Clips and scores may reflect synthetic sources; neutrality, naturalness and
speaker labels are not independently human-validated. This reference shortlist
does not prove CFG2 improves cloning with these newly selected six clips.
`select_fixed_references.py` reproduces selection; **only a deliberate rerun**
may replace the manifest/audio. Existing private uploads are not touched.

Verification for this update: **201 unit/controller/in-memory tests pass**, two
optional GPU-only unit tests are skipped. Mocked real-Chromium controls/layout
pass at 320–1440px, including 64 search configurations. The live HTTPS browser
test passes **seven real GPU generations**: default Luna + Selma Best-of-3, an
angry single-stream Selma take with the identical reference hash, and one single
take for each of the other five promoted voices. All use Full FT without LoRA;
fixed references supply 102–148 codec frames rather than the old 37-frame limit.
The selected takes have WER 0 in these smokes, not proof of general reliability.
All six previews are mono 96-kbps MP3; generated output remains session-private.
No JavaScript errors or mobile overflow. Reproduce with
`fixed_reference_live_browser_test.py`; artifact:
`test-results/fixed-references-live.json`. The original HTTPS tunnel was retained.

## Humaneness Ears quality ranking (2026-10-08)

Both [Humaneness Ears](https://huggingface.co/laion/humaneness-ears-base-medium)
variants are loaded on the local scorer GPU before readiness: **Base** (Whisper
Base, 26M) and **Medium** (Whisper Small, 97M). `small` in the original artifacts
is a historical alias for Medium, not a separate third release. Medium is the
default. Immutable revision `818506970d93c809a3295a0a02c40dee8ff2bfc3` is used;
the inspected custom architecture and Safetensors are hash-verified. Only
inference assets are downloaded, never optimizer pickle checkpoints or teachers.
Existing Torch/Transformers are reused without an environment upgrade.

The new controls sit directly below Voice. Choose:

- **VoiceCLAP + Ears:** default Content Enjoyment (CE), 50/50, adjustable 0–100%
  Ears share; Production Quality (PQ), Content Usefulness (CU), Production
  Complexity (PC), DNSMOS P.808 MOS and DNSMOS Overall can also be selected.
- **Only Ears score:** rank exclusively by the selected raw regression value,
  without VoiceCLAP, speaker, burst or WER weights. Out-of-range raw predictions
  remain ordered rather than being tied by clipping. WER is still displayed.
- **Previous ranking:** restores the previous selectable ECAPA/VoiceNet speaker
  mix, automatic/manual VoiceNet/EIV+ dimensions and optional burst alignment.
  Incompatible API combinations are rejected rather than silently ignored.

Training mean/std from the release decode standardized network outputs into raw
units. Mixed quality uses `clip((raw−1)/9,0,1)` for AudioBox axes and
`clip((raw−1)/4,0,1)` for DNSMOS; VoiceCLAP retains `clip((cosine+1)/2,0,1)`.
The default reward is `(.5*CLAP_normalized + .5*CE_normalized)*max(0,1−WER)`;
search uses the existing prefix-aware WER at intermediate steps. There are no
hidden Genuineness/Blend/speaker weights. Pure-score mode is intentionally not
text-safe: pleasant incorrect speech can outrank accurate speech. Complexity is
not synonymous with naturalness, and maximizing any proxy can be gamed.

These are Humaneness Ears **predictions of teacher targets**, not invocations of
Meta's original AudioBox model, Microsoft's original DNSMOS, or human MOS.
Input is private in-memory 16-kHz PCM; audio is not sent to Hugging Face.
Clips longer than 30 seconds are fully covered by non-overlapping <=30s windows,
with duration-weighted raw-score averaging; a short last window is padded only
for inference, not given artificial weight. In search, every candidate's current
audio prefix is scored. With VC enabled, the input is always converted PCM.
All-N output is sorted by the selected Ears reward, while Previous ranking keeps
the original all-take presentation. Search RTF includes Ears scoring; TTS-only
device RTF continues to exclude reward models.

Verification of the preceding S3 integration: **187 unit/controller/in-memory tests passed**, plus real Chromium
layout/control tests at 320–1440 px and all 64 old search N/chunk combinations.
Native GPU predictions succeed for both Ears variants. The real HTTPS default
Luna/single-live case delivers six MP3 chunks before final result, WER 0, and a
TTS-only on-device RTF of 0.239 (not an Ears-inclusive throughput claim).
`ears_live_browser_test.py` passes six real GPU/HTTPS cases: mixed Medium CE,
pure Base PQ, pure Medium P.808 MOS, mixed Base CE after Chatterbox, mixed Medium
CE winner search and pure Medium CE top-2 search. Winner/rank, exact mixture
formula, pure raw-score equality, disabled speaker weighting, every search-step
Ears score, live MP3 events and sorted All-N presentation are checked. These are
integration smokes, **not a study proving improved quality**. Artifacts:
`test-results/ears-live.json`, `playground-default-live.json`,
`inprocess-tests.json` and `ui-layout.json`.
The frozen speaker-study runner explicitly requests `ranking.mode=legacy`.

## Optional Chatterbox voice-conversion stage

The checkbox under Voice enables **TTS → Chatterbox → word fades/alignment →
reward evaluation → MP3**. Historical GROW24k default was on; the current
Humaneness Converter 40k is optional and off by default. Conversion requires a selected catalog
voice or an owned custom upload. The fixed same-speaker reference is
also the explicit VC target; no preset/corpus voice silently replaces it.
The implementation follows
[LAION's Chatterbox VC library](https://github.com/LAION-AI/chatterbox-voice-conversion)
and the original demo's source-context/crossfade approach. Existing installed
dependencies are reused; no replacement of Torch/Transformers is needed.
The model is lazily loaded from pinned `ResembleAI/chatterbox` revision
`5bb1f6ee58e50c3b8d408bc82a6d3740c2db6e18`, only the SHA-256-verified
`s3gen.safetensors` artifact, with no built-in voice conditioning. The verified
SFT and 24k adapters are then installed as documented above.

Single live takes convert each incoming PCM window using 1.6 seconds of previous
source context, holding 40 ms for a same-time linear seam crossfade. This is a
windowed streaming wrapper, **not native causal S3Gen decoding**. Short initial
chunks are buffered to at least 800 ms; a short final source is silence-padded
for the model and cropped afterwards. Native token-rounding corrections are
reported and constrained. Unlike the old seam implementation, the crossfade
does not delete 40 ms at every join. The saved/graded take is assembled from the
same converted windows, never independently re-converted with new random noise.

Best-of-N and All N convert **every candidate before** alignment/ranking.
Chunk search also converts **every candidate before every search reward**, then
forks the VC context, held seam and converted history alongside the TTS caches.
Top-2 commits only an identical **converted PCM** prefix, because identical TTS
tokens alone cannot guarantee identical stochastic CFM output. This may delay
top-2 playback and significantly increases cost with N. VC RNG is isolated from
TTS RNG on both GPUs. Errors abort the turn rather than silently delivering or
grading raw TTS as if it were converted. Target conditioning is released after
the turn; no user reference or embedding is published/cached to disk.

Chatterbox's native output is **24 kHz**, resampled to 48 kHz only for this demo's
shared PCM/alignment/scoring/MP3 pipeline; this does not restore missing bandwidth.
The native VC reference limit is **10 seconds**. The separate TTS reference cap
remains **15 seconds** (optionally 20). No automatic SIDON enhancement is added.
Additional conversion latency is reported separately; TTS-only device timing
excludes VC callbacks, while search RTF includes per-candidate VC. Fidelity,
prosody/emotions and intelligibility can change; no general improvement is promised.

Verification on 2026-10-08: **172 unit/in-memory integration tests pass**,
including post-VC alignment/scoring in all five generation modes, lossless seam
timing, independent forked states and fail-closed conversion errors. Real
Chromium controller tests cover all 64 search-mode/N/chunk combinations,
reference-dependent checkbox gating, uploads/removal and 320–1440 px layouts.
`chatterbox_live_browser_test.py` additionally passes through real public HTTPS
and GPU inference: single live, single offline, Best-of-2 offline, winner-from-2
at 800 ms and top-2-from-4 at 1600 ms. Converted MP3 chunks decode/play before
the result, stay mono 96 kbps and are inaccessible from a foreign browser session.
The top-2 case only found a safe prefix at the end: **an option labelled streaming
does not guarantee early playback for top-2**. All candidates and search-step
rewards use converted PCM; synthetic transform tests independently enforce this.
`chatterbox_upload_browser_test.py` verifies a genuine long catalog-voice clip
uploaded into its own fresh session: **15 s / 187 TTS reference frames, 10 s VC
target**, with live converted playback and subsequent removal of only that test
upload. All six short smoke cases have ASR WER 0, which is not a quality study.
Separate native-GPU checks verify that VC leaves CPU/cuda:0/cuda:1 Torch RNG
states unchanged and preserves the source length over six streaming windows.
Artifacts: `test-results/chatterbox-live.json` and `chatterbox-custom-live.json`.

Startup warms the actual single-live codec context as well as search paths, but
**new prompt/reference shapes can still trigger expensive compilation**: the
first VC live case took 98.1 s total (1.12 s in VC itself); the subsequent single
offline case took 1.83 s total (0.38 s VC). A new 15-s upload case took 29.1 s.
The short winner/top-2 search smoke RTFs were 1.32 / 1.51 including conversion
and scoring, not real-time. These are integration observations, not a warmed
throughput benchmark or evidence of better speaker/emotion fidelity.

Earlier 2026-10-08 reference/adapter integration (historical defaults): 15-second reference context (optionally 20), original
base by default, independent DPO/Ref15 toggles and additive mixtures are deployed.
**136 local tests pass**, including real WAV/FFmpeg ingestion, CPU encoder spies,
published release-contract checks, synthetic fusion/restore tests, in-memory API
requests and the actual JavaScript controllers. Actual GPU generation with a
genuine long Selma source clip verifies 15 seconds at the codec input and
187 reference frames (14.96 seconds at 12.5 Hz), not the old 37-frame limit.
Base, previous DPO, Ref15 and both run successfully; restoring the base produces
the identical deterministic MP3. Real ECAPA winner/top-2 search and alignment
also pass. Chromium mock-API checks cover all 64 search combinations and desktop/
mobile layouts. See `test-results/ref15-status.json`, `ref15-live.json` and the
separate HTTPS browser test. These are integration checks, not a speaker-fidelity
or emotion-quality study. Optional 20-second references are outside Ref15's
training limit and have not been GPU-quality-validated.

The actual HTTPS Chromium test also passes four preset-click generations,
15-second uploads, separate strengths, persistence/reload, mobile layouts,
private-session access and mono 96-kbps MP3 delivery without JavaScript errors.
An additional real Luna + dynamic Selma run with the unchanged defaults passes:
base only, N12 winner search, 1600-ms chunks, 40/60 prompt/ECAPA reward and
alignment enabled. Its naturally short matching clip uses 6.30 seconds / 78 frames
rather than being artificially extended (`test-results/ref15-default-smoke.json`).

Permanent public reports: [Hugging Face study Space](https://huggingface.co/spaces/laion/humaneness-voice-small-studies),
[GPU throughput](https://laion-humaneness-voice-small-studies.static.hf.space/performance/index.html),
and [emotion / CFG / temperature](https://laion-humaneness-voice-small-studies.static.hf.space/study/index.html).
The Space contains the measured JSON and all 1,320 raw/processed synthetic MP3s
from 660 takes, but no source references, live user uploads or credentials.
Local report endpoints `/performance/` and `/study/` remain available.

Reward-guided chunk search is available as `search_winner` and `search_beam`:
Every batch size 1–16, 800/1600-ms chunks. When explicitly selecting live
winner search, its defaults are 12 candidates, 1600-ms chunks and a 30-second cap (375 frames).
The historical performance study used a 6.6-second cap, not these new defaults.
Winner mode commits chosen chunks immediately; top-2
splits N children between the two prefixes (an extra child to the better
prefix for odd N) and commits only the common acoustic prefix. BS 1 uses
one path in either mode. The isolated throughput study measured N=4/8/12/16,
not every intermediate batch size; newly used sizes may need first-use compilation.
Search SSE events carry private mono 96-kbps MP3 segment URLs; Web Audio
schedules them. Live playback uses buffered Parakeet fades; final forced
alignment also produces the saved take. Search RTF includes ASR, shared CLAP/heads, transfers and cache cloning,
unlike the old TTS-only device RTF. One-time query embedding, Luna, MP3,
network and final alignment/evaluation remain excluded. No universal quality
win or fixed top-2 first-audio latency is promised. Published
[search report and code](https://laion-humaneness-voice-small-studies.static.hf.space/search/index.html)
include actual one-GPU and two-GPU tests plus all failed output scores.
The model's custom Qwen3-0.6B + local Talker loading contract is followed;
it is not loaded through the old 4.55B AutoModel path.

## Adaptive VoiceNet / Empathic Insight Plus reward

### Single-take speaker-identity prompting study (2026-10-08)

`speaker_study.py` freezes a 240-take design before generation: Idris/Selma,
six prompt recipes, original S3 versus the previous rank-16 DPO, and ten matched
independent seeds per speaker/model/recipe. Five emotions × English/German
form the ten paired blocks; this is not ten replications of every emotion.
The fixed conditioning clips are separate from the held-out profile anchors.
Perceived speaker hints use the existing profile-trait predictions, not the
sometimes conflicting prose voice cards. Ref15 is deliberately off.

Every request uses one take, temperature 1.0 and configured CFG 1.5; no
Best-of-N, reward-guided search, streaming, alignment/fades or quality-filtered
retries. Transcript-only has no CFG contrast. The study keeps all failed-word
outputs, scores conventional Parakeet WER and raw-PCM conditioning-reference
ECAPA, then independently scores the actual MP3 transport with pinned ECAPA
against both conditioning audio and an unused same-speaker anchor. It also
measures all 45 pairwise output cosines within each ten-take cell.

The English self-contained report `/speaker-study/` (alias
`/speaker-study.html`) embeds all 240 generated mono-96k MP3s as Base64, with
an overview table first, matched-block bootstrap intervals, per-speaker rows,
reference controls, exact prompts and filters. Original source references,
private uploads and the embedding cache are never served. Public JSON is
limited to `design.json`, `summary.json`, `results.json` and
`reference-controls.json`. `speaker_study_browser_test.py` verifies actual
HTTPS audio playback, every Base64 payload, filters and mobile layout.

The pilot ranking is declared before scores: among emotion-capable recipes
with mean WER ≤10% and each speaker's mean WER ≤20%, maximize held-out-anchor
cosine. Transcript-only is a separate no-emotion control; if no recipe passes,
the report says so instead of claiming an optimum. ECAPA and Genuineness are
proxies, not human identity/MOS verdicts. One bilingual scene and two speakers
cannot establish a universal prompt rule. At collection time the study did not
change live defaults; the subsequent user-requested default change is described above.

Run against the existing local server with
`python speaker_study.py all`; interrupted runs resume from saved trial files.
Individual actions `design`, `generate`, `score`, `report` are also available.
The source audio paths are deployment-specific; rebuild a new frozen design
directory before substituting speakers or prompts.

Completed result: all **240** single takes and independent ECAPA evaluations
are present. The pooled pilot choice is **previous DPO + explicit same-speaker
Caption**, mean raw WER **6.36%**, conditioning-reference cosine **0.336**,
held-out-anchor cosine **0.185**, and mean across-take cosine **0.417**.
This average conceals substantial speaker dependence: Idris held-out cosine
is **0.311**, Selma only **0.059**. The reference/anchor control for Selma is
itself only **0.260**. No tested recipe establishes robust Selma cloning.
Gender-only/timbre hints have no decisive paired advantage over affect-only
Caption. The selected recipe versus affect-only DPO changes held-out cosine
by **+0.022** with exploratory 95% block CI **[-0.008, +0.058]**; do not call
the prompt universally better. The tested untimed GENERAL/SCRIPT recipe has
mean raw WER **97.3% base / 83.6% DPO**. Four empty-ASR takes lack a raw
speech-gated speaker score but remain in WER, offline embeddings and audio.
All scores and their coverage are exposed in the report. Real HTTPS Chromium
verifies all 240 Base64 payloads, embedded audio playback, filters, mobile
widths 1440/390/320, and rejection of private source-reference routes.

The UI's Reward selector offers the existing VoiceCLAP scorer (unchanged
default), automatic Luna dimension selection, and manual dimension editing.
This applies to single/Best-of-N/all-takes generation and both chunk-search
modes. Automatic selection sees audible acting direction only, not the literal
transcript. A separate preview button turns Luna's proposal into an editable
manual plan before any audio is generated.

The pinned catalog exposes 57 VoiceNet dimensions and 55 EIV+ experts:
40 emotions, 11 documented attributes and four pooled quality heads. Choose
1–4 unique dimensions across either or both families, positive weights, and
maximize/minimize or a native-scale target. Examples include `vn:AROU`,
`vn:S_WHIS`, `vn:S_RANT`, `vn:S_DRAM`, `vn:S_CASU`, `vn:S_FORM`, `eiv:Anger`
and `eiv:score_speech_quality`. Raw values, normalized utility, relative
weights, selected targets and the planner's explanation are visible in the UI.

Normalization is fixed, not re-fitted to the current candidates:

```text
x = clip((raw - native_min) / (native_max - native_min), 0, 1)
utility = x                         # maximize
        = 1 - x                     # minimize
        = 1 - abs(x - target_norm)  # target
style = sum(weight * utility) / sum(weight)
quality = style_share * style + (1-style_share) * (Genuine/6 + Blend/10)/2
reward = quality * max(0, 1-WER)     # prefix-WER during search; full WER at end
```

Genuine and Blend are clipped to 0–1 after normalization. The default
`style_share=0.5` preserves the original 50/25/25 balance; the UI can change it
up to a dimensions-only quality term at 1.0. Each native range comes from the
upstream taxonomy/card. EIV+ quality ranges are expected empirical ranges,
not hard physical bounds or calibrated human MOS. Four EIV+ heads without
verified native ranges (Authenticity, Background Noise, Recording Quality,
Vulnerable vs. Emotionally Detached) are deliberately not offered. Some
documented attribute endpoint directions are ambiguous; Luna is instructed to
prefer clearer VoiceNet rubrics for those concepts, and these axes are excluded
from its catalog and rejected by the automatic-plan validator.

`dimension_predictors.py` uses the official **GELU** VoiceNet regression
architecture and **ReLU** EIV+ architecture. It reuses the byte-validated
VoiceCLAP tower, loads only selected EIV+ heads (at most four resident), and
uses pinned `laion/BUD-E-Whisper` for EIV+ rather than the earlier independent
EmoWhisper evaluator. One Whisper encode serves every chosen EIV+ head;
quality experts use mean/min/max/std pooling, emotions the full flattened
1500×768 sequence. Candidates are processed in groups of at most four to
bound memory. Audio is padded to the trained 30-second window; longer clips
use the latest 30 seconds for EIV+ (the ASR gate still uses the full prefix).
These experts are voice-acting proxies, not reliable psychological assessments.

The extra encoder can materially change RTF. Published CLAP-only timings and
quality comparisons do **not** apply to this new mode. Model downloads and
Luna planning happen before the measured search and are separately timed in
the UI; active dimension inference is included in search/scoring RTF.
Pinned revisions and all available scales are in `reward_catalog.json`.
`adaptive_reward_validation.py` compares real GPU outputs with upstream
GELU head inference and an independent native ReLU/pooled computation;
maximum observed absolute difference was below 0.000002. Reproduce the real
HTTPS integration checks with `adaptive_reward_browser_test.py`.
The real HTTPS checks cover VoiceNet-only search, mixed four-term Best-of-4,
Luna-selected Top-2, mixed BS-16 search and legacy scoring. A BS-16 / 1600-ms
mixed VoiceNet+EIV+ smoke on the live two-card deployment measured search
RTF around 1.02–1.03 in two checks; that is not an isolated throughput study
or a real-time guarantee.
All 70 unit/API tests pass. Raw developer verification artifacts remain in
`test-results/`, not in the expiring user-upload directory.

### Optional position-aware vocal-burst reward

The independent **Burst check** UI selector is off by default. Choose
**Locator v2: position/duration** or **Locator + Commercial Detector:
position/type**, combined with either VoiceCLAP or the adaptive dimensions.
Targets come from the actual GENERAL/SCRIPT burst tags, or 1–8 manually
entered label/start/duration targets (also usable with CAPTION/TRANSCRIPT).
Manual targets are evaluation requests, not automatically inserted generation
instructions: request the corresponding sound in your acting prompt too.

Script targets default to gaps between exact-matched Parakeet word timestamps,
so changes in speaking tempo do not automatically invalidate the position.
Alternatively use fixed script times. Duration tags determine those times;
untimed speech uses an explicit heuristic of 0.36 s/word, at least 0.4 s/segment,
not an exact model guarantee. Untimed burst tags use the existing renderer's
default duration; numeric durations preserve the actual conditioning script.
Manual windows always use absolute time. The initial settings are a 25% burst
share, 0.3-s tolerance, family matching, and a small final extra-event penalty.

```text
match = timing * slot_overlap * sqrt(min(actual_duration, requested_duration)
                                   / max(actual_duration, requested_duration))
        * locator_confidence * type_probability
burst = max(0, mean(one-to-one target matches, missing targets=0) - extra_penalty)
quality = (1-burst_share) * base_quality + burst_share * burst
reward = quality * max(0, 1-WER)
```

Matches use global one-to-one assignment: an event cannot satisfy two requests.
Clock timing decays linearly with onset error to zero at the tolerance limit;
word-slot timing penalizes events outside the inter-word window, and overlap
downweights events extending into neighboring speech beyond tolerance.
The commercial classifier runs on localized event crops, not the entire take;
family mode sums the probabilities of the related classes, exact mode uses one
class. Unsupported classes are explicitly reported as **location-only**, not
treated as verified types. The classifier supports 16 non-speech types plus
`no_burst`; several are unreliable in the upstream evaluation. These noisy
proxies cannot certify burst existence/type, naturalness or quality improvement.

Future targets remain pending during chunk search and do not lower the score
yet. All targets are scored at completion; missing/unresolved targets score
zero. An intentional leading non-speech interval may bypass the empty-prefix
ASR gate only before speech is due, never the final full-transcript WER gate.
Extra detections receive at most 0.15 penalty, only at completion, and this
penalty can be disabled. With no explicit targets the burst addition is
inactive and the previous ranking is preserved. Enabled CLAP+burst Best-of-N
uses the fixed search-style 50/25/25 base quality instead of the legacy
within-candidate-set normalization, making the new score comparable across
search parents and chunks.

The pinned [locator](https://huggingface.co/laion/vocalburst-locator) uses
**model_v2.pt**, threshold 0.5, merge gap 0.1 s and minimum duration 0.1 s,
not the public helper's v1 defaults. Physical PCM extent is cropped before
postprocessing; overlapping windows cover prefixes longer than 30 seconds.
Its separately trained Whisper encoder cannot be substituted with BUD-E.
The [commercial detector](https://huggingface.co/laion/vocal-burst-detector-commercial)
uses all five official BatchNorm/GELU heads, mean logits followed by softmax,
and the already byte-validated shared VoiceCLAP tower. Models are loaded lazily.
Active locator/classifier inference is included in search RTF; initial model
loading is shown separately. Old throughput tables do **not** benchmark this
option, and real-time performance is not guaranteed.

`burst_reward_validation.py` checks pinned upstream GPU logits, independent
classifier-head inference, localized synthetic samples, timing displacement
and a real 31.2-s prefix. `burst_reward_browser_test.py` checks real HTTPS
controls, manual/word-based targets, adaptive combinations, Best-of-N,
800/1600-ms winner/Top-2 searches, no-target/off regressions and mobile layout.
Developer artifacts stay under `test-results/`; no private upload is published.
The local English search report documents this extension separately from the
original measurements.

### UI navigation and search discoverability

The live UI source now separates settings from the text/audio conversation.
The first, initially open card is **Voice & important settings**, followed by
**Generation & Beam Search**, with direct
winner-search and Top-2 activation buttons. BS 1–16 and 800/1600-ms controls
remain visible before activation; inactive controls are disabled and explained,
not silently hidden. Full-utterance N (1–12) is distinct from search BS.
Only the appropriate duration control applies: speech-duration hints for normal
takes, the separate search cap for search. The composer shows the active method,
candidate count/chunk, reward and reference selection before sending.

Separate numbered cards cover prompts, reference voice, sampling/playback,
adaptive rewards and optional position-aware bursts. Desktop uses a scrollable
settings column and a conversation/composer column, rather than burying settings
inside a capped footer. Mobile uses a normal scrolling page, section shortcuts
and initially folded secondary cards. A persistent header link opens the search
card; another returns to text/audio. Search results explain RTF and retained
prefixes before exposing the complete technical JSON in a nested disclosure.

No model defaults or generation methods were removed. The reference upload,
manual prompt template, Luna preview, dimension weights/targets, burst settings,
temperature zero, CFG controls, MP3 transport and alignment controls remain.
Codec streaming for full takes is explicitly distinguished from live chunk
search. Top-2 streaming may wait for a common prefix; it is not promised to
emit every chunk immediately.

`node ui_controller_test.cjs` (Node >=18) exercises the actual production JS
functions with small DOM fixtures across all 64 search-mode/BS/chunk
combinations plus normal/adaptive/burst cases. `tests/test_ui.py` verifies the
source controls and validates every resulting request against the real `Turn`
schema. Set `SMALL_UI_TEST_NODE` if the default Node is older than 18.
`ui_layout_browser_test.py` is a separate real-Chromium desktop/mobile test with
fully mocked API responses; it makes no GPU or quality claim. The 2026-10-08
run passes at widths 1440, 1024, 390 and 320, all four LoRA presets, independent
weights, cancel/Escape/reset, reference controls and all five generation modes,
with no JavaScript errors. `ref15_live_browser_test.py` is the separate actual
HTTPS upload/preset/GPU-generation test; `ref15_live_test.py` also tests long-
reference chunk search and deterministic base restoration. Earlier performance
and quality-study reports remain historical evidence, not a re-benchmark of
this longer-reference release.

Run from this directory with `bash run.sh`. It uses the existing Python venv,
existing scorer assets and API credential file. `SMALL_PORT` defaults to 8792;
the old demo must be stopped first. TTS uses physical GPU 0, scorers GPU 1.
For background operation with persistent logs: `python service.py start`,
`python service.py tunnel`, `python service.py status`, `python service.py stop`.

### Base-default adapter overlay and independent reference voices

The header and voice-settings card open a searchable, keyboard-accessible overlay
for the six [DPO adapters](https://huggingface.co/laion/Humaneness-Voice-Small-DPO-LoRAs),
pinned to `1a3fc77df1a420710ed521f59d4467efe0ff57d6`. The latest rank-16 release,
`subset_pitch_quality_p2_rank016`, remains the one-click **previous DPO** preset
at strength 1, but **the unmodified S3 base is now the default**. The rank-64
subset, original and rebalanced rank-64/128 releases are selectable, as is the
unmodified base. Old rank-1 specialist selections are no longer accepted or
shown. Historical rank-1 test artifacts are retained, not active runtime inputs.

Choose at most one DPO, optionally add Ref15, adjust independent strengths 0–2,
and apply. Four quick buttons select base / previous DPO / Ref15 / both for the
next generation. Old auto-enabled DPO browser settings are not migrated into
this base-default release. New explicit preferences persist locally. Cancel/Escape
does not change the next-turn selection. Default/base buttons provide explicit
reset choices. Browser-local selections are sent on every request and serialized
under the generation lock, not shared as a user preference across sessions.
The DPO changes both the semantic backbone and local Talker: 144 projections,
288 adapter tensors, and the release's `alpha/rank=2` scaling. Base and adapter
SHA-256, format, target names, shapes and finiteness are checked before mutation.
The release's `adapter_bytes` metadata omits serialized-file overhead; SHA-256
is the integrity authority. Fusion starts from immutable CPU base weights and
preserves existing BF16 parameter storage for compiled/CUDA graphs. It is not
bit-identical to the official FP32 residual-branch loader. Adapter ID, strength,
revision and switch time are included in results. Non-unit strengths are
experimental; a universal quality improvement is not claimed.

The new [Ref15 rank-128 repository](https://huggingface.co/laion/Humaneness-Voice-Small-Ref15-LoRA-r128)
is installed as a separate opt-in (`adapters.ref15=true`, `ref15_weight=1`).
Its public README, inference/training loader and CONTENTS manifest were inspected.
The release is pinned to `0c6a96cf00780a76809ba557897d2f3ee18e8033`;
`adapter_model.pt` SHA-256 is
`a64656d078cf0233b1d59567c4ee75daa27e81341b78e4ba32e6e8e1d6047d9f`.
The verified payload format is `humaneness-voice-small-ref15-qkvo-r128-v1`,
trained on S3 step 32337, rank **128**, alpha **128** (scale **1**, not DPO's 2).
It changes the q/k/v/o projections of all 28 semantic layers: 112 modules,
224 A/B tensors, **no Talker changes**. Exact release fields, all target names,
actual model shapes, dtype/finiteness and checksum are validated before fusion.
Only adapter weights are downloaded, never executed training code or optimizer
state. The validated contract is recorded in `model-cache/ref15/release.json`.
Ref15 requires a catalog or uploaded reference; no-reference mode is rejected
when it has nonzero strength. Training used at most 187 frames (14.96 s).
Ref15 + DPO sums each declared delta from the same base and has passed actual
GPU generation, streaming search and base-restoration smoke checks. The mix is
still experimental: successful inference does not prove an improvement in
speaker similarity or quality. BF16 fusion is not bit-identical to the official
FP32 residual-branch implementation.

The separate voice menu restores exactly Idris, Selma, Cormac, Mira, Rasmus,
Nora, Béla, Anton, Robin and Osku. No original speaker LoRA is loaded. After
Luna produces the acting direction, the existing commercial VoiceCLAP instance
ranks the original centered condition centroids plus emotion-anchor agreement.
Candidates are strictly limited to the selected speaker and spoken language;
isolated bursts and character transformations are excluded. Only audible
delivery cues, never the literal transcript, enter retrieval. Each request
selects a new clip and puts it in the Small model's reference-audio channel.
The clip, condition/emotion, score and MP3 preview are visible in the result.
The text tower expects English acting cues; missing/German-only cues or missing
eligible clips fall back explicitly to the same speaker's fixed anchor, never
another speaker. This is a retrieval heuristic, not a speaker/emotion guarantee.
Voice choice is independent of DPO selection and reward-model selection.

### Default voice and speaker-consistency reward

Selma (`emolia_c0542`) is now the UI/API default, with the voice selector and
speaker-weight slider at the top of the sidebar (also open on mobile). The
live winner-of-12 uses 1600-ms chunks and a 30-second search limit. The ordinary
codec-streaming chunk default is also 1600 ms.
Choosing no voice explicitly means no reference audio. Custom uploads remain
private and active until removed; catalog conditioning clips are selected per
acting prompt using VoiceCLAP, strictly within the chosen voice and language.

The default ranking factor is now exactly:

```text
style = clip((centered_prompt_audio_cosine + 1) / 2, 0, 1)
speaker = clip((reference_output_speaker_cosine + 1) / 2, 0, 1)
quality = (1 - speaker_weight) * style + speaker_weight * speaker
reward = quality * max(0, 1 - WER)
speaker_weight = 0.60  # style 0.40; slider 0..1; ECAPA by default
```

During search the gate uses stable prefix-WER; completed candidates use full
WER. The bounded inverse-accuracy gate is not singular `1/WER`. Genuine/Blend
remain diagnostics, not hidden extra terms when speaker mixing is on.
Adaptive VoiceNet/EIV+ dimensions can replace the prompt-style factor; their
weighted normalized utility is mixed with identity in the same proportions.
Optional explicit burst placement remains a separately visible blend afterwards.
Speaker method `off` restores the historical quality mixture; speaker weight
zero means pure prompt-style ranking, not that historical mixture. With no
reference, insufficient recognized speech, or a reference shorter than 0.5 s,
identity is marked unavailable and effective weights become 100% style / 0%
speaker. Loading/integrity errors are surfaced, not silently swallowed.

Optional `voicenet` reuses the commercial audio tower and 15 pinned regression
heads: `AGEV`, `GEND`, `REGS`, `R_CHST`, `R_HEAD`, `R_ORAL`, `R_NASL`, `R_THRT`,
`R_MASK`, `FULL`, `BRGT`, `WARM`, `ROUG`, `METL`, `HARM`. Register is a perceived
pitch-register proxy, not measured F0. Timing, emotion, tempo, pitch-range,
recording noise and speaking-style heads are deliberately excluded. The native
0..6 scalars are uniformly mapped to -1..1 before cosine, preventing trivial
high similarity from all-positive vectors. These attributes are perceptual
proxies, not unique or strictly emotion-invariant speaker identity; some timbre
heads have weak variation, so high cosine does not establish identity. This is
an experimental consistency reward, not a measured quality improvement.

Default `ecapa` uses the real 192-D speaker embeddings from
[`speechbrain/spkrec-ecapa-voxceleb`](https://huggingface.co/speechbrain/spkrec-ecapa-voxceleb),
pinned to `0f99f2d0ebe89ac095bcc5903c4dd8f72b367286`. It is loaded lazily on the
scorer GPU; audio is mono 16 kHz, encoded in batches with true length ratios.
No external audio/biometric service is called. Short emotional chunks remain
less reliable than longer clean speech. In both methods, the reference target
is now the exact fixed conditioning clip chosen for the profile;
custom mode uses the uploaded clip. Reference vectors are
content-hash cached (bounded, process-local), never published, and evicted on
upload removal/replacement. Catalog anchors and custom identity references use
at most the configured 15-second reference cap (20 seconds if explicitly set).

For streaming search, identity is evaluated on the newest 800/1600-ms chunk;
for single/Best-of-N/all-finished-candidates it uses the whole output. VoiceNet
therefore needs an additional batched audio encode in search, rather than
pretending cumulative-prefix embeddings are chunk embeddings. ECAPA uses its
own encoder. Speaker preparation is reported separately and excluded from
search RTF; per-chunk speaker scoring is included. Historical published RTF
measurements do NOT establish the throughput of these new rewards.

### Default cutoff and fades, including live search

`align=true` remains the UI/API default. Finished takes use the original Qwen
forced-alignment lead suppression and tail cutoff/fade before final scoring.
Live search now also buffers 1 second, initially 1.3 seconds for lead
detection. It uses existing Parakeet v3 word timestamps to silence/fade unwanted
lead and stop/fade the tail only when the normalized stable transcript exactly
matches the full target. The final word must lie inside committed PCM with room
for the 0.12-s air margin and 0.15-s fade; unfinished words cannot be cut merely
because a forced aligner fitted the whole script onto a prefix. Requested edge
bursts/pauses retain the old extra allowance. A successful cutoff also stops
further TTS search, rather than merely hiding subsequent streamed samples.
Remaining held-back PCM is flushed from the final forced-aligned waveform.
Already emitted samples cannot be retracted; late alignment is explicitly
reported and the held raw tail is faded rather than stopped abruptly.
Disabling alignment is a passthrough. This trades some first-audio
latency for editable edges. First-playable timing and live alignment decisions
are present in search results; raw acoustic-commit timing is retained separately.

The previous real speaker/cutoff measurements below used the former Idris /
VoiceNet 66/34 defaults. They are not a comparison or validation of the new
Selma / ECAPA 40/60 ranking weights. Default/schema and mocked Chromium tests
have been updated for Selma, ECAPA, 0.60 speaker weight and 1600-ms chunks.

Speaker/cutoff verification: 107 unit/API/controller tests pass. Chromium
tests cover mobile voice selection above the conversation, the mixture slider,
ECAPA selection and the existing 64 search combinations. `speaker_live_test.py`
passed 11 real TTS/ASR cases, including VoiceNet and ECAPA Best-of-N, 800/1600-ms
search, no reference, zero speaker weight and adaptive dimension rewards.
These are functional smoke tests, not evidence of improved speaker quality.
First ECAPA use took about 137 seconds in the initial run and 70 seconds after
the next server restart, including cold initialization/generation; a subsequent
ECAPA search took about 2.9 seconds. Do not assume a warm-start RTF applies to
first use. Latest local evidence is in `test-results/speaker-live.json`.
Five additional real HTTPS browser generations checked 800/1600-ms searches at
BS 1/3/4/8/16, decoded MP3 chunk durations against actual emitted playback,
mono 96-kbit/s encoding and cross-session isolation. All five final ASR WERs
were zero; one Top-2 run actually triggered the Parakeet cutoff (3.88-s fade
start, 4.03-s streamed output). This is a short functional sample, not a quality
or throughput study. Evidence: `test-results/search-browser.json`. Existing
search RTF denominators refer to raw generated PCM, not post-cutoff playback.

Earlier DPO/reference verification: 96 unit/API/controller tests passed. Chrome checks
cover the DPO overlay, all 64 search-mode/N/chunk combinations, reference
source selection and layouts at 1440/1024/390/320 pixels. A separate HTTPS
Chrome check verified the live defaults, ten voices, adapter menu and saved
voice choice. `dpo_live_test.py` passed 12 real TTS cases covering all six
adapters, no-reference generation, anger-versus-sadness retrieval for Mira,
800-ms winner search, 1600-ms top-2 search, and byte-identical base restoration.
`reference_live_test.py` passed four real cases covering automatic custom reuse,
no-reference override, private MP3 access/deletion, and German Mira retrieval
with default live N=12. These are functional smokes, not emotion/naturalness
or speaker-similarity studies. The older sampling study was performed on the
base checkpoint, not the new DPO default.

`python service.py wait-start` starts a singleton background watcher, with
`logs/waiter.log` and `logs/waiter.pid`. It prefetches public adapters on CPU,
then checks both GPUs every 15 seconds. Three consecutive observations with
no foreign compute process, at most 256 MiB residual memory and utilization
at most 5% are required. It starts the demo and preserves/reuses the tunnel.
An existing idle demo may be gracefully restarted; active turns are not killed.
Foreign GPU jobs are never signalled. The default waiting timeout is one hour;
`python wait_for_gpus.py --timeout 0` waits indefinitely in the foreground.
`python service.py stop` also stops the watcher so it cannot restart a stopped demo.
The tunnel URL is printed in `logs/tunnel.log`.
`HF_HOME` defaults to `/mnt/nvme/hf_cache`. The first launch downloads source,
processor assets and the selected S3 weights, not all ten stages.

Luna defaults to `gpt-5.6-luna` at the same OpenAI-compatible Hyprlab endpoint
as the old demo, with reasoning disabled. Configure `SMALL_LUNA_BASE`,
`SMALL_LUNA_MODEL`, and `SMALL_LUNA_API_KEY` or `SMALL_LUNA_KEY_FILE` to override.
Credentials stay on the server. Chat and exact-text performance use Luna;
manual prompts bypass the API. The selected DPO adapter is loaded; no specialist
speaker, emotion, VoiceNet or burst LoRA is loaded.
Luna requests have a 30 s network timeout and a 35 s overall deadline (including
JSON retries). Failure releases the server, reports a readable error and retains
the user's input for retry. Other sessions see the actual job phase/elapsed time
instead of an unexplained busy state. The UI blocks sends while another job runs.

`actor_prompt.py` adapts the original `../llm_agent.py` acting sections: soft
natural baseline, physical emotion and containment, emotional arcs, intensity
bands, per-sentence cues, intra-sentence pauses, hesitation and worked acting
recipes. Tool calls, reference selection, adapter banks and old-checkpoint
performance claims are removed. Small's README supplies the supported surfaces
and their experimental-control caveats. Chat interprets acting briefs and enacts
the scene; it does not merely repeat the assignment. Explicit supplied lines
remain exact. Use the exact-text mode only for words to speak, not instructions.

## Prompt formats

The default is the old demo's structured direction format:

```text
GENERAL: warm, natural, softly amused, conversational
SCRIPT:
(clearly amused, easy and unforced) [4.0 seconds duration] Ich hätte [0.4 seconds pause] wirklich nicht gedacht, dass wir es schaffen.
```

The release also trained these surfaces, all selectable in the UI:

```text
CAPTION: Warm, natural, softly amused conversation.
TRANSCRIPT: "Ich hätte wirklich nicht gedacht, dass wir es schaffen."
```

```text
CAPTION: Tags: warm, softly_amused, conversational. Caption: A genuine moment of quiet relief.
TRANSCRIPT: "Ich hätte wirklich nicht gedacht, dass wir es schaffen."
```

```text
TRANSCRIPT: "Ich hätte wirklich nicht gedacht, dass wir es schaffen."
```

English direction, English or German speech. A manual prompt is used literally,
with its spoken transcript extracted rather than rewritten. The release's
`ScorePacker` builds the trained `<user_inst>` fields; `Text` repeats SCRIPT for
structured prompts and follows the inherited packer for the other surfaces.
Explicit timing and vocal bursts remain experimental. The model README reports
substantially better intelligibility for caption prompts than GENERAL/SCRIPT.

## Reference audio and expiry

Choose a fixed catalog voice, no voice/no reference, or an uploaded custom
reference. Uploading automatically activates custom mode. That clip is reused
on every subsequent request until removed or another voice source is selected.
The UI exposes one custom slot; legacy API slot B remains supported.
A reference is encoded once and cached as 12-codebook RVQ frames. Leading
silence is removed and audio is converted to mono 48 kHz and capped at **15 s**
by default, independently of adapters. `SMALL_REFERENCE_SECONDS=20` raises the
cap to 20 s; other values must remain between 0.24 and 20 s. The former 37-frame
guard is removed. The 12.5-Hz codec accepts up to 188 frames for 15 s and 250
for 20 s; actual codec rounding is reported, not forced. The tested real codec
produces 187 frames from a 15-second waveform. WAV reading uses SoundFile, then the pinned
processor's waveform encoder, avoiding its TorchCodec-dependent path reader.
Files must be under 20 MB. Clips shorter than the cap remain short; nothing is
looped or concatenated. Result `reference.tts_context` and the UI show actual
used seconds/frame count, so a short catalog clip cannot masquerade as 15 s.
Old uploads were irreversibly cropped to 2.88 s: upload the original file again
to use its later samples. Actual 15-second GPU codec and TTS checks pass for
base, DPO, Ref15 and the mixture. A greater supported cap does not prove that
the base or DPO model performs well on long references; broad speaker/emotion
quality studies and actual 20-second GPU checks remain unperformed.
References are never artificially concatenated into a longer reference. Replacing
a custom slot removes its previous file; explicit removal deletes WAV and MP3.

Uploads and generated MP3s are scoped to the browser session. Custom references
do not expire: a private mode-0600 ledger restores their ownership after server
restart. Access requires the same browser cookie (refreshed with a one-year
lifetime); clearing cookies loses access. Generated takes, automatic profile
previews and orphaned files expire after 3600 seconds of inactivity, controlled
by `SMALL_TTL_SECONDS`, and their codec caches are dropped. Active references
are pinned during generation. Download any takes you want to keep. Never
publish the ledger, uploaded references or runtime directory.

## Generation and selection

CFG is checked by default, strength **2.0**, including single-take generation.
The existing channel-wise loop uses `u + g * (c - u)` for audio logits. Both
branches share sampled codebook tokens, reference audio, transcript, duration
budget, pauses and burst requests. Only delivery directions change; the stop
decision comes from the conditional branch with Small's original sampling.
Caption prompts use a neutral caption in the same format on the comparison
branch, following the release's published reference-conditioned CFG probe.
No old-model stop bias is inherited. `g=1` disables guidance. For a transcript-only
prompt both branches are identical, so CFG has no effect and the UI says so.

Single take, best-of-N, or all N candidates are selectable. N ranges from 1 to 12,
default 3. Candidates are generated in batches of up to twelve and
all are playable and downloadable as **mono MP3, 96 kbit/s, 48 kHz**. Reference
previews are also MP3; their internal source stays PCM WAV for codec conditioning.
Alignment and judging use uncompressed floating-point audio before MP3 encoding.
Best-of-N ranks using the original Judge:
within-set normalized genuineness + blend + 2×CLAP + 2×burst (when requested),
multiplied by inverse WER. Parakeet v3 supplies transcription. These are model
scores; choosing the perceptually best Small take is not established by the
old 4.55B measurements. Generation completes before audio playback, especially
when selecting between candidates. The optional streaming GPU path incrementally
decodes PCM but browser playback still waits for alignment and judging. On-device
performance is measured separately from browser/network latency, as described below.

## Word boundaries and fades

**Sprachgrenzen & Fades** is enabled by default, independently of CFG. The same
Apache-2.0 Qwen3-ForcedAligner-0.6B backend as the old server runs on GPU 1 and
aligns the known spoken transcript, not a guessed ASR transcript. Every take is
processed BEFORE scoring, playback and download using the original
`align_engine.trim`: unwanted lead-in is silenced with a 100 ms ramp reaching
full level at the first word; after the last word there is 120 ms air followed
by a 150 ms fade-out and removal of excess tail. Minimum lead/tail thresholds
and structural plausibility checks are inherited from `../config.py`.

Explicit opening/closing bursts and pauses are protected. Small's wrapper also
handles manually supplied untimed bursts and scripts without duration tags;
mid-sentence bursts do not protect unrelated edges. Closing cues receive the
old server's 2× duration slack. Word content and pauses inside speech are not
edited. Qwen returns timestamps, NOT calibrated word confidence: structural
checks catch gross failures, not every alignment error. If alignment is missing
or fails, that take remains unchanged and the report discloses why. No MMS
non-commercial fallback is loaded. Each take's expandable report shows word
times, fade positions, original/final duration; the checkbox permits comparison.
The API field is `align` (default true), result timing is `alignment_ms`, and
each candidate includes its `alignment` report. Existing `MOSS_ALIGN_*` ramp and
threshold environment overrides also apply; the new checkbox/default is separate
from the old server's global enable switch.

## HTTP

`GET /api/state`, `POST /api/reference?slot=A` (multipart `file`),
`DELETE /api/reference/{id}`, `GET /api/audio/{id}`, `POST /api/asr` (audio bytes).
`POST /api/turn` accepts JSON and returns progress/result events as SSE:

```json
{
  "message": "Say something softly encouraging in German.",
  "format": "caption",
  "input_mode": "chat",
  "language": "de",
  "generation": "best",
  "n": 8,
  "guidance": 1.5,
  "seed": 777,
  "reference_id": null
}
```

Keep the session cookie when using curl; uploaded references and result URLs
belong to that cookie. Optional `seconds` sets a 1–60 s frame budget, not a
guarantee of output duration. Temperature ranges from **0 to 1.5**, default 1.0;
zero means true greedy/argmax sampling for audio and stop/continue tokens, not a
tiny positive-temperature approximation. Different seeds may therefore produce
identical best-of-N candidates at zero. Positive temperatures retain the release's
top-p 0.95, top-k 50, audio repetition penalty 1.0 and stop-token temperature 1.0.
Audio responses use `audio/mpeg` with `.mp3` filenames, including reference previews.

Sources: [model and inference contract](https://huggingface.co/laion/Humaneness-Voice-Small),
[Luna API model](https://developers.openai.com/api/docs/models/gpt-5.6-luna).

## Verified on this host, 30 September 2026

### Measured performance and emotion study

English HTML reports are available at `/performance/` and `/study/`, or locally
as `performance/index.html` and `study/index.html`. They include raw JSON links,
methods, limitations and (for the emotion study) every generated listening sample.
Do not expose `runtime/`, reference recordings, API credentials or raw upload files.

The optimized decoder fuses conditional/neutral CFG branches, captures the fixed
12-codebook local step in CUDA graphs, compiles the dynamic global decoder, and
batch-decodes the causal codec with supported BF16 encoder/decoder weights while
retaining FP32 quantizers. Model weights and TTS BF16 precision are unchanged.
Different GEMM batch shapes / fused pointwise rounding can change stochastic
trajectories: this is not bit-identical legacy inference. The study includes a
20-take original-backend quality diagnostic; it is not a perceptual equivalence
trial. Set `SMALL_FAST_INFERENCE=0 SMALL_COMPILE_GLOBAL=0` for the original token
loop and FP32 codec weights. `generate_legacy` also retains the exact old method.
`SMALL_COMPILE_GLOBAL=0` alone retains fused CFG / local CUDA graphs without global
compilation. `SMALL_BATCH_SIZE` defaults to 12; an OOM retries smaller batches.
Startup warms common paths before ready. New shapes may still compile, and the
UI truthfully includes cold compilation / graph capture inside the generation span.

On one RTX 3090, the warmed eight-second workload measured single-stream RTF
approximately 0.30 with streaming and 0.20 offline, versus 1.09 and 0.97 originally.
Ten parallel streams achieved around 28.3 audio seconds per device second. B=48
succeeded in the short test and B=64 ran out of memory; the exact maximum is not
determined. Twelve 60-second streams took about 28.4 device seconds, per-stream
RTF 0.474, with 11.45 GiB peak allocated memory. These are synchronized batches,
not a claim of a live asynchronous multi-user streaming scheduler.

GPU-Pfad selects offline or actual incremental PCM decoding. PCM-Chunk selects
2/5/10/20 frames (160/400/800/1600 ms); default 10. One-frame chunks were slower
than real time and are not offered in the UI. First-PCM readiness at B=1 was about
179 ms for five frames (RTF 0.415) or 251 ms for ten frames (RTF 0.30). MP3 transport
remains mono 96 kbps. API fields: `streaming` (default false), `chunk_frames`
(default 10). The result `performance` object contains CUDA-event device time,
raw audio duration, aggregate RTF, throughput and per-batch timings. This includes
TTS prefill, global/local decode, codec, and launch gaps; it excludes reference
encoding, packing, Luna, alignment, scoring, CPU PCM copies, MP3 and network.
The RTF denominator is all raw generated audio BEFORE fades/tail removal.

The 660-take emotion pilot comprises a predeclared 320-take CFG × temperature
grid, 160 selected-setting language/reference checks, 160 exploratory terse-cue
structured-recipe takes, and 20 original-backend takes. Five emotions, two prompt
surfaces and repeated seeds are evaluated. No best-of-N filtering is applied.
Practical starting range: concise CAPTION/exact TRANSCRIPT, CFG 1.5–2, audio
temperature 1.0. The main-grid emotion-first preference CFG 2 / T=1.2 did not
consistently generalize; defaults stay CFG 1.5 / T=1. Genuineness scores are low
in absolute terms; no human naturalness or emotion validation is claimed. See
the report for uncertainty, reference sensitivity and structured-prompt findings.

Scored WER now uses conventional Unicode-aware word Levenshtein distance:
one substitution counts once, German letters are preserved, and WER may exceed
100% with insertions. The older root scorer's SequenceMatcher distance is retained
as `legacy_word_distance` for inspection, but not used for new rankings. Study ASR
uses anti-aliased resampling for both raw/postprocessed PCM; production retains
the old scorer's box downsample. Report WER and old historical values therefore
are not directly interchangeable. Emotion scoring matches the public EIV card's
EmoWhisper-AnS-Small-v0.1 encoder and ReLU heads, rather than silently inheriting
the older server's generic Whisper backbone. Every score is a model proxy, not MOS.

Reproduce with `throughput_benchmark.py --phase baseline`, `--phase optimize`,
`--phase compile`, `--phase sweep --compile-global`, `--phase latency --compile-global`,
and `--phase long --frames 750 --repeats 2 --compile-global`, while stopping this
demo's model process (not its tunnel). `performance_validation.py` checks real
CUDA code equality / codec outputs. `emotion_study.py design`, `generate`, and
`score --follow` reproduce the frozen main design; `holdout --configs ...` and
`diagnostics` add explicitly labelled follow-ups. Use batch 4 in the experiment,
which is enforced independently of the demo's batch 12. For legacy controls use
`SMALL_FAST_INFERENCE=0 SMALL_COMPILE_GLOBAL=0 SMALL_BATCH_SIZE=4` with
`emotion_study.py generate --stage legacy`. `reports.py both` rebuilds the HTML.
Run model benchmarks only without other GPU 0 work; an open live demo confounds them.

36 unit/API tests pass after the performance changes. HTTPS Chromium checks cover
both responsive HTML reports, sample filtering/MP3 playback, RTF display,
streaming parameters, and the original busy/error/retry UI behavior. Historical
verification below refers to the pre-optimization code and its old WER formula.
Real HTTPS API/browser tests also pass for Best-of-3 with temperature 0 and
400 ms incremental PCM, and Best-of-12 with temperature 1 and offline PCM;
both verify mono MP3 96 kbps playback, real on-device metrics, all-candidate
alignment, reference ownership/deletion, prompt editing and mobile layout.
The 12-take smoke generated 53.28 aggregate audio seconds in 1.86 device seconds,
in one batch, without OOM. This variable-length smoke is distinct from the
fixed eight-second isolated benchmarks above. Reference B in the study is from
a hiss-labelled speech-containing corpus clip, so its cohort is a challenging
reference-sensitivity check, not a verified clean-neutral reference replication;
exact source filenames/hashes are recorded in the study design.

26 unit/API tests pass (prompt surfaces, literal transcript preservation,
CFG preserving timed and untimed bursts, upload ownership, deletion, expiry,
parameter limits, deriving chat words from the actual director script, old fades
with word content untouched, edge bursts, all-candidate editing BEFORE judging,
MP3 codec/channels/bitrate, zero-temperature sampling, and timeout busy release).
The live API schema accepts temperature 0 and defaults guidance to 1.5. The
public HTTPS UI's busy/Enter guard, timeout/409 display and retry were tested
without occupying the GPU. Additional real alignment testing found 86 ms for
seven German words: unchanged speech samples, lead suppression and tail fade,
WER 0.286 before vs 0.000 after, at the SAME seed and CFG 3. Real Luna produced
a new German confession from an acting brief with intra-sentence hesitation;
the best of three structured takes had WER 0.200. A subsequent Luna request
timed out upstream: this motivated the shorter deadlines and readable UI errors.

Initial-build results below predate the alignment/MP3/default updates:
Real Chromium tests passed at desktop and mobile sizes both locally and through
the HTTPS Cloudflare tunnel. The public browser test uploaded two references,
generated best-of-8 at guidance 3.0, exposed eight playable/downloadable WAVs,
edited the generated prompt, reset the chat and deleted both uploads, with no
JavaScript errors. The original Gemma and 4.55B server are stopped.

Initial real-model API run, seed 777, guidance 3.0 unless marked otherwise:

| Case | Candidates | Best WER | Total time |
|---|---:|---:|---:|
| English caption, no reference, CFG off | 1 | 0.000 | 2.86 s |
| English caption, no reference, CFG 3 | 1 | 1.000 (no recognized speech) | 4.58 s |
| German caption directed by Luna, reference A | 5 | 0.000 | 13.37 s |
| German structured manual prompt, reference B | 2 | 0.000 | 4.38 s |
| English structured Luna chat, no reference | 1 | 0.211 | 10.52 s |
| German structured Luna chat, reference A | 3 | 0.077 | 9.16 s |
| English tags + caption, reference A | 1 | 0.000 | 4.45 s |
| German transcript only, reference B | 1 | 0.000 | 2.53 s |

These few prompts do not establish a general quality ranking. In particular,
CFG 3 without a reference failed on the English caption above despite producing
a nonempty waveform. A clean reference is recommended; the UI shows a warning
when ASR detects high word error. The CFG default was subsequently changed to
1.5 at the user's request. The initial run also verified that changing CFG changes audio bytes,
five candidates are distinct across the 4+1 generation boundary, ASR accepts a
generated WAV, and a different browser session cannot download these takes.
Exact requests, rendered prompts, scores, hashes and test WAVs are under
`test-results/`; these developer-created test artifacts are intentionally kept,
unlike user uploads and live result MP3s in the expiring `runtime/` directory.

Reproduce with the existing venv: `python -m pytest tests -q` and, with the
server running, `python smoke_test.py`. For the browser check use Playwright
and the installed Chromium: `python browser_test.py --url https://YOUR-TUNNEL --n 8`.
`python ui_state_test.py` tests error/retry behavior against the public UI with
intercepted responses and no generation. `python acting_smoke_test.py` exercises
real acting briefs and deterministic alignment on/off; upstream Luna availability
can fail that integration test independently of local model readiness.
