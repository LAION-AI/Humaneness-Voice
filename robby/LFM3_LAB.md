# Experimental LFM3 audio / video lab

This opt-in lab is independent of Robby's conversation, memory, ASR and speaker identity. Open **Experimental audio & video lab** in the lower settings, press Record and Stop, choose a window (default 3 s) and stride (default 1 s), then press Analyze. Recording uses low-resolution video and audio for at most 30 s. The browser displays one player per complete window, each restricted to its time range, and both model descriptions and measured timing. Audio and video run in parallel on separate GPUs; windows are processed sequentially per model. The preceding two descriptions are supplied as uncertain context to that same model. There is no cross-modal person tracking or biometric recognition; proposed person IDs are approximate and can drift. Partial final tails are omitted, except that a clip shorter than one window is analyzed once.

The server receives the recording only on Analyze, limits it to 24 MiB/30 s, permits one job at a time, and deletes its private temporary files on normal completion or errors. The server sends a seekable MP4 of the same recording back to the browser; the browser retains it locally until the page is closed or another recording starts. A hard process crash may leave temporary files in `.runtime/lfm3-*`; do not publish them. Seekable playback is needed because Chromium's MediaRecorder WebM often has no duration metadata. Capturing people requires their permission. This demo endpoint has no user accounts: anyone with access to the demo can submit a recording, one at a time. It never adds audio/video or descriptions to Robby's conversation memory.

## Models and reproducible setup

Both checkpoint revisions are pinned and run in separate worker processes using BF16:

- [LiquidAI/LFM3-Audio-2B](https://huggingface.co/LiquidAI/LFM3-Audio-2B), revision `9d49701c372d6011d900f6abb40c6b296fb706dd`, GPU 0.
- [LiquidAI/LFM3-VL-2B](https://huggingface.co/LiquidAI/LFM3-VL-2B), revision `f83d2b0a2296aaa277f394990a9b8b949ddf9691`, GPU 1. Per model card, the video encoder samples at 2 fps, despite the 5–10 fps recording.

Both have the Liquid LFM Open License v1.0; consult each model card and license before use or redistribution. Remote custom model code is executed locally from the pinned downloads. The example environment here uses the existing torch 2.6 CUDA stack plus a separate Python package target with `transformers==5.21.0`, `huggingface-hub==1.31.0`, `tokenizers==0.23.3`, `av==19.0.2`, and a compatible recent `regex`. PyAV 19 requires a small compatibility shim for the installed torchvision decoder. The audio model warns that causal-conv1d is not installed and uses a slower PyTorch reference implementation. Neither model changes Robby's existing Python environment.

Example installation (choose a cache on a drive with at least 10 GB free, and review custom code):

```bash
export HF_HOME=/mnt/nvme/moss-15-v2/.cache/lfm3
export LFM3_PYTHONPATH=/mnt/nvme/moss-15-v2/lfm3-env
uv pip install --python "$ROBBY_PYTHON" --target "$LFM3_PYTHONPATH" --no-deps \
  transformers==5.21.0 huggingface-hub==1.31.0 tokenizers==0.23.3 \
  av==19.0.2 'regex>=2025.10.22'
"$ROBBY_PYTHON" - <<'PY'
from huggingface_hub import snapshot_download
for repo, revision in [
 ('LiquidAI/LFM3-Audio-2B','9d49701c372d6011d900f6abb40c6b296fb706dd'),
 ('LiquidAI/LFM3-VL-2B','f83d2b0a2296aaa277f394990a9b8b949ddf9691')]:
    snapshot_download(repo,revision=revision,ignore_patterns=['assets/*','*.svg','*.webp'])
PY
PYTHONPATH="$LFM3_PYTHONPATH" "$ROBBY_PYTHON" lfm3_service.py
```

`ROBBY_PYTHON` must be set to the existing Robby Python with PyTorch, aiohttp and model dependencies. Configure `LFM3_AUDIO_GPU`, `LFM3_VIDEO_GPU`, `HF_HOME` and `LFM3_PYTHONPATH` as needed. The service listens on loopback port 8810; the Robby server proxies `/api/lfm3/*` and must be restarted after adding routes. It has its own lifecycle; restart it after a machine reboot. The source export excludes weights and recordings.

## Measurements and limitations

A 3-second synthetic beeping/color-pattern clip tested on two RTX 3090s returned audio in about 1.8–2.0 s and video in about 2.2–2.7 s including preprocessing; the models ran concurrently. Another warmed run detected an authored synthetic melody in 2.28 s audio processing (43 output tokens, 37 tokens/s, including prefill). On an authored mixture of speech and quiet music it identified the speech but missed the music. These single samples are not an accuracy or throughput benchmark. Results depend on output length, GPU contention and warmup. Token/s divides output tokens by model generation wall time including prefill; it is not decode-only speed. FFmpeg conversion, upload and queuing are included in the overall wall time, not in per-model timing.

The model can hallucinate, omit quiet sound, misdescribe visual content, return non-JSON text or misassociate people across windows. Unstructured output is shown verbatim instead of inventing structured fields. It is not a safety detector, identity verifier or validated diarization system. It can return a label such as “none” instead of an empty field. The first browser run failed because MediaRecorder WebM lacked duration; the current version remuxes it and returns seekable MP4. Browser tests use fake devices and synthetic audio; real-device behavior needs user testing.

Reproduce the end-to-end browser test with `browser_lfm3_lab_check.py` using Playwright and Google Chrome, after both services are running. It records from fake browser devices, changes window and stride, submits to the live HTTPS service, checks multiple annotated windows and tests later-window playback. Reports named `static/lfm3-*.json` contain only authored synthetic test material, never user recordings.

In an additional two-window test using one static public-domain portrait, the vision model returned `person_1` in both windows. This is a minimal continuity check, not evidence that it can reliably track multiple moving people. With the two LFM workers resident on the GPUs, the regular Robby fake-microphone wake, TTS, camera and permission browser checks also passed. The 584 offline regression checks passed. No physical camera or human multi-speaker recording was tested.

## Facade upload and fast-caption update (October 28)

The port 8798 FastAPI facade used to reject every non-Bolt POST, so the lab POST received HTTP 405 when this facade was the demo origin. It now forwards **only** `/api/lfm3/analyze` with a 25 MiB request limit; other POST routes remain blocked. Run `browser_lfm3_lab_check.py --url https://YOUR-FACADE` as well as the direct-origin test.

Vision reasoning was already explicitly disabled with `enable_thinking=False`. The audio chat template used for this task has no reasoning prompt. The new video prompt asks for concise fields and avoids guesses about gender, age and identity; video preprocessing uses 2 fps, matching the model's documented sampling. Three warmed runs on the same three-second synthetic music/video clip produced vision total times of 1,789 / 1,818 / 1,774 ms before and 1,164 / 1,128 / 1,157 ms afterward; output fell from 87 to 55 tokens. Audio was about 0.89–0.91 s in both sets. This is a small deterministic fixture, not a representative quality or real-time benchmark. Both versions parsed as JSON and detected the synthetic music. A two-window static portrait test retained its person label with the concise prompt, but the model described the photographed person's pose incorrectly; visual descriptions and IDs remain fallible. See `static/lfm3-speed-before.json`, `static/lfm3-speed-final.json` and `static/lfm3-speed-final-portrait.json` for the authored test results. Recordings and the MP4 payload are excluded from these receipts.

## Detailed descriptions (current)

The current prompts replace the earlier terse video prompt: video should describe visible foreground, background, objects, people, movement and changes in detail, while avoiding unsupported identity, demographic or intent claims. Audio asks separately for foreground, background, intelligible speech in the original language, delivery, apparent tone and speed, and music, melody, instrumentation, rhythm, style, tempo and mood. Emotions and music labels are uncertain impressions, not facts. The model is told not to guess song titles or unintelligible words. The output limit is 512 tokens. Output can still be incomplete, inaccurate or non-JSON.

When the first pass detects speech or music, a second audio query asks for detail; when speech is detected it also asks independently about faint music, because the first pass can miss background music. If the passes disagree, the result explicitly says the music finding is uncertain. Windows below an RMS threshold of 0.0003 (approximately -70.5 dBFS) are labeled silence or very quiet audio instead of being fed to the audio model. Very quiet real sounds may also fall below this threshold. These extra audio calls and longer video descriptions cost time compared with the previous concise mode.

On authored three-second fixtures, the model described the synthetic melody, rhythm and mood; identified the words “Hi Robbie” and described their delivery; and the follow-up found music underneath that speech, which the first pass had missed. A speech-only fixture produced “no music” on the follow-up. The muted portrait clip activated the silence gate. The model still made potentially inaccurate visual/voice judgments, so these are illustrative checks, not calibrated accuracy measurements. The `static/lfm3-detail-*.json` receipts contain only synthetic media descriptions and no audio/video bytes. Tests ran locally on the RTX 3090s; a warm speech-plus-music sample used about 3.27 s of summed audio model processing and the more detailed synthetic video sample took about 4.1 s (including preprocessing). Browser and real-world end-to-end timings vary.

A final repeat after the disagreement indicator was added is in `static/lfm3-detail-release.json`: the first pass missed background music, while the follow-up suggested it, and the displayed result records that disagreement. The same test exposed an unsupported inference about a speaker being nervous; treat such descriptions as uncertain model output. The facade browser check passed with six windows in 16.7 s on fake devices.
