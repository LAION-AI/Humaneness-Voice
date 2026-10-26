# gpu-unlock and the local Nemotron E4B: reproducible check

Date: 2026-10-26. This directory is independent of the running Robby source tree.

## What has actually been tested

The inspected upstream revision is **`af28cbbfd3d922a659227ecfb7369adfd924cf03`** of [Zagitta/gpu-unlock](https://github.com/Zagitta/gpu-unlock/tree/af28cbbfd3d922a659227ecfb7369adfd924cf03).

Both installed GPUs are **RTX 3090 / GA102, PCI device `10de:2204`**, which appears in upstream's supported-device table. The host uses **Linux 5.4.0-208-generic** and NVIDIA **550.54.15**. Both cards are currently bound to the NVIDIA driver. `gpu_unlock` is not loaded; its boot option is absent. The actual selector-register state was not inspected, so module absence alone is not proof of a locked card.

The project's CPU/mock checks passed **five top-level tests**, including six compiled C component/mock programs. No NVIDIA firmware was downloaded, and these checks do not exercise a physical GPU. A separate **real kernel-module compilation** against the installed host headers failed at missing `pci_dev_lock` and `pci_dev_unlock` declarations. The later `linux/aperture.h` dependency is also absent from these headers. This is a demonstrated compatibility blocker, not a speculative performance concern. See [inspection.json](inspection.json), [upstream-tests.log](upstream-tests.log) and [the compiler log](kernel-build.log).

No module was installed or loaded. No boot settings, drivers, GPU clocks/power limits, services or running jobs were changed. No unlocked-versus-locked speedup has been measured.

## What the project changes

Upstream is an early-boot kernel module, rather than a per-process CUDA optimization. Before the stock NVIDIA driver attaches, it changes Ampere instruction-rate selectors using the project's firmware path and performs a PCIe function reset. Its source explicitly refuses a device already attached to a driver. [README and boot flow](https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/README.md), [device/driver checks](https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/kernel/gpu_unlock/main.c), [selector/reset implementation](https://github.com/Zagitta/gpu-unlock/blob/af28cbbfd3d922a659227ecfb7369adfd924cf03/kernel/gpu_unlock/unlock_610.c).

Consequently, a real trial here needs a compatible kernel plus scheduled boot/maintenance work; loading a module next to the live LLM is not a valid intervention. The running audio demo and other users' GPU training were preserved.

Upstream's performance reports concern larger floating-point matrix throughput and LLM prompt processing; they do not promise equivalent token-decoding gains. Our local model uses quantized **Q4_K_M** weights. **Inference:** any benefit must be measured separately for prefill, first token and serial decoding; a dense-BF16 GEMM result cannot be transferred directly to this application. The local [llama.cpp quantized vector path](https://github.com/ggml-org/llama.cpp/blob/5d9e5ac30e469d44c0a5a52556de0ead03aaa5b0/ggml/src/ggml-cuda/mmvq.cu) also makes that distinction relevant.

## Baseline measurement

`benchmark_local_4b.py` queries the **existing unchanged** llama.cpp service at `127.0.0.1:8799`; it never activates gpu-unlock. It records the exact GGUF hash, llama.cpp commit, system-prompt hash, generation settings, prompt/output token counts, real server prefill/decode timings, client first-content latency and concurrent GPU load.

It uses three synthetic prompt sizes, one excluded warm-up per case, then three serial measured requests. Every request has a 96-token output cap, temperature 0, seed 20261026 and prompt caching disabled. A length-capped output is expected in this throughput test and must not be counted as task-quality success. The actual Robby system prompt is read as a literal without importing or altering the application.

On another machine use its Python environment and set `--url`, `--robby-source`, `--llama-source`, `--upstream-source` and, if needed, `--model-file`. The source directories are used for the public system-prompt literal and Git revision metadata; no service is launched. The GGUF must be locally readable for hashing. Example:

```bash
python benchmark_local_4b.py --url http://127.0.0.1:8799 \
  --robby-source /path/to/Humaneness-Voice/robby \
  --llama-source /path/to/llama.cpp --upstream-source /path/to/gpu-unlock \
  --model-file /path/to/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf \
  --label baseline-new-run --out baseline-new-run.json
```

Choose a fresh output filename for every attempt; the portable script refuses to overwrite an existing report. `benchmark_local_4b.executed.py` preserves the exact initially executed script with this host's default paths. The portable version only adds explicit path options, output-preservation and process-path sanitization; the measured requests and timing procedure are unchanged.

```bash
cd /mnt/nvme/moss-15-v2/gpu-unlock-study
/home/c4r33u19/moss15v2/.venv/bin/python benchmark_local_4b.py \
  --label unchanged-shared-gpu --repetitions 3 --out baseline.json
```

Run only after latency-sensitive conversational tests finish. Existing independent training jobs on both cards mean a result can still be a **shared-GPU baseline**, not an isolated hardware-capacity measurement. Keep every GPU sample and repetition. Do not compare a busy baseline with an idle after-unlock result.

The executed run completed after the parent's audio/browser suite ended. Results below are medians of three measured repetitions per case, excluding warm-up:

| Input | Client first content | Server prefill | Server decoding | Client total for 96 tokens |
|---|---:|---:|---:|---:|
| Short prompt | 135.3 ms | 323.8 tokens/s | 38.3 tokens/s | 2645.6 ms |
| Actual Robby system prompt | 619.9 ms | 2126.6 tokens/s | 38.3 tokens/s | 3109.7 ms |
| Longer synthetic context | 859.9 ms | 2727.1 tokens/s | 35.1 tokens/s | 3598.3 ms |

All nine measured requests produced 96 output tokens with zero cached prompt tokens. Outputs were identical across the three repetitions of each case. This checks repeatability within this unchanged service/run; there is no post-unlock output comparison. Short-prompt prefill tokens/s reflects fixed overhead and must not be compared directly to larger batches of prompt tokens. TTFT includes local HTTP/server scheduling, while the server's token rates use its own timer. The medians of different columns need not add up.

Raw evidence is in [baseline.json](baseline.json), with a concise version in [results.json](results.json) and a readable [HTML report](report.html). This baseline cannot establish what gpu-unlock would change. A contention-free before/after experiment remains outstanding.

## Valid future A/B procedure

1. Arrange a maintenance window with the machine/job owners. Preserve the current working boot entry and service launch configuration. This task did not authorize disrupting other users' training.
2. First obtain a kernel on which this pinned module builds. Establish **baseline A on that same kernel and driver**, without the unlock enabled. Otherwise a kernel upgrade would be confounded with the unlock.
3. Use the identical GGUF, llama.cpp executable/commit, launch arguments, prompt files, precision, sampler, concurrency, power limit and thermal state in both conditions. Wait for other workloads to finish normally; do not kill them to obtain a clean result.
4. Follow the pinned upstream installation/boot procedure during that maintenance window, targeting only the intended card (`0000:01:00.0` on this host) rather than all GPUs. Confirm success and selector values from the kernel log before labeling condition B “unlocked”. No installation or reboot command was executed in this study.
5. Repeat the same script with `--label after-unlock --out after-unlock.json`. Verify model/request hashes and output-token equality before comparing medians. Repeat across additional boots or alternate order when feasible; three in-process repetitions do not establish reboot-level reproducibility.
6. Report separate prefill tokens/s, decode tokens/s, client TTFT and total wall time, with every repetition and GPU load. Re-run Robby's actual first-sentence and ASR/LLM/TTS latency tests afterward. Do not infer assistant quality or end-to-end speed from this microbenchmark.

The local server launch recorded for reproduction is:

```text
llama-server -m /mnt/nvme/robby-bolt/models/Nemotron-3.5-Bolt-E4B-Q4_K_M.gguf
  --alias nemotron-bolt --host 127.0.0.1 --port 8799 -ngl 999
  -c 16384 -np 2 -fa on -b 512 -ub 256 -t 6 --jinja --no-webui
```

This is a record of the existing service, not an instruction to start a duplicate on the same port. The model file is 6,180,446,208 bytes; its E4B name refers to the released model, while its full GGUF size is not 4 GB.

## Reproduce the source/compile checks without installation

```bash
git clone https://github.com/Zagitta/gpu-unlock upstream
git -C upstream checkout af28cbbfd3d922a659227ecfb7369adfd924cf03
make -C upstream test ASSETS="$PWD/no-firmware"
make -C upstream/kernel/gpu_unlock
```

The last command only compiles against the running kernel headers; it does not load the module. On the tested 5.4 host it exits 2. `make install`, initramfs changes, module loading, driver unloading and GPU reset are intentionally absent from these executed commands.

Source/CPU-check storage was approximately 1.6 MB. No model weights or firmware copies were downloaded for this inspection. Test data is synthetic text only; no microphones, enrolled speaker embeddings or API keys are included.
