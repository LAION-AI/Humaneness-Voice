"""Isolated, synchronized GPU throughput benchmark; no LLM/network/scorers.

Results are persisted after every trial so interrupted runs remain auditable.
"""
import argparse
from contextlib import nullcontext
import gc
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time

import numpy as np
import torch
import transformers

from engine import SmallEngine
from performance_core import FastGenerator, StreamingCodec
from prompting import build
import settings

OUT = settings.HERE / "performance"
CAPTION = {"text": "The train was late, but we made it home before the rain started. I never thought we would be this lucky.",
           "caption": "Warm, natural, softly relieved conversation.", "general": "Warm natural conversation."}


class Benchmark:
    def __init__(self, frames=100):
        # Load the unoptimized baseline explicitly; benchmark phases enable
        # each optimization separately, regardless of demo defaults.
        settings.FAST_INFERENCE = False
        settings.COMPILE_GLOBAL = False
        self.engine = SmallEngine()
        self.fast = FastGenerator(self.engine)
        self.frames = frames
        self.spec = build(CAPTION, "caption", "en", frames / 12.5)
        source = next(Path("/mnt/nvme/moss-15-v2-assets/refs3/audio/emolia_c1699").rglob("*.mp3"))
        OUT.mkdir(exist_ok=True)
        reference = OUT / "benchmark-reference.wav"
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(source),
            "-af", "silenceremove=start_periods=1:start_duration=0.02:start_threshold=-45dB",
            "-t", "2.88", "-ac", "1", "-ar", "48000", str(reference)], check=True)
        self.reference = self.engine.reference(reference)
        self.records = []
        self.notes = []
        self.environment = {"date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "gpu": torch.cuda.get_device_name(0), "vram_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
            "torch": torch.__version__, "cuda": torch.version.cuda, "transformers": transformers.__version__,
            "python": platform.python_version(), "model": settings.REPO, "revision": settings.REVISION,
            "stage": settings.STAGE, "codec_revision": "f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
            "tts_dtype": "bfloat16", "codec_baseline_weights": "float32", "codec_compute": "bfloat16",
            "frames": frames, "fixed_audio_seconds_per_stream": frames / 12.5, "cfg": 1.5,
            "temperature": 1.0, "reference_frames": len(self.reference), "caption": CAPTION,
            "gpu_isolated": True, "server_stopped": True,
            "gpu_processes": subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"], text=True).strip()}

    def save(self, suffix):
        OUT.mkdir(exist_ok=True)
        (OUT / f"{suffix}.json").write_text(json.dumps({"environment": self.environment,
            "notes": self.notes, "trials": self.records}, indent=2))

    def prepare(self, n, guidance):
        branches = [self.engine.pack(self.spec["prompt"], self.spec, self.reference, n)]
        if guidance != 1:
            branches.append(self.engine.pack(self.spec["unconditional"], self.spec, self.reference, n))
        return branches

    @torch.inference_mode()
    def trial(self, n=1, variant="legacy", streaming=False, chunk_frames=10,
              guidance=1.5, seed=777, fixed=True, frames=None, record=True):
        frames = frames or self.frames
        branches = self.prepare(n, guidance)
        params = {"max_new_tokens": frames if fixed else frames + 60,
            "audio_temperature": 1., "audio_top_k": 50, "audio_top_p": .95,
            "audio_repetition_penalty": 1., "text_temperature": 1., "text_top_k": 50, "text_top_p": 1.}
        torch.cuda.synchronize(self.engine.device)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.reset_peak_memory_stats(self.engine.device)
        start, end, token_end = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        chunk_events = []
        outputs = []
        count = 0
        generated = []
        capture_before = self.fast.capture_seconds
        codec = self.engine.proc.audio_tokenizer
        context = StreamingCodec(codec, n, self.engine.model.config.audio_pad_token_id) if streaming else nullcontext()
        wall_start = time.perf_counter()
        start.record()
        with torch.autocast("cuda", dtype=torch.bfloat16), context as decoder:
            if variant == "legacy":
                iterator = self.engine._stream_frames_cfg(branches, params,
                    chunk_frames=chunk_frames if streaming else params["max_new_tokens"] + 1,
                    min_frames=frames if fixed else 0, guidance=guidance)
            else:
                iterator = self.fast.stream_frames(branches, params,
                    chunk_frames=chunk_frames if streaming else params["max_new_tokens"] + 1,
                    min_frames=frames if fixed else 0, guidance=guidance, graph=variant == "graph")
            for generated in iterator:
                if len(generated) <= count:
                    continue
                new_codes = torch.stack(generated[count:], dim=1)
                count = len(generated)
                if streaming:
                    pcm, lengths = decoder.decode(new_codes)
                    outputs.append((pcm, lengths))
                    event = torch.cuda.Event(enable_timing=True)
                    event.record()
                    chunk_events.append((event, count))
            token_end.record()
            codes = torch.stack(generated, dim=1)
            if not streaming:
                # Legacy really decodes each candidate separately. Optimized
                # path decodes all rows in one codec batch, never silently B=1.
                groups = [codes[i:i+1] for i in range(n)] if variant == "legacy" else [codes]
                for group in groups:
                    valid = group[:, :, 0].ne(self.engine.model.config.audio_pad_token_id)
                    result = codec.decode(group.masked_fill(~valid.unsqueeze(-1), 0).permute(2, 0, 1).contiguous(),
                        padding_mask=valid, num_quantizers=12, return_dict=True, chunk_duration=8)
                    outputs.append((result.audio.float().mean(1), result.audio_lengths))
        end.record()
        end.synchronize()
        wall_seconds = time.perf_counter() - wall_start
        device_seconds = start.elapsed_time(end) / 1000
        token_seconds = start.elapsed_time(token_end) / 1000
        valid_counts = codes[:, :, 0].ne(self.engine.model.config.audio_pad_token_id).sum(-1).cpu().tolist()
        durations = [c / 12.5 for c in valid_counts]
        timestamps = [(start.elapsed_time(event)/1000, position) for event, position in chunk_events]
        pcm_valid = all(torch.isfinite(audio).all().item() for audio, _ in outputs)
        assert pcm_valid
        # Save only synthetic test audio, never the user's uploads.
        digest = hashlib.sha256(codes.cpu().numpy().tobytes()).hexdigest()
        result = {"variant": variant, "n": n, "streaming": streaming, "chunk_frames": chunk_frames if streaming else None,
            "guidance": guidance, "seed": seed, "fixed_length": fixed, "requested_frames": frames,
            "audio_seconds": durations, "aggregate_audio_seconds": sum(durations),
            "device_seconds": device_seconds, "model_wall_seconds": wall_seconds,
            "device_rtf": device_seconds / sum(durations),
            "per_stream_rtf": device_seconds / statistics.mean(durations),
            "audio_seconds_per_second": sum(durations) / device_seconds,
            "token_loop_seconds": token_seconds, "codec_seconds": device_seconds-token_seconds if not streaming else None,
            "first_audio_seconds": timestamps[0][0] if timestamps else device_seconds,
            "chunks": [{"device_seconds": t, "frame": f} for t, f in timestamps],
            "max_inter_chunk_seconds": max((timestamps[i][0] - timestamps[i-1][0] for i in range(1, len(timestamps))), default=None),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
            "graph_capture_seconds": self.fast.capture_seconds - capture_before,
            "codes_sha256": digest, "status": "ok"}
        if record:
            self.records.append(result)
        print(json.dumps({k: result[k] for k in ("variant", "n", "streaming", "device_seconds", "device_rtf", "per_stream_rtf", "first_audio_seconds", "peak_allocated_gib", "graph_capture_seconds")}), flush=True)
        return result, codes, outputs

    def warm(self, n, variant, streaming, guidance=1.5):
        self.trial(n, variant, streaming, guidance=guidance, frames=20, record=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["baseline", "optimize", "compile", "sweep", "validate", "latency", "long"], default="baseline")
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--compile-global", action="store_true")
    args = parser.parse_args()
    bench = Benchmark(args.frames)
    if args.compile_global:
        from torch import _dynamo
        _dynamo.config.cache_size_limit = 64
        started = time.monotonic()
        bench.engine.model.transformer.forward = torch.compile(
            bench.engine.model.transformer.forward, dynamic=True, mode="default")
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        bench.warm(1, "graph", False)
        bench.notes.append({"global_compile_enabled": True, "compile_warm_seconds": time.monotonic() - started})
    if args.phase == "baseline":
        for streaming, n in [(False, 1), (True, 1), (True, 4)]:
            bench.warm(n, "legacy", streaming)
            for repeat in range(args.repeats):
                bench.trial(n, "legacy", streaming, seed=777+repeat)
                bench.save("baseline")
    elif args.phase == "optimize":
        for variant in ["legacy", "fused", "graph"]:
            bench.warm(1, variant, False)
            for repeat in range(args.repeats):
                bench.trial(1, variant, False, seed=777+repeat)
                bench.save("optimizations")
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        bench.notes.append("Codec encoder/decoder weights BF16, quantizer unchanged FP32, using codec's supported setter")
        bench.warm(1, "graph", False)
        for repeat in range(args.repeats):
            result, _, _ = bench.trial(1, "graph", False, seed=777+repeat)
            result["variant"] = "graph_codec_bf16"
            bench.save("optimizations")
    elif args.phase == "compile":
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        bench.warm(1, "graph", False)
        for repeat in range(args.repeats):
            bench.trial(1, "graph", False, seed=777+repeat)
            bench.save("compile")
        started = time.monotonic()
        bench.engine.model.transformer.forward = torch.compile(
            bench.engine.model.transformer.forward, dynamic=True, mode="default")
        bench.warm(1, "graph", False)
        bench.notes.append({"global_compile_cold_seconds": time.monotonic() - started})
        for repeat in range(args.repeats):
            result, _, _ = bench.trial(1, "graph", False, seed=777+repeat)
            result["variant"] = "graph_global_compile"
            bench.save("compile")
    elif args.phase == "sweep":
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        for n in list(range(1, 11)) + [12, 16, 24, 32, 48, 64]:
            try:
                bench.fast.clear_graphs()
                gc.collect()
                torch.cuda.empty_cache()
                bench.warm(n, "graph", True)
                for repeat in range(args.repeats if n <= 10 else 1):
                    bench.trial(n, "graph", True, seed=777+repeat)
                    bench.save("sweep")
            except torch.cuda.OutOfMemoryError:
                bench.records.append({"variant": "graph", "n": n, "streaming": True, "status": "oom"})
                bench.save("sweep")
                bench.fast.clear_graphs()
                gc.collect()
                torch.cuda.empty_cache()
                break
        for guidance in [1.5, 1.0]:
            for streaming in [False, True]:
                bench.warm(1, "graph", streaming, guidance)
                for repeat in range(args.repeats):
                    bench.trial(1, "graph", streaming, guidance=guidance, seed=777+repeat)
                    bench.save("sweep")
    elif args.phase == "latency":
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        for n in [1, 10]:
            for chunk in [1, 2, 5, 10, 20]:
                bench.trial(n, "graph", True, chunk_frames=chunk, frames=20, record=False)
                for repeat in range(args.repeats):
                    bench.trial(n, "graph", True, chunk_frames=chunk, seed=777+repeat)
                    bench.save("latency")
    elif args.phase == "long":
        bench.engine.proc.audio_tokenizer.set_codec_weight_dtype("bf16")
        for n in [1, 12]:
            bench.warm(n, "graph", True)
            for repeat in range(args.repeats):
                bench.trial(n, "graph", True, seed=777+repeat)
                bench.save("long")
    elif args.phase == "validate":
        comparisons = []
        for n in [1, 4]:
            for variant in ["legacy", "fused", "graph"]:
                bench.warm(n, variant, False)
                result, codes, outputs = bench.trial(n, variant, False, frames=30)
                comparisons.append((variant, n, codes.cpu(), outputs[0][0].cpu()))
        # Full equality is not assumed after a different GEMM batch shape;
        # graph/non-graph identical math must preserve frame-local results.
        bench.notes.append("Code hashes and recorded natural takes permit numerical/quality review; batch GEMM rounding can change stochastic trajectories")
        bench.save("validation")


if __name__ == "__main__":
    main()
