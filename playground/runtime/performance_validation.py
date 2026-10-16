"""Numerical tests on the real pinned CUDA model and codec."""
import json
import torch
from throughput_benchmark import Benchmark, OUT
from performance_core import StreamingCodec


def comparison(a, b):
    assert a.shape == b.shape, (a.shape, b.shape)
    diff = (a.float() - b.float()).flatten()
    return {"samples": a.numel(), "rms_difference": float(diff.square().mean().sqrt()),
            "maximum_difference": float(diff.abs().max()), "rms_signal": float(a.float().square().mean().sqrt())}


@torch.inference_mode()
def main():
    bench = Benchmark(30)
    result = {"environment": bench.environment, "checks": []}
    for n in [1, 4]:
        _, eager, _ = bench.trial(n, "fused", False, frames=30)
        _, graph, _ = bench.trial(n, "graph", False, frames=30)
        assert torch.equal(eager, graph), "Graph changed fused token sequence"
        result["checks"].append({"check": "fused_eager_vs_cuda_graph_codes", "batch": n, "exact": True})
    codec = bench.engine.proc.audio_tokenizer
    # Deliberately terminate rows at different lengths, including a row which
    # finishes in the middle of an incremental chunk.
    lengths = [30, 19, 7, 25]
    valid = torch.arange(30, device=graph.device)[None] < torch.tensor(lengths, device=graph.device)[:, None]
    pad = bench.engine.model.config.audio_pad_token_id
    codes = graph.masked_fill(~valid[..., None], pad)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        full = codec.decode(codes.masked_fill(~valid[..., None], 0).permute(2, 0, 1).contiguous(),
            padding_mask=valid, return_dict=True, chunk_duration=8)
        baseline = full.audio.float().mean(1).clone()
        assert full.audio_lengths.tolist() == [l * 3840 for l in lengths]
        pieces = [[] for _ in lengths]
        with StreamingCodec(codec, 4, pad) as stream:
            for offset in range(0, 30, 10):
                pcm, size = stream.decode(codes[:, offset:offset+10])
                for i in range(4):
                    pieces[i].append(pcm[i, :int(size[i])].clone())
        for i, length in enumerate(lengths):
            a = baseline[i, :length*3840]
            b = torch.cat(pieces[i])
            check = comparison(a, b)
            result["checks"].append({"check": "incremental_vs_full_pcm", "row": i, "frames": length, **check})
            # Different convolution shapes can cause BF16 rounding; require a
            # small absolute signal error, not a false bit-identical claim.
            assert check["rms_difference"] < .005
        codec.set_codec_weight_dtype("bf16")
        low = codec.decode(codes.masked_fill(~valid[..., None], 0).permute(2, 0, 1).contiguous(),
            padding_mask=valid, return_dict=True, chunk_duration=8).audio.float().mean(1)
        check = comparison(baseline, low)
        result["checks"].append({"check": "codec_fp32_weights_vs_supported_bf16", **check})
        assert check["rms_difference"] < .005
    # Compilation must not be assumed bit-exact after fused pointwise kernels.
    from torch import _dynamo
    _dynamo.config.cache_size_limit = 64
    bench.engine.model.transformer.forward = torch.compile(bench.engine.model.transformer.forward, dynamic=True)
    _, compiled, _ = bench.trial(4, "graph", False, frames=30)
    result["checks"].append({"check": "compiled_global_vs_eager_codes", "exact": bool(torch.equal(graph, compiled)),
        "same_token_fraction": float(graph.eq(compiled).float().mean()),
        "note": "BF16 rounding can change stochastic trajectories; quality is evaluated separately."})
    (OUT / "validation.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result["checks"], indent=2), flush=True)


if __name__ == "__main__":
    main()
