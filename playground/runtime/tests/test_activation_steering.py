"""CPU unit tests: exact no-op, masks, CFG ordering and hook lifecycle."""
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from activation_steering import ActivationCollector, LayerIntervention, add_vector, research_stream_frames
from performance_core import FastGenerator


class Block(nn.Module):
    def forward(self, hidden_states, **kwargs):
        return hidden_states + 1, "unchanged cache"


class Transformer(nn.Module):
    def __init__(self, depth=3):
        super().__init__()
        self.layers = nn.ModuleList(Block() for _ in range(depth))
        self.recorded_inputs = []

    def forward(self, inputs_embeds, **kwargs):
        self.recorded_inputs.append(inputs_embeds.detach().clone())
        hidden = inputs_embeds
        for layer in self.layers:
            hidden, _ = layer(hidden)
        return SimpleNamespace(last_hidden_state=hidden, past_key_values="cache")


def test_collector_counts_only_actual_frames_and_conditional_values():
    c = ActivationCollector(2)
    c.stage("layer_0", torch.tensor([[1., 2.], [100., 200.]]))
    c.stage("semantic", torch.tensor([[3., 4., 5.], [30., 40., 50.]]))
    c.commit(torch.tensor([True, False]))
    c.stage("layer_0", torch.tensor([[3., 4.], [7., 8.]]))
    c.commit(torch.tensor([True, True]))
    c.stage("layer_0", torch.full((2, 2), float("nan")))
    c.commit(torch.tensor([False, False]))
    result = c.export()
    assert result["counts"].tolist() == [2, 1]
    assert result["means"]["layer_0"].tolist() == [[2., 3.], [7., 8.]]
    assert result["point_counts"]["semantic"].tolist() == [1, 0]
    assert result["means"]["semantic"].tolist() == [[3., 4., 5.], [0., 0., 0.]]
    assert not c.pending


def test_duplicate_stage_and_uncommitted_export_fail_closed():
    c = ActivationCollector(1)
    c.stage("layer_0", torch.ones(1, 3))
    with pytest.raises(RuntimeError):
        c.stage("layer_0", torch.zeros(1, 3))
    with pytest.raises(RuntimeError):
        c.export()
    c.discard_pending()
    assert c.export()["counts"].tolist() == [0]


def test_capture_path_does_not_copy_to_cpu_or_read_scalar(monkeypatch):
    c = ActivationCollector(2)
    values = torch.ones(2, 4, dtype=torch.bfloat16)
    valid = torch.tensor([True, False])
    def forbid(*args, **kwargs):
        raise AssertionError("per-token host synchronization")
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", forbid)
        patch.setattr(torch.Tensor, "item", forbid)
        c.stage("layer_0", values)
        c.commit(valid)
    assert c.export()["counts"].tolist() == [1, 0]


def test_export_is_an_immutable_snapshot_even_for_cpu_collection():
    c = ActivationCollector(1)
    c.stage("layer_0", torch.ones(1, 2)); c.commit(torch.tensor([True]))
    report = c.export()
    c.stage("layer_0", torch.full((1, 2), 3.)); c.commit(torch.tensor([True]))
    assert report["counts"].tolist() == [1]
    assert report["sums"]["layer_0"].tolist() == [[1., 1.]]


@pytest.mark.parametrize("layers", [(0,), (0, 1), (0, 1, 2)])
def test_one_two_three_layer_combinations_and_per_layer_scaling(layers):
    model = Transformer(3)
    values = torch.ones(1, 2, 2)
    baseline = model(values).last_hidden_state
    factors = {index: .25 * (index + 1) for index in layers}
    with LayerIntervention(model, 1, 1, vectors={i: torch.ones(2) for i in layers}, factors=factors) as ctx:
        ctx.active = torch.tensor([True])
        out = model(values).last_hidden_state
    assert torch.equal(out[:, :-1], baseline[:, :-1])
    torch.testing.assert_close(out[:, -1] - baseline[:, -1], torch.full((1, 2), sum(factors.values())))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("mode", ["raw", "rms_relative"])
def test_zero_factor_is_exact_object_identity(dtype, mode):
    values = torch.randn(4, 8).to(dtype)
    assert add_vector(values, torch.randn(8), 0., mode) is values


def test_rms_relative_factor_has_declared_added_rms():
    values = torch.tensor([[1., 2., 3.], [4., 5., 6.]])
    out = add_vector(values, torch.tensor([2., -1., 0.]), .25, "rms_relative")
    ratio = (out - values).square().mean(-1).sqrt() / values.square().mean(-1).sqrt()
    torch.testing.assert_close(ratio, torch.full((2,), .25))


def test_rms_relative_layer_addition_is_identical_for_cfg_partners():
    model = Transformer(1)
    values = torch.tensor([[[1., 2.]], [[10., 20.]]])
    baseline = model(values).last_hidden_state
    with LayerIntervention(model, 1, 2, vectors={0: torch.tensor([1., 2.])}, factors=.25,
                           normalization="rms_relative") as ctx:
        ctx.active = torch.tensor([True])
        out = model(values).last_hidden_state
    deltas = out[:, -1] - baseline[:, -1]
    torch.testing.assert_close(deltas[0], deltas[1], rtol=1e-5, atol=1e-6)


def test_hooks_only_last_position_conditional_capture_both_cfg_steering():
    model = Transformer(2)
    values = torch.arange(24.).reshape(4, 3, 2)
    c = ActivationCollector(2)
    baseline = model(values).last_hidden_state
    with LayerIntervention(model, 2, 2, c, {0: torch.tensor([10., 20.])}, .5) as ctx:
        # Unrelated forwards are inert until loop explicitly marks predictor.
        assert torch.equal(model(values).last_hidden_state, baseline)
        ctx.active = torch.tensor([True, False])
        result = model(values).last_hidden_state
        c.commit(torch.tensor([True, False]))
        assert torch.equal(result[:, :-1], baseline[:, :-1])
        delta = result[:, -1] - baseline[:, -1]
        assert delta.tolist() == [[5., 10.], [0., 0.], [5., 10.], [0., 0.]]
    means = c.export()["means"]
    assert means["layer_0"][0].tolist() == (values[0, -1] + 1).tolist()
    assert means["layer_1"][0].tolist() == (values[0, -1] + 2 + torch.tensor([5., 10.])).tolist()
    assert means["layer_0"][1].tolist() == [0., 0.]
    assert all(not layer._forward_hooks for layer in model.layers)


def test_cleanup_on_exception_and_compiled_transformer_rejection():
    model = Transformer()
    c = ActivationCollector(1)
    with pytest.raises(RuntimeError):
        with LayerIntervention(model, 1, 1, c) as ctx:
            ctx.active = torch.tensor([True])
            model(torch.ones(1, 2, 4))
            raise RuntimeError("generation failed before sampled stop")
    assert not c.pending
    assert all(not layer._forward_hooks for layer in model.layers)
    model._orig_mod = object()
    with pytest.raises(ValueError, match="eager"):
        with LayerIntervention(model, 1, 1):
            pass


class FakeModel:
    def __init__(self):
        self.config = SimpleNamespace(pad_token_id=0, audio_pad_token_id=-1,
            audio_assistant_slot_token_id=5, audio_end_token_id=6)
        self.transformer = Transformer(2)
        self.local_transformer = SimpleNamespace(ln_f=nn.LayerNorm(2))

    def _build_inputs_embeds(self, ids):
        return ids[..., :2].float()

    def _global_hidden_to_local(self, hidden):
        return hidden

    def _build_generation_row(self, n, device, frame):
        return torch.cat((torch.full((n, 1, 1), 5, device=device), frame[:, None]), -1)


class FakeFast:
    def __init__(self):
        self.calls = 0
        self.inputs = []

    def frame(self, local_input, n, nb, guidance, p, graph=False):
        self.inputs.append(local_input.clone())
        # Row0: two frames then EOS; row1: one frame then EOS, then an
        # intentionally invalid revival code which must remain masked.
        tokens = ([5, 5], [5, 6], [6, 5])[self.calls]
        self.calls += 1
        return torch.tensor(tokens), torch.full((n, 2), self.calls)


def fake_engine():
    model = FakeModel()
    return SimpleNamespace(model=model, fast=FakeFast())


def branches():
    return [(torch.tensor([[[8, 3, 4], [9, 4, 5]], [[12, 7, 8], [13, 8, 9]]]), torch.ones(2, 2, dtype=torch.bool)),
            (torch.tensor([[[100, 20, 21]], [[200, 30, 31]]]), torch.ones(2, 1, dtype=torch.bool))]


def exhaust(iterator):
    frames = []
    for frames in iterator:
        pass
    return torch.stack(frames, 1)


def test_research_loop_zero_factor_exact_production_parity_and_no_prompt_eos_tails():
    p = {"max_new_tokens": 8, "audio_repetition_penalty": 1.}
    baseline = fake_engine()
    reference = exhaust(FastGenerator.stream_frames(SimpleNamespace(engine=baseline, frame=baseline.fast.frame),
        branches(), p, chunk_frames=2, guidance=2.))
    engine = fake_engine()
    c = ActivationCollector(2)
    out = exhaust(research_stream_frames(engine, branches(), p, 2, collector=c,
        layer_vectors={0: torch.ones(2)}, factors=0.))
    assert torch.equal(out, reference)
    assert out[1, 1].tolist() == [-1, -1]
    assert all(torch.equal(left, right) for left, right in zip(engine.fast.inputs, baseline.fast.inputs))
    report = c.export()
    assert report["counts"].tolist() == [2, 1]
    # Last prefill position predicts first frame; no other prompt positions.
    assert report["means"]["layer_0"][1].tolist() == [14., 9.]
    assert report["means"]["layer_0"][0].tolist() == [8., 3.5]
    assert all(not layer._forward_hooks for layer in engine.model.transformer.layers)


def test_optional_frame_zero_exclusion_and_semantic_callback_shared_commit():
    engine = fake_engine()
    c = ActivationCollector(2)
    class Semantic:
        def __init__(self): self.active_masks = []
        def set_active(self, mask): self.active_masks.append(mask.clone())
        def __call__(self, point, values):
            return values
    semantic = Semantic()
    exhaust(research_stream_frames(engine, branches(), {"max_new_tokens": 8, "audio_repetition_penalty": 1.}, 2,
        collector=c, include_first_prediction=False, semantic_transform=semantic))
    assert c.export()["counts"].tolist() == [1, 0]
    assert c.export()["point_counts"]["semantic_global"].tolist() == [1, 0]
    assert c.export()["point_counts"]["semantic_local"].tolist() == [1, 0]
    assert [mask.tolist() for mask in semantic.active_masks] == [[True, True], [True, True], [True, False]]


def test_abandoned_generation_cleans_hooks_and_pending_capture():
    engine = fake_engine()
    c = ActivationCollector(2)
    iterator = research_stream_frames(engine, branches(), {"max_new_tokens": 8, "audio_repetition_penalty": 1.}, 1, collector=c)
    next(iterator)
    assert any(layer._forward_hooks for layer in engine.model.transformer.layers)
    iterator.close()
    assert all(not layer._forward_hooks for layer in engine.model.transformer.layers)
    assert not c.pending
