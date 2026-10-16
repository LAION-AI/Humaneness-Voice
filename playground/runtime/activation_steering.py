"""Standalone, eager-only activation-steering research utilities.

Nothing in this module changes the live server. Global block vectors refer to
the OUTPUT of Qwen decoder blocks, before the final global RMSNorm. The global
block width is 1024 in the pinned Small export; the projected semantic vector
fed into the local decoder has width 2560. Acoustic output is 12 codes/frame.

Capture the final *predictor* position only, then commit after the binary stop
head identifies a real acoustic frame. The final prefill position predicts
frame zero and is included; the other prompt/reference positions are never
captured or steered. EOS, already-finished rows and padded batch tails are not
averaged. CFG batches are branch-major: conditional rows first, then neutral.
Collection is conditional-only; default steering is applied to both branches.

Raw mean addition is deliberately distinguished from RMS-relative addition:
raw alpha multiplies the saved activation vector; RMS-relative alpha expresses
added-vector RMS as a fraction of the current row's activation RMS. Neither
is guaranteed to encode speaker identity rather than content/style.
"""
from __future__ import annotations

from contextlib import AbstractContextManager
from collections.abc import Mapping
from typing import Callable

import torch

from performance_core import combine_branches


class ActivationCollector:
    """GPU-resident token sums/counts with one compact CPU export per batch.

``stage(point, values)`` takes conditional-row values of shape [N, H].
``commit(continues)`` accepts the exact actual-frame mask [N] from the stop head.
No .item(), CPU copy, or device synchronization occurs during either operation.
The caller scores completed audio and selects high-similarity rows afterwards.
"""

    def __init__(self, batch_size: int):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.batch_size = batch_size
        self.pending: dict[str, torch.Tensor] = {}
        self.sums: dict[str, torch.Tensor] = {}
        self.point_counts: dict[str, torch.Tensor] = {}
        self.counts: torch.Tensor | None = None

    def stage(self, point: str, values: torch.Tensor):
        if values.ndim != 2 or values.shape[0] != self.batch_size:
            raise ValueError("stage requires conditional rows [batch_size, hidden]")
        point = str(point)
        if point in self.pending:
            raise RuntimeError(f"Prediction was not committed before staging {point}")
        # Clone is needed: a later in-place operation must not change the sample
        # measured by this hook. Usually the hooks themselves use out-of-place.
        self.pending[point] = values.detach().clone()

    def commit(self, valid: torch.Tensor):
        if valid.ndim != 1 or valid.shape[0] != self.batch_size:
            raise ValueError("commit requires actual-frame mask [batch_size]")
        if valid.dtype != torch.bool:
            raise ValueError("actual-frame mask must be bool")
        if self.counts is None:
            self.counts = torch.zeros(self.batch_size, dtype=torch.int64, device=valid.device)
        if self.counts.device != valid.device:
            raise ValueError("collector cannot span devices")
        self.counts += valid.to(torch.int64)
        for point, values in self.pending.items():
            if values.device != valid.device:
                raise ValueError("values and actual-frame mask must share a device")
            if point not in self.sums:
                self.sums[point] = torch.zeros_like(values, dtype=torch.float32)
                self.point_counts[point] = torch.zeros_like(self.counts)
            if self.sums[point].shape != values.shape:
                raise ValueError("activation point changed shape within one batch")
            self.sums[point] += values.float().masked_fill(~valid[:, None], 0)
            self.point_counts[point] += valid.to(torch.int64)
        self.pending.clear()

    def discard_pending(self):
        """Discard an uncommitted forward after a generation exception."""
        self.pending.clear()

    def export(self):
        if self.pending:
            raise RuntimeError("cannot export an uncommitted prediction")
        counts = torch.zeros(self.batch_size, dtype=torch.int64) if self.counts is None else self.counts.detach().cpu().clone()
        sums = {key: value.detach().float().cpu().clone() for key, value in self.sums.items()}
        point_counts = {key: value.detach().cpu().clone() for key, value in self.point_counts.items()}
        means = {key: value / point_counts[key].clamp_min(1)[:, None] for key, value in sums.items()}
        return {"counts": counts, "point_counts": point_counts, "sums": sums, "means": means,
                "capture": "conditional final predictor position; actual acoustic frames only"}


def add_vector(values: torch.Tensor, vector: torch.Tensor, factor: float,
               normalization: str = "raw", active: torch.Tensor | None = None,
               rms_reference_rows: int | None = None):
    """Add a saved vector to [rows, hidden], preserving exact alpha=0 parity."""
    if normalization not in {"raw", "rms_relative"}:
        raise ValueError("normalization must be raw or rms_relative")
    if values.ndim != 2 or vector.ndim != 1 or vector.shape[0] != values.shape[-1]:
        raise ValueError("vector width must match [rows, hidden] values")
    if active is not None and (active.dtype != torch.bool or active.shape != values.shape[:1]):
        raise ValueError("active mask must be bool [rows]")
    if factor == 0:
        return values
    vector = vector.to(device=values.device, dtype=torch.float32)
    if normalization == "rms_relative":
        vector_rms = vector.square().mean().sqrt().clamp_min(1e-12)
        if rms_reference_rows is not None:
            if rms_reference_rows < 1 or values.shape[0] % rms_reference_rows:
                raise ValueError("RMS reference rows must evenly divide the CFG batch")
            # Conditional RMS is shared with its neutral CFG partner: do not
            # introduce an extra branch-asymmetric guidance perturbation.
            row_rms = values[:rms_reference_rows].float().square().mean(-1, keepdim=True).sqrt()
            row_rms = row_rms.repeat(values.shape[0] // rms_reference_rows, 1)
        else:
            row_rms = values.float().square().mean(-1, keepdim=True).sqrt()
        delta = vector[None, :] * (float(factor) * row_rms / vector_rms)
    else:
        delta = vector[None, :] * float(factor)
    if active is not None:
        delta = delta * active[:, None]
    return values + delta.to(values.dtype)


def assert_eager_transformer(transformer):
    """Fail closed rather than mixing hooks with torch.compile/CUDA capture."""
    forward = transformer.forward
    if (hasattr(transformer, "_orig_mod") or
            hasattr(forward, "_torchdynamo_orig_callable") or
            hasattr(forward, "_orig_mod")):
        raise ValueError("activation research requires an eager global transformer (COMPILE_GLOBAL=False)")


class LayerIntervention(AbstractContextManager):
    """Short-lived global block hooks, with deterministic cleanup.

``vectors`` maps zero-based layer indices to [H] tensors. Capture points are
``layer_0``, ... . ``collect_layers=None`` captures every global layer. Vector
factors may be a scalar or mapping layer index -> scalar. State is staged
*before* injection, so collected observations have a precise intervention
point definition. All hooks act only while ``active`` is set by the research
generation loop; a direct external/prefill forward otherwise remains unchanged.
"""

    def __init__(self, transformer, batch_size: int, branches: int = 2,
                 collector: ActivationCollector | None = None,
                 vectors: Mapping[int, torch.Tensor] | None = None,
                 factors: float | Mapping[int, float] = 0.,
                 normalization: str = "raw", collect_layers=None,
                 steer_branches: str = "both"):
        if branches not in {1, 2}:
            raise ValueError("supported CFG branches: one conditional, or conditional + neutral")
        if steer_branches not in {"both", "conditional"}:
            raise ValueError("steer_branches must be both or conditional")
        if normalization not in {"raw", "rms_relative"}:
            raise ValueError("normalization must be raw or rms_relative")
        if batch_size < 1 or (collector is not None and collector.batch_size != batch_size):
            raise ValueError("collector and intervention batch sizes must agree")
        self.transformer, self.batch_size, self.branches = transformer, batch_size, branches
        self.collector = collector
        self.vectors = {int(key): value for key, value in (vectors or {}).items()}
        self.factors, self.normalization = factors, normalization
        self.collect_layers = set(range(len(transformer.layers))) if collect_layers is None else set(collect_layers)
        if any(i < 0 or i >= len(transformer.layers) for i in set(self.vectors) | self.collect_layers):
            raise ValueError("layer index outside global transformer")
        self.steer_branches = steer_branches
        self.active: torch.Tensor | None = None
        self.handles = []

    def __enter__(self):
        assert_eager_transformer(self.transformer)
        if self.handles:
            raise RuntimeError("intervention cannot be entered twice")
        try:
            for index in sorted(set(self.vectors) | (self.collect_layers if self.collector is not None else set())):
                self.handles.append(self.transformer.layers[index].register_forward_hook(self._hook(index)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def _hook(self, index):
        def hook(module, args, output):
            if self.active is None:
                return output
            hidden = output[0] if isinstance(output, (tuple, list)) else output
            expected_rows = self.batch_size * self.branches
            if hidden.ndim != 3 or hidden.shape[0] != expected_rows:
                raise ValueError("global hook expects branch-major [N * branches, T, H]")
            last = hidden[:, -1]
            if self.collector is not None and index in self.collect_layers:
                self.collector.stage(f"layer_{index}", last[:self.batch_size])
            if index not in self.vectors:
                return output
            factor = self.factors[index] if isinstance(self.factors, Mapping) else self.factors
            if float(factor) == 0.:
                return output
            mask = self.active.repeat(self.branches)
            if self.steer_branches == "conditional" and self.branches == 2:
                mask = torch.cat((self.active, torch.zeros_like(self.active)))
            last = add_vector(last, self.vectors[index], float(factor), self.normalization, mask,
                              rms_reference_rows=self.batch_size)
            new_hidden = torch.cat((hidden[:, :-1], last[:, None]), dim=1)
            if isinstance(output, tuple):
                return (new_hidden, *output[1:])
            if isinstance(output, list):
                return [new_hidden, *output[1:]]
            return new_hidden
        return hook

    def __exit__(self, *args):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.active = None
        if self.collector is not None:
            self.collector.discard_pending()
        return False


@torch.inference_mode()
def research_stream_frames(engine, branches, p, chunk_frames, guidance=2.,
                           collector: ActivationCollector | None = None,
                           layer_vectors=None, factors=0., normalization="raw",
                           collect_layers=None, semantic_transform: Callable | None = None,
                           graph=False, include_first_prediction=True, min_frames=0):
    """Isolated production-equivalent fused-CFG loop plus scoped hooks.

    ``semantic_transform(point, values)`` optionally modifies ``semantic_global``
    (post-final-norm) or ``semantic_local`` (post-projection) before ``fast.frame``.
    Both points are automatically staged on the SAME collector, before their
    respective intervention, then committed with the actual-frame mask. Local
CUDA graphs are safe when the callback occurs outside their capture; global
transformer must remain eager. The callback itself is never graph-captured.
``include_first_prediction=False`` excludes the frame-zero predictor from
collection only; it does not change generation or steering.

Yields cumulative lists of [N, 12] frames, as FastGenerator.stream_frames does.
"""
    mdl, cfg = engine.model, engine.model.config
    assert_eager_transformer(mdl.transformer)
    if p["audio_repetition_penalty"] != 1.:
        raise ValueError("research fast path supports unchanged repetition penalty 1 only")
    nb, n = len(branches), branches[0][0].shape[0]
    if nb not in {1, 2} or any(ids.shape[0] != n for ids, _ in branches):
        raise ValueError("branches must have equal batch size and be conditional-first")
    if chunk_frames < 1:
        raise ValueError("chunk_frames must be positive")
    ids, initial_mask = combine_branches(branches, cfg.pad_token_id, cfg.audio_pad_token_id)
    budget, length = int(p["max_new_tokens"]), ids.shape[1]
    mask_buffer = torch.zeros(n * nb, length + budget, dtype=torch.bool, device=ids.device)
    mask_buffer[:, :length] = initial_mask
    frames, past, last = [], None, 0
    finished = torch.zeros(n, dtype=torch.bool, device=ids.device)
    model_input = ids
    if semantic_transform is not None and hasattr(semantic_transform, "set_batch"):
        semantic_transform.set_batch(n, nb)
    with LayerIntervention(mdl.transformer, n, nb, collector, layer_vectors, factors,
                           normalization, collect_layers) as intervention:
        for step in range(budget):
            intervention.active = ~finished
            if semantic_transform is not None and hasattr(semantic_transform, "set_active"):
                semantic_transform.set_active(~finished)
            mask = mask_buffer[:, :length + step]
            embeddings = mdl._build_inputs_embeds(model_input)
            out = mdl.transformer(inputs_embeds=embeddings, input_ids=None,
                attention_mask=mask, past_key_values=past, use_cache=True,
                position_ids=None, output_attentions=False, output_hidden_states=False,
                return_dict=True, cu_seqlens=None, num_sequences=None)
            past = out.past_key_values
            global_input = out.last_hidden_state[:, -1]
            if collector is not None:
                collector.stage("semantic_global", global_input[:n])
            if semantic_transform is not None:
                global_input = semantic_transform("semantic_global", global_input)
            local_input = mdl._global_hidden_to_local(global_input).to(mdl.local_transformer.ln_f.weight.dtype)
            if collector is not None:
                collector.stage("semantic_local", local_input[:n])
            if semantic_transform is not None:
                local_input = semantic_transform("semantic_local", local_input)
            token, sampled = engine.fast.frame(local_input, n, nb, guidance, p, graph=graph)
            continues = token.eq(cfg.audio_assistant_slot_token_id) & ~finished
            finished = finished | token.eq(cfg.audio_end_token_id)
            if step < min_frames:
                continues = torch.ones_like(continues)
                finished = torch.zeros_like(finished)
            if collector is not None:
                collector.commit(continues if step or include_first_prediction else torch.zeros_like(continues))
            if not bool(continues.any().item()):
                break
            frame = sampled.masked_fill(~continues[:, None], cfg.audio_pad_token_id)
            frames.append(frame)
            row = mdl._build_generation_row(n, ids.device, frame)
            row[:, 0, 0] = torch.where(continues, row[:, 0, 0], cfg.pad_token_id)
            model_input = row.repeat(nb, 1, 1)
            mask_buffer[:, length + step] = continues.repeat(nb)
            if len(frames) - last >= chunk_frames:
                last = len(frames)
                yield frames
        if len(frames) != last:
            yield frames
