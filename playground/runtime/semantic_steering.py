"""Research-only semantic-frame steering; never installed in the live demo.

Two different sites are deliberately named: ``semantic_global`` is the 1024-D
post-final-RMSNorm Qwen state; ``semantic_local`` is the 2560-D ``proj_out``
output that seeds the local acoustic decoder. They are not acoustic tokens.
Use SemanticVectorTransform as ``semantic_transform`` in activation_steering's
eager research loop. That loop captures and commits states after stop prediction, only for rows
whose audio frame is emitted. No prompt sequence, EOS, or finished-row filler
is counted; the final prefill position is counted if it predicts frame zero.
"""
from __future__ import annotations

import json
import math

import torch


SEMANTIC_POINTS = ("semantic_global", "semantic_local")
RAW_ALPHAS = (.25, .5, .75, 1.)


def mean_direction(sample_means, selected, *, subtract_selected=None):
    """Equal-sample FP32 mean, optionally high-minus-low; never token weighted.

    Inputs are [samples, hidden] means already excluding invalid frames. Select
    using *collection* scores only, not later search/holdout results. Empty
    groups raise instead of silently lowering a speaker-similarity threshold.
    """
    means = torch.as_tensor(sample_means, dtype=torch.float32)
    if means.ndim != 2 or not means.shape[1] or not torch.isfinite(means).all():
        raise ValueError("sample_means must be a finite [samples, hidden] matrix")

    def average(mask):
        mask = torch.as_tensor(mask, dtype=torch.bool, device=means.device)
        if mask.shape != (means.shape[0],) or not bool(mask.any()):
            raise ValueError("A selection needs at least one valid collection sample")
        return means[mask].mean(0)

    high = average(selected)
    if subtract_selected is None:
        return high
    high_mask = torch.as_tensor(selected, dtype=torch.bool, device=means.device)
    low_mask = torch.as_tensor(subtract_selected, dtype=torch.bool, device=means.device)
    if low_mask.shape != high_mask.shape or bool((high_mask & low_mask).any()):
        raise ValueError("Contrast groups must have equal-length, disjoint masks")
    return high - average(low_mask)


class SemanticVectorTransform:
    """Point-aware callback adding one vector to all active CFG branches.

    Capturing and committing only valid emitted predictions belongs to the
    eager research loop; this callback does not collect or register hooks. Raw
    addition is exactly the requested alpha * average activation. The optional
    ``rms_matched`` control gives alpha a relative-strength meaning by matching
    delta RMS to the current conditional frame RMS, using the *same* scale for
    its unconditional CFG partner. It is not the primary raw-mean experiment.

    Global addition is reprojected before seeding the local decoder, rather than
    confusing a 1024-D backbone state with its 2560-D projected representation.
    Neither intervention rewrites past KV states. Audio-frame additions can
    also alter the binary stop head and therefore utterance duration.
    """

    def __init__(self, vector=None, point="semantic_local", alpha=.25, normalization="raw"):
        if point not in SEMANTIC_POINTS:
            raise ValueError(f"Unknown semantic point: {point}")
        if normalization not in ("raw", "rms_matched", "rms_relative"):
            raise ValueError("normalization must be raw or rms_relative")
        if not math.isfinite(float(alpha)):
            raise ValueError("alpha must be finite")
        self.n, self.branches, self.active = None, None, None
        self.point, self.alpha, self.normalization = point, float(alpha), normalization
        self.vector = None if vector is None else torch.as_tensor(vector).detach().float().clone()
        if self.vector is not None and (self.vector.ndim != 1 or not self.vector.numel()
                                       or not torch.isfinite(self.vector).all()):
            raise ValueError("vector must be a finite nonempty 1-D activation")
        self._device_vectors = {}
        self.calls = 0
        self.intervention_calls = 0

    def set_batch(self, n, branches):
        if not isinstance(n, int) or n < 1 or branches not in (1, 2):
            raise ValueError("Expected positive conditional batch size and one or two CFG branches")
        self.n, self.branches = n, branches
        self.active = None

    def set_active(self, active):
        if self.n is None or active.shape != (self.n,) or active.dtype != torch.bool:
            raise ValueError("active must be bool [conditional batch size]")
        self.active = active

    def _delta(self, states):
        key = str(states.device)
        if key not in self._device_vectors:
            self._device_vectors[key] = self.vector.to(device=states.device, dtype=torch.float32)
        vector = self._device_vectors[key]
        if vector.numel() != states.shape[-1]:
            raise ValueError(f"{self.point} needs {states.shape[-1]} dimensions; got {vector.numel()}")
        if self.normalization == "raw":
            delta = self.alpha * vector.unsqueeze(0)
        else:
            # One conditional scale per candidate, shared with all CFG branches.
            rms = states[:self.n].float().square().mean(-1).sqrt()
            direction_rms = vector.square().mean().sqrt()
            scale = self.alpha * rms / direction_rms.clamp_min(1e-12)
            delta = scale.repeat(self.branches).unsqueeze(-1) * vector.unsqueeze(0)
        if self.active is not None:
            delta = delta * self.active.to(states.device).repeat(self.branches).unsqueeze(-1)
        return delta.to(states.dtype)

    def __call__(self, point, states):
        if point not in SEMANTIC_POINTS:
            raise ValueError(f"Unknown semantic point: {point}")
        if self.n is None:
            raise RuntimeError("Call set_batch before generation")
        if states.ndim != 2 or states.shape[0] != self.n * self.branches:
            raise ValueError("Semantic states must contain complete conditional-first CFG batches")
        self.calls += 1
        if point != self.point or self.vector is None or self.alpha == 0:
            # Required exact no-op control: no rounding, reprojecting, or RNG.
            return states
        self.intervention_calls += 1
        return states + self._delta(states)


def describe():
    return {
        "sites": {"semantic_global": "post-final-Qwen-RMSNorm, before proj_out (1024-D)",
                  "semantic_local": "proj_out output, acoustic decoder frame seed (2560-D)"},
        "primary": "semantic_local equal-sample raw high-similarity mean addition",
        "alphas": list(RAW_ALPHAS),
        "controls": ["exact alpha=0", "raw high-minus-low similarity contrast",
                     "RMS-matched contrast as exploratory control"],
        "selection": "collection-only cosine threshold; never percent identity probability",
        "execution": "research-only eager, no CUDA graphs and no live model changes",
        "counted": "conditional last-position states whose predicted audio frames are emitted",
        "excluded": "all other prefill positions, EOS, completed candidate filler",
        "limitations": ["Mean mixes content, prosody, identity and model baseline; not a proven speaker direction",
                        "Stop head and duration may change; report failures and untrimmed WER",
                        "Raw alpha magnitude differs from RMS-matched relative strength",
                        "No previous KV-state rewriting"],
    }


if __name__ == "__main__":
    print(json.dumps(describe(), indent=2))
