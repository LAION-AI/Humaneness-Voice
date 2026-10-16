"""Verified, request-scoped layer-16 speaker contrast steering.

Permanent dormant hooks are installed before the global forward is compiled.
Only tensor-buffer values change between requests; module identities, hooks,
parameters and local CUDA graphs remain unchanged. Only the final predictor
position is modified, including frame zero's final prefill predictor. The
global attention mask supplies active rows for fast, legacy and beam paths.

These contrasts are exploratory interventions, not proven speaker-only vectors.
No remote downloads and no fallback to a different speaker/reference vector.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

import settings


BANK = settings.HERE / "speaker-steering"
LAYER = 16
ALPHA = .75


def _sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class SpeakerSteeringState(nn.Module):
    """No parameters and no persisted buffers; stable identities for compile."""

    def __init__(self, hidden_size, max_rows=64):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.max_rows = int(max_rows)
        self.register_buffer("direction", torch.zeros(hidden_size, dtype=torch.float32), persistent=False)
        self.register_buffer("alpha", torch.tensor(ALPHA, dtype=torch.float32), persistent=False)
        self.register_buffer("enabled", torch.tensor(False, dtype=torch.bool), persistent=False)
        self.register_buffer("active", torch.zeros(max_rows, dtype=torch.bool), persistent=False)
        self.handles = []

    def install(self, transformer, layer=LAYER):
        if self.handles:
            raise RuntimeError("Speaker steering hooks must be installed only once")
        if hasattr(transformer.forward, "_torchdynamo_orig_callable"):
            raise ValueError("Install dormant speaker hooks before compiling global forward")
        if not 0 <= layer < len(transformer.layers):
            raise ValueError("Speaker steering layer does not exist")
        # Register state on the transformer so buffer references are tracked by
        # torch.compile, but excluded from released checkpoint/state_dict keys.
        transformer.add_module("_speaker_steering_state", self)
        self.handles.append(transformer.register_forward_pre_hook(self._before, with_kwargs=True))
        self.handles.append(transformer.layers[layer].register_forward_hook(self._after))
        return self

    def _before(self, module, args, kwargs):
        mask = kwargs.get("attention_mask")
        if mask is None or mask.ndim != 2:
            # Fail closed for unrelated training/packed calls lacking the
            # inference path's explicit 2D query-validity mask.
            self.active.zero_()
        else:
            rows = mask.shape[0]
            if rows > self.max_rows:
                raise ValueError("Speaker steering batch exceeds reserved active-row buffer")
            self.active[:rows].copy_(mask[:, -1].bool())

    def _after(self, module, args, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        if hidden.ndim != 3 or hidden.shape[-1] != self.hidden_size:
            raise ValueError("Speaker steering expects global [rows, positions, hidden]")
        rows = hidden.shape[0]
        if rows > self.max_rows:
            raise ValueError("Speaker steering batch exceeds reserved active-row buffer")
        last = hidden[:, -1]
        delta = (self.direction * self.alpha).to(dtype=hidden.dtype)
        mask = self.enabled & self.active[:rows]
        steered = torch.where(mask[:, None], last + delta[None], last)
        new_hidden = torch.cat((hidden[:, :-1], steered[:, None]), dim=1)
        if isinstance(output, tuple): return (new_hidden, *output[1:])
        if isinstance(output, list): return [new_hidden, *output[1:]]
        return new_hidden

    @torch.no_grad()
    def configure(self, vector, alpha=ALPHA):
        # Clear first; a failed validation must never retain a previous voice.
        self.enabled.fill_(False)
        self.active.zero_()
        self.direction.zero_()
        if vector is None:
            return
        vector = torch.as_tensor(vector).detach().float()
        if vector.shape != (self.hidden_size,) or not bool(torch.isfinite(vector).all()):
            raise ValueError("Speaker steering vector must be finite and match the global width")
        if not np.isfinite(alpha) or not 0 <= float(alpha) <= 1:
            raise ValueError("Speaker steering alpha must be finite and in [0,1]")
        self.direction.copy_(vector.to(self.direction.device))
        self.alpha.fill_(float(alpha))
        self.enabled.fill_(float(alpha) != 0)


def install_speaker_steering(model):
    state = SpeakerSteeringState(int(model.config.hidden_size)).to(next(model.parameters()).device)
    return state.install(model.transformer)


def _manifest(bank=BANK):
    path = Path(bank) / "manifest.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if data.get("version") != 1 or not isinstance(data.get("voices"), dict):
        return None
    return data


def _public_validation(value):
    if not isinstance(value, dict):
        return None
    allowed = {"passed", "eligible", "n", "failures", "heldout_positive_interval", "gate_reasons", "means", "matched"}
    return {key: value[key] for key in allowed if key in value}


def catalog_speaker_steering(bank=BANK):
    """Public statuses only: never expose file paths, selected IDs or tensors."""
    manifest = _manifest(bank)
    if manifest is None:
        return {"default_enabled": True, "available": False, "reason": "bank_unavailable", "voices": [],
                "layer": LAYER, "alpha": ALPHA}
    voices = []
    for voice, entry in manifest["voices"].items():
        voices.append({"voice": voice, "name": entry.get("name", voice),
                       "status": entry.get("status", "unavailable"), "reason": entry.get("reason"),
                       "warning": entry.get("warning"), "validation": _public_validation(entry.get("validation")),
                       "layer": manifest.get("layer", LAYER), "alpha": entry.get("alpha", manifest.get("alpha", ALPHA))})
    return {"default_enabled": True, "available": True, "voices": voices,
            "layer": manifest.get("layer", LAYER), "alpha": manifest.get("alpha", ALPHA),
            "note": "Exploratory speaker contrast steering; enabled does not guarantee improved similarity or quality."}


def resolve_speaker_steering(engine, voice_id, reference_path, *, enabled=True, custom=False, bank=BANK):
    """Resolve exactly one verified voice/reference/base-model vector or bypass.

    Return (CPU FP32 tensor | None, public report). Experimental valid vectors
    can be tried as requested; bank entries may explicitly disallow activation.
    Checkpoint mismatch, custom/no-reference, LoRAs and corruption fail closed.
    """
    report = {"enabled": bool(enabled), "active": False, "voice": voice_id,
              "layer": LAYER, "alpha": ALPHA, "kind": "high_minus_low_contrast"}
    def bypass(reason):
        return None, {**report, "reason": reason}
    if not enabled: return bypass("disabled_by_user")
    if custom: return bypass("custom_reference_no_matching_vector")
    if not voice_id: return bypass("no_catalog_voice")
    if reference_path is None: return bypass("no_reference")
    adapter_bank = getattr(engine, "adapter_bank", None)
    if adapter_bank is not None and getattr(adapter_bank, "active_key", ()):
        return bypass("adapters_active_vector_base_mismatch")
    manifest = _manifest(bank)
    if manifest is None: return bypass("bank_unavailable")
    checkpoint = getattr(engine, "checkpoint_info", {}).get("sha256")
    if not checkpoint or manifest.get("checkpoint_sha256") != checkpoint:
        return bypass("checkpoint_mismatch")
    if manifest.get("layer") != LAYER:
        return bypass("unsupported_layer")
    entry = manifest["voices"].get(voice_id)
    if not isinstance(entry, dict): return bypass("voice_vector_unavailable")
    if entry.get("allow_activation") is False:
        return bypass(entry.get("reason") or "voice_activation_disallowed")
    if entry.get("status") not in {"ready", "experimental", "unvalidated", "quality_failed"}:
        return bypass(entry.get("reason") or "voice_vector_not_ready")
    try:
        if not entry.get("reference_sha256") or _sha256(reference_path) != entry["reference_sha256"]:
            return bypass("reference_mismatch")
        bank_root = Path(bank).resolve()
        vector_file = (bank_root / entry["vector_file"]).resolve()
        if not vector_file.is_relative_to(bank_root): return bypass("invalid_vector_path")
        if not entry.get("sha256") or _sha256(vector_file) != entry["sha256"]:
            return bypass("vector_integrity_mismatch")
        with np.load(vector_file, allow_pickle=False) as archive:
            vector = np.asarray(archive[entry.get("vector_key", "layer_16__contrast")], dtype=np.float32).copy()
        dimension = int(entry.get("dimension", 0))
        expected = int(engine.model.config.hidden_size)
        if dimension != expected or vector.shape != (expected,) or not np.isfinite(vector).all():
            return bypass("invalid_vector")
        alpha = float(entry.get("alpha", manifest.get("alpha", ALPHA)))
        if not np.isfinite(alpha) or not 0 <= alpha <= 1: return bypass("invalid_alpha")
    except (OSError, KeyError, ValueError, TypeError):
        return bypass("invalid_vector_bank_entry")
    warning = entry.get("warning")
    if entry.get("status") != "ready" and not warning:
        warning = "Experimental direction: improvement is not established and quality may regress."
    return torch.from_numpy(vector), {**report, "active": alpha != 0, "alpha": alpha,
        "status": entry["status"], "reason": "verified_catalog_vector", "warning": warning,
        "validation": _public_validation(entry.get("validation")), "vector_sha256": entry["sha256"],
        "reference_sha256": entry["reference_sha256"], "checkpoint_sha256": checkpoint,
        "application": "Block16 output; last predictor only; active conditional and neutral CFG rows; no parameter changes"}
