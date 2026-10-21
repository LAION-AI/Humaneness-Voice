"""Incremental speaker cue, deliberately separate from TYGER turn-taking.

Features are four new frozen Nemotron layer-15 vectors per 320-ms update.
Silence/overlap return no identity. Confidence is an activity/context indicator,
not an authentication probability. Do not persist private microphone features.
"""
from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from .models import SpeakerPredictor,load_state


class Predictor:
    def __init__(self, checkpoint, device="cpu", min_rms=0.003, accumulate=True):
        state = torch.load(Path(checkpoint), map_location="cpu", weights_only=False)
        cfg = state["config"]
        self.model = SpeakerPredictor(cfg["architecture"], cfg["width"],centered=cfg.get("centered_target",False))
        load_state(self.model,state["model"])
        self.model.to(device).eval()
        self.device = device
        self.context = cfg["context"]
        self.frames = deque(maxlen=self.context)
        self.min_rms = min_rms
        self.accumulate = accumulate
        self.average = None
        self.speech_updates = 0
        self.voiced_frames = 0
        self.threshold = float(state.get("activity_threshold", 0.6))

    def reset(self):
        self.frames.clear()
        self.average = None
        self.speech_updates = 0
        self.voiced_frames = 0

    @torch.inference_mode()
    def update(self, features, energy=None):
        """Accept [new_frames,1024] numpy/tensor; energy is microphone RMS."""
        values = torch.as_tensor(features, dtype=torch.float32).cpu()
        if values.ndim != 2 or values.shape[1] != 1024:
            raise ValueError("Expected new Nemotron layer-15 frames [T,1024]")
        for frame in values:
            self.frames.append(frame)
        if energy is not None and energy < self.min_rms:
            self.average = None
            self.speech_updates = 0
            self.voiced_frames = 0
            return {"embedding": None, "confidence": 0., "kind": "silence",
                    "context_ms": len(self.frames) * 80}
        if len(self.frames) < 4:
            return {"embedding": None, "confidence": 0., "kind": "warming",
                    "context_ms": len(self.frames) * 80}
        x = torch.stack(list(self.frames))
        if len(x) < self.context:
            x = torch.cat([x[:1].expand(self.context - len(x), -1), x])
        emb, logits = self.model(x[None].to(self.device))
        probs = logits.softmax(-1)[0].cpu().numpy()
        kind = ("speech", "silence", "overlap")[int(probs.argmax())]
        coverage = min(len(self.frames),self.voiced_frames+len(values)) / self.context
        confidence = float(probs[0]) * min(1.,coverage)
        if kind != "speech" or probs[0] < self.threshold:
            self.average = None
            self.speech_updates = 0
            self.voiced_frames = 0
            return {"embedding": None, "confidence": confidence, "kind": kind,
                    "context_ms": len(self.frames) * 80}
        vector = emb[0].cpu().numpy()
        self.voiced_frames = min(self.context,self.voiced_frames+len(values))
        # Avoid blindly averaging across an obvious speaker change.
        if self.average is None or np.dot(self.average, vector) < 0.3:
            if self.average is not None:
                self.voiced_frames = min(self.context,len(values))
                confidence = float(probs[0])*self.voiced_frames/self.context
            self.average = vector
            self.speech_updates = 1
        elif self.accumulate:
            self.speech_updates = min(self.speech_updates + 1, 8)
            alpha = 1 / self.speech_updates
            self.average = (1 - alpha) * self.average + alpha * vector
            self.average /= max(np.linalg.norm(self.average), 1e-9)
        else:
            self.average = vector
        return {"embedding": self.average.copy(), "confidence": confidence,
                "kind": "speech", "context_ms": len(self.frames) * 80,
                "activity_probabilities": probs.tolist()}
