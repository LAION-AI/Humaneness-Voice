"""Small causal Nemotron-L15 -> ECAPA-192 predictors; no ASR/teacher weights."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class SpeakerPredictor(nn.Module):
    def __init__(self, architecture="gru", width=128, input_dim=1024, output_dim=192, centered=False):
        super().__init__()
        self.architecture = architecture
        self.width = width
        self.centered = centered
        self.norm = nn.LayerNorm(input_dim)
        if architecture == "mlp":
            self.encoder = nn.Sequential(nn.Linear(input_dim * 2, width), nn.GELU(),
                                         nn.Linear(width, width), nn.GELU())
        elif architecture == "gru":
            self.project = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
            self.encoder = nn.GRU(width, width, batch_first=True)
        elif architecture == "tcn":
            self.project = nn.Sequential(nn.Linear(input_dim, width), nn.GELU())
            self.encoder = nn.ModuleList([
                nn.Sequential(nn.Conv1d(width, width, 3, groups=width, dilation=d),
                              nn.GELU(), nn.Conv1d(width, width, 1), nn.GELU())
                for d in (1, 2, 4)])
        else:
            raise ValueError(architecture)
        self.embedding = nn.Linear(width, output_dim)
        self.activity = nn.Linear(width, 3)  # speech / silence / ambiguous overlap
        self.register_buffer("teacher_mean",torch.zeros(output_dim))
        self.register_buffer("teacher_radius",torch.ones(()))

    def forward(self, features, lengths=None):
        x = self.norm(features)
        if self.architecture == "mlp":
            h = self.encoder(torch.cat([x.mean(1), x.std(1, unbiased=False)], -1))
        elif self.architecture == "gru":
            h, _ = self.encoder(self.project(x))
            h = h[:, -1]
        else:
            h = self.project(x).transpose(1, 2)
            for dilation, layer in zip((1, 2, 4), self.encoder):
                h = h + layer(F.pad(h, (2 * dilation, 0)))
            h = h[:, :, -1]
        direction = F.normalize(self.embedding(h),dim=-1)
        embedding = F.normalize(self.teacher_mean+self.teacher_radius*direction,dim=-1) if self.centered else direction
        return embedding, self.activity(h)


def load_state(model, state):
    """Compatibility with the first uncentered pilot checkpoints."""
    missing, unexpected = model.load_state_dict(state,strict=False)
    if unexpected or set(missing)-{"teacher_mean","teacher_radius"}:
        raise ValueError(f"Unexpected checkpoint mismatch: missing={missing}, unexpected={unexpected}")


def parameter_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
