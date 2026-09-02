"""Small PointNet frame encoder + optional GRU. 1-channel-safe, no RGB tricks."""
from __future__ import annotations

import torch
import torch.nn as nn


class FramePointNet(nn.Module):
    def __init__(self, in_ch: int = 5, emb: int = 128):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_ch, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, emb),
        )
        self.emb = emb

    def forward(self, x, mask):
        # x: B,T,P,C or B,P,C
        squeeze = False
        if x.ndim == 3:
            x = x.unsqueeze(1)
            mask = mask.unsqueeze(1)
            squeeze = True
        h = self.mlp(x)
        fill = torch.finfo(h.dtype).min
        h = h.masked_fill(~mask.unsqueeze(-1), fill)
        out = h.max(dim=2).values
        valid = mask.any(dim=-1)
        out = out.masked_fill(~valid.unsqueeze(-1), 0)
        return out.squeeze(1) if squeeze else out


class PointNetPool(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 5, emb: int = 128):
        super().__init__()
        self.enc = FramePointNet(in_ch, emb)
        self.fc = nn.Linear(emb, num_classes)

    def forward(self, x, mask):
        h = self.enc(x, mask)
        if h.ndim == 3:
            h = h.mean(dim=1)
        return self.fc(h)


class PointNetGRU(nn.Module):
    def __init__(self, num_classes: int, in_ch: int = 5, emb: int = 128, hidden: int = 128):
        super().__init__()
        self.enc = FramePointNet(in_ch, emb)
        self.gru = nn.GRU(emb, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)

    def forward(self, x, mask):
        if x.ndim == 3:
            x = x.unsqueeze(1)
            mask = mask.unsqueeze(1)
        h = self.enc(x, mask)
        out, _ = self.gru(h)
        return self.fc(out[:, -1])


class VoxelCNNGRU(nn.Module):
    """Per-frame 2D CNN on 10x32x32 occupancy, then GRU. Official voxel layout."""

    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(10, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        # x: B,T,10,32,32
        b, t = x.shape[:2]
        h = self.conv(x.reshape(b * t, 10, 32, 32)).flatten(1).reshape(b, t, 32)
        out, _ = self.gru(h)
        return self.fc(out[:, -1])
