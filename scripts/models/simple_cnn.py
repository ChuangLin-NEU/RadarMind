"""Small CNN baseline for 224x224 single-channel spectrograms."""
from __future__ import annotations

import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self, num_classes: int = 11, channels: int = 1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        x = self.net(x)
        return self.fc(x.flatten(1))
