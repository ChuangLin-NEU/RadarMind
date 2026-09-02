"""Standard ResNet18 adapted to 1-channel spectrograms."""
from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


class ResNet18Spec(nn.Module):
    def __init__(self, num_classes: int = 11, channels: int = 1):
        super().__init__()
        self.backbone = resnet18(weights=None)
        self.backbone.conv1 = nn.Conv2d(channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        return self.backbone(x)
