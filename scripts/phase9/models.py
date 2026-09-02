"""RadarBridge v2: lightweight radar-specific adapters + shared ResNet18 trunk."""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from models.resnet18_spec import ResNet18Spec  # noqa: E402

RADARS = ["10GHz", "24GHz", "77GHz"]


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv1 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        h = self.act(self.bn1(self.conv1(x)))
        h = self.bn2(self.conv2(h))
        return self.act(x + h)


class RadarAdapter(nn.Module):
    """1×H×W → 1×H×W. Low-level sensor normalization only."""

    def __init__(self, hidden: int = 16):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, hidden, 3, padding=1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(inplace=True),
        )
        self.res1 = ResidualBlock(hidden)
        self.res2 = ResidualBlock(hidden)
        self.proj = nn.Conv2d(hidden, 1, 1, bias=True)

    def forward(self, x, return_feat: bool = False):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.stem(x)
        h = self.res1(h)
        h = self.res2(h)
        y = self.proj(h)
        if return_feat:
            return y, h
        return y


class SharedEncoder(nn.Module):
    """ResNet18 trunk. Input is 1-channel spectrogram (adapter output or raw R2)."""

    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.backbone = ResNet18Spec(num_classes, 1)

    def forward(self, x, return_emb: bool = False):
        bb = self.backbone.backbone
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim == 4 and x.size(1) != 1:
            x = x[:, :1]
        h = bb.conv1(x)
        h = bb.bn1(h)
        h = bb.relu(h)
        h = bb.maxpool(h)
        h = bb.layer1(h)
        h = bb.layer2(h)
        h = bb.layer3(h)
        h = bb.layer4(h)
        h = bb.avgpool(h)
        emb = torch.flatten(h, 1)
        logits = bb.fc(emb)
        if return_emb:
            return logits, emb
        return logits


class RadarBridgeV2(nn.Module):
    """
    Known-radar Sensor-Adapter Framework.
    Requires radar_id to pick Adapter_10 / Adapter_24 / Adapter_77.
    Not a universal / unseen-radar encoder.
    """

    def __init__(self, use_adapters: bool = True, allowed_radars=None):
        super().__init__()
        self.use_adapters = use_adapters
        self.allowed = list(allowed_radars or RADARS)
        self.adapters = nn.ModuleDict({r: RadarAdapter() for r in self.allowed}) if use_adapters else nn.ModuleDict()
        self.encoder = SharedEncoder(7)

    def adapter_params(self) -> int:
        return sum(p.numel() for p in self.adapters.parameters())

    def encoder_params(self) -> int:
        return sum(p.numel() for p in self.encoder.parameters())

    def apply_adapter(self, x, radars, override=None):
        """
        x: (B,H,W) or (B,1,H,W)
        radars: list[str] length B, or a single str
        override: None | 'identity' | a radar key (force that adapter)
        returns adapted x (B,1,H,W), feat (B,C,H,W) or None
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if (not self.use_adapters) or override == "identity":
            return x, None
        if override is not None:
            if override not in self.adapters:
                raise KeyError(f"adapter {override} was not trained")
            return self.adapters[override](x, return_feat=True)
        if isinstance(radars, str):
            return self.adapters[radars](x, return_feat=True)
        out = None
        feat = None
        for r in sorted(set(radars)):
            if r not in self.adapters:
                raise KeyError(f"adapter {r} was not trained")
            idx = [i for i, rr in enumerate(radars) if rr == r]
            y, f = self.adapters[r](x[idx], return_feat=True)
            if out is None:
                out = y.new_zeros(x.shape)
                feat = f.new_zeros((x.size(0),) + f.shape[1:])
            out[idx] = y
            feat[idx] = f
        return out, feat

    def forward(self, x, radars, override=None, return_all=False):
        xa, feat = self.apply_adapter(x, radars, override=override)
        logits, emb = self.encoder(xa, return_emb=True)
        if return_all:
            return logits, emb, xa, feat
        return logits, emb


def pairwise_cosine_distance(z_a, z_b):
    a = F.normalize(z_a, dim=1)
    b = F.normalize(z_b, dim=1)
    return (1.0 - (a * b).sum(dim=1)).mean()


def exact_pair_loss(z10, z24, z77):
    return (pairwise_cosine_distance(z10, z24) + pairwise_cosine_distance(z10, z77) + pairwise_cosine_distance(z24, z77)) / 3.0


def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())
