#!/usr/bin/env python3
"""Phase 7 helpers. Writes only under results/phase7_radar_invariance/. Never touches frozen trees."""
from __future__ import annotations

import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from models.resnet18_spec import ResNet18Spec  # noqa: E402

OUT = ROOT / "results" / "phase7_radar_invariance"
P2_CACHE = ROOT / "results" / "phase2_cross_radar" / "cache"
P2_RUNS = ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"
P3_METRICS = ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv"
SPLIT_CSV = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"
SEED = 42
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RADARS = ["10GHz", "24GHz", "77GHz"]

# Locked chrome crop (top, bottom, left, right) in native pixels. Not signal-adaptive.
CROP = {
    "10GHz": (1, 1, 1, 1),
    "24GHz": (8, 8, 0, 0),
    "77GHz": (55, 58, 0, 0),
}
CARRIER_LABEL_HZ = {"10GHz": 10e9, "24GHz": 24e9, "77GHz": 77.56365e9}
# 77: (77.1799e9 + 77.9474e9) / 2 from datToImage_77.m
R0_CROSS = {
    "10→10": 0.843,
    "24→24": 0.898,
    "77→77": 0.922,
    "77→10": 0.275,
    "10→77": 0.373,
    "77→24": 0.163,
    "24→77": 0.039,
    "10→24": 0.143,
    "24→10": 0.353,
}


class EmbNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Spec(num_classes, 1)

    def forward(self, x, return_emb=False):
        bb = self.backbone.backbone
        if x.ndim == 3:
            x = x.unsqueeze(1)
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


class GradientReversalFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.lambd * grad, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversalFn.apply(x, lambd)


class M3Net(nn.Module):
    def __init__(self, n_act: int, n_radar: int):
        super().__init__()
        self.encoder = EmbNet(n_act)
        self.radar_head = nn.Linear(512, n_radar)

    def forward(self, x, lambd_adv: float = 0.0, return_emb: bool = False):
        logits, emb = self.encoder(x, return_emb=True)
        radar_logits = self.radar_head(grl(emb, lambd_adv))
        if return_emb:
            return logits, emb, radar_logits
        return logits, radar_logits


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cache_name(freq: str, kind: str) -> Path:
    key = freq.replace("GHz", "")
    if kind == "R0":
        return P2_CACHE / f"ci4r_{key}ghz_png224.npz"
    return OUT / "cache" / f"ci4r_{key}ghz_r1_png224.npz"


def load_band(freq: str, kind: str = "R1") -> dict:
    z = np.load(cache_name(freq, kind), allow_pickle=True)
    return {
        "X": z["X"],
        "y": z["y"],
        "split": np.array([str(s) for s in z["split"]]),
        "timestamp": np.array([str(s) for s in z["timestamp"]]),
        "sample_id": np.array([str(s) for s in z["sample_id"]]),
        "action": np.array([str(s) for s in z["action"]]),
        "source_file": np.array([str(s) for s in z["source_file"]]) if "source_file" in z.files else None,
    }


def load_packs(kind: str = "R1") -> dict[str, dict]:
    return {f: load_band(f, kind) for f in RADARS}


def views_for(packs: dict, radars: list[str], split: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for radar in radars:
        d = packs[radar]
        mask = d["split"] == split
        for i in np.where(mask)[0]:
            ts = str(d["timestamp"][i])
            slot = out.setdefault(ts, {})
            if radar not in slot:
                slot[radar] = (
                    np.asarray(d["X"][i], dtype=np.float32),
                    int(d["y"][i]),
                    str(d["action"][i]),
                )
    return out


def infonce(za: torch.Tensor, zb: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    za = F.normalize(za, dim=1)
    zb = F.normalize(zb, dim=1)
    logits = za @ zb.t() / temp
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


def load_split() -> dict[str, str]:
    rows = list(csv.DictReader(SPLIT_CSV.open(encoding="utf-8")))
    return {str(r["timestamp"]): r["split"] for r in rows}


def append_csv(path: Path, fields: list[str], row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))
