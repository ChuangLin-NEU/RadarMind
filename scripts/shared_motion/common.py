#!/usr/bin/env python3
"""Shared helpers. Reads Phase 2 PNG cache; does not rewrite it."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "cross_radar"))
from models.resnet18_spec import ResNet18Spec  # noqa: E402
from train_cross_radar import EmbNet, load_band  # noqa: E402

OUT = ROOT / "results" / "phase3_shared_motion"
P2_CACHE = ROOT / "results" / "phase2_cross_radar" / "cache"
SEED = 42
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RADARS = ["10GHz", "24GHz", "77GHz"]

AUG_SPEC = {
    "name": "physics_aware_sensor_randomization_v01",
    "A_amplitude": {"gain_min": 0.65, "gain_max": 1.40, "note": "radar gain / response"},
    "B_doppler_axis_scale": {"p": 0.6, "scale_min": 0.88, "scale_max": 1.12, "axis": 0},
    "C_temporal_resample": {"p": 0.6, "scale_min": 0.88, "scale_max": 1.12, "axis": 1},
    "D_resolution_blur": {"p": 0.5, "sigma_min": 0.40, "sigma_max": 1.10},
    "E_radar_like_noise": {"std_min": 0.03, "std_max": 0.10},
    "forbidden": ["large_rotation", "color_jitter", "imagenet_flip"],
}


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_packs() -> dict[str, dict]:
    return {f: load_band(f) for f in RADARS}


def views_for(packs: dict, radars: list[str], split: str) -> dict[str, dict]:
    """timestamp -> {radar: (X, y, action)} . First file only if duplicates."""
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


def _fit_hw(z: np.ndarray, h: int, w: int) -> np.ndarray:
    out = np.zeros((h, w), np.float32)
    zh, zw = z.shape
    src_y = max(0, (zh - h) // 2)
    src_x = max(0, (zw - w) // 2)
    dst_y = max(0, (h - zh) // 2)
    dst_x = max(0, (w - zw) // 2)
    hh = min(h, zh)
    ww = min(w, zw)
    out[dst_y : dst_y + hh, dst_x : dst_x + ww] = z[src_y : src_y + hh, src_x : src_x + ww]
    return out


def _scale_axis(x: np.ndarray, axis: int, scale: float) -> np.ndarray:
    h, w = x.shape
    if axis == 0:
        new_h = max(8, int(round(h * scale)))
        yy = np.linspace(0, h - 1, new_h)
        y0 = np.floor(yy).astype(np.int64)
        y1 = np.minimum(y0 + 1, h - 1)
        a = (yy - y0).astype(np.float32)[:, None]
        z = (1.0 - a) * x[y0] + a * x[y1]
        return _fit_hw(z.astype(np.float32), h, w)
    new_w = max(8, int(round(w * scale)))
    xx = np.linspace(0, w - 1, new_w)
    x0 = np.floor(xx).astype(np.int64)
    x1 = np.minimum(x0 + 1, w - 1)
    a = (xx - x0).astype(np.float32)
    z = (1.0 - a) * x[:, x0] + a * x[:, x1]
    return _fit_hw(z.astype(np.float32), h, w)


def _blur(x: np.ndarray, sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(sigma * 3)))
    t = np.arange(-r, r + 1, dtype=np.float32)
    k = np.exp(-0.5 * (t / sigma) ** 2)
    k = k / k.sum()
    pad = np.pad(x, ((r, r), (0, 0)), mode="edge")
    tmp = np.zeros_like(x)
    for i, ki in enumerate(k):
        tmp += ki * pad[i : i + x.shape[0]]
    pad2 = np.pad(tmp, ((0, 0), (r, r)), mode="edge")
    out = np.zeros_like(x)
    for i, ki in enumerate(k):
        out += ki * pad2[:, i : i + x.shape[1]]
    return out.astype(np.float32)


def physics_aug(x: np.ndarray, rng: np.random.RandomState) -> np.ndarray:
    spec = AUG_SPEC
    y = x.astype(np.float32) * float(rng.uniform(spec["A_amplitude"]["gain_min"], spec["A_amplitude"]["gain_max"]))
    if rng.rand() < spec["B_doppler_axis_scale"]["p"]:
        y = _scale_axis(
            y,
            0,
            float(rng.uniform(spec["B_doppler_axis_scale"]["scale_min"], spec["B_doppler_axis_scale"]["scale_max"])),
        )
    if rng.rand() < spec["C_temporal_resample"]["p"]:
        y = _scale_axis(
            y,
            1,
            float(rng.uniform(spec["C_temporal_resample"]["scale_min"], spec["C_temporal_resample"]["scale_max"])),
        )
    if rng.rand() < spec["D_resolution_blur"]["p"]:
        y = _blur(y, float(rng.uniform(spec["D_resolution_blur"]["sigma_min"], spec["D_resolution_blur"]["sigma_max"])))
    noise_std = float(rng.uniform(spec["E_radar_like_noise"]["std_min"], spec["E_radar_like_noise"]["std_max"]))
    y = y + rng.normal(0.0, noise_std, size=y.shape).astype(np.float32)
    return y.astype(np.float32)


def infonce(za: torch.Tensor, zb: torch.Tensor, temp: float = 0.07) -> torch.Tensor:
    za = F.normalize(za, dim=1)
    zb = F.normalize(zb, dim=1)
    logits = za @ zb.t() / temp
    labels = torch.arange(za.size(0), device=za.device)
    return 0.5 * (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels))


class NanLossError(RuntimeError):
    pass


__all__ = [
    "AUG_SPEC",
    "CLASS_NAMES",
    "EmbNet",
    "NanLossError",
    "OUT",
    "P2_CACHE",
    "RADARS",
    "ROOT",
    "ResNet18Spec",
    "SEED",
    "infonce",
    "load_packs",
    "nn",
    "physics_aug",
    "set_seed",
    "views_for",
]
