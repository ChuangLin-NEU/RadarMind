"""Shared point-cloud helpers. No dataset-specific guessing."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

SEED = 0
FEAT_NAMES = ("x", "y", "z", "velocity", "intensity")
FEAT_SLICES = {"XYZ": 3, "XYZV": 4, "XYZVI": 5}


def write_csv(path: Path, rows: Sequence[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def append_csv(path: Path, row: dict, fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + "_part.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def seed_from(*parts) -> int:
    h = hashlib.md5("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def sample_frame(pts: np.ndarray, p_fixed: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """Return (P, C) float32 and (P,) bool mask. pts is (N, C); N may be 0."""
    c = pts.shape[1] if pts.ndim == 2 and pts.size else 5
    out = np.zeros((p_fixed, c), dtype=np.float32)
    mask = np.zeros((p_fixed,), dtype=np.bool_)
    n = 0 if pts is None or pts.size == 0 else int(pts.shape[0])
    if n == 0:
        return out, mask
    if n >= p_fixed:
        idx = rng.choice(n, size=p_fixed, replace=False)
        out[:] = pts[idx]
        mask[:] = True
    else:
        out[:n] = pts
        mask[:n] = True
    return out, mask


def stack_window(frames: List[np.ndarray], start: int, t_len: int, p_fixed: int, rng: np.random.RandomState) -> tuple[np.ndarray, np.ndarray]:
    """frames[i] is (Ni, C). Missing / short index → empty frame."""
    c = 5
    x = np.zeros((t_len, p_fixed, c), dtype=np.float32)
    m = np.zeros((t_len, p_fixed), dtype=np.bool_)
    n_avail = len(frames)
    for i in range(t_len):
        t = start + i
        pts = frames[t] if 0 <= t < n_avail else np.zeros((0, c), dtype=np.float32)
        x[i], m[i] = sample_frame(pts, p_fixed, rng)
    return x, m


def feat_stats(arr: np.ndarray) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.float64).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"min": None, "max": None, "mean": None, "std": None}
    return {
        "min": float(a.min()),
        "max": float(a.max()),
        "mean": float(a.mean()),
        "std": float(a.std()),
    }


def dump_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
