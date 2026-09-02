"""LOAO + subject-independent masks. Unseen action never enters train/val/threshold."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from scripts.pointcloud.mri import CLASS_NAMES


def subject_assign(z, split_csv: Path) -> np.ndarray:
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    mp = {r["action_segment_id"]: r["split"] for r in rows}
    segs = [str(s) for s in z["action_segment_id"]]
    return np.array([mp.get(s, "heldout") for s in segs])


def known_classes(held_out: str) -> list[str]:
    if held_out not in CLASS_NAMES:
        raise ValueError(f"unknown held-out action: {held_out}")
    return [c for c in CLASS_NAMES if c != held_out]


def remap_y(y: np.ndarray, actions: np.ndarray, known: list[str]) -> np.ndarray:
    mp = {c: i for i, c in enumerate(known)}
    out = np.full(len(y), -1, dtype=np.int64)
    for i, a in enumerate(actions):
        if a in mp:
            out[i] = mp[a]
    return out


def loao_masks(actions: np.ndarray, assign: np.ndarray, held_out: str) -> dict:
    known = actions != held_out
    unseen = actions == held_out
    return {
        "train": (assign == "train") & known,
        "val": (assign == "val") & known,
        "test_known": (assign == "test") & known,
        "test_unknown": (assign == "test") & unseen,
        "blocked_unseen_non_test": unseen & (assign != "test"),
    }
