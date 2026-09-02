"""RadHAR ROS-style txt. New frame when point_id == 0 (official voxels.py / DAP-Net)."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List

import numpy as np

# README: x,y,z in m; velocity in m/s; intensity in dB.
CLASS_NAMES = ["boxing", "jack", "jump", "squats", "walk"]


def parse_filename_tokens(name: str) -> Dict[str, str]:
    """Do not invent subject. Filenames are inconsistent (leading underscores)."""
    stem = Path(name).stem
    return {
        "filename_stem": stem,
        "subject": "UNKNOWN",
        "session": "UNKNOWN",
        "recording": stem,
        "participant_token": "UNKNOWN",
    }


def parse_mmwave_txt(path: Path) -> List[np.ndarray]:
    """List of (Ni, 5) [x,y,z,velocity,intensity]. Official: split ---, new frame at point_id 0."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    blocks = [b.strip() for b in raw.split("---") if b.strip()]
    frames: List[List[List[float]]] = []
    started = False
    for blk in blocks:
        kv = {}
        for ln in blk.splitlines():
            ln = ln.strip()
            if ":" in ln:
                k, v = ln.split(":", 1)
                kv[k.strip().lower()] = v.strip().strip('"')
        try:
            pid = int(float(kv.get("point_id", kv.get("point id", "-1"))))
        except ValueError:
            continue
        if pid == 0:
            frames.append([])
            started = True
        if not started:
            continue
        try:
            feat = [
                float(kv["x"]),
                float(kv["y"]),
                float(kv["z"]),
                float(kv.get("velocity", "nan")),
                float(kv.get("intensity", "nan")),
            ]
        except (KeyError, ValueError):
            continue
        if any(np.isnan(feat)):
            continue
        frames[-1].append(feat)
    return [np.asarray(p, dtype=np.float32) if p else np.zeros((0, 5), dtype=np.float32) for p in frames]


def first_stamp(path: Path) -> Dict[str, str]:
    """Read first ROS stamp if present. Not a subject id."""
    secs = nsecs = "UNKNOWN"
    with path.open(encoding="utf-8", errors="ignore") as f:
        text = f.read(4000)
    m = re.search(r"secs:\s*(\d+)", text)
    n = re.search(r"nsecs:\s*(\d+)", text)
    if m:
        secs = m.group(1)
    if n:
        nsecs = n.group(1)
    return {"stamp_secs": secs, "stamp_nsecs": nsecs}
