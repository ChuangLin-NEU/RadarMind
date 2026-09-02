"""mRI raw radar CSV + videolabels JSON. Alignment uses Frame #, not Time."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# Dryad README: Frame # = frame index; Time = wall timestamp.
# DAP-Net mRI2csv.py: inclusive [a, b] on Frame #.
CANONICAL = {
    "t pose": "T pose",
    "t pose 2": "T pose",
    "pose_1": "pose_1",
    "pose_2": "pose_2",
    "pose_3": "pose_3",
    "pose_4": "pose_4",
    "pose_5": "pose_5",
    "pose_6": "pose_6",
    "pose_7": "pose_7",
    "pose_8": "pose_8",
    "pose_9": "pose_9",
    "pose_10": "pose_10",
    "free_form": "free_form",
    "walk": "walk",
}
CLASS_NAMES = [
    "T pose",
    "pose_1",
    "pose_2",
    "pose_3",
    "pose_4",
    "pose_5",
    "pose_6",
    "pose_7",
    "pose_8",
    "pose_9",
    "pose_10",
    "free_form",
    "walk",
]


def canonical_action(raw: str) -> str:
    key = str(raw).strip().lower()
    return CANONICAL.get(key, raw)


def parse_subject_id(name: str) -> str:
    m = re.search(r"subject\s*(\d+)", str(name), re.I)
    if not m:
        return "UNKNOWN"
    return f"subject{int(m.group(1))}"


def find_radar_csv(radar_dir: Path, subject: str) -> Path | None:
    hits = sorted(radar_dir.glob(f"{subject}_*.csv")) + sorted(radar_dir.glob(f"{subject}.csv"))
    return hits[0] if hits else None


def load_label_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_radar_csv(path: Path) -> Dict[int, np.ndarray]:
    """Frame # → (N, 5) float32 [x,y,z,doppler,intensity] plus side maps."""
    frames: Dict[int, List[List[float]]] = defaultdict(list)
    nobj: Dict[int, int] = {}
    times: Dict[int, List[float]] = defaultdict(list)
    with path.open(encoding="utf-8", errors="replace") as f:
        header = f.readline()
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            try:
                fr = int(float(parts[0]))
                no = int(float(parts[1]))
                x, y, z = float(parts[2]), float(parts[3]), float(parts[4])
                dpl, inten, t = float(parts[5]), float(parts[6]), float(parts[7])
            except ValueError:
                continue
            frames[fr].append([x, y, z, dpl, inten])
            nobj.setdefault(fr, no)
            times[fr].append(t)
    out = {fr: np.asarray(pts, dtype=np.float32) for fr, pts in frames.items()}
    meta = {
        "nobj_declared": nobj,
        "time_median": {fr: float(np.median(ts)) for fr, ts in times.items()},
        "header": header.strip(),
    }
    return out, meta


def segments_from_json(label_data: dict) -> List[dict]:
    subject = str(label_data.get("subject", "UNKNOWN"))
    rows = []
    for i, (raw, rng) in enumerate(label_data.get("labels", {}).items()):
        if not (isinstance(rng, (list, tuple)) and len(rng) == 2):
            continue
        a, b = int(rng[0]), int(rng[1])
        rows.append(
            {
                "subject": subject,
                "action_raw": raw,
                "action": canonical_action(raw),
                "action_segment_id": f"{subject}_{i:02d}_{raw.replace(' ', '_')}",
                "segment_start": a,
                "segment_end": b,  # inclusive, DAP-Net
                "n_label_frames": b - a + 1,
            }
        )
    return rows


def overlap_pairs(segs: List[dict]) -> List[Tuple[str, str, int, int]]:
    out = []
    for i, a in enumerate(segs):
        for b in segs[i + 1 :]:
            lo = max(a["segment_start"], b["segment_start"])
            hi = min(a["segment_end"], b["segment_end"])
            if lo <= hi:
                out.append((a["action_segment_id"], b["action_segment_id"], lo, hi))
    return out
