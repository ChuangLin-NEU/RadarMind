#!/usr/bin/env python3
"""Official RadHAR voxel windows as uint8 pack. Resume per recording."""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.common import atomic_savez  # noqa: E402
from pointcloud.radhar import CLASS_NAMES, parse_mmwave_txt  # noqa: E402
from pointcloud.voxels_official import voxel_windows  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rad = Path(os.environ.get("RADARMIND_RADHAR_ROOT", "").strip() or ".")
    split_csv = Path(os.environ.get("RADARMIND_SPLIT_DIR", "").strip() or (root / "data_splits")) / "radhar_official.csv"
    cache_dir = root / "results" / "phase1_pointcloud" / "cache"
    out = Path(os.environ.get("RADARMIND_RADHAR_VOXEL_OUT", "").strip() or (cache_dir / "radhar_voxels.npz"))
    part = Path(os.environ.get("RADARMIND_RADHAR_VOXEL_PARTS", "").strip() or (cache_dir / "radhar_vox_parts"))
    part.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    xs, ys, roles = [], [], []
    for i, r in enumerate(rows):
        dest = part / (Path(r["source_file"]).name + ".npz")
        if dest.exists() and dest.stat().st_size > 0:
            z = np.load(dest)
            wins = z["X"]
        else:
            fp = rad / r["source_file"]
            frames = parse_mmwave_txt(fp)
            wins = voxel_windows(frames, 60, 10)
            if wins.shape[0] == 0:
                print("[skip-short]", r["source_file"], flush=True)
                continue
            atomic_savez(dest, X=wins)
            print("[vox]", i + 1, r["source_file"], wins.shape, flush=True)
        xs.append(wins)
        ys.append(np.full((wins.shape[0],), CLASS_NAMES.index(r["action"]), np.int64))
        roles.append(np.array([r["split_role"]] * wins.shape[0]))
    X = np.concatenate(xs, 0)
    y = np.concatenate(ys, 0)
    role = np.concatenate(roles, 0)
    print("[stack]", X.shape, X.nbytes / 1024**3, "GB", flush=True)
    atomic_savez(out, X=X, y=y, split=role)
    print("[done]", out, out.stat().st_size, flush=True)


if __name__ == "__main__":
    main()
