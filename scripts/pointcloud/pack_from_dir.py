#!/usr/bin/env python3
"""Rebuild packed npz + index from per-window files (resume-safe)."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pointcloud.common import atomic_savez, write_csv  # noqa: E402


def pack_dir(win_dir: Path, dest: Path, index_csv: Path, dataset: str) -> None:
    files = sorted(win_dir.glob("*.npz"))
    print("[scan]", win_dir, len(files), flush=True)
    xs, ms, ys, ids, subs, acts, segs, srcs, splits = [], [], [], [], [], [], [], [], []
    index = []
    for i, p in enumerate(files):
        try:
            z = np.load(p, allow_pickle=True)
            xs.append(z["X"])
            ms.append(z["mask"])
            ys.append(int(z["y"]))
            ids.append(str(z["sample_id"]))
            subs.append(str(z["subject"]) if "subject" in z.files else "UNKNOWN")
            acts.append(str(z["action"]))
            segs.append(str(z["action_segment_id"]) if "action_segment_id" in z.files else "")
            srcs.append(str(z["source_file"]) if "source_file" in z.files else "")
            splits.append(str(z["split"]) if "split" in z.files else "")
            index.append(
                {
                    "sample_id": str(z["sample_id"]),
                    "cache_path": str(p),
                    "status": "ok",
                    "subject": str(z["subject"]) if "subject" in z.files else "UNKNOWN",
                    "action": str(z["action"]),
                    "action_segment_id": str(z["action_segment_id"]) if "action_segment_id" in z.files else "",
                    "source_file": str(z["source_file"]) if "source_file" in z.files else "",
                    "split": str(z["split"]) if "split" in z.files else "",
                }
            )
        except Exception as exc:  # noqa: BLE001
            print("[skip]", p, exc, flush=True)
        if (i + 1) % 1000 == 0:
            print("[read]", i + 1, flush=True)
    if not xs:
        print("[empty]", dest)
        return
    write_csv(
        index_csv,
        index,
        ["sample_id", "cache_path", "status", "subject", "action", "action_segment_id", "source_file", "split"],
    )
    print("[stack]", len(xs), xs[0].shape, flush=True)
    atomic_savez(
        dest,
        X=np.stack(xs),
        mask=np.stack(ms),
        y=np.asarray(ys, np.int64),
        sample_id=np.array(ids),
        subject=np.array(subs),
        action=np.array(acts),
        action_segment_id=np.array(segs),
        source_file=np.array(srcs),
        split=np.array(splits),
        dataset=np.array(dataset),
    )
    print("[pack]", dest, dest.stat().st_size, flush=True)


def main() -> None:
    root = Path(__file__).resolve().parents[2] / "results" / "phase1_pointcloud" / "cache"
    mri_tmp = root / "mri_windows.npz.tmp.npz"
    mri_ok = root / "mri_windows.npz"
    if mri_tmp.exists() and not mri_ok.exists():
        mri_tmp.replace(mri_ok)
        print("[rename-mri]", mri_ok, mri_ok.stat().st_size, flush=True)
        # index only
        idx = []
        for p in sorted((root / "mri").glob("*.npz")):
            z = np.load(p, allow_pickle=True)
            idx.append(
                {
                    "sample_id": str(z["sample_id"]),
                    "cache_path": str(p),
                    "status": "ok",
                    "subject": str(z["subject"]) if "subject" in z.files else "UNKNOWN",
                    "action": str(z["action"]),
                    "action_segment_id": str(z["action_segment_id"]) if "action_segment_id" in z.files else "",
                    "source_file": str(z["source_file"]) if "source_file" in z.files else "",
                    "split": str(z["split"]) if "split" in z.files else "",
                }
            )
        write_csv(root / "mri_window_index.csv", idx, list(idx[0].keys()))
        print("[index-mri]", len(idx), flush=True)
    elif not mri_ok.exists():
        pack_dir(root / "mri", mri_ok, root / "mri_window_index.csv", "mRI")
    pack_dir(root / "radhar", root / "radhar_windows.npz", root / "radhar_window_index.csv", "RadHAR")


if __name__ == "__main__":
    main()
