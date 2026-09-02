#!/usr/bin/env python3
"""Measure official spectrogram PNG statistics. No prior guess about 24 GHz."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3_shared_motion"
INV = ROOT / "results" / "phase2_cross_radar" / "ci4r_radar_inventory.csv"
SPLIT = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    assign = {r["timestamp"]: r["split"] for r in csv.DictReader(SPLIT.open(encoding="utf-8"))}
    agreed = set(assign)
    rows = []
    for r in csv.DictReader(INV.open(encoding="utf-8")):
        if r["timestamp"] not in agreed:
            continue
        png = ROOT / r["source_file"]
        if not png.is_file():
            continue
        im = Image.open(png)
        arr = np.asarray(im.convert("L"), dtype=np.float64)
        rgb = np.asarray(im.convert("RGB"), dtype=np.float64)
        rows.append(
            {
                "radar_frequency": r["radar_frequency"],
                "action": r["action"],
                "timestamp": r["timestamp"],
                "split": assign[r["timestamp"]],
                "native_w": im.size[0],
                "native_h": im.size[1],
                "mode": im.mode,
                "gray_mean": float(arr.mean()),
                "gray_std": float(arr.std()),
                "gray_min": float(arr.min()),
                "gray_max": float(arr.max()),
                "row_energy_std": float(arr.mean(1).std()),
                "col_energy_std": float(arr.mean(0).std()),
                "rgb_mean": float(rgb.mean()),
            }
        )
    dest = OUT / "radar_domain_statistics.csv"
    fields = list(rows[0].keys()) if rows else []
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    # summary
    from collections import defaultdict

    by = defaultdict(list)
    for r in rows:
        by[r["radar_frequency"]].append(r)
    summary = []
    for freq, xs in sorted(by.items()):
        summary.append(
            {
                "radar_frequency": freq,
                "n": len(xs),
                "native_w_mean": float(np.mean([x["native_w"] for x in xs])),
                "native_h_mean": float(np.mean([x["native_h"] for x in xs])),
                "gray_mean": float(np.mean([x["gray_mean"] for x in xs])),
                "gray_std": float(np.mean([x["gray_std"] for x in xs])),
                "row_energy_std": float(np.mean([x["row_energy_std"] for x in xs])),
                "col_energy_std": float(np.mean([x["col_energy_std"] for x in xs])),
            }
        )
    with (OUT / "radar_domain_statistics_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
        w.writeheader()
        w.writerows(summary)
    print(summary, flush=True)


if __name__ == "__main__":
    main()
