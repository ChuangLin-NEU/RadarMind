#!/usr/bin/env python3
"""Build R1 fair-PNG cache from the same Phase 2 files. Does not rewrite Phase 2 cache."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import (  # noqa: E402
    CARRIER_LABEL_HZ,
    CLASS_NAMES,
    CROP,
    OUT,
    P2_CACHE,
    RADARS,
    write_csv,
)

FIELDS = [
    "sample_id",
    "timestamp",
    "radar",
    "action",
    "split",
    "source_file",
    "representation_source",
    "representation_version",
    "native_height",
    "native_width",
    "crop_top",
    "crop_bottom",
    "crop_left",
    "crop_right",
    "cropped_height",
    "cropped_width",
    "native_time_resolution",
    "native_doppler_resolution",
    "time_axis_unit",
    "doppler_axis_unit",
    "carrier_frequency",
    "output_shape",
    "channel_count",
    "interpolation",
    "normalization",
    "physical_alignment",
]


def load_r1_gray224(path: Path, radar: str) -> tuple[np.ndarray, dict]:
    im = Image.open(path).convert("L")
    w, h = im.size
    top, bot, left, right = CROP[radar]
    box = (left, top, w - right, h - bot)
    if box[2] <= box[0] or box[3] <= box[1]:
        raise ValueError(f"empty crop {radar} {path} native={w}x{h} crop={CROP[radar]}")
    cropped = im.crop(box)
    cw, ch = cropped.size
    arr = np.asarray(cropped.resize((224, 224), Image.BICUBIC), dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    meta = {
        "native_height": h,
        "native_width": w,
        "crop_top": top,
        "crop_bottom": bot,
        "crop_left": left,
        "crop_right": right,
        "cropped_height": ch,
        "cropped_width": cw,
        "native_time_resolution": cw,
        "native_doppler_resolution": ch,
        "time_axis_unit": "pixel",
        "doppler_axis_unit": "normalized_doppler_coordinate",
        "carrier_frequency": CARRIER_LABEL_HZ[radar],
        "output_shape": "1x224x224",
        "channel_count": 1,
        "interpolation": "BICUBIC",
        "normalization": "per_image_zscore",
        "physical_alignment": "none",
        "representation_source": "PLOTTED_PNG",
        "representation_version": "R1",
    }
    return arr, meta


def main() -> None:
    dest = OUT / "cache"
    dest.mkdir(parents=True, exist_ok=True)
    man_rows = []
    summary = {}
    for freq in RADARS:
        key = freq.replace("GHz", "")
        src = np.load(P2_CACHE / f"ci4r_{key}ghz_png224.npz", allow_pickle=True)
        xs, metas = [], []
        n = len(src["y"])
        for i in range(n):
            png = ROOT / str(src["source_file"][i])
            x, meta = load_r1_gray224(png, freq)
            xs.append(x)
            rec = {
                "sample_id": str(src["sample_id"][i]),
                "timestamp": str(src["timestamp"][i]),
                "radar": freq,
                "action": str(src["action"][i]),
                "split": str(src["split"][i]),
                "source_file": str(src["source_file"][i]),
                **meta,
            }
            metas.append(rec)
            man_rows.append(rec)
        np.savez_compressed(
            dest / f"ci4r_{key}ghz_r1_png224.npz",
            X=np.stack(xs).astype(np.float32),
            y=src["y"],
            sample_id=src["sample_id"],
            timestamp=src["timestamp"],
            prefix_group=src["prefix_group"],
            action=src["action"],
            split=src["split"],
            source_file=src["source_file"],
            class_names=src["class_names"],
            native_height=np.asarray([m["native_height"] for m in metas], np.int32),
            native_width=np.asarray([m["native_width"] for m in metas], np.int32),
            cropped_height=np.asarray([m["cropped_height"] for m in metas], np.int32),
            cropped_width=np.asarray([m["cropped_width"] for m in metas], np.int32),
            representation_source=np.array(["PLOTTED_PNG"] * n),
            carrier_frequency=np.asarray([m["carrier_frequency"] for m in metas], np.float64),
        )
        from collections import Counter

        roles = Counter(str(s) for s in src["split"])
        summary[freq] = {"n": n, "split": dict(roles), "crop": list(CROP[freq]), "source": "PLOTTED_PNG"}
        print("[r1]", freq, n, roles, flush=True)
    write_csv(OUT / "fair_representation_manifest.csv", FIELDS, man_rows)
    (dest / "r1_pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("wrote", OUT / "fair_representation_manifest.csv", flush=True)


if __name__ == "__main__":
    main()
