#!/usr/bin/env python3
"""Energy inside vs outside R2-D common Doppler band. Uses native STFT, not R2 cache."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

FD_LO = -256.0
FD_HI = 255.875


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--native", required=True)
    ap.add_argument("--mapping", required=True)
    ap.add_argument("--split", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    native = Path(args.native)
    mapping = list(csv.DictReader(Path(args.mapping).open(encoding="utf-8")))
    split = {str(r["timestamp"]): r for r in csv.DictReader(Path(args.split).open(encoding="utf-8"))}
    rows = []
    for r in mapping:
        ts, radar = str(r["timestamp"]), r["radar"]
        p = native / radar / f"{ts}.npz"
        if not p.is_file():
            rows.append(
                {
                    "timestamp": ts,
                    "radar": radar,
                    "action": r.get("action", ""),
                    "split": split.get(ts, {}).get("split", ""),
                    "retained_energy_fraction": "",
                    "error": "missing_native",
                }
            )
            continue
        z = np.load(p)
        db = np.asarray(z["spectrogram_db"], np.float64)
        fd = np.asarray(z["doppler_axis_hz"], np.float64)
        power = np.power(10.0, db / 10.0)
        inside = (fd >= FD_LO) & (fd <= FD_HI)
        e_in = float(power[inside, :].sum())
        e_out = float(power[~inside, :].sum())
        tot = e_in + e_out
        frac = e_in / tot if tot > 0 else 0.0
        rows.append(
            {
                "timestamp": ts,
                "radar": radar,
                "action": r.get("action", ""),
                "split": split.get(ts, {}).get("split", ""),
                "energy_inside_common_band": e_in,
                "energy_outside_common_band": e_out,
                "retained_energy_fraction": frac,
                "n_bins_inside": int(inside.sum()),
                "n_bins_total": int(fd.size),
                "error": "",
            }
        )
    fields = [
        "timestamp",
        "radar",
        "action",
        "split",
        "energy_inside_common_band",
        "energy_outside_common_band",
        "retained_energy_fraction",
        "n_bins_inside",
        "n_bins_total",
        "error",
    ]
    out = Path(args.out)
    write_csv(out, fields, rows)
    summary = {}
    for radar in ("10GHz", "24GHz", "77GHz"):
        fr = [float(r["retained_energy_fraction"]) for r in rows if r["radar"] == radar and r["retained_energy_fraction"] != ""]
        summary[radar] = {
            "n": len(fr),
            "mean": float(np.mean(fr)) if fr else None,
            "std": float(np.std(fr)) if fr else None,
            "min": float(np.min(fr)) if fr else None,
            "max": float(np.max(fr)) if fr else None,
        }
    (out.parent / "doppler_band_retention_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
