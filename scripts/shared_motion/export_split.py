#!/usr/bin/env python3
"""Copy the Phase 2 acquisition split as the frozen Phase 3 split. Do not reshuffle."""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data_splits" / "ci4r_cross_radar_timestamp.csv"
DST = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"


def main() -> None:
    rows = list(csv.DictReader(SRC.open(encoding="utf-8")))
    fields = ["acquisition_id", "timestamp", "action", "split", "protocol", "seed"]
    DST.parent.mkdir(exist_ok=True)
    with DST.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow(
                {
                    "acquisition_id": r["timestamp"],
                    "timestamp": r["timestamp"],
                    "action": r["action"],
                    "split": r["split"],
                    "protocol": "acquisition_timestamp_shared_motion",
                    "seed": r.get("seed", "42"),
                }
            )
    print("wrote", DST, "n", len(rows), flush=True)


if __name__ == "__main__":
    main()
