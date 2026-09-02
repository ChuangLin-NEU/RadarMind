#!/usr/bin/env python3
"""R0 (frozen Phase 2) vs R1 source-only comparison + gate.json."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import OUT, R0_CROSS, read_csv, write_csv  # noqa: E402

JOBS = [
    ("10→10", "r1_within_10"),
    ("24→24", "r1_within_24"),
    ("77→77", "r1_within_77"),
    ("77→10", "r1_cross_77_to_10"),
    ("10→77", "r1_cross_10_to_77"),
    ("77→24", "r1_cross_77_to_24"),
    ("24→77", "r1_cross_24_to_77"),
    ("10→24", "r1_cross_10_to_24"),
    ("24→10", "r1_cross_24_to_10"),
]


def main() -> None:
    runs = {r["run_id"]: r for r in read_csv(OUT / "training_runs.csv") if r.get("run_id")}
    rows = []
    within_ok = True
    r1_built = (OUT / "cache" / "ci4r_10ghz_r1_png224.npz").is_file()
    for proto, rid in JOBS:
        r0 = R0_CROSS[proto]
        rec = runs.get(rid, {})
        try:
            r1 = float(rec["accuracy"])
        except (TypeError, ValueError, KeyError):
            r1 = None
        rows.append(
            {
                "protocol": proto,
                "R0_accuracy": r0,
                "R1_accuracy": r1 if r1 is not None else "",
                "delta": (r1 - r0) if r1 is not None else "",
                "r1_run_id": rid,
                "r0_source": "phase2_frozen",
            }
        )
        if proto.endswith(proto.split("→")[0]) and proto[0:2] == proto[-2:]:
            pass
        if proto in ("10→10", "24→24", "77→77"):
            if r1 is None or r1 < 0.70:
                within_ok = False
    write_csv(
        OUT / "representation_fairness_comparison.csv",
        ["protocol", "R0_accuracy", "R1_accuracy", "delta", "r1_run_id", "r0_source"],
        rows,
    )
    gate = {
        "r1_constructed": r1_built,
        "r1_source": "PLOTTED_PNG",
        "physical_alignment": False,
        "within_radar_reasonable": within_ok,
        "enter_part_d": bool(r1_built and within_ok),
        "reason": (
            "R1 is fair-PNG crop + common post; within-radar Acc>=0.70"
            if (r1_built and within_ok)
            else "R1 missing or within-radar Acc<0.70; do not train M3"
        ),
    }
    (OUT / "gate.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    print(json.dumps(gate, indent=2), flush=True)
    for r in rows:
        print(r["protocol"], "R0", r["R0_accuracy"], "R1", r["R1_accuracy"], "d", r["delta"], flush=True)


if __name__ == "__main__":
    main()
