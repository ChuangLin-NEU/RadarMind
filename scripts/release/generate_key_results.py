#!/usr/bin/env python3
"""Write docs/generated/key_results.md from the frozen snapshot. No second truth."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SNAP = ROOT / "results" / "final_delivery" / "metric_snapshot_full.json"
OUT = ROOT / "docs" / "generated" / "key_results.md"


def D(metrics: dict, key: str) -> str:
    return str(metrics[key]["display"])


def main() -> int:
    snap = json.loads(SNAP.read_text(encoding="utf-8"))
    m = snap["metrics"]
    lines = [
        "<!-- Generated from results/final_delivery/metric_snapshot_full.json. Do not edit by hand. -->",
        "",
        "| Result | Value | Snapshot key |",
        "|---|---|---|",
        f"| mRI subject-independent accuracy | {D(m, 'MRI_SI_ACC')} | `MRI_SI_ACC` |",
        f"| mRI subject-independent macro-F1 | {D(m, 'MRI_SI_F1')} | `MRI_SI_F1` |",
        f"| RadHAR official XYZV accuracy | {D(m, 'RADHAR_XYZV_ACC')} | `RADHAR_XYZV_ACC` |",
        f"| RadHAR official XYZV macro-F1 | {D(m, 'RADHAR_XYZV_F1')} | `RADHAR_XYZV_F1` |",
        f"| CI4R 10 GHz within (R0) | {D(m, 'R0_10_WITHIN')} | `R0_10_WITHIN` |",
        f"| CI4R 24 GHz within (R0) | {D(m, 'R0_24_WITHIN')} | `R0_24_WITHIN` |",
        f"| CI4R 77 GHz within (R0) | {D(m, 'R0_77_WITHIN')} | `R0_77_WITHIN` |",
        f"| CI4R 24→77 source-only (R0) | {D(m, 'R0_24_TO_77')} | `R0_24_TO_77` |",
        f"| M1 known-radar activity accuracy | {D(m, 'M1_ACC')} | `M1_ACC` |",
        f"| M1 same-action retrieval | {D(m, 'M1_RET')} | `M1_RET` |",
        f"| M1 3-radar Radar-ID | {D(m, 'M1_RID')} | `M1_RID` |",
        f"| R2-D 3-radar Radar-ID | {D(m, 'RID_R2D_3CLASS')} | `RID_R2D_3CLASS` |",
        f"| R2-V 24/77 Radar-ID | {D(m, 'RID_R2V_2477')} | `RID_R2V_2477` |",
        f"| Continuous frame Acc / F1 | {D(m, 'CONT_FRAME_ACC')} / {D(m, 'CONT_FRAME_F1')} | `CONT_FRAME_ACC`, `CONT_FRAME_F1` |",
        f"| Continuous window Acc / F1 | {D(m, 'CONT_WIN_ACC')} / {D(m, 'CONT_WIN_F1')} | `CONT_WIN_ACC`, `CONT_WIN_F1` |",
        f"| Continuous segment F1 | {D(m, 'CONT_SEG_F1')} | `CONT_SEG_F1` |",
        f"| Continuous boundary error (s) | {D(m, 'CONT_BND_SEC')} | `CONT_BND_SEC` |",
        f"| Prototype LOAO AUROC | {D(m, 'PROTO_AUROC')} ± {D(m, 'PROTO_AUROC_STD')} | `PROTO_AUROC`, `PROTO_AUROC_STD` |",
        "",
        "Source: `results/final_delivery/metric_snapshot_full.json`.",
        "",
    ]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
