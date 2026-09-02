#!/usr/bin/env python3
"""Fingerprint / evolution / empty R2 tables. No training."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"
P7 = ROOT / "results" / "phase7_radar_invariance"


def read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> None:
    pix = {r["representation"]: r for r in read_csv(P7 / "representation_radar_id.csv")}
    fair = {r["protocol"]: r for r in read_csv(P7 / "representation_fairness_comparison.csv")}
    p3 = read_csv(ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv")
    p3m0 = next((r for r in p3 if r.get("model") == "M0" and r.get("protocol") == "multi_radar_shared"), {})
    p7p = read_csv(P7 / "probe_results.csv")
    p7m0 = next((r for r in p7p if r.get("model") == "M0"), {})

    write_csv(
        OUT / "representation_fingerprint.csv",
        ["representation", "kind", "radar_id_accuracy", "n_test", "note"],
        [
            {
                "representation": "R0",
                "kind": "pixels_16x16_logreg",
                "radar_id_accuracy": pix.get("R0", {}).get("accuracy", "1.0"),
                "n_test": pix.get("R0", {}).get("n_test", ""),
                "note": "Phase 7; official PNG 224",
            },
            {
                "representation": "R1",
                "kind": "pixels_16x16_logreg",
                "radar_id_accuracy": pix.get("R1", {}).get("accuracy", "1.0"),
                "n_test": pix.get("R1", {}).get("n_test", ""),
                "note": "Phase 7; cropped PNG 224",
            },
            {
                "representation": "R2",
                "kind": "not_run",
                "radar_id_accuracy": "",
                "n_test": "",
                "note": "R2 not built; BLOCKED_BY_DATA on this host",
            },
            {
                "representation": "R0",
                "kind": "motion_embedding_probe",
                "radar_id_accuracy": p3m0.get("radar_probe_accuracy", "1.0"),
                "n_test": "",
                "note": "Phase 3 M0; frozen",
            },
            {
                "representation": "R1",
                "kind": "motion_embedding_probe",
                "radar_id_accuracy": p7m0.get("radar_probe_accuracy", "1.0"),
                "n_test": "",
                "note": "Phase 7 M0; frozen",
            },
            {
                "representation": "R2",
                "kind": "motion_embedding_probe",
                "radar_id_accuracy": "",
                "n_test": "",
                "note": "Gate 2 failed; no HAR train",
            },
        ],
    )

    evo = []
    for proto in ("10→10", "24→24", "77→77", "77→10", "10→77", "77→24", "24→77", "10→24", "24→10"):
        r = fair.get(proto, {})
        evo.append(
            {
                "protocol": proto,
                "R0_accuracy": r.get("R0_accuracy", ""),
                "R1_accuracy": r.get("R1_accuracy", ""),
                "R2_accuracy": "",
                "R0_radar_id": pix.get("R0", {}).get("accuracy", "1.0"),
                "R1_radar_id": pix.get("R1", {}).get("accuracy", "1.0"),
                "R2_radar_id": "",
                "note": "R2 empty: no physical cache",
            }
        )
    write_csv(
        OUT / "representation_evolution.csv",
        ["protocol", "R0_accuracy", "R1_accuracy", "R2_accuracy", "R0_radar_id", "R1_radar_id", "R2_radar_id", "note"],
        evo,
    )
    write_csv(
        OUT / "paired_similarity.csv",
        ["pair_kind", "metric", "value", "note"],
        [
            {
                "pair_kind": "same_acquisition_cross_radar",
                "metric": "R2_not_computed",
                "value": "",
                "note": "Need R2 arrays. Do not compute similarity on PNG and call it physical.",
            }
        ],
    )
    write_csv(
        OUT / "representation_manifest.csv",
        ["representation", "n_samples", "source", "physical_axes", "status"],
        [
            {"representation": "R0", "n_samples": "327x3 PNG", "source": "PLOTTED_PNG whole figure", "physical_axes": "none", "status": "FROZEN_PHASE2"},
            {"representation": "R1", "n_samples": "same files cropped", "source": "PLOTTED_PNG fair crop", "physical_axes": "none", "status": "FROZEN_PHASE7"},
            {"representation": "R2", "n_samples": "0", "source": "none on this host", "physical_axes": "none", "status": "BLOCKED_BY_DATA"},
        ],
    )
    gate2 = {
        "A_numerical_source": False,
        "B_physical_axes_evidence": False,
        "C_common_grid": False,
        "D_sanity_plots": False,
        "enter_min_train": False,
        "reason": "Raw/MAT members listed in archive but not extracted here. Phase1 77 GHz 224 cache is not a physical grid.",
    }
    (OUT / "gate2.json").write_text(json.dumps(gate2, indent=2), encoding="utf-8")
    print("tables ok", flush=True)


if __name__ == "__main__":
    main()
