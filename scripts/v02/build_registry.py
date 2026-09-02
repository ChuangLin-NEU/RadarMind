#!/usr/bin/env python3
"""Pointer-only experiment registry. Do not copy large files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V02 = ROOT / "results" / "radarmind_v02"


def _j(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    V02.mkdir(parents=True, exist_ok=True)
    p1a_r = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "random_record" / "resnet18" / "metrics.json") or {}
    p1a_p = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "prefix_tail4_disjoint" / "resnet18" / "metrics.json") or {}
    p1b = _j(ROOT / "results" / "phase1_pointcloud" / "baselines" / "mri_si_gru_xyz" / "metrics.json") or {}
    rad = _j(ROOT / "results" / "phase1_pointcloud" / "baselines" / "radhar_official_gru_xyzv" / "metrics.json") or {}
    p5 = _j(ROOT / "results" / "phase5_open_set" / "eval_summary.json")
    p5full = _j(ROOT / "results" / "phase5_open_set" / "full_loao.json") or {}
    p4 = ROOT / "results" / "phase4_continuous" / "phase4_continuous_report.md"
    entries = [
        {
            "id": "phase1a_ci4r",
            "phase": "Phase 1A",
            "dataset": "CI4R 77 GHz",
            "protocol": "random-record / prefix-group-disjoint",
            "model": "ResNet18",
            "checkpoint": "results/phase1_ci4r/baselines/random_record/resnet18/best.pt",
            "metrics": {"random_acc": p1a_r.get("accuracy"), "prefix_acc": p1a_p.get("accuracy")},
            "result_path": "results/phase1_ci4r/",
            "status": "DONE",
        },
        {
            "id": "phase1b_mri",
            "phase": "Phase 1B",
            "dataset": "mRI",
            "protocol": "subject-independent",
            "model": "PointNet-GRU XYZ",
            "checkpoint": "results/phase1_pointcloud/baselines/mri_si_gru_xyz/best.pt",
            "metrics": {"accuracy": p1b.get("accuracy"), "macro_f1": p1b.get("macro_f1")},
            "result_path": "results/phase1_pointcloud/",
            "status": "DONE",
        },
        {
            "id": "phase1b_radhar",
            "phase": "Phase 1B",
            "dataset": "RadHAR",
            "protocol": "official Train/Test",
            "model": "PointNet-GRU XYZV",
            "checkpoint": "results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/best.pt",
            "metrics": {"accuracy": rad.get("accuracy"), "macro_f1": rad.get("macro_f1")},
            "result_path": "results/phase1_pointcloud/",
            "status": "DONE",
        },
        {
            "id": "phase2b_cross_radar",
            "phase": "Phase 2B",
            "dataset": "CI4R official PNG 7-class",
            "protocol": "within / zero-shot cross-radar / CORAL",
            "model": "ResNet18 official PNG",
            "checkpoint": None,
            "metrics": {"note": "see results/phase2_cross_radar/training_runs.csv"},
            "result_path": "results/phase2_cross_radar/",
            "status": "DONE",
        },
        {
            "id": "phase3_shared_motion",
            "phase": "Phase 3",
            "dataset": "CI4R 10/24/77 GHz official PNG",
            "protocol": "shared encoder + LORO",
            "model": "M2 shared",
            "checkpoint": "results/phase3_shared_motion/runs/M2_shared/best.pt",
            "metrics": {"note": "Radar-ID probe 1.000"},
            "result_path": "results/phase3_shared_motion/",
            "status": "DONE",
        },
        {
            "id": "phase4_continuous",
            "phase": "Phase 4",
            "dataset": "mRI labeled-interval windows",
            "protocol": "labeled-interval continuous replay",
            "model": "mri_si_gru_xyz (13-class, not retrained)",
            "checkpoint": "results/phase1_pointcloud/baselines/mri_si_gru_xyz/best.pt",
            "metrics": {"note": str(p4.relative_to(ROOT)) if p4.is_file() else None},
            "result_path": "results/phase4_continuous/",
            "status": "DONE",
        },
        {
            "id": "phase5_open_set",
            "phase": "Phase 5",
            "dataset": "mRI",
            "protocol": "LOAO + subject-independent",
            "model": "PointNet-GRU 12-class per fold",
            "checkpoint": "results/phase5_open_set/folds/*/best.pt",
            "metrics": {"n_folds": len(p5) if p5 else 0, "full_loao": p5full.get("full_loao")},
            "result_path": "results/phase5_open_set/",
            "status": "DONE" if p5 else "PENDING",
        },
    ]
    payload = {"product": "RadarMind", "version": "0.2", "note": "pointers only; do not copy caches or checkpoints", "experiments": entries}
    (V02 / "experiment_registry.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[registry]", V02 / "experiment_registry.json", flush=True)


if __name__ == "__main__":
    main()
