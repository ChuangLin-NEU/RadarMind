#!/usr/bin/env python3
"""Phase 4 paths. Reuses Phase 1B manifests/cache. Does not rewrite Phase 1–3."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
P1B = ROOT / "results" / "phase1_pointcloud"
OUT = ROOT / "results" / "phase4_continuous"
SEG_CSV = P1B / "mri_segments.csv"
QC_CSV = P1B / "mri_qc.csv"
SI_SPLIT = ROOT / "data_splits" / "mri_subject_independent.csv"
SUMMARY = P1B / "mri_summary.json"
T = 32
STRIDE = 16  # Phase 1B cache; raw CSV not on this machine, cannot rebuild stride=8
FPS_NOTE = "approx_fps from mri_qc.csv (Time span / unique frames)"

CLASS_NAMES = [
    "T pose",
    "pose_1",
    "pose_2",
    "pose_3",
    "pose_4",
    "pose_5",
    "pose_6",
    "pose_7",
    "pose_8",
    "pose_9",
    "pose_10",
    "free_form",
    "walk",
]


def subject_split() -> dict[str, str]:
    rec = json.loads(SUMMARY.read_text(encoding="utf-8"))
    return {str(k): str(v) for k, v in rec["si_subjects"].items()}


def qc_rows() -> list[dict]:
    return list(csv.DictReader(QC_CSV.open(encoding="utf-8")))


def segment_rows() -> list[dict]:
    return list(csv.DictReader(SEG_CSV.open(encoding="utf-8")))


def fps_of(source_file: str) -> float:
    for r in qc_rows():
        if r["source_file"] == source_file:
            return float(r["approx_fps"])
    return 10.0


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
