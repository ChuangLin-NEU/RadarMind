#!/usr/bin/env python3
"""Pick 3 held-out actions from Phase 1B SI metrics + coverage. Do not guess."""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.pointcloud.mri import CLASS_NAMES  # noqa: E402

P1B = ROOT / "results" / "phase1_pointcloud"
OUT = ROOT / "results" / "phase5_open_set"
MET = P1B / "baselines" / "mri_si_gru_xyz" / "metrics.json"
CM = P1B / "baselines" / "mri_si_gru_xyz" / "cm.csv"
SEG = P1B / "mri_segments.csv"
SI = ROOT / "data_splits" / "mri_subject_independent.csv"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    met = json.loads(MET.read_text(encoding="utf-8"))
    per = {r["class"]: r for r in met["per_class"]}
    cm = np.loadtxt(CM, delimiter=",")
    off = cm.copy()
    np.fill_diagonal(off, 0)
    leak = {CLASS_NAMES[i]: float(off[i].sum() / max(cm[i].sum(), 1)) for i in range(len(CLASS_NAMES))}
    pose_to_pose = {}
    pose_idx = [i for i, n in enumerate(CLASS_NAMES) if n.startswith("pose_")]
    for i in pose_idx:
        pose_to_pose[CLASS_NAMES[i]] = float(off[i, pose_idx].sum() / max(cm[i].sum(), 1))
    segs = list(csv.DictReader(SEG.open(encoding="utf-8")))
    split_rows = list(csv.DictReader(SI.open(encoding="utf-8")))
    subj_split = {}
    for r in split_rows:
        subj_split[r["subject"]] = r["split"]
    cov = Counter()
    test_cov = Counter()
    for r in segs:
        cov[r["action"]] += 1
        if subj_split.get(r["subject"]) == "test":
            test_cov[r["action"]] += 1
    easy = max((c for c in CLASS_NAMES if c.startswith("pose_")), key=lambda c: per[c]["f1"])
    confused = max(pose_to_pose, key=pose_to_pose.get)
    semantic = "walk" if test_cov.get("walk", 0) >= 2 else "free_form"
    selected = [
        {
            "action": easy,
            "role": "A_easy",
            "reason": f"SI PointNet-GRU XYZ closed-set F1={per[easy]['f1']:.3f}, highest among pose_*",
            "closed_set_f1": per[easy]["f1"],
            "n_segments": cov[easy],
            "n_test_segments": test_cov[easy],
        },
        {
            "action": confused,
            "role": "B_confused_pose",
            "reason": f"largest fraction of errors going to other pose_* ({pose_to_pose[confused]:.3f}); closed-set F1={per[confused]['f1']:.3f}",
            "closed_set_f1": per[confused]["f1"],
            "pose_to_pose_error_frac": pose_to_pose[confused],
            "n_segments": cov[confused],
            "n_test_segments": test_cov[confused],
        },
        {
            "action": semantic,
            "role": "C_semantic",
            "reason": f"{semantic} is not a numbered pose; SI F1={per[semantic]['f1']:.3f}; kept as semantic held-out",
            "closed_set_f1": per[semantic]["f1"],
            "n_segments": cov[semantic],
            "n_test_segments": test_cov[semantic],
        },
    ]
    # uniqueness
    seen = set()
    uniq = []
    for s in selected:
        if s["action"] in seen:
            alt = next(a for a in ("walk", "free_form", "T pose") if a not in seen)
            s = dict(s)
            s["action"] = alt
            s["reason"] += f"; collided, switched to {alt}"
        seen.add(s["action"])
        uniq.append(s)
    payload = {
        "protocol": "leave_one_action_out + subject_independent",
        "source_metrics": str(MET.relative_to(ROOT)),
        "selected": uniq,
        "per_class_f1": {k: per[k]["f1"] for k in CLASS_NAMES},
        "note": "First three folds. Full 13-fold runs if these finish quickly.",
    }
    (OUT / "selected_open_set_actions.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(uniq, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
