#!/usr/bin/env python3
"""Pick 3 TEST-subject recordings. No train subject. No synthetic concat for metrics."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuous.common import OUT  # noqa: E402


def main() -> None:
    runs = list(csv.DictReader((OUT / "continuous_runs.csv").open(encoding="utf-8")))
    segs = list(csv.DictReader((OUT / "segment_metrics.csv").open(encoding="utf-8")))
    bnds = list(csv.DictReader((OUT / "boundary_metrics.csv").open(encoding="utf-8")))
    test = [r for r in runs if r["mode"] == "confidence_aware"]
    by = {r["recording"]: r for r in test}

    def f(row, key):
        try:
            return float(row[key])
        except (TypeError, ValueError, KeyError):
            return 0.0

    bnd = {r["recording"]: r for r in bnds if r["mode"] == "confidence_aware"}
    ranked_acc = sorted(test, key=lambda r: f(r, "window_accuracy"), reverse=True)
    ranked_bnd = sorted(test, key=lambda r: f(bnd.get(r["recording"], {}), "median_frames"), reverse=True)
    ranked_bad = sorted(test, key=lambda r: f(r, "window_macro_f1"))
    picks = []
    used = set()
    for role, pool, why in (
        ("Demo A", ranked_acc, "highest window accuracy among test recordings"),
        ("Demo B", ranked_bnd, "largest median boundary error among test recordings"),
        ("Demo C", ranked_bad, "lowest window macro-F1 among test recordings"),
    ):
        hit = next((r for r in pool if r["recording"] not in used), None)
        if hit is None:
            continue
        used.add(hit["recording"])
        pred = json.loads((OUT / "predicted_timelines" / f"{hit['recording']}.json").read_text(encoding="utf-8"))
        gt = json.loads((OUT / "ground_truth_timelines" / f"{hit['recording']}.json").read_text(encoding="utf-8"))
        acts = [e["activity"] for e in gt["events"] if e["activity"] != "UNLABELED"]
        picks.append(
            {
                "demo_id": role.replace(" ", "_").lower(),
                "title": role,
                "role": role,
                "recording": hit["recording"],
                "subject": hit["subject"],
                "dataset": "mri",
                "protocol": "subject_independent",
                "split": "test",
                "synthetic": False,
                "why": why,
                "n_labeled_actions": len(set(acts)),
                "actions": "|".join(acts),
                "window_accuracy": hit.get("window_accuracy"),
                "window_macro_f1": hit.get("window_macro_f1"),
                "n_windows": hit.get("n_windows"),
                "window_ids": [w["sample_id"] for w in pred["modes"]["confidence_aware"]["windows"]],
            }
        )
    dest = OUT / "demo_registry.json"
    dest.write_text(
        json.dumps(
            {
                "demos": picks,
                "note": "All demos are real TEST-subject recordings. Not concatenated. Not used as train.",
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(picks, flush=True)


if __name__ == "__main__":
    main()
