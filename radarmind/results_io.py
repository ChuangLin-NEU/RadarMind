"""Read Phase 1 metrics. Do not hardcode Accuracy / F1 in the UI."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from radarmind.paths import PHASE1A_PREFIX_METRICS, PHASE1A_RANDOM_METRICS, PHASE1B_RUNS, ROOT


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _runs() -> list[dict]:
    if not PHASE1B_RUNS.is_file():
        return []
    with PHASE1B_RUNS.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _f(row: dict, key: str) -> float | None:
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def pick_run(dataset: str, split: str, features: str | None = None, temporal: str | None = None, model: str | None = None) -> dict | None:
    hits = []
    for r in _runs():
        if r.get("dataset") != dataset:
            continue
        if r.get("split") != split:
            continue
        if features is not None and r.get("features") != features:
            continue
        if temporal is not None and str(r.get("temporal_length")) != str(temporal):
            continue
        if model is not None and r.get("model") != model:
            continue
        hits.append(r)
    return hits[0] if hits else None


def generalization_table() -> list[dict]:
    rows = []
    if PHASE1A_RANDOM_METRICS.is_file() and PHASE1A_PREFIX_METRICS.is_file():
        a = _read_json(PHASE1A_RANDOM_METRICS)
        b = _read_json(PHASE1A_PREFIX_METRICS)
        rows.append(
            {
                "dataset": "CI4R 77 GHz",
                "protocol_a": "random-record",
                "protocol_b": "prefix-group-disjoint",
                "protocol_b_note": "prefix-group-disjoint ≠ subject-independent",
                "model": "ResNet18",
                "acc_a": a.get("accuracy"),
                "f1_a": a.get("macro_f1"),
                "acc_b": b.get("accuracy"),
                "f1_b": b.get("macro_f1"),
                "gap_acc": float(a["accuracy"]) - float(b["accuracy"]),
                "gap_f1": float(a["macro_f1"]) - float(b["macro_f1"]),
                "source_a": str(PHASE1A_RANDOM_METRICS.relative_to(ROOT)),
                "source_b": str(PHASE1A_PREFIX_METRICS.relative_to(ROOT)),
            }
        )
    rnd = pick_run("mRI", "random_segment", "XYZVI", "32", "pointnet_gru")
    si = pick_run("mRI", "subject_independent", "XYZ", "32", "pointnet_gru")
    si_vi = pick_run("mRI", "subject_independent", "XYZVI", "32", "pointnet_gru")
    if rnd and si:
        rows.append(
            {
                "dataset": "mRI",
                "protocol_a": "random-segment",
                "protocol_b": "subject-independent",
                "protocol_b_note": "official subject ids",
                "model": "PointNet-GRU (random XYZVI / SI XYZ)",
                "acc_a": _f(rnd, "accuracy"),
                "f1_a": _f(rnd, "macro_f1"),
                "acc_b": _f(si, "accuracy"),
                "f1_b": _f(si, "macro_f1"),
                "gap_acc": (_f(rnd, "accuracy") or 0) - (_f(si, "accuracy") or 0),
                "gap_f1": (_f(rnd, "macro_f1") or 0) - (_f(si, "macro_f1") or 0),
                "source_a": "results/phase1_pointcloud/training_runs.csv",
                "source_b": "results/phase1_pointcloud/training_runs.csv",
                "same_setup_xyzvi_gap_acc": ((_f(rnd, "accuracy") or 0) - (_f(si_vi, "accuracy") or 0)) if si_vi else None,
                "same_setup_xyzvi_gap_f1": ((_f(rnd, "macro_f1") or 0) - (_f(si_vi, "macro_f1") or 0)) if si_vi else None,
            }
        )
    pn = pick_run("RadHAR", "official_train_test", "XYZV", "60", "pointnet_gru")
    vx = pick_run("RadHAR", "official_train_test", "occupancy", "60", "voxel_cnn_gru")
    if pn:
        rows.append(
            {
                "dataset": "RadHAR",
                "protocol_a": "official train/test",
                "protocol_b": "official train/test",
                "protocol_b_note": "no subject metadata",
                "model": "PointNet-GRU XYZV vs Voxel CNN-GRU",
                "acc_a": _f(pn, "accuracy"),
                "f1_a": _f(pn, "macro_f1"),
                "acc_b": _f(vx, "accuracy") if vx else None,
                "f1_b": _f(vx, "macro_f1") if vx else None,
                "gap_acc": None,
                "gap_f1": None,
                "source_a": "results/phase1_pointcloud/training_runs.csv",
                "source_b": "results/phase1_pointcloud/training_runs.csv",
            }
        )
    return rows


def temporal_rows() -> list[dict]:
    out = []
    t1 = pick_run("mRI", "subject_independent", "XYZVI", "1", "pointnet_pool")
    t32 = pick_run("mRI", "subject_independent", "XYZVI", "32", "pointnet_gru")
    if t1 and t32:
        out.append(
            {
                "dataset": "mRI",
                "t_short": 1,
                "t_long": 32,
                "acc_short": _f(t1, "accuracy"),
                "acc_long": _f(t32, "accuracy"),
                "f1_short": _f(t1, "macro_f1"),
                "f1_long": _f(t32, "macro_f1"),
            }
        )
    r1 = pick_run("RadHAR", "official_train_test", "XYZVI", "1", "pointnet_pool")
    r60 = pick_run("RadHAR", "official_train_test", "XYZV", "60", "pointnet_gru")
    if r1 and r60:
        out.append(
            {
                "dataset": "RadHAR",
                "t_short": 1,
                "t_long": 60,
                "acc_short": _f(r1, "accuracy"),
                "acc_long": _f(r60, "accuracy"),
                "f1_short": _f(r1, "macro_f1"),
                "f1_long": _f(r60, "macro_f1"),
            }
        )
    return out


def ablation_block(dataset: str, split: str) -> dict:
    feats = []
    for feat in ("XYZ", "XYZV", "XYZVI"):
        r = pick_run(dataset, split, feat, None, "pointnet_gru")
        if r:
            feats.append(
                {
                    "features": feat,
                    "accuracy": _f(r, "accuracy"),
                    "macro_f1": _f(r, "macro_f1"),
                    "temporal_length": r.get("temporal_length"),
                }
            )
    text = _ablation_text(dataset, feats)
    return {"dataset": dataset, "split": split, "rows": feats, "text": text}


def _ablation_text(dataset: str, rows: list[dict]) -> str:
    by = {r["features"]: r for r in rows}
    if not {"XYZ", "XYZV", "XYZVI"} <= set(by):
        if not rows:
            return f"{dataset}：training_runs.csv 里没有对应的 PointNet-GRU 行。"
        parts = [f"{r['features']} Accuracy={r['accuracy']:.3f} Macro-F1={r['macro_f1']:.3f}" for r in rows]
        return f"{dataset} 现有特征结果：" + "；".join(parts) + "。"
    xyz, xyzv, xyzvi = by["XYZ"], by["XYZV"], by["XYZVI"]
    bits = []
    dv = (xyzv["accuracy"] or 0) - (xyz["accuracy"] or 0)
    di = (xyzvi["accuracy"] or 0) - (xyzv["accuracy"] or 0)
    if dv > 0.01:
        bits.append(f"加上 velocity 后 Accuracy 提高 {dv:.3f}")
    elif dv < -0.01:
        bits.append(f"加上 velocity 后 Accuracy 下降 {abs(dv):.3f}，主指标没有提高")
    else:
        bits.append("加上 velocity 后 Accuracy 变化不到 0.01")
    if di > 0.01:
        bits.append(f"再加 intensity 后 Accuracy 提高 {di:.3f}")
    elif di < -0.01:
        bits.append(f"再加 intensity 后 Accuracy 下降 {abs(di):.3f}，没有进一步提升")
    else:
        bits.append("再加 intensity 后 Accuracy 变化不到 0.01")
    return f"{dataset}（读 training_runs.csv）：" + "；".join(bits) + "。"


def research_status() -> dict:
    openset = ROOT / "results" / "phase5_open_set" / "eval_summary.json"
    done = [
        "CI4R 77 GHz HAR",
        "mRI point-cloud HAR",
        "RadHAR point-cloud HAR",
        "mRI cross-subject",
        "CI4R cross-radar",
        "Shared Motion Representation v0.1",
        "Continuous labeled-interval inference",
        "Motion Timeline",
        "DeepSeek interpretation",
        "Replay Application",
    ]
    if openset.is_file():
        done.append("Open-set recognition MVP")
    if (ROOT / "results" / "phase7_radar_invariance" / "phase7_radar_invariance_report.md").is_file():
        done.append("Radar invariance audit (Phase 7)")
    p8 = ROOT / "results" / "phase8_physical_representation" / "answers.json"
    blocked = []
    partial = [
        "sensor-invariant motion representation",
        "continuous full-stream segmentation",
        "open-world understanding",
        "open-vocabulary recognition",
    ]
    if p8.is_file():
        a8 = json.loads(p8.read_text(encoding="utf-8"))
        stt = a8.get("physical_representation", "")
        if stt == "BLOCKED_BY_DATA":
            blocked.append("Physical Representation (Phase 8)")
        elif stt in ("DONE", "R2D_BUILT"):
            done.append("Physical Representation R2-D (Phase 8B)")
        elif stt == "PARTIAL":
            partial.insert(0, "Physical Representation (Phase 8)")
    p10z = ROOT / "results" / "phase10_closure" / "answers.json"
    cross_radar = {
        "Benchmark": "DONE",
        "Representation audit": "DONE",
        "Physical Doppler reconstruction": "DONE",
        "Velocity alignment": "PARTIAL — 24/77 only",
        "Sensor calibration": "FAILED",
        "Empirical sensor translation": "FAILED",
        "Geometry-aware modeling": "BLOCKED BY DATA",
        "Next-data protocol": "READY",
        "CI4R research": "CLOSED",
        "Cross-radar diagnosis": "COMPLETE",
        "Empirical translation": "STOPPED",
        "CURRENT BOTTLENECK": "SPATIAL / ACQUISITION METADATA",
        "Acquisition Toolkit": "READY" if (ROOT / "results" / "acquisition_toolkit_v01" / "answers.json").is_file() else "PARTIAL",
        "Next": "REAL DATA COLLECTION",
        "Phase 11": "WAITING FOR REAL DATA",
    }
    if p10z.is_file():
        done.append("Cross-radar scientific closure (Phase 10Z)")
        blocked.append("Geometry-aware measurement model (no range/aspect/orientation)")
        partial = [p for p in partial if p != "sensor-invariant motion representation"]
        partial.insert(0, "24/77 velocity axis only; 10 GHz fc unresolved")
    tk = ROOT / "results" / "acquisition_toolkit_v01" / "answers.json"
    if tk.is_file():
        done.append("Acquisition Toolkit v0.1 (MOCK)")
    return {
        "DONE": done,
        "PARTIAL": partial,
        "BLOCKED_BY_DATA": blocked,
        "PENDING": [
            "cross-environment",
            "MM-Fi radar",
            "MiliPoint",
            "FAU",
            "Live Radar",
        ],
        "CROSS_RADAR": cross_radar,
    }
