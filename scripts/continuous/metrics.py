#!/usr/bin/env python3
"""Mode A: labeled intervals only. UNLABELED is never a 14th class."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuous.common import CLASS_NAMES, OUT, write_csv  # noqa: E402


def _load_gt(recording: str) -> dict:
    return json.loads((OUT / "ground_truth_timelines" / f"{recording}.json").read_text(encoding="utf-8"))


def _load_pred(recording: str) -> dict:
    return json.loads((OUT / "predicted_timelines" / f"{recording}.json").read_text(encoding="utf-8"))


def _paint_gt(gt: dict) -> dict[int, str]:
    out = {}
    for e in gt["events"]:
        for f in range(int(e["start_frame"]), int(e["end_frame"]) + 1):
            out[f] = e["activity"]
    return out


def _paint_pred(windows: list[dict]) -> dict[int, list[str]]:
    hits: dict[int, list[str]] = {}
    for w in windows:
        for f in range(int(w["start_frame"]), int(w["end_frame"]) + 1):
            hits.setdefault(f, []).append(w["activity"])
    return hits


def _majority(xs: list[str]) -> str:
    c: dict[str, int] = {}
    for x in xs:
        c[x] = c.get(x, 0) + 1
    return max(c, key=c.get)


def frame_scores(gt_map: dict[int, str], pred_hits: dict[int, list[str]]) -> dict:
    y, p = [], []
    for f, lab in gt_map.items():
        if lab == "UNLABELED":
            continue
        if f not in pred_hits:
            continue
        y.append(lab)
        p.append(_majority(pred_hits[f]))
    if not y:
        return {"accuracy": None, "macro_f1": None, "n": 0}
    return {
        "accuracy": float(accuracy_score(y, p)),
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
        "n": int(len(y)),
    }


def window_scores(windows: list[dict]) -> dict:
    y = [w["gt"] for w in windows]
    p = [w["activity"] for w in windows]
    return {
        "accuracy": float(accuracy_score(y, p)) if y else None,
        "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)) if y else None,
        "n": int(len(y)),
    }


def _iou(a0, a1, b0, b1) -> float:
    inter = max(0, min(a1, b1) - max(a0, b0) + 1)
    union = (a1 - a0 + 1) + (b1 - b0 + 1) - inter
    return inter / union if union else 0.0


def segment_scores(gt_events: list[dict], pred_events: list[dict], iou_thr=0.5) -> dict:
    gt = [e for e in gt_events if e["activity"] != "UNLABELED"]
    pred = [e for e in pred_events if e["activity"] != "UNLABELED"]
    used = set()
    tp = 0
    for g in gt:
        best_i, best = -1, 0.0
        for i, p in enumerate(pred):
            if i in used or p["activity"] != g["activity"]:
                continue
            v = _iou(g["start_frame"], g["end_frame"], p["start_frame"], p["end_frame"])
            if v > best:
                best, best_i = v, i
        if best >= iou_thr:
            tp += 1
            used.add(best_i)
    prec = tp / len(pred) if pred else 0.0
    rec = tp / len(gt) if gt else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {
        "segment_precision": float(prec),
        "segment_recall": float(rec),
        "segment_f1": float(f1),
        "n_gt": len(gt),
        "n_pred": len(pred),
        "tp": tp,
    }


def boundary_errors(gt_events: list[dict], pred_events: list[dict], fps: float) -> dict:
    gt_b = []
    pred_b = []
    gtl = [e for e in gt_events if e["activity"] != "UNLABELED"]
    pdl = [e for e in pred_events if e["activity"] != "UNLABELED"]
    for i in range(1, len(gtl)):
        gt_b.append((int(gtl[i]["start_frame"]), gtl[i]["activity"], gtl[i - 1]["activity"]))
    for i in range(1, len(pdl)):
        pred_b.append((int(pdl[i]["start_frame"]), pdl[i]["activity"], pdl[i - 1]["activity"]))
    errs = []
    used = set()
    for gf, gto, gfr in gt_b:
        best_i, best = -1, 10**9
        for i, (pf, pto, pfr) in enumerate(pred_b):
            if i in used:
                continue
            if pto != gto or pfr != gfr:
                continue
            d = abs(pf - gf)
            if d < best:
                best, best_i = d, i
        if best_i >= 0:
            used.add(best_i)
            errs.append(best)
        else:
            # nearest same-target boundary
            cands = [abs(pf - gf) for pf, pto, _ in pred_b if pto == gto]
            if cands:
                errs.append(min(cands))
    if not errs:
        return {"mean_frames": None, "median_frames": None, "mean_seconds": None, "median_seconds": None, "n": 0}
    arr = np.asarray(errs, dtype=np.float64)
    return {
        "mean_frames": float(arr.mean()),
        "median_frames": float(np.median(arr)),
        "mean_seconds": float(arr.mean() / fps),
        "median_seconds": float(np.median(arr) / fps),
        "n": int(len(arr)),
        "errors_frames": [float(x) for x in arr],
    }


def main() -> None:
    pred_dir = OUT / "predicted_timelines"
    recs = sorted(p.stem for p in pred_dir.glob("*.json"))
    run_rows, seg_rows, bnd_rows = [], [], []
    all_y, all_p = [], []
    all_err = []
    for rec in recs:
        pred = _load_pred(rec)
        if pred.get("split") != "test":
            continue
        gt = _load_gt(rec)
        gt_map = _paint_gt(gt)
        fps = float(pred["frame_rate"])
        for mode in ("raw", "isolated_flip", "confidence_aware"):
            wins = pred["modes"][mode]["windows"]
            ev = pred["modes"][mode]["events"]
            wsc = window_scores(wins)
            fsc = frame_scores(gt_map, _paint_pred(wins))
            ssc = segment_scores(gt["events"], ev)
            bsc = boundary_errors(gt["events"], ev, fps)
            run_rows.append(
                {
                    "recording": rec,
                    "subject": pred["subject"],
                    "split": "test",
                    "mode": mode,
                    "n_windows": wsc["n"],
                    "window_accuracy": wsc["accuracy"],
                    "window_macro_f1": wsc["macro_f1"],
                    "frame_accuracy": fsc["accuracy"],
                    "frame_macro_f1": fsc["macro_f1"],
                    "n_labeled_frames_eval": fsc["n"],
                    "note": "Mode A: labeled intervals only",
                }
            )
            seg_rows.append({"recording": rec, "subject": pred["subject"], "mode": mode, **ssc})
            bnd_rows.append(
                {
                    "recording": rec,
                    "subject": pred["subject"],
                    "mode": mode,
                    "mean_frames": bsc["mean_frames"],
                    "median_frames": bsc["median_frames"],
                    "mean_seconds": bsc["mean_seconds"],
                    "median_seconds": bsc["median_seconds"],
                    "n_boundaries": bsc["n"],
                }
            )
            if mode == "confidence_aware":
                all_y.extend([w["gt"] for w in wins])
                all_p.extend([w["activity"] for w in wins])
                all_err.extend(bsc.get("errors_frames") or [])
    write_csv(
        OUT / "continuous_runs.csv",
        run_rows,
        [
            "recording",
            "subject",
            "split",
            "mode",
            "n_windows",
            "window_accuracy",
            "window_macro_f1",
            "frame_accuracy",
            "frame_macro_f1",
            "n_labeled_frames_eval",
            "note",
        ],
    )
    write_csv(
        OUT / "segment_metrics.csv",
        seg_rows,
        ["recording", "subject", "mode", "segment_precision", "segment_recall", "segment_f1", "n_gt", "n_pred", "tp"],
    )
    write_csv(
        OUT / "boundary_metrics.csv",
        bnd_rows,
        [
            "recording",
            "subject",
            "mode",
            "mean_frames",
            "median_frames",
            "mean_seconds",
            "median_seconds",
            "n_boundaries",
        ],
    )
    if all_y:
        (OUT / "figures").mkdir(exist_ok=True)
        (OUT / "cache").mkdir(exist_ok=True)
        cm = confusion_matrix(all_y, all_p, labels=CLASS_NAMES)
        np.savetxt(OUT / "figures" / "continuous_confusion.csv", cm, fmt="%d", delimiter=",")
        (OUT / "cache").mkdir(exist_ok=True)
        np.savez(OUT / "cache" / "test_window_labels.npz", y=np.array(all_y), p=np.array(all_p), errors=np.array(all_err))
    print("wrote metrics", len(run_rows), flush=True)


if __name__ == "__main__":
    main()
