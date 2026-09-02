#!/usr/bin/env python3
"""Full-recording inference on existing T=32 / stride=16 cache. No retrain. No new large cache."""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radarmind.data.base import DatasetRegistry  # noqa: E402
from radarmind.intelligence.extract import build_intelligence  # noqa: E402
from radarmind.models.registry import ModelRegistry  # noqa: E402
from scripts.continuous.common import CLASS_NAMES, OUT, STRIDE, T, fps_of, segment_rows, subject_split  # noqa: E402
from scripts.continuous.smooth import apply as apply_smooth  # noqa: E402

_WIN = re.compile(r"_w(\d+)$")


def _seg_index() -> dict[str, dict]:
    return {r["action_segment_id"]: r for r in segment_rows()}


def window_frames(sample_id: str, segs: dict[str, dict]) -> tuple[int, int, str, str, str]:
    m = _WIN.search(sample_id)
    off = int(m.group(1)) if m else 0
    sid = _WIN.sub("", sample_id)
    # action_segment_id may include spaces replaced
    row = segs.get(sid)
    if row is None:
        for k, v in segs.items():
            if sample_id.startswith(k + "_w"):
                row = v
                sid = k
                break
    if row is None:
        raise KeyError(sample_id)
    a = int(row["segment_start"]) + off
    return a, a + T - 1, row["source_file"], row["recording"], row["action"]


def merge_events(windows: list[dict], fps: float) -> list[dict]:
    if not windows:
        return []
    events = []
    cur = {
        "activity": windows[0]["activity"],
        "start_frame": windows[0]["start_frame"],
        "end_frame": windows[0]["end_frame"],
        "confs": [windows[0]["confidence"]],
        "uncs": [windows[0]["uncertainty"]],
        "intel": windows[0].get("motion_intelligence", {}),
        "n_windows": 1,
    }
    for w in windows[1:]:
        if w["activity"] == cur["activity"]:
            cur["end_frame"] = max(cur["end_frame"], w["end_frame"])
            cur["confs"].append(w["confidence"])
            cur["uncs"].append(w["uncertainty"])
            cur["intel"] = w.get("motion_intelligence", cur["intel"])
            cur["n_windows"] += 1
        else:
            events.append(_event_from(cur, fps))
            cur = {
                "activity": w["activity"],
                "start_frame": w["start_frame"],
                "end_frame": w["end_frame"],
                "confs": [w["confidence"]],
                "uncs": [w["uncertainty"]],
                "intel": w.get("motion_intelligence", {}),
                "n_windows": 1,
            }
    events.append(_event_from(cur, fps))
    return events


def _event_from(cur: dict, fps: float) -> dict:
    confs = cur["confs"]
    uncs = cur["uncs"]
    mean_c = float(np.mean(confs))
    min_c = float(np.min(confs))
    stats = (cur.get("intel") or {}).get("evidence") or {}
    return {
        "start_frame": int(cur["start_frame"]),
        "end_frame": int(cur["end_frame"]),
        "start_time": float(cur["start_frame"] / fps),
        "end_time": float((cur["end_frame"] + 1) / fps),
        "activity": cur["activity"],
        "mean_confidence": mean_c,
        "min_confidence": min_c,
        "confidence": mean_c,
        "uncertainty": float(np.mean(uncs)) if uncs else 0.0,
        "low_confidence": mean_c < 0.5,
        "n_windows": cur["n_windows"],
        "motion_statistics": {
            k: stats.get(k)
            for k in ("mean_velocity", "peak_abs_velocity", "motion_energy", "mean_points_per_frame")
            if k in stats
        },
    }


def window_f1(windows: list[dict]) -> float:
    y = [w["gt"] for w in windows]
    p = [w["activity"] for w in windows]
    if not y:
        return 0.0
    return float(f1_score(y, p, average="macro", zero_division=0))


def pick_conf_min(val_recs: dict[str, list[dict]]) -> float:
    best_t, best = 0.35, -1.0
    for t in (0.25, 0.35, 0.45, 0.55):
        scores = []
        for wins in val_recs.values():
            labs = [w["raw_activity"] for w in wins]
            confs = [w["confidence"] for w in wins]
            sm = apply_smooth("confidence_aware", labs, confs, conf_min=t)
            tmp = [{**w, "activity": a} for w, a in zip(wins, sm)]
            scores.append(window_f1(tmp))
        score = float(np.mean(scores)) if scores else 0.0
        if score > best:
            best, best_t = score, t
    rec = {"conf_min": best_t, "val_macro_f1": best, "grid": [0.25, 0.35, 0.45, 0.55], "note": "chosen on val subjects only"}
    (OUT / "smoothing_choice.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print("[smooth-pick]", rec, flush=True)
    return best_t


def series_of(windows: list[dict], fps: float) -> list[dict]:
    out = []
    for w in windows:
        ev = (w.get("motion_intelligence") or {}).get("evidence") or {}
        ms = ev.get("motion_statistics") or ev.get("kinematics") or ev
        out.append(
            {
                "time": (w["start_frame"] + w["end_frame"]) / 2.0 / fps,
                "center_frame": int((w["start_frame"] + w["end_frame"]) // 2),
                "activity": w["activity"],
                "confidence": w["confidence"],
                "uncertainty": w["uncertainty"],
                "mean_velocity": ms.get("mean_velocity"),
                "peak_velocity": ms.get("peak_abs_velocity"),
                "motion_energy": ms.get("motion_energy"),
                "gt": w["gt"],
            }
        )
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "predicted_timelines").mkdir(exist_ok=True)
    split = subject_split()
    segs = _seg_index()
    adapter = DatasetRegistry.load_dataset("mri")
    models = ModelRegistry()
    want = {s for s, role in split.items() if role in ("val", "test")}
    grouped: dict[str, list] = defaultdict(list)
    for sid in adapter.sample_ids():
        sample = adapter.get(sid)
        if sample.subject_id not in want:
            continue
        a, b, src, rec, gt = window_frames(sid, segs)
        grouped[src].append(
            {
                "sample_id": sid,
                "subject": sample.subject_id,
                "source_file": src,
                "recording": rec,
                "start_frame": a,
                "end_frame": b,
                "center_frame": (a + b) // 2,
                "gt": gt,
                "split": split[sample.subject_id],
                "sample": sample,
            }
        )
    for src in grouped:
        grouped[src].sort(key=lambda w: (w["start_frame"], w["sample_id"]))

    print("recordings", {k: len(v) for k, v in grouped.items()}, flush=True)
    for src, wins in grouped.items():
        for w in wins:
            p = models.predict(w["sample"], model_name="mri_pointnet_gru", protocol="subject_independent")
            intel = build_intelligence(w["sample"], p).to_dict()
            w["raw_activity"] = p.activity
            w["confidence"] = float(p.confidence)
            w["entropy"] = float(p.prediction_entropy) if p.prediction_entropy != "UNKNOWN" else None
            w["uncertainty"] = float(p.uncertainty) if p.uncertainty != "UNKNOWN" else 0.0
            w["motion_intelligence"] = intel
            w["class_probabilities"] = p.class_probabilities
        print("[infer]", src, len(wins), flush=True)

    val_recs = {k: v for k, v in grouped.items() if v and v[0]["split"] == "val"}
    conf_min = pick_conf_min(val_recs)

    for src, wins in grouped.items():
        fps = fps_of(src)
        modes = {}
        for name in ("raw", "isolated_flip", "confidence_aware"):
            labs = apply_smooth(name, [w["raw_activity"] for w in wins], [w["confidence"] for w in wins], conf_min=conf_min)
            rows = []
            for w, act in zip(wins, labs):
                row = {
                    "sample_id": w["sample_id"],
                    "start_frame": w["start_frame"],
                    "end_frame": w["end_frame"],
                    "center_frame": w["center_frame"],
                    "activity": act,
                    "raw_activity": w["raw_activity"],
                    "confidence": w["confidence"],
                    "entropy": w["entropy"],
                    "uncertainty": w["uncertainty"],
                    "gt": w["gt"],
                    "motion_intelligence": w["motion_intelligence"],
                }
                rows.append(row)
            modes[name] = {
                "windows": rows,
                "events": merge_events(rows, fps),
                "series": series_of(rows, fps),
            }
        payload = {
            "recording": wins[0]["recording"],
            "subject": wins[0]["subject"],
            "source_file": src,
            "split": wins[0]["split"],
            "protocol": "subject_independent",
            "model_name": "mri_pointnet_gru",
            "window": {"T": T, "stride": STRIDE, "note": "Phase 1B cache; raw CSV not reread"},
            "smoothing": {"confidence_aware_conf_min": conf_min, "selected_on": "val subjects"},
            "frame_rate": fps,
            "modes": modes,
        }
        dest = OUT / "predicted_timelines" / f"{wins[0]['recording']}.json"
        dest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        y = [w["gt"] for w in wins]
        p = [w["raw_activity"] for w in wins]
        print(
            "[saved]",
            dest.name,
            "acc",
            float(accuracy_score(y, p)),
            "f1",
            float(f1_score(y, p, average="macro", zero_division=0)),
            flush=True,
        )


if __name__ == "__main__":
    main()
