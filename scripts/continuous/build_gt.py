#!/usr/bin/env python3
"""Recording manifest + ground-truth timelines from Phase 1B CSVs. Frame # only."""
from __future__ import annotations

import json
from collections import defaultdict

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.continuous.common import OUT, qc_rows, segment_rows, subject_split, write_csv  # noqa: E402


def unlabeled_intervals(frame_min: int, frame_max: int, segs: list[dict]) -> list[dict]:
    ordered = sorted(segs, key=lambda s: int(s["segment_start"]))
    covered: list[tuple[int, int]] = []
    for s in ordered:
        a, b = int(s["segment_start"]), int(s["segment_end"])
        if covered and a <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], b))
        else:
            covered.append((a, b))
    gaps = []
    cursor = frame_min
    for a, b in covered:
        if cursor < a:
            gaps.append({"start_frame": cursor, "end_frame": a - 1, "activity": "UNLABELED"})
        cursor = max(cursor, b + 1)
    if cursor <= frame_max:
        gaps.append({"start_frame": cursor, "end_frame": frame_max, "activity": "UNLABELED"})
    return gaps


def labeled_events(segs: list[dict]) -> list[dict]:
    ordered = sorted(segs, key=lambda s: (int(s["segment_start"]), int(s["segment_end"])))
    events = []
    for i, s in enumerate(ordered):
        a, b = int(s["segment_start"]), int(s["segment_end"])
        if i + 1 < len(ordered):
            nxt = int(ordered[i + 1]["segment_start"])
            if nxt <= b:
                b = nxt - 1
        if b < a:
            continue
        events.append(
            {
                "start_frame": a,
                "end_frame": b,
                "activity": s["action"],
                "action_raw": s["action_raw"],
                "action_segment_id": s["action_segment_id"],
            }
        )
    return events


def build_timeline(rec: dict, segs: list[dict], fps: float) -> dict:
    labeled = labeled_events(segs)
    unlabeled = unlabeled_intervals(int(rec["frame_min"]), int(rec["frame_max"]), segs)
    intervals = labeled + unlabeled
    intervals.sort(key=lambda x: x["start_frame"])
    events = []
    for iv in intervals:
        a, b = iv["start_frame"], iv["end_frame"]
        events.append(
            {
                **iv,
                "start_time": a / fps,
                "end_time": (b + 1) / fps,
                "n_frames": b - a + 1,
                "source": "videolabels Frame #",
            }
        )
    return {
        "recording": rec["recording"],
        "subject": rec["subject"],
        "source_file": rec["source_file"],
        "split": rec["split"],
        "frame_min": int(rec["frame_min"]),
        "frame_max": int(rec["frame_max"]),
        "frame_rate": fps,
        "label_rule": "JSON integers are Frame # inclusive. Time column is not used.",
        "unlabeled_policy": "UNLABELED intervals are kept. They are not assigned to neighboring actions.",
        "events": events,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "ground_truth_timelines").mkdir(exist_ok=True)
    split = subject_split()
    segs = segment_rows()
    by_src: dict[str, list] = defaultdict(list)
    for s in segs:
        by_src[s["source_file"]].append(s)
    man_rows = []
    for q in qc_rows():
        src = q["source_file"]
        slist = by_src.get(src, [])
        subject = q["subject"]
        fps = float(q["approx_fps"])
        rec = {
            "subject": subject,
            "source_radar_csv": src,
            "source_file": src,
            "recording": slist[0]["recording"] if slist else src.replace(".csv", ""),
            "session": slist[0]["session"] if slist else "",
            "start_frame": int(q["frame_min"]),
            "end_frame": int(q["frame_max"]),
            "frame_min": int(q["frame_min"]),
            "frame_max": int(q["frame_max"]),
            "frame_rate": fps,
            "approx_fps": fps,
            "labeled_segments": len(slist),
            "unlabeled_intervals": len(unlabeled_intervals(int(q["frame_min"]), int(q["frame_max"]), slist)),
            "n_unlabeled_frames": int(float(q["n_unlabeled_frames"])),
            "actions": "|".join(s["action"] for s in sorted(slist, key=lambda x: int(x["segment_start"]))),
            "total_frames": int(q["n_frames"]),
            "split": split.get(subject, ""),
            "protocol": "subject_independent",
        }
        man_rows.append(rec)
        tl = build_timeline(rec, slist, fps)
        dest = OUT / "ground_truth_timelines" / f"{rec['recording']}.json"
        dest.write_text(json.dumps(tl, indent=2, ensure_ascii=False), encoding="utf-8")
    fields = [
        "subject",
        "source_radar_csv",
        "recording",
        "session",
        "start_frame",
        "end_frame",
        "frame_rate",
        "labeled_segments",
        "unlabeled_intervals",
        "n_unlabeled_frames",
        "actions",
        "total_frames",
        "split",
        "protocol",
    ]
    write_csv(OUT / "mri_recording_manifest.csv", man_rows, fields)
    print("recordings", len(man_rows), "gt", OUT / "ground_truth_timelines", flush=True)


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[2]))
    main()
