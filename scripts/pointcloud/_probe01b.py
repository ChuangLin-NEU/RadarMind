#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import os

MRI = Path(os.environ.get("RADARMIND_MRI_ROOT", ""))
RAD = Path(os.environ.get("RADARMIND_RADHAR_ROOT", ""))


def main() -> None:
    labs = MRI / "raw_data" / "videolabels"
    radar = MRI / "raw_data" / "radar"
    all_keys = Counter()
    per_sub = {}
    for jf in sorted(labs.glob("*.json")):
        d = json.loads(jf.read_text(encoding="utf-8"))
        keys = list(d.get("labels", {}).keys())
        per_sub[d.get("subject", jf.stem)] = keys
        all_keys.update(keys)
        for k, v in d.get("labels", {}).items():
            print("SEG", d.get("subject"), k, v, type(v).__name__, len(v) if isinstance(v, list) else "")
    print("KEY_COUNTS", dict(all_keys))
    print("N_SUBJECTS", len(per_sub))

    # subject1 frame vs json
    csv = next(radar.glob("subject1_*.csv"))
    print("CSV", csv)
    # stream: unique frames min max, first/last time
    frames = []
    times = []
    nobj = []
    rows_per = Counter()
    with csv.open(encoding="utf-8") as f:
        header = f.readline()
        for i, line in enumerate(f):
            parts = line.strip().split(",")
            if len(parts) < 8:
                continue
            fr = int(float(parts[0]))
            no = int(float(parts[1]))
            frames.append(fr)
            nobj.append(no)
            times.append(float(parts[7]))
            rows_per[fr] += 1
    frames_u = sorted(set(frames))
    print("subject1 n_rows", len(frames), "unique_frames", len(frames_u), "min", frames_u[0], "max", frames_u[-1])
    print("time min/max", min(times), max(times), "dt_span", max(times) - min(times))
    diffs = np.diff(frames_u)
    print("frame_id gaps>1", int((diffs > 1).sum()), "max_gap", int(diffs.max()) if len(diffs) else None)
    mismatch = sum(1 for fr, n in rows_per.items() if n != nobj[frames.index(fr)])
    # better: per-frame first # Obj vs count
    first_nobj = {}
    with csv.open(encoding="utf-8") as f:
        f.readline()
        for line in f:
            parts = line.strip().split(",")
            fr = int(float(parts[0]))
            no = int(float(parts[1]))
            first_nobj.setdefault(fr, no)
    mm = sum(1 for fr, n in rows_per.items() if n != first_nobj[fr])
    print("#Obj vs rows mismatch frames", mm, "of", len(rows_per))
    pts = np.array(list(rows_per.values()))
    print("pts/frame min/med/max", pts.min(), np.median(pts), pts.max())
    # json 98-139 coverage
    cov = sum(1 for t in range(98, 140) if t in rows_per)
    print("frames 98-139 present", cov, "/43")
    # Time is unix, not 98-139
    print("any Time in [98,139]?", any(98 <= t <= 139 for t in times[:5000]))

    # RadHAR filenames
    names = []
    for p in (RAD / "Data").rglob("*.txt"):
        names.append((p.parent.parent.name, p.parent.name, p.name))
    print("n_txt", len(names))
    print("name examples", names[:15])
    stems = [n[2] for n in names]
    print("stem prefixes", Counter(s.split("_")[0] for s in stems))

    # one txt frame count via point_id==0
    p = RAD / "Data" / "Train" / "boxing" / "20_boxing_1.txt"
    raw = p.read_text(encoding="utf-8", errors="ignore")
    n0 = raw.lower().count("point_id: 0\n") + raw.lower().count("point_id: 0\r")
    print("boxing1 point_id0", n0, "seps", raw.count("---"))


if __name__ == "__main__":
    main()
