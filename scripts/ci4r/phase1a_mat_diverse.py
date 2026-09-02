#!/usr/bin/env python3
"""Read archive MAT/PNG from several actions and tail4 tokens. No raw IQ."""
from __future__ import annotations

import argparse
import csv
import io
import os
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np

CLASS_NAMES = ["Towards", "Away", "Pick", "Bend", "Sit", "Kneel", "Crawl", "Toes", "Limp", "SStep", "Scissor"]
ACTION_FOLDERS = {
    "Towards": "05_Walking_Towards_radar",
    "Away": "06_walking_away_from_Radar",
    "Pick": "07_Picking_up_an_object",
    "Bend": "08_Bending",
    "Sit": "09_Sitting",
    "Kneel": "10_Kneeling",
    "Crawl": "11_Crawling",
    "Toes": "16_Walking_on_both_toes",
    "Limp": "17_limping_with_RL_Stiff",
    "SStep": "18_Short_steps",
    "Scissor": "19_Scissors_gait",
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", "")))
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    import scipy.io as sio
    from PIL import Image

    rows = list(csv.DictReader(Path(args.manifest).open(encoding="utf-8")))
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["action"], r["name_token_tail4"])].append(r)
    picked = []
    for a in CLASS_NAMES:
        for tail in ("0010", "0014", "0029", "0030"):
            items = buckets.get((a, tail), [])
            if items:
                picked.append(items[0])
                break
    recs = []
    with zipfile.ZipFile(args.zip) as zf:
        names = {i.filename.replace("\\", "/") for i in zf.infolist()}
        for r in picked:
            stem = Path(r["archive_member"]).stem
            folder = ACTION_FOLDERS[r["action"]]
            mats = [p for p in names if p.endswith(".mat") and stem in p and "activity_spect_matrices" in p]
            rec = {
                "sample_id": r["sample_id"],
                "action": r["action"],
                "name_token_tail4": r["name_token_tail4"],
                "mat_spec": mats[0] if mats else "MISSING",
            }
            if mats:
                m = sio.loadmat(io.BytesIO(zf.read(mats[0])), squeeze_me=True, struct_as_record=False)
                keys = [k for k in m.keys() if not k.startswith("__")]
                rec["keys"] = ";".join(keys)
                v = m.get("sx1")
                if v is not None:
                    rec["shape"] = list(np.shape(v))
                    rec["dtype"] = str(v.dtype)
                    rec["complex"] = bool(np.iscomplexobj(v))
                    arr = np.abs(v) if np.iscomplexobj(v) else np.asarray(v)
                    rec["abs_min"] = float(np.min(arr))
                    rec["abs_max"] = float(np.max(arr))
                    rec["log10_peak"] = float(np.log10(np.max(arr) + 1e-12))
                    rec["already_log"] = bool(np.max(arr) < 100 and np.min(arr) < 0)
            recs.append(rec)
            print("[mat]", rec, flush=True)
    fields = sorted({k for r in recs for k in r})
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(recs)
    print("[done]", len(recs), out)


if __name__ == "__main__":
    main()
