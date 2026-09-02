#!/usr/bin/env python3
"""One-shot probe of mRI / RadHAR layout on the lab machine."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import os

MRI = Path(os.environ.get("RADARMIND_MRI_ROOT", ""))
RAD = Path(os.environ.get("RADARMIND_RADHAR_ROOT", ""))


def main() -> None:
    print("MRI", MRI.exists(), MRI)
    radar = MRI / "raw_data" / "radar"
    labs = MRI / "raw_data" / "videolabels"
    print("radar files", sorted(p.name for p in radar.glob("*")) if radar.is_dir() else "MISSING")
    print("label files", sorted(p.name for p in labs.glob("*")) if labs.is_dir() else "MISSING")
    readmes = list(MRI.rglob("README*")) + list(MRI.parent.glob("README*"))
    print("readmes", [str(p) for p in readmes])
    jf = labs / "subject1.json"
    if not jf.exists():
        cands = list(labs.glob("*.json"))
        jf = cands[0] if cands else None
        print("using", jf)
    if jf and jf.exists():
        d = json.loads(jf.read_text(encoding="utf-8"))
        print("json_keys", list(d.keys()))
        print("subject", d.get("subject"))
        labs_d = d.get("labels", {})
        print("n_actions", len(labs_d))
        for k, v in labs_d.items():
            print("  LABEL", k, v)
    csvs = sorted(radar.glob("*.csv")) if radar.is_dir() else []
    print("n_csv", len(csvs), "examples", [p.name for p in csvs[:5]])
    if csvs:
        p = csvs[0]
        text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        print("csv0", p.name, "nlines", len(text), "header", text[0] if text else "")
        for ln in text[1:6]:
            print("  ", ln[:200])
    print("aligned_radar", list((MRI / "aligned_data" / "radar").glob("*"))[:8] if (MRI / "aligned_data" / "radar").is_dir() else None)

    print("RAD", RAD.exists())
    print("rad children", [p.name for p in RAD.iterdir()])
    data = RAD / "Data"
    if data.is_dir():
        for split in ("Train", "Test"):
            for act in sorted((data / split).iterdir()) if (data / split).is_dir() else []:
                n = len(list(act.glob("*.txt")))
                print(f"  {split}/{act.name} n_txt={n}")
    vox = RAD / "DataPreprocessing" / "voxels.py"
    print("voxels.py", vox.exists(), vox.stat().st_size if vox.exists() else 0)
    txts = list((data / "Train").rglob("*.txt")) if data.is_dir() else []
    if txts:
        t = txts[0]
        raw = t.read_text(encoding="utf-8", errors="replace")
        print("txt0", t, "size", t.stat().st_size, "seps", raw.count("---"))
        print(raw[:800])
        print("NAME", t.name)


if __name__ == "__main__":
    main()
