#!/usr/bin/env python3
"""Fill gaps from Phase 0: CI4R 77GHz sample, nested mRI zip, MiliPoint pickle, counts."""
from __future__ import annotations

import csv
import json
import pickle
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

import os

ROOT = Path(os.environ.get("RADARMIND_DATA_ROOT", ""))
OUT = Path(os.environ.get("RADARMIND_DATA_AUDIT_OUT", "results/data_audit"))
WORK = Path(os.environ.get("RADARMIND_DATA_AUDIT_WORK", "results/data_audit/samples"))
EXT = Path(os.environ.get("RADARMIND_DATA_EXTRACTED", ""))


def ci4r_counts_and_77ghz():
    zpath = ROOT / "ci4r" / "raw_data.zip"
    second = Counter()
    classes = defaultdict(Counter)
    bins_77 = []
    dats = []
    with zipfile.ZipFile(zpath) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p]
            if len(parts) >= 2:
                second[parts[1]] += 1
            if len(parts) >= 3:
                classes[parts[1]][parts[2]] += 1
            if name.lower().endswith(".bin") and ("77" in name.lower() or "iwr" in name.lower()):
                bins_77.append((name, info.file_size))
            if name.lower().endswith(".dat"):
                dats.append((name, info.file_size))
    # pick a few large 77ghz bins
    bins_77.sort(key=lambda x: -x[1])
    samples = []
    dest = WORK / "CI4R" / "bins77"
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zpath) as zf:
        for name, sz in bins_77[:3]:
            out = dest / Path(name).name
            with zf.open(name) as src, out.open("wb") as fh:
                fh.write(src.read())
            raw = np.fromfile(str(out), dtype=np.int16)
            rec = {
                "member": name,
                "file_size": sz,
                "int16_len": int(raw.size),
                "div8": raw.size % 8 == 0,
                "min": int(raw.min()) if raw.size else None,
                "max": int(raw.max()) if raw.size else None,
            }
            if raw.size >= 8 and raw.size % 8 == 0:
                lanes = raw.reshape((8, -1), order="F")
                iq = lanes[3, :].astype(np.complex64) + 1j * lanes[7, :].astype(np.complex64)
                rec["iq_len"] = int(iq.size)
                rec["div256"] = iq.size % 256 == 0
                if iq.size % 256 == 0:
                    cube = iq.reshape((256, -1), order="F")
                    rec["nts256_shape"] = list(cube.shape)
                    rec["iq_abs_min"] = float(np.abs(cube).min())
                    rec["iq_abs_max"] = float(np.abs(cube).max())
            samples.append(rec)
    return {
        "second_dirs": dict(second),
        "classes": {k: dict(v) for k, v in classes.items()},
        "n_77_bin": len(bins_77),
        "largest_77": bins_77[:8],
        "dat_example": dats[:5],
        "samples_77": samples,
    }


def extract_mri_inner():
    inner = EXT / "mRI" / "dataset_release.zip"
    dest = EXT / "mRI_release"
    readme = (EXT / "mRI" / "README.md").read_text(encoding="utf-8", errors="replace") if (EXT / "mRI" / "README.md").exists() else ""
    if inner.exists() and not dest.exists():
        dest.mkdir(parents=True)
        with zipfile.ZipFile(inner) as zf:
            zf.extractall(dest)
    files = [p for p in dest.rglob("*") if p.is_file()] if dest.exists() else []
    by_ext = Counter(p.suffix.lower() or "<none>" for p in files)
    examples = [str(p.relative_to(dest)) for p in files[:40]] if dest.exists() else []
    jsons = list(dest.rglob("*.json")) if dest.exists() else []
    csvs = list(dest.rglob("*.csv")) if dest.exists() else []
    samples = []
    for p in jsons[:5]:
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            samples.append({"path": str(p.relative_to(dest)), "type": type(obj).__name__, "keys": list(obj.keys())[:30] if isinstance(obj, dict) else None, "len": len(obj) if isinstance(obj, list) else None})
        except Exception as exc:
            samples.append({"path": str(p.relative_to(dest)), "error": str(exc)})
    for p in csvs[:4]:
        lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
        samples.append({"path": str(p.relative_to(dest)), "n_lines": len(lines), "header": lines[0][:300] if lines else "", "line1": lines[1][:300] if len(lines) > 1 else ""})
    return {
        "readme_head": readme[:2500],
        "n_files": len(files),
        "by_ext": dict(by_ext),
        "examples": examples,
        "samples": samples,
    }


def peek_milipoint():
    root = EXT / "MiliPoint"
    labels = np.load(root / "action_label.npy")
    uniq, cnt = np.unique(labels, return_counts=True)
    out = {
        "label_shape": list(labels.shape),
        "label_dtype": str(labels.dtype),
        "label_unique": [int(x) for x in uniq[:60]],
        "label_counts_head": {int(a): int(b) for a, b in list(zip(uniq, cnt))[:20]},
        "pkl": [],
    }
    for name in ["0.pkl", "1.pkl", "2.pkl"]:
        p = root / name
        if not p.exists():
            continue
        with p.open("rb") as f:
            obj = pickle.load(f)
        rec = {"file": name, "type": type(obj).__name__, "size": p.stat().st_size}
        if isinstance(obj, dict):
            rec["keys"] = list(obj.keys())[:30]
            for k, v in list(obj.items())[:4]:
                if hasattr(v, "shape"):
                    rec[f"key_{k}_shape"] = list(v.shape)
                    rec[f"key_{k}_dtype"] = str(getattr(v, "dtype", ""))
                else:
                    rec[f"key_{k}_type"] = type(v).__name__
                    rec[f"key_{k}_len"] = len(v) if hasattr(v, "__len__") else None
        elif isinstance(obj, (list, tuple)):
            rec["len"] = len(obj)
            rec["elem0_type"] = type(obj[0]).__name__ if obj else None
            if obj and hasattr(obj[0], "shape"):
                rec["elem0_shape"] = list(obj[0].shape)
                rec["elem0_dtype"] = str(obj[0].dtype)
            if obj and isinstance(obj[0], dict):
                rec["elem0_keys"] = list(obj[0].keys())[:20]
        elif hasattr(obj, "shape"):
            rec["shape"] = list(obj.shape)
            rec["dtype"] = str(obj.dtype)
            rec["min"] = float(np.min(obj)) if np.issubdtype(obj.dtype, np.number) else None
            rec["max"] = float(np.max(obj)) if np.issubdtype(obj.dtype, np.number) else None
        out["pkl"].append(rec)
    return out


def radhar_counts():
    data = EXT / "RadHAR" / "RadHAR-master" / "Data"
    counts = {}
    if data.exists():
        for split in ["Train", "Test"]:
            d = data / split
            if not d.is_dir():
                continue
            counts[split] = {}
            for act in sorted(p for p in d.iterdir() if p.is_dir()):
                counts[split][act.name] = len(list(act.glob("*.txt")))
    voxels_py = (EXT / "RadHAR" / "RadHAR-master" / "DataPreprocessing" / "voxels.py")
    voxel_head = voxels_py.read_text(encoding="utf-8", errors="replace")[:2500] if voxels_py.exists() else ""
    return {"counts": counts, "voxel_script_head": voxel_head}


def mmfi_counts():
    env = set()
    sub = set()
    act = set()
    n = 0
    with (OUT / "MMFi_file_inventory.csv").open(encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            if row.get("is_dir") == "1":
                continue
            n += 1
            name = row["member"]
            parts = name.split("/")
            for p in parts:
                if p.startswith("E") and p[1:].isdigit():
                    env.add(p)
                if p.startswith("S") and p[1:].isdigit():
                    sub.add(p)
                if p.startswith("A") and p[1:].isdigit():
                    act.add(p)
    return {"n_png": n, "env": sorted(env), "subjects": sorted(sub), "actions": sorted(act)}


def main():
    out = {
        "ci4r": ci4r_counts_and_77ghz(),
        "mri": extract_mri_inner(),
        "milipoint": peek_milipoint(),
        "radhar": radhar_counts(),
        "mmfi": mmfi_counts(),
    }
    dest = OUT / "followup_facts.json"
    dest.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print("wrote", dest)
    print("ci4r second", out["ci4r"]["second_dirs"])
    print("ci4r n77", out["ci4r"]["n_77_bin"])
    print("mri n", out["mri"]["n_files"], out["mri"]["by_ext"])
    print("mmfi", out["mmfi"])
    print("radhar", out["radhar"]["counts"])
    print("mili pkl", out["milipoint"]["pkl"])


if __name__ == "__main__":
    main()
