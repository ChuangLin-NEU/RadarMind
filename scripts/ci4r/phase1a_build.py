#!/usr/bin/env python3
"""Phase 1A data side: manifest, QC, sample preprocess, MAT/PNG check, splits, spec cache."""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import random
import shutil
import sys
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from ci4r.preprocess_77ghz import airhar_micro_doppler, zscore  # noqa: E402
from ci4r.raw_reader import iq_qc, load_ci4r_77ghz_raw  # noqa: E402

ACTION_FOLDERS = {
    "05_Walking_Towards_radar": ("Towards", 0),
    "06_walking_away_from_Radar": ("Away", 1),
    "07_Picking_up_an_object": ("Pick", 2),
    "08_Bending": ("Bend", 3),
    "09_Sitting": ("Sit", 4),
    "10_Kneeling": ("Kneel", 5),
    "11_Crawling": ("Crawl", 6),
    "16_Walking_on_both_toes": ("Toes", 7),
    "17_limping_with_RL_Stiff": ("Limp", 8),
    "18_Short_steps": ("SStep", 9),
    "19_Scissors_gait": ("Scissor", 10),
}
CLASS_NAMES = ["Towards", "Away", "Pick", "Bend", "Sit", "Kneel", "Crawl", "Toes", "Limp", "SStep", "Scissor"]
SEED = 0


def write_csv(path: Path, rows: List[dict], fields: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def scan_manifest(zf: zipfile.ZipFile) -> List[dict]:
    rows = []
    for info in zf.infolist():
        if info.is_dir():
            continue
        name = info.filename.replace("\\", "/")
        if not name.startswith("Cross-frequency/77ghz/"):
            continue
        if not name.endswith("_Raw_0.bin"):
            continue
        parts = [p for p in name.split("/") if p]
        if len(parts) < 4:
            continue
        folder = parts[2]
        if folder.startswith("activity_") or folder not in ACTION_FOLDERS:
            continue
        fname = parts[3]
        stem = Path(fname).stem  # 04000000_1574696114_Raw_0
        bits = stem.split("_")
        prefix = bits[0] if bits else "UNKNOWN"
        unix = bits[1] if len(bits) > 1 and bits[1].isdigit() else "UNKNOWN"
        action, aid = ACTION_FOLDERS[folder]
        sid = f"ci4r77_{prefix}_{unix}_{action}"
        rows.append(
            {
                "sample_id": sid,
                "archive_member": name,
                "subject": "UNKNOWN",
                "action": action,
                "action_id": aid,
                "action_folder": folder,
                "session": "UNKNOWN",
                "repetition": "UNKNOWN",
                "radar_frequency": "77GHz",
                "file_size": info.file_size,
                "filename_prefix": prefix,
                "name_token_tail4": prefix[-4:] if len(prefix) >= 4 else prefix,
                "unix_token": unix,
            }
        )
    rows.sort(key=lambda r: (r["action_id"], r["filename_prefix"], r["unix_token"]))
    return rows


def distributions(rows: List[dict], out: Path) -> None:
    cls = Counter(r["action"] for r in rows)
    write_csv(
        out / "class_distribution.csv",
        [{"action": a, "n": cls[a]} for a in CLASS_NAMES],
        ["action", "n"],
    )
    # subject column is UNKNOWN; report prefix tail4 as grouping token only
    sub = Counter(r["name_token_tail4"] for r in rows)
    write_csv(
        out / "subject_distribution.csv",
        [{"name_token_tail4": k, "n": sub[k], "subject_field": "UNKNOWN"} for k in sorted(sub)],
        ["name_token_tail4", "n", "subject_field"],
    )
    cross = Counter((r["name_token_tail4"], r["action"]) for r in rows)
    write_csv(
        out / "subject_action_counts.csv",
        [{"name_token_tail4": a, "action": b, "n": n} for (a, b), n in sorted(cross.items())],
        ["name_token_tail4", "action", "n"],
    )


def read_member(zf: zipfile.ZipFile, name: str) -> bytes:
    with zf.open(name) as f:
        return f.read()


def qc_raw(zf: zipfile.ZipFile, rows: List[dict], out: Path, k: int = 12) -> List[dict]:
    # diverse by action and tail4
    buckets = defaultdict(list)
    for r in rows:
        buckets[(r["action"], r["name_token_tail4"])].append(r)
    keys = list(buckets)
    random.Random(SEED).shuffle(keys)
    picked = []
    i = 0
    while len(picked) < k and keys:
        key = keys[i % len(keys)]
        if buckets[key]:
            picked.append(buckets[key].pop(0))
        i += 1
        if i > 10000:
            break
    recs = []
    for r in picked:
        blob = read_member(zf, r["archive_member"])
        loaded = load_ci4r_77ghz_raw(blob)
        q = iq_qc(loaded["iq"])
        recs.append({**r, **q, "reader_ok": loaded["metadata"].get("ok"), "error": loaded["metadata"].get("error", "")})
        print("[qc]", r["sample_id"], q["shape"], flush=True)
    write_csv(
        out / "raw_signal_qc.csv",
        recs,
        [
            "sample_id",
            "archive_member",
            "action",
            "name_token_tail4",
            "shape",
            "dtype",
            "real_min",
            "real_max",
            "imag_min",
            "imag_max",
            "amp_min",
            "amp_max",
            "nan_count",
            "inf_count",
            "zero_fraction",
            "reader_ok",
            "error",
        ],
    )
    return picked


def write_schema(out: Path) -> None:
    text = """# CI4R 77 GHz signal schema

Evidence: `third_party/AIRHAR/datasets/data_prepare_CI4R.py` (`bin2np`, `RGB_mD`) and `third_party/CI4R-MULTI3/datToImage_77.m`, plus Phase 1A reads of `*_Raw_0.bin`.

## On-disk layout

- dtype: `int16` (DCA1000 two's complement). MATLAB: `fread(..., 'int16')`. AIRHAR: `np.fromfile(..., dtype=np.int16)`.
- Interleave: reshape to `(8, N)` Fortran / MATLAB column-major.
- `datToImage_77.m`: `numLanes = 4`; `Data(1:4,:) + 1j*Data(5:8,:)`. Four RX, I then Q.
- AIRHAR `bin2np`: keeps **only the fourth lane** (`data[3,:] + 1j*data[7,:]`).

Observed file size for usable action-folder `*_Raw_0.bin`: **131072000** bytes = 65536000 int16.

```
int16[65536000]
  -> reshape (8, 8192000) order F
  -> 4 RX complex streams of length 8192000
  -> AIRHAR lane 4 -> complex[8192000]
  -> reshape (256, 32000) order F
```

## What is 256?

**ADC samples per chirp (NTS).** Both AIRHAR (`NTS = 256`) and MATLAB (`NTS = 256`) say this. Range FFT is along this axis.

## What is 32000?

**Number of chirps / pulses in this recording**, after one RX is selected: `8192000 / 256 = 32000`. This is measured from the files, and AIRHAR also writes `N = 32000`.

## Comments that do **not** match these files

MATLAB / AIRHAR also write:

- `NPpF = 128` pulses per frame
- `NoF = 500` frames
- `SweepTime = 40e-3`

`128 * 500 = 64000` pulses, which would be **20 s** at `dT = SweepTime/NPpF`. The `*_Raw_0.bin` we opened have **32000** pulses (about **10 s** at that `dT`). Plot code still does `axis([0 20 ...])`.

So:

- `256` = NTS: **resolved**
- `32000` = chirps in file: **resolved from data**
- frame packing `32000 = frames × pulses_per_frame`: **UNRESOLVED** (128×250 would fit; 128×500 does not)
- Tx count: **UNRESOLVED** (not in these two loaders)
- Rx count: **4** in MATLAB; AIRHAR uses 1 of 4

Official reshape used in this phase (AIRHAR):

`raw int16 -> [8, N] -> IQ lane 4 -> [256 ADC, 32000 chirps]`

Not claimed: `[frame, chirp, rx, adc]` as a stored tensor. That would require a frame size that matches 32000, which the comments do not.
"""
    (out / "ci4r_signal_schema.md").write_text(text, encoding="utf-8")


def preprocess_subset(zf: zipfile.ZipFile, rows: List[dict], out: Path, n: int = 36) -> List[dict]:
    buckets = defaultdict(list)
    for r in rows:
        buckets[r["action"]].append(r)
    picked = []
    for a in CLASS_NAMES:
        items = buckets[a]
        random.Random(SEED).shuffle(items)
        picked.extend(items[: max(1, n // 11)])
    picked = picked[:n]
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    recs = []
    specs = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    for r in picked:
        blob = read_member(zf, r["archive_member"])
        iq = load_ci4r_77ghz_raw(blob)["iq"]
        spec = airhar_micro_doppler(iq)
        zs = zscore(spec)
        recs.append(
            {
                **r,
                "spec_shape": list(spec.shape),
                "spec_min": float(spec.min()),
                "spec_max": float(spec.max()),
                "z_min": float(zs.min()),
                "z_max": float(zs.max()),
                "spec_bytes": int(spec.nbytes),
            }
        )
        specs.append((r, spec))
        print("[prep]", r["sample_id"], spec.shape, spec.min(), spec.max(), flush=True)
    write_csv(
        out / "preprocess_subset.csv",
        recs,
        ["sample_id", "action", "name_token_tail4", "spec_shape", "spec_min", "spec_max", "z_min", "z_max", "spec_bytes"],
    )
    if plt is not None:
        # a few representative panels
        by_act = defaultdict(list)
        for r, spec in specs:
            by_act[r["action"]].append((r, spec))
        acts = [a for a in CLASS_NAMES if by_act[a]][:6]
        fig, axes = plt.subplots(len(acts), 2, figsize=(8, 2.2 * len(acts)))
        if len(acts) == 1:
            axes = np.array([axes])
        for i, a in enumerate(acts):
            for j, (r, spec) in enumerate(by_act[a][:2]):
                ax = axes[i, j]
                ax.imshow(spec, aspect="auto", origin="lower", cmap="gray", vmin=-45, vmax=0)
                ax.set_title(f"{a} / tail4={r['name_token_tail4']}", fontsize=8)
                ax.axis("off")
        fig.tight_layout()
        fig.savefig(figdir / "fig4_airhar_spec_examples.png", dpi=120)
        plt.close(fig)
    return picked


def inspect_mat_png(zf: zipfile.ZipFile, rows: List[dict], out: Path, n_pair: int = 8) -> None:
    names = {i.filename.replace("\\", "/") for i in zf.infolist()}
    recs = []
    val_rows = []
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from PIL import Image
    except Exception:
        plt = None
        Image = None
    try:
        import scipy.io as sio
    except Exception:
        sio = None

    pairs = []
    for r in rows:
        stem = Path(r["archive_member"]).stem
        png = f"Cross-frequency/77ghz/activity_spectogram_77GHz/{_png_folder(r['action_folder'])}/{stem}.png"
        mat_fwcc = f"Cross-frequency/77ghz/activity_HWCC/{r['action_folder']}/{stem}_FWCC.mat"
        # matrices folder naming unknown; try a few
        mat_spec_cands = [
            p
            for p in names
            if p.endswith(".mat") and stem in p and "activity_spect_matrices" in p
        ]
        if png in names or mat_fwcc in names or mat_spec_cands:
            pairs.append((r, png if png in names else "", mat_fwcc if mat_fwcc in names else "", mat_spec_cands[:2]))
        if len(pairs) >= n_pair:
            break

    # if few pairs from start of list, take evenly
    if len(pairs) < n_pair:
        for r in rows[:: max(1, len(rows) // n_pair)]:
            stem = Path(r["archive_member"]).stem
            png = f"Cross-frequency/77ghz/activity_spectogram_77GHz/{_png_folder(r['action_folder'])}/{stem}.png"
            mat_fwcc = f"Cross-frequency/77ghz/activity_HWCC/{r['action_folder']}/{stem}_FWCC.mat"
            mat_spec_cands = [p for p in names if p.endswith(".mat") and stem in p and "activity_spect_matrices" in p]
            if png in names:
                pairs.append((r, png, mat_fwcc if mat_fwcc in names else "", mat_spec_cands[:2]))
            if len(pairs) >= n_pair:
                break

    shown = 0
    for r, png, mat_fwcc, mat_specs in pairs[:n_pair]:
        blob = read_member(zf, r["archive_member"])
        spec = airhar_micro_doppler(load_ci4r_77ghz_raw(blob)["iq"])
        rec = {
            "sample_id": r["sample_id"],
            "png": png or "MISSING",
            "mat_fwcc": mat_fwcc or "MISSING",
            "mat_spec": ";".join(mat_specs) if mat_specs else "MISSING",
        }
        if sio and mat_fwcc:
            rawm = read_member(zf, mat_fwcc)
            m = sio.loadmat(io.BytesIO(rawm), squeeze_me=True, struct_as_record=False)
            keys = [k for k in m.keys() if not k.startswith("__")]
            rec["fwcc_keys"] = keys
            for k in keys[:4]:
                v = m[k]
                if hasattr(v, "shape"):
                    rec[f"fwcc_{k}_shape"] = list(np.shape(v))
                    rec[f"fwcc_{k}_dtype"] = str(getattr(v, "dtype", ""))
        if sio and mat_specs:
            rawm = read_member(zf, mat_specs[0])
            m = sio.loadmat(io.BytesIO(rawm), squeeze_me=True, struct_as_record=False)
            keys = [k for k in m.keys() if not k.startswith("__")]
            rec["specmat_keys"] = keys
            rec["specmat_path"] = mat_specs[0]
            for k in keys[:6]:
                v = m[k]
                if hasattr(v, "shape") and getattr(v, "size", 0) > 8:
                    rec["specmat_key"] = k
                    rec["specmat_shape"] = list(np.shape(v))
                    rec["specmat_dtype"] = str(v.dtype)
                    rec["specmat_complex"] = bool(np.iscomplexobj(v))
                    arr = np.abs(v) if np.iscomplexobj(v) else np.asarray(v, dtype=np.float32)
                    rec["specmat_min"] = float(np.min(arr))
                    rec["specmat_max"] = float(np.max(arr))
                    break
        corr = mae = ssim = "UNKNOWN"
        if Image and png:
            img = Image.open(io.BytesIO(read_member(zf, png))).convert("L")
            rec["png_size"] = list(img.size)
            rec["png_mode"] = "L"
            g = np.asarray(img, dtype=np.float32)
            rec["png_min"] = float(g.min())
            rec["png_max"] = float(g.max())
            # official PNG is a plotted figure (colorbar/axes). Compare after resize only as sanity.
            try:
                import cv2

                g224 = cv2.resize(g, (224, 224)).astype(np.float32)
            except Exception:
                g224 = np.array(img.resize((224, 224)))
            a = (spec - spec.mean()) / (spec.std() + 1e-8)
            b = (g224 - g224.mean()) / (g224.std() + 1e-8)
            corr = float(np.mean(a * b))
            mae = float(np.mean(np.abs(a - b)))
            rec["corr_zscore_vs_pngL"] = corr
            rec["nmae_zscore_vs_pngL"] = mae
            if plt is not None and shown < 4:
                fig, ax = plt.subplots(1, 2, figsize=(8, 3))
                ax[0].imshow(spec, aspect="auto", origin="lower", cmap="gray", vmin=-45, vmax=0)
                ax[0].set_title("AIRHAR from raw IQ")
                ax[1].imshow(g, aspect="auto", cmap="gray")
                ax[1].set_title("archive PNG (plotted)")
                for x in ax:
                    x.axis("off")
                fig.tight_layout()
                fig.savefig(out / "figures" / f"preprocess_validation_{shown}.png", dpi=120)
                plt.close(fig)
                shown += 1
        recs.append(rec)
        val_rows.append(rec)
        print("[matpng]", r["sample_id"], rec.get("fwcc_keys"), rec.get("specmat_shape"), rec.get("corr_zscore_vs_pngL"), flush=True)

    write_csv(out / "mat_schema.csv", recs, sorted({k for r in recs for k in r.keys()}))
    write_csv(out / "preprocessing_validation.csv", val_rows, sorted({k for r in val_rows for k in r.keys()}))


def _png_folder(action_folder: str) -> str:
    # archive uses slightly different names under activity_spectogram_77GHz
    alt = {
        "05_Walking_Towards_radar": "05_walking_towards",
        "06_walking_away_from_Radar": "06_walking_away",
        "07_Picking_up_an_object": "07_picking_obj",
        "08_Bending": "08_Bending",
        "09_Sitting": "09_sitting",
        "10_Kneeling": "10_kneeling",
        "11_Crawling": "11_crawling",
        "16_Walking_on_both_toes": "16_walking_on_toes",
        "17_limping_with_RL_Stiff": "17_limping_RL_stiff",
        "18_Short_steps": "18_short_steps",
        "19_Scissors_gait": "19_scissor_gait",
    }
    return alt.get(action_folder, action_folder)


def make_splits(rows: List[dict], split_dir: Path) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(SEED)
    # Protocol 1: stratified record split 70/15/15
    by_a = defaultdict(list)
    for r in rows:
        by_a[r["action"]].append(r["sample_id"])
    assign = {}
    for a, ids in by_a.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(0.70 * n))
        n_va = int(round(0.15 * n))
        if n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        tr, va, te = ids[:n_tr], ids[n_tr : n_tr + n_va], ids[n_tr + n_va :]
        for i in tr:
            assign[i] = "train"
        for i in va:
            assign[i] = "val"
        for i in te:
            assign[i] = "test"
    p1 = [{**r, "split": assign[r["sample_id"]], "protocol": "random_record", "seed": SEED} for r in rows]
    write_csv(
        split_dir / "ci4r_random.csv",
        p1,
        ["sample_id", "archive_member", "action", "name_token_tail4", "split", "protocol", "seed"],
    )

    # Protocol 2: prefix-tail4 disjoint. Not official subject IDs.
    # 0029/0030 have all 11 classes and most samples; 0010 also has all 11; 0014 misses Towards; 0000 is Towards-only.
    mapping = {"0029": "train", "0030": "val", "0010": "test"}
    p2 = []
    for r in rows:
        tail = r["name_token_tail4"]
        sp = mapping.get(tail, "heldout")
        p2.append({**r, "split": sp, "protocol": "prefix_tail4_disjoint", "seed": SEED})
    write_csv(
        split_dir / "ci4r_subject_independent.csv",
        p2,
        ["sample_id", "archive_member", "action", "name_token_tail4", "split", "protocol", "seed"],
    )

    # Protocol 3: LOSO on the four multi-class tokens
    tokens = ["0010", "0014", "0029", "0030"]
    for i, held in enumerate(tokens, 1):
        rows_f = []
        for r in rows:
            if r["name_token_tail4"] == "0000":
                continue
            sp = "test" if r["name_token_tail4"] == held else "train"
            rows_f.append({**r, "split": sp, "protocol": f"loso_{held}", "seed": SEED})
        write_csv(
            split_dir / f"ci4r_loso_fold{i:02d}.csv",
            rows_f,
            ["sample_id", "archive_member", "action", "name_token_tail4", "split", "protocol", "seed"],
        )


def coverage_figs(rows: List[dict], out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    cls = Counter(r["action"] for r in rows)
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.bar(CLASS_NAMES, [cls[a] for a in CLASS_NAMES], color="#3d5a80")
    ax.set_ylabel("n records")
    ax.set_title("CI4R 77 GHz Raw_0.bin by action")
    fig.tight_layout()
    fig.savefig(figdir / "class_distribution.png", dpi=140)
    fig.savefig(figdir / "fig2_class_distribution.png", dpi=140)
    plt.close(fig)

    tails = sorted({r["name_token_tail4"] for r in rows})
    mat = np.zeros((len(tails), len(CLASS_NAMES)), dtype=int)
    idx_t = {t: i for i, t in enumerate(tails)}
    idx_a = {a: i for i, a in enumerate(CLASS_NAMES)}
    for r in rows:
        mat[idx_t[r["name_token_tail4"]], idx_a[r["action"]]] += 1
    fig, ax = plt.subplots(figsize=(9, 3.2))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(11), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(tails)), [f"tail4={t}" for t in tails])
    ax.set_title("filename prefix tail4 × action (subject field is UNKNOWN)")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(figdir / "subject_action_matrix.png", dpi=140)
    fig.savefig(figdir / "fig3_subject_action_matrix.png", dpi=140)
    plt.close(fig)

    # pipeline schematic
    fig, ax = plt.subplots(figsize=(10, 2.2))
    ax.axis("off")
    boxes = [
        "raw int16\n8-lane DCA1000",
        "AIRHAR bin2np\nIQ (256, 32000)",
        "range FFT +\nsum bins 23:31",
        "STFT 4096\n20log clip",
        "224×224\nHAR CNN",
    ]
    for i, t in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i * 2.0, 0.3), 1.7, 1.4, fill=False, lw=1.2))
        ax.text(i * 2.0 + 0.85, 1.0, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i * 2.0 + 1.95, 1.0), xytext=(i * 2.0 + 1.7, 1.0), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(0, 2)
    ax.set_title("Fig.1 CI4R 77 GHz path used in Phase 1A (AIRHAR)")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_pipeline.png", dpi=140)
    plt.close(fig)


def cache_all_specs(zf: zipfile.ZipFile, rows: List[dict], cache_path: Path) -> Dict:
    """Write compact float32 spectrograms. Skip empty/short bins. Resume via per-id npy."""
    part_dir = cache_path.parent / "spec_npy"
    part_dir.mkdir(parents=True, exist_ok=True)
    excluded = []
    xs, ys, ids, tails = [], [], [], []
    n = len(rows)
    print(f"[cache] n_manifest={n} part_dir={part_dir}", flush=True)
    for i, r in enumerate(rows):
        sid = r["sample_id"]
        npy = part_dir / f"{sid}.npy"
        try:
            sz = int(r.get("file_size") or 0)
        except ValueError:
            sz = 0
        if sz < 256 * 8 * 2:
            excluded.append({**r, "reason": f"file_size={sz}"})
            print(f"[cache-skip] {sid} file_size={sz}", flush=True)
            continue
        if npy.exists() and npy.stat().st_size > 0:
            spec_z = np.load(npy)
        else:
            blob = read_member(zf, r["archive_member"])
            loaded = load_ci4r_77ghz_raw(blob)
            iq = loaded["iq"]
            if iq.size == 0 or iq.shape[1] < 256 + 56:
                excluded.append({**r, "reason": f"short_iq shape={list(iq.shape)}"})
                print(f"[cache-skip] {sid} short_iq {iq.shape}", flush=True)
                continue
            spec_z = zscore(airhar_micro_doppler(iq))
            np.save(npy, spec_z)
        xs.append(spec_z)
        ys.append(int(r["action_id"]))
        ids.append(sid)
        tails.append(r["name_token_tail4"])
        if (i + 1) % 10 == 0 or i == 0:
            print(f"[cache] {i+1}/{n} kept={len(ids)} skip={len(excluded)} {r['action']}", flush=True)
    X = np.stack(xs, axis=0).astype(np.float32)
    y = np.asarray(ys, dtype=np.int64)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        X=X,
        y=y,
        sample_id=np.array(ids),
        tail4=np.array(tails),
        class_names=np.array(CLASS_NAMES),
    )
    write_csv(
        cache_path.parent.parent / "excluded_samples.csv",
        excluded,
        ["sample_id", "archive_member", "action", "file_size", "reason"],
    )
    return {"n_kept": int(len(ids)), "n_excluded": int(len(excluded)), "path": str(cache_path), "bytes": cache_path.stat().st_size}


def cost_report(n: int, disk_free: int) -> dict:
    raw_one = 131072000
    spec_one = 224 * 224 * 4
    return {
        "n_usable": n,
        "raw_one_bytes": raw_one,
        "full_raw_extract_GB": n * raw_one / 1024**3,
        "airhar_spec_one_bytes": spec_one,
        "full_spec_cache_GB": n * spec_one / 1024**3,
        "disk_free_GB": disk_free / 1024**3,
        "expansion_if_extract_raw": (n * raw_one) / (n * raw_one),
        "recommend": "lazy read zip + cache 224x224 float32 (z-scored AIRHAR spec). Do not extract all raw bins.",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--zip", default=os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", "")))
    ap.add_argument("--out", default=os.environ.get("RADARMIND_PHASE1A_OUT", str(repo / "results" / "phase1_ci4r")))
    ap.add_argument("--split-dir", default=os.environ.get("RADARMIND_SPLIT_DIR", str(repo / "data_splits")))
    ap.add_argument("--cache-all", action="store_true")
    ap.add_argument("--cache-only", action="store_true", help="reuse existing manifest CSV; only build spec cache")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("[open]", args.zip, flush=True)
    with zipfile.ZipFile(args.zip) as zf:
        man_csv = out / "ci4r_77ghz_manifest.csv"
        if args.cache_only and man_csv.exists():
            rows = list(csv.DictReader(man_csv.open(encoding="utf-8")))
            print("[manifest-reuse]", len(rows), flush=True)
            cost = cost_report(sum(1 for r in rows if int(r.get("file_size") or 0) > 0), shutil.disk_usage(str(out)).free)
            (out / "disk_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
            info = cache_all_specs(zf, rows, out / "cache" / "ci4r77_airhar_spec_zscore.npz")
            print("[cache-done]", info, flush=True)
            print("[done] n=", len(rows), flush=True)
            return
        rows = scan_manifest(zf)
        print("[manifest]", len(rows), flush=True)
        write_csv(
            out / "ci4r_77ghz_manifest.csv",
            rows,
            [
                "sample_id",
                "archive_member",
                "subject",
                "action",
                "session",
                "repetition",
                "radar_frequency",
                "file_size",
                "filename_prefix",
                "name_token_tail4",
                "unix_token",
                "action_id",
                "action_folder",
            ],
        )
        distributions(rows, out)
        write_schema(out)
        qc_raw(zf, rows, out, k=12)
        preprocess_subset(zf, rows, out, n=36)
        inspect_mat_png(zf, rows, out, n_pair=8)
        make_splits(rows, Path(args.split_dir))
        coverage_figs(rows, out)
        cost = cost_report(len(rows), shutil.disk_usage(str(out)).free)
        (out / "disk_cost.json").write_text(json.dumps(cost, indent=2), encoding="utf-8")
        print("[cost]", json.dumps(cost), flush=True)
        if args.cache_all:
            if cost["full_spec_cache_GB"] < 2 and cost["disk_free_GB"] > 5:
                info = cache_all_specs(zf, rows, out / "cache" / "ci4r77_airhar_spec_zscore.npz")
                print("[cache-done]", info, flush=True)
            else:
                print("[cache-skip] cost too high", flush=True)
    print("[done] n=", len(rows), flush=True)


if __name__ == "__main__":
    main()
