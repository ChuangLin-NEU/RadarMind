#!/usr/bin/env python3
"""Phase 1B data side: mRI + RadHAR manifests, QC, splits, figures, window cache."""
from __future__ import annotations

import argparse
import os
import random
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from pointcloud.common import (  # noqa: E402
    SEED,
    atomic_savez,
    dump_json,
    feat_stats,
    sample_frame,
    seed_from,
    stack_window,
    write_csv,
)
from pointcloud.mri import (  # noqa: E402
    CLASS_NAMES as MRI_CLASSES,
    canonical_action,
    find_radar_csv,
    load_label_json,
    load_radar_csv,
    overlap_pairs,
    parse_subject_id,
    segments_from_json,
)
from pointcloud.radhar import CLASS_NAMES as RAD_CLASSES  # noqa: E402
from pointcloud.radhar import first_stamp, parse_filename_tokens, parse_mmwave_txt  # noqa: E402
from pointcloud.voxels_official import voxelize_frame  # noqa: E402

MRI_T, MRI_P, MRI_STRIDE = 32, 64, 16
RAD_T, RAD_P, RAD_STRIDE = 60, 32, 10


def _plt():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def build_mri(mri_root: Path, out: Path, split_dir: Path) -> dict:
    radar_dir = mri_root / "raw_data" / "radar"
    lab_dir = mri_root / "raw_data" / "videolabels"
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    seg_rows = []
    qc_rows = []
    feat_chunks = {k: [] for k in ("x", "y", "z", "velocity", "intensity")}
    pts_per_frame = []
    subject_action = Counter()
    failed = []

    jsons = sorted(lab_dir.glob("*.json"))
    for jf in jsons:
        try:
            lab = load_label_json(jf)
            subject = parse_subject_id(lab.get("subject") or jf.stem)
            csv_path = find_radar_csv(radar_dir, subject)
            if csv_path is None:
                failed.append({"dataset": "mRI", "source_file": str(jf), "reason": "radar_csv_missing"})
                continue
            frames, meta = load_radar_csv(csv_path)
            segs = segments_from_json(lab)
            overlaps = overlap_pairs(segs)
            frame_to_segs = defaultdict(list)
            for s in segs:
                for t in range(s["segment_start"], s["segment_end"] + 1):
                    frame_to_segs[t].append(s)
            all_fr = sorted(frames)
            gaps = int(np.sum(np.diff(all_fr) > 1)) if len(all_fr) > 1 else 0
            n_empty_declared = 0
            mismatch = 0
            nan_c = inf_c = dup_c = 0
            zvel = zint = 0
            npts_all = 0
            for fr, pts in frames.items():
                n = len(pts)
                pts_per_frame.append(n)
                npts_all += n
                declared = meta["nobj_declared"].get(fr, n)
                if n == 0:
                    n_empty_declared += 1
                if n != declared:
                    mismatch += 1
                if np.isnan(pts).any():
                    nan_c += int(np.isnan(pts).sum())
                if np.isinf(pts).any():
                    inf_c += int(np.isinf(pts).sum())
                if n > 1:
                    uniq = np.unique(np.round(pts, 6), axis=0)
                    dup_c += n - len(uniq)
                zvel += int(np.sum(pts[:, 3] == 0))
                zint += int(np.sum(pts[:, 4] == 0))
                if len(feat_chunks["x"]) < 400000:
                    feat_chunks["x"].append(pts[:, 0])
                    feat_chunks["y"].append(pts[:, 1])
                    feat_chunks["z"].append(pts[:, 2])
                    feat_chunks["velocity"].append(pts[:, 3])
                    feat_chunks["intensity"].append(pts[:, 4])
                labs = frame_to_segs.get(fr, [])
                if not labs:
                    action = "UNLABELED"
                    sid = ""
                    a0 = a1 = ""
                elif len(labs) == 1:
                    action = labs[0]["action"]
                    sid = labs[0]["action_segment_id"]
                    a0, a1 = labs[0]["segment_start"], labs[0]["segment_end"]
                else:
                    action = "|".join(s["action"] for s in labs)
                    sid = "|".join(s["action_segment_id"] for s in labs)
                    a0, a1 = labs[0]["segment_start"], labs[0]["segment_end"]
                frame_rows.append(
                    {
                        "subject": subject,
                        "source_file": str(csv_path.name),
                        "frame_id": fr,
                        "timestamp": meta["time_median"].get(fr, ""),
                        "num_points": n,
                        "action": action,
                        "action_segment_id": sid,
                        "segment_start": a0,
                        "segment_end": a1,
                    }
                )
            times = list(meta["time_median"].values())
            dt = (max(times) - min(times)) if times else None
            fps = (len(all_fr) / dt) if dt and dt > 0 else None
            qc_rows.append(
                {
                    "subject": subject,
                    "source_file": csv_path.name,
                    "n_frames": len(frames),
                    "frame_min": all_fr[0] if all_fr else "",
                    "frame_max": all_fr[-1] if all_fr else "",
                    "n_gaps": gaps,
                    "empty_frames": n_empty_declared,
                    "nobj_mismatch_frames": mismatch,
                    "nan_count": nan_c,
                    "inf_count": inf_c,
                    "duplicate_points": dup_c,
                    "zero_velocity_frac": (zvel / npts_all) if npts_all else "",
                    "zero_intensity_frac": (zint / npts_all) if npts_all else "",
                    "pts_min": int(min(len(v) for v in frames.values())) if frames else 0,
                    "pts_max": int(max(len(v) for v in frames.values())) if frames else 0,
                    "pts_median": float(np.median([len(v) for v in frames.values()])) if frames else "",
                    "time_span_s": dt,
                    "approx_fps": fps,
                    "n_segments": len(segs),
                    "n_overlap_pairs": len(overlaps),
                    "n_unlabeled_frames": sum(1 for fr in frames if fr not in frame_to_segs),
                }
            )
            for s in segs:
                ids = [t for t in range(s["segment_start"], s["segment_end"] + 1) if t in frames]
                pts_n = int(sum(len(frames[t]) for t in ids))
                t0 = meta["time_median"].get(ids[0]) if ids else None
                t1 = meta["time_median"].get(ids[-1]) if ids else None
                rec_date = csv_path.stem.split("_")[-1] if "_" in csv_path.stem else "UNKNOWN"
                seg_rows.append(
                    {
                        **s,
                        "source_file": csv_path.name,
                        "session": rec_date,
                        "recording": csv_path.stem,
                        "n_radar_frames": len(ids),
                        "n_missing_in_interval": s["n_label_frames"] - len(ids),
                        "total_points": pts_n,
                        "duration_s": (t1 - t0) if (t0 is not None and t1 is not None) else "",
                        "overlap_flag": int(any(s["action_segment_id"] in p[:2] for p in overlaps)),
                    }
                )
                subject_action[(subject, s["action"])] += 1
            print("[mri]", subject, "frames", len(frames), "segs", len(segs), "overlaps", len(overlaps), flush=True)
        except Exception as exc:  # noqa: BLE001
            failed.append({"dataset": "mRI", "source_file": str(jf), "reason": str(exc)})
            print("[mri-fail]", jf, exc, flush=True)

    write_csv(
        out / "mri_manifest.csv",
        frame_rows,
        ["subject", "source_file", "frame_id", "timestamp", "num_points", "action", "action_segment_id", "segment_start", "segment_end"],
    )
    write_csv(
        out / "mri_segments.csv",
        seg_rows,
        [
            "subject",
            "source_file",
            "session",
            "recording",
            "action_raw",
            "action",
            "action_segment_id",
            "segment_start",
            "segment_end",
            "n_label_frames",
            "n_radar_frames",
            "n_missing_in_interval",
            "total_points",
            "duration_s",
            "overlap_flag",
        ],
    )
    write_csv(out / "mri_qc.csv", qc_rows, list(qc_rows[0].keys()) if qc_rows else ["subject"])
    write_csv(out / "failed_samples.csv", failed, ["dataset", "source_file", "reason"]) if failed else None

    schema = """# mRI signal / label schema

Evidence: Dryad README (inner `dataset_release`) and `competitors/dap_net_official/dataset/makecsv/mRI2csv.py`.

## Hierarchy

```
subject (JSON field `subject`, e.g. subject1)
  └── recording: one raw CSV `raw_data/radar/{subject}_{MMDD}.csv`
        └── action segment: videolabels JSON `labels[name] = [start, end]` inclusive on Frame #
              └── frame: all CSV rows with that Frame #
                    └── points: columns X,Y,Z,Doppler,Intensity
```

One subject file = one recording. The `_MMDD` token is stored as `session` from the filename only.

## How labels attach

JSON integers are **Frame #**, not `Time`.

- `Time` is a wall-clock unix stamp (~1.65e9). It is never in the JSON ranges (e.g. 98–139).
- DAP-Net compares `Frame # == t` for `t` in `[a, b]` inclusive.
- Radar `Frame #` is not gap-free (subject1: 164 gaps, max gap 51). Missing IDs in a JSON interval are unlabeled holes, not extra points.
- `# Obj` matches the number of rows of that Frame # on the files checked.

## Overlap and unlabeled

Adjacent JSON intervals often share the end/start frame (e.g. pose_1 [169,640] and pose_2 [640,1122]). Those frames are marked with `action` joined by `|` in the frame manifest.

Frames before the first interval, after the last, and in interior gaps are `UNLABELED`.

`T pose 2` appears on 13/20 subjects as a short closing T-pose. Training uses canonical name `T pose` (same pose words). The raw JSON key is kept in `action_raw`.

## Rate

`Time` span / unique frames is about **10 Hz** (not 30 Hz). DAP-Net window T=32 is ~3.2 s.
"""
    (out / "mri_schema.md").write_text(schema, encoding="utf-8")

    # splits at segment level
    rng = random.Random(SEED)
    assign_rand = {}
    by_a = defaultdict(list)
    for s in seg_rows:
        by_a[s["action"]].append(s["action_segment_id"])
    for a, ids in by_a.items():
        ids = list(ids)
        rng.shuffle(ids)
        n = len(ids)
        n_tr = int(round(0.70 * n))
        n_va = int(round(0.15 * n))
        if n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        for i in ids[:n_tr]:
            assign_rand[i] = "train"
        for i in ids[n_tr : n_tr + n_va]:
            assign_rand[i] = "val"
        for i in ids[n_tr + n_va :]:
            assign_rand[i] = "test"
    p1 = [{**s, "split": assign_rand[s["action_segment_id"]], "protocol": "random_segment", "seed": SEED} for s in seg_rows]
    write_csv(
        split_dir / "mri_random_segment.csv",
        p1,
        ["action_segment_id", "subject", "action", "source_file", "segment_start", "segment_end", "split", "protocol", "seed"],
    )

    subs = sorted({s["subject"] for s in seg_rows})
    rng2 = random.Random(SEED)
    rng2.shuffle(subs)
    # 14 / 3 / 3
    map_si = {}
    for i, su in enumerate(subs):
        map_si[su] = "train" if i < 14 else ("val" if i < 17 else "test")
    p2 = [{**s, "split": map_si[s["subject"]], "protocol": "subject_independent", "seed": SEED} for s in seg_rows]
    write_csv(
        split_dir / "mri_subject_independent.csv",
        p2,
        ["action_segment_id", "subject", "action", "source_file", "segment_start", "segment_end", "split", "protocol", "seed"],
    )
    for i, held in enumerate(sorted({s["subject"] for s in seg_rows}), 1):
        rows = []
        for s in seg_rows:
            rows.append({**s, "split": "test" if s["subject"] == held else "train", "protocol": f"loso_{held}", "seed": SEED})
        write_csv(
            split_dir / f"mri_loso_fold{i:02d}.csv",
            rows,
            ["action_segment_id", "subject", "action", "source_file", "segment_start", "segment_end", "split", "protocol", "seed"],
        )

    plt = _plt()
    if plt and pts_per_frame:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(pts_per_frame, bins=32, color="#3d5a80")
        ax.set_xlabel("points / frame")
        ax.set_ylabel("frames")
        fig.tight_layout()
        fig.savefig(figdir / "mri_points_per_frame.png", dpi=140)
        plt.close(fig)
        acts = MRI_CLASSES
        subs_s = sorted({s["subject"] for s in seg_rows})
        mat = np.zeros((len(subs_s), len(acts)), dtype=int)
        ia = {a: i for i, a in enumerate(acts)}
        iu = {u: i for i, u in enumerate(subs_s)}
        for s in seg_rows:
            if s["action"] in ia:
                mat[iu[s["subject"]], ia[s["action"]]] += 1
        fig, ax = plt.subplots(figsize=(9, 5))
        im = ax.imshow(mat, cmap="Blues", aspect="auto")
        ax.set_xticks(range(len(acts)), acts, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(subs_s)), subs_s, fontsize=7)
        ax.set_title("Fig.3 mRI subject × canonical action (segment counts)")
        fig.colorbar(im, ax=ax, fraction=0.03)
        fig.tight_layout()
        fig.savefig(figdir / "mri_subject_action_matrix.png", dpi=140)
        fig.savefig(figdir / "fig3_mri_subject_action.png", dpi=140)
        plt.close(fig)

    feat_schema_rows = []
    raw_names = {"x": "X", "y": "Y", "z": "Z", "velocity": "Doppler", "intensity": "Intensity"}
    for feat in raw_names:
        st = feat_stats(np.concatenate(feat_chunks[feat]) if feat_chunks[feat] else np.array([]))
        feat_schema_rows.append(
            {
                "dataset": "mRI",
                "feature": feat,
                "raw_name": raw_names[feat],
                "unit": "UNIT_UNKNOWN",
                "min": st["min"],
                "max": st["max"],
                "mean": st["mean"],
                "std": st["std"],
                "evidence": "Dryad README: X,Y,Z=3D coordinates; Doppler=Doppler velocity; Intensity=reflected power; Time=wall timestamp. No unit string. JSON ranges match Frame # (DAP-Net).",
            }
        )
    dump_json(
        out / "mri_summary.json",
        {
            "n_subjects": len({s["subject"] for s in seg_rows}),
            "n_segments": len(seg_rows),
            "n_frames_manifest": len(frame_rows),
            "median_pts_frame": float(np.median(pts_per_frame)) if pts_per_frame else None,
            "selected_P": MRI_P,
            "selected_T": MRI_T,
            "si_subjects": map_si,
        },
    )
    return {
        "seg_rows": seg_rows,
        "feat_schema_rows": feat_schema_rows,
        "pts_per_frame": pts_per_frame,
        "feat_chunks": feat_chunks,
        "failed": failed,
        "map_si": map_si,
    }


def build_radhar(rad_root: Path, out: Path, split_dir: Path, feat_schema_rows: list) -> dict:
    data = rad_root / "Data"
    figdir = out / "figures"
    recs = []
    qc = []
    failed = []
    pts_pf = []
    feat_chunks = {k: [] for k in ("x", "y", "z", "velocity", "intensity")}
    n_win = 0
    for split in ("Train", "Test"):
        for act in RAD_CLASSES:
            d = data / split / act
            if not d.is_dir():
                continue
            for fp in sorted(d.glob("*.txt")):
                try:
                    frames = parse_mmwave_txt(fp)
                    nfr = len(frames)
                    tot = int(sum(len(f) for f in frames))
                    empty = sum(1 for f in frames if len(f) == 0)
                    tok = parse_filename_tokens(fp.name)
                    stamp = first_stamp(fp)
                    mean_p = (tot / nfr) if nfr else 0
                    recs.append(
                        {
                            "split": split.lower(),
                            "source_file": str(fp.relative_to(rad_root)).replace("\\", "/"),
                            "action": act,
                            "subject": "UNKNOWN",
                            "session": "UNKNOWN",
                            "recording": tok["recording"],
                            "frames": nfr,
                            "total_points": tot,
                            "mean_points_per_frame": mean_p,
                            "stamp_secs": stamp["stamp_secs"],
                            "protocol_note": "official_train_test_split",
                        }
                    )
                    nan = inf = dup = zv = zi = 0
                    for f in frames:
                        pts_pf.append(len(f))
                        if f.size == 0:
                            continue
                        nan += int(np.isnan(f).sum())
                        inf += int(np.isinf(f).sum())
                        if len(f) > 1:
                            dup += len(f) - len(np.unique(np.round(f, 6), axis=0))
                        zv += int(np.sum(f[:, 3] == 0))
                        zi += int(np.sum(f[:, 4] == 0))
                        if sum(len(x) for x in feat_chunks["x"]) < 400000:
                            feat_chunks["x"].append(f[:, 0])
                            feat_chunks["y"].append(f[:, 1])
                            feat_chunks["z"].append(f[:, 2])
                            feat_chunks["velocity"].append(f[:, 3])
                            feat_chunks["intensity"].append(f[:, 4])
                    qc.append(
                        {
                            "source_file": recs[-1]["source_file"],
                            "split": split.lower(),
                            "action": act,
                            "n_frames": nfr,
                            "empty_frames": empty,
                            "pts_min": int(min((len(f) for f in frames), default=0)),
                            "pts_max": int(max((len(f) for f in frames), default=0)),
                            "pts_median": float(np.median([len(f) for f in frames])) if frames else "",
                            "nan_count": nan,
                            "inf_count": inf,
                            "duplicate_points": dup,
                            "zero_velocity_frac": (zv / tot) if tot else "",
                            "zero_intensity_frac": (zi / tot) if tot else "",
                            "n_windows_t60_s10": max(0, (nfr - RAD_T) // RAD_STRIDE + 1) if nfr >= RAD_T else 0,
                        }
                    )
                    n_win += qc[-1]["n_windows_t60_s10"]
                    print("[radhar]", split, act, fp.name, "F", nfr, "P", tot, flush=True)
                except Exception as exc:  # noqa: BLE001
                    failed.append({"dataset": "RadHAR", "source_file": str(fp), "reason": str(exc)})
                    print("[radhar-fail]", fp, exc, flush=True)

    write_csv(
        out / "radhar_manifest.csv",
        recs,
        ["split", "source_file", "action", "subject", "session", "recording", "frames", "total_points", "mean_points_per_frame"],
    )
    write_csv(out / "radhar_qc.csv", qc, list(qc[0].keys()) if qc else ["source_file"])
    if failed:
        write_csv(out / "failed_samples.csv", failed, ["dataset", "source_file", "reason"])

    # official split at recording level; val = 20% of Train files, seed 0, stratified by action
    rng = random.Random(SEED)
    assign = {}
    by_a = defaultdict(list)
    for r in recs:
        if r["split"] == "train":
            by_a[r["action"]].append(r["source_file"])
        else:
            assign[r["source_file"]] = "test"
    for a, ids in by_a.items():
        ids = list(ids)
        rng.shuffle(ids)
        n_va = max(1, int(round(0.20 * len(ids))))
        for i in ids[:n_va]:
            assign[i] = "val"
        for i in ids[n_va:]:
            assign[i] = "train"
    rows = [{**r, "split_role": assign[r["source_file"]], "protocol": "official_train_test", "seed": SEED} for r in recs]
    write_csv(
        split_dir / "radhar_official.csv",
        rows,
        ["source_file", "action", "split", "split_role", "recording", "frames", "protocol", "seed"],
    )

    plt = _plt()
    if plt:
        c = Counter(r["action"] for r in recs)
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(RAD_CLASSES, [c[a] for a in RAD_CLASSES], color="#ee6c4d")
        ax.set_ylabel("txt recordings")
        fig.tight_layout()
        fig.savefig(figdir / "radhar_class_distribution.png", dpi=140)
        plt.close(fig)

    raw_names = {"x": "x", "y": "y", "z": "z", "velocity": "velocity", "intensity": "intensity"}
    units = {"x": "m", "y": "m", "z": "m", "velocity": "m/s", "intensity": "dB"}
    ev = "RadHAR README: x front m, y right-positive m, z up m, velocity range-rate m/s, intensity dB. New frame when point_id==0."
    for feat in raw_names:
        st = feat_stats(np.concatenate(feat_chunks[feat]) if feat_chunks[feat] else np.array([]))
        feat_schema_rows.append(
            {
                "dataset": "RadHAR",
                "feature": feat,
                "raw_name": raw_names[feat],
                "unit": units[feat],
                "min": st["min"],
                "max": st["max"],
                "mean": st["mean"],
                "std": st["std"],
                "evidence": ev,
            }
        )
    dump_json(
        out / "radhar_summary.json",
        {
            "n_recordings": len(recs),
            "n_train_txt": sum(1 for r in recs if r["split"] == "train"),
            "n_test_txt": sum(1 for r in recs if r["split"] == "test"),
            "subject_metadata": False,
            "median_pts_frame": float(np.median(pts_pf)) if pts_pf else None,
            "selected_P": RAD_P,
            "selected_T": RAD_T,
            "approx_windows_official": n_win,
        },
    )
    return {"recs": recs, "feat_chunks": feat_chunks, "pts_per_frame": pts_pf, "failed": failed, "assign": assign}


def domain_and_taxonomy(out: Path, mri_feat, rad_feat, mri_pts, rad_pts) -> None:
    figdir = out / "figures"
    plt = _plt()
    if plt:
        fig, ax = plt.subplots(figsize=(6, 3.2))
        ax.hist(mri_pts, bins=30, alpha=0.6, label="mRI", density=True)
        ax.hist(rad_pts, bins=30, alpha=0.6, label="RadHAR", density=True)
        ax.set_xlabel("points / frame")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "mri_vs_radhar_point_density.png", dpi=140)
        fig.savefig(figdir / "fig8_domain_density.png", dpi=140)
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 3.2))
        if mri_feat["velocity"]:
            ax.hist(np.concatenate(mri_feat["velocity"]), bins=40, alpha=0.6, label="mRI Doppler", density=True)
        if rad_feat["velocity"]:
            ax.hist(np.concatenate(rad_feat["velocity"]), bins=40, alpha=0.6, label="RadHAR velocity", density=True)
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "mri_vs_radhar_velocity_distribution.png", dpi=140)
        plt.close(fig)
        fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
        for ax, dim in zip(axes, ("x", "y", "z")):
            if mri_feat[dim]:
                ax.hist(np.concatenate(mri_feat[dim]), bins=30, alpha=0.6, label="mRI", density=True)
            if rad_feat[dim]:
                ax.hist(np.concatenate(rad_feat[dim]), bins=30, alpha=0.6, label="RadHAR", density=True)
            ax.set_title(dim)
        axes[0].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(figdir / "mri_vs_radhar_spatial_distribution.png", dpi=140)
        plt.close(fig)

    tax = [
        {"dataset": "mRI", "original_action": "T pose", "normalized_action": "T pose", "possible_cross_dataset_match": "", "confidence": "", "notes": "static pose; no RadHAR class"},
        {"dataset": "mRI", "original_action": "T pose 2", "normalized_action": "T pose", "possible_cross_dataset_match": "", "confidence": "", "notes": "short closing T-pose on 13/20 subjects; merged to T pose for training"},
        *[
            {
                "dataset": "mRI",
                "original_action": f"pose_{i}",
                "normalized_action": f"pose_{i}",
                "possible_cross_dataset_match": "",
                "confidence": "",
                "notes": "mRI paper pose set; not an exercise name",
            }
            for i in range(1, 11)
        ],
        {"dataset": "mRI", "original_action": "free_form", "normalized_action": "free_form", "possible_cross_dataset_match": "", "confidence": "", "notes": "unscripted motion"},
        {"dataset": "mRI", "original_action": "walk", "normalized_action": "walk", "possible_cross_dataset_match": "RadHAR walk", "confidence": "medium", "notes": "both walking; protocol/distance/speed not aligned in the packages"},
        {"dataset": "RadHAR", "original_action": "boxing", "normalized_action": "boxing", "possible_cross_dataset_match": "", "confidence": "", "notes": ""},
        {"dataset": "RadHAR", "original_action": "jack", "normalized_action": "jumping_jack", "possible_cross_dataset_match": "", "confidence": "", "notes": "jumping jack"},
        {"dataset": "RadHAR", "original_action": "jump", "normalized_action": "jump", "possible_cross_dataset_match": "", "confidence": "", "notes": ""},
        {"dataset": "RadHAR", "original_action": "squats", "normalized_action": "squats", "possible_cross_dataset_match": "", "confidence": "", "notes": ""},
        {"dataset": "RadHAR", "original_action": "walk", "normalized_action": "walk", "possible_cross_dataset_match": "mRI walk", "confidence": "medium", "notes": "only plausible name-level overlap"},
    ]
    write_csv(
        out / "action_taxonomy.csv",
        tax,
        ["dataset", "original_action", "normalized_action", "possible_cross_dataset_match", "confidence", "notes"],
    )


def example_figures(mri_root: Path, rad_root: Path, out: Path) -> None:
    plt = _plt()
    if plt is None:
        return
    figdir = out / "figures"
    # mRI: one walk frame + one pose from two subjects
    radar_dir = mri_root / "raw_data" / "radar"
    lab_dir = mri_root / "raw_data" / "videolabels"
    panels = []
    for sub, want in (("subject1", "walk"), ("subject10", "pose_1")):
        lab = load_label_json(lab_dir / f"{sub}.json")
        segs = [s for s in segments_from_json(lab) if s["action"] == want]
        csv_path = find_radar_csv(radar_dir, sub)
        if not segs or csv_path is None:
            continue
        frames, _ = load_radar_csv(csv_path)
        mid = (segs[0]["segment_start"] + segs[0]["segment_end"]) // 2
        pts = frames.get(mid)
        if pts is None:
            continue
        panels.append((f"mRI {sub} {want} f{mid}", pts))
    for split, act, globn in (("Train", "walk", "*walk*1.txt"), ("Train", "boxing", "*boxing_1.txt")):
        hits = list((rad_root / "Data" / split / act).glob(globn))
        if not hits:
            hits = list((rad_root / "Data" / split / act).glob("*.txt"))
        if not hits:
            continue
        frs = parse_mmwave_txt(hits[0])
        mid = len(frs) // 2
        panels.append((f"RadHAR {act} {hits[0].name}", frs[mid] if frs else np.zeros((0, 5))))

    n = min(4, len(panels))
    if n:
        fig, axes = plt.subplots(n, 3, figsize=(9, 2.2 * n))
        if n == 1:
            axes = np.array([axes])
        for i in range(n):
            title, pts = panels[i]
            for j, (a, b, lab) in enumerate(((0, 1, "XY"), (0, 2, "XZ"), (1, 2, "YZ"))):
                ax = axes[i, j]
                if pts.size:
                    sc = ax.scatter(pts[:, a], pts[:, b], c=pts[:, 3], s=8, cmap="coolwarm")
                ax.set_title(f"{title} {lab}", fontsize=7)
                ax.set_aspect("equal", adjustable="box")
        fig.tight_layout()
        fig.savefig(figdir / "fig1_pointcloud_examples.png", dpi=130)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 2.0))
    ax.axis("off")
    boxes = ["raw points\nCSV / txt", "sample P\n+ mask", "PointNet\nper frame", "pool / GRU", "HAR"]
    for i, t in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i * 2.0, 0.25), 1.7, 1.4, fill=False, lw=1.2))
        ax.text(i * 2.0 + 0.85, 0.95, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=(i * 2.0 + 1.95, 0.95), xytext=(i * 2.0 + 1.7, 0.95), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.1, 10)
    ax.set_ylim(0, 2)
    ax.set_title("Fig.2 point-cloud branch")
    fig.tight_layout()
    fig.savefig(figdir / "fig2_pipeline.png", dpi=140)
    plt.close(fig)


def cache_mri_windows(mri_root: Path, segs: list, cache_dir: Path, failed_path: Path) -> dict:
    from pointcloud.common import append_csv

    radar_dir = mri_root / "raw_data" / "radar"
    by_sub = defaultdict(list)
    for s in segs:
        by_sub[s["subject"]].append(s)
    n_ok = n_skip = 0
    index = []
    for subject, slist in by_sub.items():
        csv_path = find_radar_csv(radar_dir, subject)
        if csv_path is None:
            continue
        frames, _ = load_radar_csv(csv_path)
        for s in slist:
            sid = s["action_segment_id"]
            a, b = int(s["segment_start"]), int(s["segment_end"])
            seq = [frames.get(t, np.zeros((0, 5), np.float32)) for t in range(a, b + 1)]
            if len(seq) < MRI_T:
                continue
            for start in range(0, len(seq) - MRI_T + 1, MRI_STRIDE):
                wid = f"{sid}_w{start:04d}"
                dest = cache_dir / "mri" / f"{wid}.npz"
                if dest.exists() and dest.stat().st_size > 0:
                    n_skip += 1
                    index.append({"sample_id": wid, "cache_path": str(dest), "status": "exists", "subject": subject, "action": s["action"], "action_segment_id": sid})
                    continue
                try:
                    rng = np.random.RandomState(seed_from(wid))
                    x, m = stack_window(seq, start, MRI_T, MRI_P, rng)
                    atomic_savez(
                        dest,
                        X=x,
                        mask=m.astype(np.uint8),
                        y=np.int64(MRI_CLASSES.index(s["action"])),
                        sample_id=np.array(wid),
                        subject=np.array(subject),
                        action=np.array(s["action"]),
                        action_segment_id=np.array(sid),
                        source_file=np.array(s["source_file"]),
                    )
                    n_ok += 1
                    index.append({"sample_id": wid, "cache_path": str(dest), "status": "ok", "subject": subject, "action": s["action"], "action_segment_id": sid})
                except Exception as exc:  # noqa: BLE001
                    append_csv(failed_path, {"dataset": "mRI", "source_file": wid, "reason": str(exc)}, ["dataset", "source_file", "reason"])
            print("[cache-mri]", sid, flush=True)
    write_csv(cache_dir / "mri_window_index.csv", index, ["sample_id", "cache_path", "status", "subject", "action", "action_segment_id"])
    return {"n_ok": n_ok, "n_skip": n_skip, "n_index": len(index)}


def cache_radhar_windows(rad_root: Path, recs: list, assign: dict, cache_dir: Path, failed_path: Path) -> dict:
    from pointcloud.common import append_csv

    n_ok = n_skip = 0
    index = []
    for r in recs:
        fp = rad_root / r["source_file"]
        role = assign.get(r["source_file"])
        if role not in ("train", "val", "test"):
            continue
        try:
            frames = parse_mmwave_txt(fp)
        except Exception as exc:  # noqa: BLE001
            append_csv(failed_path, {"dataset": "RadHAR", "source_file": r["source_file"], "reason": str(exc)}, ["dataset", "source_file", "reason"])
            continue
        if len(frames) < RAD_T:
            continue
        y = RAD_CLASSES.index(r["action"])
        for start in range(0, len(frames) - RAD_T + 1, RAD_STRIDE):
            wid = f"radhar_{r['recording']}_w{start:04d}"
            dest = cache_dir / "radhar" / f"{wid}.npz"
            if dest.exists() and dest.stat().st_size > 0:
                n_skip += 1
                index.append({"sample_id": wid, "cache_path": str(dest), "status": "exists", "split": role})
                continue
            try:
                rng = np.random.RandomState(seed_from(wid))
                x, m = stack_window(frames, start, RAD_T, RAD_P, rng)
                atomic_savez(
                    dest,
                    X=x,
                    mask=m.astype(np.uint8),
                    y=np.int64(y),
                    sample_id=np.array(wid),
                    subject=np.array("UNKNOWN"),
                    action=np.array(r["action"]),
                    source_file=np.array(r["source_file"]),
                    split=np.array(role),
                )
                n_ok += 1
                index.append({"sample_id": wid, "cache_path": str(dest), "status": "ok", "split": role, "action": r["action"], "source_file": r["source_file"]})
            except Exception as exc:  # noqa: BLE001
                append_csv(failed_path, {"dataset": "RadHAR", "source_file": wid, "reason": str(exc)}, ["dataset", "source_file", "reason"])
        print("[cache-radhar]", r["source_file"], flush=True)
    write_csv(cache_dir / "radhar_window_index.csv", index, ["sample_id", "cache_path", "status", "split", "action", "source_file"])
    return {"n_ok": n_ok, "n_skip": n_skip, "n_index": len(index)}


def pack_windows(index_csv: Path, cache_root: Path, dest: Path, dataset: str) -> None:
    import csv as _csv

    rows = [r for r in _csv.DictReader(index_csv.open(encoding="utf-8")) if r.get("status") in ("ok", "exists")]
    xs, ms, ys, ids, splits, subs, acts, segs, srcs = [], [], [], [], [], [], [], [], []
    for r in rows:
        p = Path(r["cache_path"])
        if not p.exists():
            p = cache_root / Path(r["cache_path"]).name
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        xs.append(z["X"])
        ms.append(z["mask"])
        ys.append(int(z["y"]))
        ids.append(str(z["sample_id"]))
        splits.append(str(z["split"]) if "split" in z.files else "")
        subs.append(str(z["subject"]) if "subject" in z.files else "UNKNOWN")
        acts.append(str(z["action"]))
        segs.append(str(z["action_segment_id"]) if "action_segment_id" in z.files else "")
        srcs.append(str(z["source_file"]) if "source_file" in z.files else r.get("source_file", ""))
    if not xs:
        print("[pack-empty]", dest, flush=True)
        return
    atomic_savez(
        dest,
        X=np.stack(xs),
        mask=np.stack(ms),
        y=np.asarray(ys, np.int64),
        sample_id=np.array(ids),
        split=np.array(splits),
        subject=np.array(subs),
        action=np.array(acts),
        action_segment_id=np.array(segs),
        source_file=np.array(srcs),
        dataset=np.array(dataset),
    )
    print("[pack]", dest, "n", len(ys), dest.stat().st_size, flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    repo = Path(__file__).resolve().parents[2]
    ap.add_argument("--mri-root", default=os.environ.get("RADARMIND_MRI_ROOT", ""))
    ap.add_argument("--radhar-root", default=os.environ.get("RADARMIND_RADHAR_ROOT", ""))
    ap.add_argument("--out", default=os.environ.get("RADARMIND_PHASE1B_OUT", str(repo / "results" / "phase1_pointcloud")))
    ap.add_argument("--split-dir", default=os.environ.get("RADARMIND_SPLIT_DIR", str(repo / "data_splits")))
    ap.add_argument("--cache-windows", action="store_true")
    ap.add_argument("--pack", action="store_true")
    args = ap.parse_args()
    if not str(args.mri_root).strip() or not str(args.radhar_root).strip():
        raise SystemExit("set --mri-root / --radhar-root or RADARMIND_MRI_ROOT / RADARMIND_RADHAR_ROOT")
    out = Path(args.out)
    split_dir = Path(args.split_dir)
    out.mkdir(parents=True, exist_ok=True)
    split_dir.mkdir(parents=True, exist_ok=True)
    mri_root = Path(args.mri_root)
    rad_root = Path(args.radhar_root)

    m = build_mri(mri_root, out, split_dir)
    r = build_radhar(rad_root, out, split_dir, m["feat_schema_rows"])
    write_csv(
        out / "point_feature_schema.csv",
        m["feat_schema_rows"],
        ["dataset", "feature", "raw_name", "unit", "min", "max", "mean", "std", "evidence"],
    )
    domain_and_taxonomy(out, m["feat_chunks"], r["feat_chunks"], m["pts_per_frame"], r["pts_per_frame"])
    example_figures(mri_root, rad_root, out)

    n_mri_seg = len(m["seg_rows"])
    # rough window counts
    mri_win_est = 0
    for s in m["seg_rows"]:
        L = int(s.get("n_radar_frames") or 0)
        if L >= MRI_T:
            mri_win_est += 1 + (L - MRI_T) // MRI_STRIDE
    rad_win_est = int(sum(max(0, (int(x["frames"]) - RAD_T) // RAD_STRIDE + 1) if int(x["frames"]) >= RAD_T else 0 for x in r["recs"]))
    one_mri = MRI_T * MRI_P * 5 * 4
    one_rad = RAD_T * RAD_P * 5 * 4
    free = shutil.disk_usage(str(out)).free
    cost = {
        "mri_P": MRI_P,
        "mri_T": MRI_T,
        "radhar_P": RAD_P,
        "radhar_T": RAD_T,
        "mri_window_est": mri_win_est,
        "radhar_window_est": rad_win_est,
        "mri_cache_GB": mri_win_est * one_mri / 1024**3,
        "radhar_cache_GB": rad_win_est * one_rad / 1024**3,
        "disk_free_GB": free / 1024**3,
        "recommend": "per-window npz + packed npz. Do not dump PNG sequences.",
    }
    dump_json(out / "disk_cost.json", cost)
    print("[cost]", cost, flush=True)

    failed_path = out / "failed_samples.csv"
    if args.cache_windows and cost["mri_cache_GB"] + cost["radhar_cache_GB"] < 8 and cost["disk_free_GB"] > 10:
        cache_dir = out / "cache"
        info_m = cache_mri_windows(mri_root, m["seg_rows"], cache_dir, failed_path)
        # also tag random-segment split onto the same windows via second index? windows inherit segment; pack later by joining split files
        info_r = cache_radhar_windows(rad_root, r["recs"], r["assign"], cache_dir, failed_path)
        print("[cache-done]", info_m, info_r, flush=True)
        if args.pack:
            pack_windows(cache_dir / "mri_window_index.csv", cache_dir / "mri", cache_dir / "mri_windows_si.npz", "mRI")
            pack_windows(cache_dir / "radhar_window_index.csv", cache_dir / "radhar", cache_dir / "radhar_windows_official.npz", "RadHAR")
    print("[done]", flush=True)


if __name__ == "__main__":
    main()
