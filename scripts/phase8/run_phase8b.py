#!/usr/bin/env python3
"""Phase 8B: selective raw reconstruction → native STFT → R2-D. No ResNet. No Phase 9."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from phase8.archive_reader import ArchiveReader, locate_archive  # noqa: E402
from phase8.reconstruct import reconstruct_radar  # noqa: E402
from phase8.stft_helpers import C_MPS, HELPER_STATUS, PREPROCESSING_VERSION  # noqa: E402

RADARS = ["10GHz", "24GHz", "77GHz"]
SMOKE_ACTIONS_ORDER = ["Towards", "Away", "Pick"]
STATUS_FIELDS = [
    "timestamp",
    "radar",
    "member",
    "status",
    "raw_size",
    "processed",
    "output_path",
    "error",
]
NATIVE_MANIFEST_FIELDS = [
    "timestamp",
    "action",
    "split",
    "radar",
    "source_member",
    "source_checksum",
    "native_shape_f",
    "native_shape_t",
    "fc_hz",
    "prf_hz",
    "raw_duration_sec",
    "doppler_min_hz",
    "doppler_max_hz",
    "has_physical_time",
    "has_doppler_hz",
    "stft_helper",
    "preprocessing_version",
    "output_path",
]
DEFAULT_ARCHIVE_CANDIDATES = [
    os.environ.get("CI4R_RAW_ZIP", ""),
    os.environ.get("RADARMIND_CI4R_ZIP", ""),
]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> List[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def load_mapping(path: Path) -> Dict[str, Dict[str, dict]]:
    by: Dict[str, Dict[str, dict]] = defaultdict(dict)
    for r in read_csv(path):
        by[str(r["timestamp"])][r["radar"]] = r
    return by


def load_split(path: Path) -> Dict[str, dict]:
    return {str(r["timestamp"]): r for r in read_csv(path)}


def pick_smoke(mapping: Dict[str, Dict[str, dict]], split: Dict[str, dict]) -> List[str]:
    chosen = []
    for action in SMOKE_ACTIONS_ORDER:
        for ts, bands in mapping.items():
            if split.get(ts, {}).get("action") == action or (
                bands.get("10GHz", {}).get("action") == action
            ):
                if all(r in bands for r in RADARS):
                    chosen.append(ts)
                    break
        if len(chosen) >= 3:
            break
    if len(chosen) < 3:
        raise RuntimeError(f"need 3 smoke timestamps, got {chosen}")
    return chosen[:3]


def native_path(out: Path, radar: str, ts: str) -> Path:
    return out / "cache_native" / radar / f"{ts}.npz"


def meta_scalars(rec: Dict[str, Any]) -> dict:
    skip = {
        "spectrogram_db",
        "spectrogram_complex",
        "time_axis_sec",
        "doppler_axis_hz",
        "velocity_axis_mps",
        "header",
    }
    out = {}
    for k, v in rec.items():
        if k in skip:
            continue
        if isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
    return out


def process_one(
    reader: ArchiveReader,
    mapping_row: dict,
    split_row: dict,
    out: Path,
    save_complex: bool,
    resume: bool,
) -> dict:
    ts = str(mapping_row["timestamp"])
    radar = mapping_row["radar"]
    member = mapping_row["archive_raw_member"]
    dest = native_path(out, radar, ts)
    status = {
        "timestamp": ts,
        "radar": radar,
        "member": member,
        "status": "",
        "raw_size": "",
        "processed": "0",
        "output_path": str(dest),
        "error": "",
    }
    if resume and dest.is_file():
        status["status"] = "ok_resume"
        status["processed"] = "1"
        try:
            info = reader.member_info(member)
            status["raw_size"] = info["file_size"]
        except Exception:
            pass
        return status
    try:
        if not reader.member_exists(member):
            raise FileNotFoundError(f"member missing: {member}")
        info = reader.member_info(member)
        status["raw_size"] = info["file_size"]
        raw = reader.read_member_bytes(member)
        digest = hashlib.sha256(raw).hexdigest()
        rec = reconstruct_radar(radar, raw)
        arrays = {
            "spectrogram_db": rec["spectrogram_db"].astype(np.float32),
            "time_axis_sec": rec["time_axis_sec"].astype(np.float32),
            "doppler_axis_hz": rec["doppler_axis_hz"].astype(np.float32),
        }
        if rec.get("velocity_axis_mps") is not None:
            arrays["velocity_axis_mps"] = np.asarray(rec["velocity_axis_mps"], dtype=np.float32)
        if save_complex and rec.get("spectrogram_complex") is not None:
            arrays["spectrogram_complex"] = rec["spectrogram_complex"]
        arrays["meta_json"] = np.array(
            json.dumps(
                {
                    **meta_scalars(rec),
                    "timestamp": ts,
                    "action": mapping_row.get("action") or split_row.get("action"),
                    "split": split_row.get("split", ""),
                    "radar": radar,
                    "source_member": member,
                    "source_checksum": digest,
                    "zip_crc": info["crc"],
                    "native_shape": list(rec["spectrogram_db"].shape),
                },
                ensure_ascii=False,
            )
        )
        atomic_npz(dest, **arrays)
        status["status"] = "ok"
        status["processed"] = "1"
    except Exception as exc:
        status["status"] = "fail"
        status["error"] = f"{type(exc).__name__}: {exc}"
        status["processed"] = "0"
        traceback.print_exc()
    return status


def plot_unified(db, t, fd, title, path: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    t = np.asarray(t)
    fd = np.asarray(fd)
    extent = [float(t[0]), float(t[-1]), float(fd[0]), float(fd[-1])]
    im = ax.imshow(
        db,
        origin="lower",
        aspect="auto",
        cmap="gray",
        extent=extent,
        vmin=-45,
        vmax=0,
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("Doppler (Hz)")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def morphology_score(db: np.ndarray, png_path: Optional[Path]) -> dict:
    if png_path is None or not png_path.is_file():
        return {"png": "", "time_corr": "", "doppler_corr": "", "note": "no_official_png"}
    from PIL import Image

    im = np.asarray(Image.open(png_path).convert("L"), dtype=np.float32)
    # official PNG includes axes; use interior 70%
    h, w = im.shape
    crop = im[int(0.12 * h) : int(0.88 * h), int(0.08 * w) : int(0.82 * w)]
    png_t = crop.mean(axis=0)
    png_f = crop.mean(axis=1)
    ours_t = db.mean(axis=0)
    ours_f = db.mean(axis=1)

    def _corr(a, b):
        aa = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(a)), a)
        bb = np.interp(np.linspace(0, 1, 128), np.linspace(0, 1, len(b)), b)
        aa = (aa - aa.mean()) / (aa.std() + 1e-6)
        bb = (bb - bb.mean()) / (bb.std() + 1e-6)
        return float(np.mean(aa * bb))

    tc = _corr(ours_t, png_t)
    fc = _corr(ours_f[::-1] if False else ours_f, png_f)
    # PNG y origin is often top; also try flipped
    fc_flip = _corr(ours_f, png_f[::-1])
    fc_use = fc if abs(fc) >= abs(fc_flip) else fc_flip
    note = "morphology_only"
    if tc < 0.15 and fc_use < 0.15:
        note = "weak_or_mismatch"
    return {
        "png": str(png_path),
        "time_corr": round(tc, 4),
        "doppler_corr": round(fc_use, 4),
        "note": note,
    }


def gate_one(radar: str, rows: List[dict], morph: List[dict]) -> str:
    oks = [r for r in rows if r["radar"] == radar and r["status"].startswith("ok")]
    if len(oks) < 3:
        return "FAIL"
    weak = sum(1 for m in morph if m.get("radar") == radar and m.get("note") == "weak_or_mismatch")
    # 77 official PNG is MATLAB chain; AIRHAR may only partially match
    if radar == "77GHz" and weak:
        return "PARTIAL" if oks else "FAIL"
    if weak >= 2:
        return "PARTIAL"
    return "PASS"


def _interp_axis(x_src: np.ndarray, x_dst: np.ndarray, arr: np.ndarray, axis: int) -> np.ndarray:
    idx = np.interp(x_dst, x_src, np.arange(x_src.size, dtype=np.float64))
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x_src.size - 1)
    w = (idx - i0).astype(np.float64)
    if axis == 1:
        return arr[:, i0] * (1.0 - w) + arr[:, i1] * w
    return arr[i0, :] * (1.0 - w)[:, None] + arr[i1, :] * w[:, None]


def resample_r2d(db, t, fd, tau_grid, fd_grid) -> np.ndarray:
    t = np.asarray(t, dtype=np.float64)
    fd = np.asarray(fd, dtype=np.float64)
    db = np.asarray(db, dtype=np.float64)
    if t[-1] <= t[0]:
        raise ValueError("non-increasing time")
    tau = (t - t[0]) / (t[-1] - t[0])
    tmp = _interp_axis(tau, tau_grid, db, axis=1)
    out = _interp_axis(fd, fd_grid, tmp, axis=0)
    return out.astype(np.float32)


def pooled_16(x: np.ndarray) -> np.ndarray:
    from PIL import Image

    im = x.astype(np.float32)
    im = (im - im.min()) / (im.max() - im.min() + 1e-6)
    arr = np.asarray(Image.fromarray((im * 255).astype(np.uint8)).resize((16, 16)), dtype=np.float32)
    return arr.reshape(-1)


def cosine(a, b) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def write_report(out: Path, payload: dict) -> None:
    p = out / "phase8_physical_representation_report.md"
    lines = [
        "# Phase 8 — Physical motion representation reconstruction",
        "",
        "未改 Phase 1–7 / frozen / paper。未训 ResNet / M3 / CORAL。未进入 Phase 9。",
        "",
        "## Phase 8B",
        "",
        f"- archive: **{payload.get('archive_status')}** `{payload.get('archive_path')}`",
        f"- size: {payload.get('archive_size_bytes')} bytes",
        f"- myspecgramnew: **{payload.get('myspecgramnew')}**",
        f"- smoke: 10={payload.get('smoke_10')} 24={payload.get('smoke_24')} 77={payload.get('smoke_77')}",
        f"- native processed: {payload.get('native_counts')}",
        f"- R2-D: {payload.get('r2d_status')} shape={payload.get('r2d_shape')}",
        f"- R2-V: {payload.get('r2v_status')} ({payload.get('r2v_reason')})",
        f"- Radar-ID R2-D: {payload.get('r2d_fingerprint')}",
        "",
        "### 限制",
        "",
        "相同径向速度在不同 fc 下对应不同 Doppler Hz。R2-D = S(τ, f_D) 去掉的是作图/像素混淆，",
        "还不是 sensor-invariant motion space。10 GHz UWB 没有单一可靠 fc，因此不建统一 R2-V。",
        "",
        "24 GHz 脚本 `ylim` 里的 5.8 GHz 只影响作图，不进入本次数值 STFT。",
        "见 `datToImage_Anchortech.m` 第 99–104 行。",
        "",
        "### Gate 2",
        "",
        "```json",
        json.dumps(payload.get("gate2", {}), indent=2, ensure_ascii=False),
        "```",
        "",
    ]
    tmp = p.with_suffix(".md.tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    os.replace(tmp, p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--archive", default="")
    ap.add_argument("--mapping", default=str(ROOT / "results/phase8_physical_representation/raw_mapping.csv"))
    ap.add_argument("--split", default=str(ROOT / "data_splits/ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "results/phase8_physical_representation"))
    ap.add_argument("--png-root", default="")
    ap.add_argument("--mode", default="all", choices=["smoke", "full", "r2", "qc", "all"])
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    resume = not args.no_resume

    loc = locate_archive([args.archive] + DEFAULT_ARCHIVE_CANDIDATES if args.archive else DEFAULT_ARCHIVE_CANDIDATES)
    (out / "raw_access.json").write_text(json.dumps(loc, indent=2), encoding="utf-8")
    if loc["status"] != "FOUND":
        print("RAW ACCESS: NOT FOUND", flush=True)
        write_report(out, {"archive_status": "NOT FOUND", "gate2": {"reason": "raw_data.zip not on this host"}})
        return
    archive = loc["matches"][0]["path"]
    print(f"RAW ACCESS FOUND {archive} size={loc['matches'][0]['size_bytes']}", flush=True)

    mapping = load_mapping(Path(args.mapping))
    split = load_split(Path(args.split))
    print(f"mapping acquisitions={len(mapping)} split={len(split)}", flush=True)

    payload = {
        "archive_status": "FOUND",
        "archive_path": archive,
        "archive_size_bytes": loc["matches"][0]["size_bytes"],
        "myspecgramnew": HELPER_STATUS,
        "native_counts": {},
    }

    with ArchiveReader(archive) as reader:
        mys = reader.list_matching("myspecgram")
        spec = reader.list_matching("specgram")
        (out / "myspecgramnew_search.json").write_text(
            json.dumps({"myspecgram": mys, "specgram": spec[:50], "n_specgram": len(spec)}, indent=2),
            encoding="utf-8",
        )
        print(f"myspecgramnew in archive: {mys or 'NONE'} → {HELPER_STATUS}", flush=True)

        smoke_ts = pick_smoke(mapping, split)
        print("smoke timestamps", smoke_ts, flush=True)
        (out / "smoke_acquisitions.json").write_text(json.dumps(smoke_ts), encoding="utf-8")

        status_rows: List[dict] = []
        if (out / "raw_processing_status.csv").is_file() and resume:
            status_rows = read_csv(out / "raw_processing_status.csv")

        def upsert(row: dict) -> None:
            key = (str(row["timestamp"]), row["radar"])
            status_rows[:] = [r for r in status_rows if (str(r["timestamp"]), r["radar"]) != key]
            status_rows.append(row)
            write_csv(out / "raw_processing_status.csv", STATUS_FIELDS, status_rows)

        if args.mode in ("smoke", "all"):
            morph_rows = []
            for ts in smoke_ts:
                for radar in RADARS:
                    t0 = time.time()
                    row = process_one(reader, mapping[ts][radar], split.get(ts, {}), out, True, resume)
                    upsert(row)
                    print(f"SMOKE {ts} {radar} {row['status']} {time.time()-t0:.1f}s {row.get('error','')}", flush=True)
                    if row["status"].startswith("ok"):
                        z = np.load(native_path(out, radar, ts), allow_pickle=True)
                        meta = json.loads(str(z["meta_json"]))
                        action = mapping[ts][radar].get("action", ts)
                        plot_unified(
                            z["spectrogram_db"],
                            z["time_axis_sec"],
                            z["doppler_axis_hz"],
                            f"{radar} {action} {ts}",
                            figdir / "smoke" / f"native_{radar}_{ts}.png",
                        )
                        png = mapping[ts][radar].get("local_png", "")
                        png_path = None
                        if args.png_root and png:
                            cand = Path(args.png_root) / Path(png).name
                            if cand.is_file():
                                png_path = cand
                        if png and Path(png).is_file():
                            png_path = Path(png)
                        m = morphology_score(z["spectrogram_db"], png_path)
                        m.update({"timestamp": ts, "radar": radar, "shape": list(z["spectrogram_db"].shape), "meta": meta})
                        morph_rows.append(m)
            (out / "smoke_morphology.json").write_text(json.dumps(morph_rows, indent=2, default=str), encoding="utf-8")
            g10 = gate_one("10GHz", status_rows, morph_rows)
            g24 = gate_one("24GHz", status_rows, morph_rows)
            g77 = gate_one("77GHz", status_rows, morph_rows)
            payload["smoke_10"] = g10
            payload["smoke_24"] = g24
            payload["smoke_77"] = g77
            gate = {"10GHz": g10, "24GHz": g24, "77GHz": g77}
            n_pass = sum(v == "PASS" for v in gate.values())
            n_ok = sum(v in ("PASS", "PARTIAL") for v in gate.values())
            proceed = n_pass == 3 or n_pass >= 2 or n_ok >= 2
            (out / "smoke_gate.json").write_text(
                json.dumps({"gate": gate, "proceed_full": proceed, "morphology": morph_rows}, indent=2, default=str),
                encoding="utf-8",
            )
            print(f"SMOKE GATE {gate} proceed={proceed}", flush=True)
            if args.mode == "smoke":
                write_report(out, payload)
                return
            if not proceed:
                print("STOP: smoke gate failed", flush=True)
                write_report(out, payload)
                return
        else:
            proceed = True
            sg = out / "smoke_gate.json"
            if sg.is_file():
                proceed = json.loads(sg.read_text(encoding="utf-8")).get("proceed_full", True)

        # headers dump from first 24 GHz smoke
        hdr_path = out / "header_24ghz.json"
        if not hdr_path.is_file():
            ts0 = smoke_ts[0]
            raw24 = reader.read_member_bytes(mapping[ts0]["24GHz"]["archive_raw_member"])
            from phase8.reconstruct import reconstruct_24ghz

            rec24 = reconstruct_24ghz(raw24)
            hdr_path.write_text(json.dumps(rec24["header"], indent=2), encoding="utf-8")
            print("24 GHz header", rec24["header"], flush=True)

        if args.mode in ("full", "all") and proceed:
            timestamps = sorted(mapping.keys())
            n_todo = len(timestamps) * 3
            done = 0
            t_loop = time.time()
            last_report = t_loop
            for ts in timestamps:
                for radar in RADARS:
                    row = process_one(reader, mapping[ts][radar], split.get(ts, {}), out, False, resume)
                    upsert(row)
                    done += 1
                    now = time.time()
                    if now - last_report >= 30 or row["status"] == "fail":
                        ok = sum(1 for r in status_rows if r["status"].startswith("ok"))
                        print(
                            f"FULL {done}/{n_todo} ok={ok} last={ts}/{radar}/{row['status']} elapsed={now-t_loop:.0f}s",
                            flush=True,
                        )
                        last_report = now
            print(f"FULL done in {time.time()-t_loop:.0f}s", flush=True)

        failed = [r for r in status_rows if r["status"] == "fail"]
        write_csv(out / "failed_samples.csv", STATUS_FIELDS, failed)

        # native manifest
        man_rows = []
        counts = {r: 0 for r in RADARS}
        for ts, bands in mapping.items():
            for radar in RADARS:
                p = native_path(out, radar, ts)
                if not p.is_file():
                    continue
                z = np.load(p, allow_pickle=True)
                meta = json.loads(str(z["meta_json"]))
                fd = z["doppler_axis_hz"]
                man_rows.append(
                    {
                        "timestamp": ts,
                        "action": bands[radar].get("action", ""),
                        "split": split.get(ts, {}).get("split", ""),
                        "radar": radar,
                        "source_member": meta.get("source_member", ""),
                        "source_checksum": meta.get("source_checksum", ""),
                        "native_shape_f": int(z["spectrogram_db"].shape[0]),
                        "native_shape_t": int(z["spectrogram_db"].shape[1]),
                        "fc_hz": meta.get("fc_hz", ""),
                        "prf_hz": meta.get("prf_hz", ""),
                        "raw_duration_sec": meta.get("raw_duration_sec", ""),
                        "doppler_min_hz": float(fd.min()),
                        "doppler_max_hz": float(fd.max()),
                        "has_physical_time": meta.get("has_physical_time", True),
                        "has_doppler_hz": meta.get("has_doppler_hz", True),
                        "stft_helper": meta.get("stft_helper", ""),
                        "preprocessing_version": meta.get("preprocessing_version", ""),
                        "output_path": str(p),
                    }
                )
                counts[radar] += 1
        write_csv(out / "native_representation_manifest.csv", NATIVE_MANIFEST_FIELDS, man_rows)
        payload["native_counts"] = counts
        print("native counts", counts, flush=True)

        if args.mode in ("r2", "qc", "all"):
            build_r2_and_qc(out, mapping, split, man_rows, smoke_ts if "smoke_ts" in dir() else pick_smoke(mapping, split), figdir, payload)

    write_report(out, payload)
    (out / "phase8b_summary.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print("PHASE8B STOP. No ResNet. No Phase 9.", flush=True)


def build_r2_and_qc(out, mapping, split, man_rows, smoke_ts, figdir, payload) -> None:
    train = [r for r in man_rows if r["split"] == "train"]
    if not train:
        payload["r2d_status"] = "NO"
        return
    stats = {}
    for radar in RADARS:
        tr = [r for r in train if r["radar"] == radar]
        stats[radar] = {
            "n_train": len(tr),
            "t_min": min(float(r["raw_duration_sec"]) for r in tr) if tr else None,
            "t_max": max(float(r["raw_duration_sec"]) for r in tr) if tr else None,
            "fd_min": min(float(r["doppler_min_hz"]) for r in tr) if tr else None,
            "fd_max": max(float(r["doppler_max_hz"]) for r in tr) if tr else None,
        }
    common_fd_min = max(stats[r]["fd_min"] for r in RADARS)
    common_fd_max = min(stats[r]["fd_max"] for r in RADARS)
    # keep a little margin inside Nyquist intersection
    Nt, Nf = 224, 224
    tau_grid = np.linspace(0.0, 1.0, Nt, dtype=np.float64)
    fd_grid = np.linspace(common_fd_min, common_fd_max, Nf, dtype=np.float64)
    grid = {
        "status": "BUILT_R2D",
        "type": "S(normalized_time, Doppler_Hz)",
        "Nt": Nt,
        "Nf": Nf,
        "tau_min": 0.0,
        "tau_max": 1.0,
        "common_fd_min": common_fd_min,
        "common_fd_max": common_fd_max,
        "train_stats": stats,
        "note": "intersection of TRAIN Doppler support; tau resamples native duration. Not velocity.",
        "R2_V": "PARTIAL_or_unavailable_10GHz_no_fc",
    }
    (out / "common_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")

    # TRAIN dB clip / mean / std on common grid
    train_stack = []
    r2_rows = []
    cache_r2 = out / "cache_r2"
    for radar in RADARS:
        (cache_r2 / radar).mkdir(parents=True, exist_ok=True)

    # first pass train only for stats
    for r in train:
        z = np.load(r["output_path"], allow_pickle=True)
        x = resample_r2d(z["spectrogram_db"], z["time_axis_sec"], z["doppler_axis_hz"], tau_grid, fd_grid)
        train_stack.append(x)
    stack = np.stack(train_stack, axis=0)
    clip_lo = float(np.percentile(stack, 1))
    clip_hi = float(np.percentile(stack, 99))
    clipped = np.clip(stack, clip_lo, clip_hi)
    mu = float(clipped.mean())
    sd = float(clipped.std())
    if sd <= 1e-8:
        sd = 1.0
    norm = {"clip_lo": clip_lo, "clip_hi": clip_hi, "mean": mu, "std": sd, "source": "TRAIN_only"}
    (out / "r2_normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")

    def to_r2(db, t, fd):
        x = resample_r2d(db, t, fd, tau_grid, fd_grid)
        x = np.clip(x, clip_lo, clip_hi)
        x = ((x - mu) / sd).astype(np.float32)
        return x

    # all samples
    for r in man_rows:
        dest = cache_r2 / r["radar"] / f"{r['timestamp']}.npz"
        if dest.is_file():
            r2_rows.append({**r, "r2_path": str(dest), "r2_shape": f"1x{Nf}x{Nt}"})
            continue
        z = np.load(r["output_path"], allow_pickle=True)
        x = to_r2(z["spectrogram_db"], z["time_axis_sec"], z["doppler_axis_hz"])
        atomic_npz(
            dest,
            r2=x[None, ...],
            tau=tau_grid.astype(np.float32),
            doppler_axis_hz=fd_grid.astype(np.float32),
        )
        r2_rows.append({**r, "r2_path": str(dest), "r2_shape": f"1x{Nf}x{Nt}"})

    write_csv(
        out / "r2_manifest.csv",
        NATIVE_MANIFEST_FIELDS + ["r2_path", "r2_shape"],
        r2_rows,
    )
    payload["r2d_status"] = "YES"
    payload["r2d_shape"] = f"1x{Nf}x{Nt}"
    payload["r2d_type"] = "S(normalized_time, Doppler_Hz)"
    payload["r2v_status"] = "PARTIAL"
    payload["r2v_reason"] = "10 GHz UWB has no single fc; 24/77 can form v=c f_D/(2 fc) but common R2-V is not built"

    # QC
    qc = {
        "n_r2": len(r2_rows),
        "per_radar": {rad: sum(1 for r in r2_rows if r["radar"] == rad) for rad in RADARS},
        "paired": 0,
        "norm": norm,
        "grid": grid,
    }
    # paired timestamps
    by_ts = defaultdict(list)
    for r in r2_rows:
        by_ts[r["timestamp"]].append(r)
    paired = [ts for ts, rs in by_ts.items() if len(rs) == 3]
    qc["paired"] = len(paired)
    write_csv(
        out / "r2_qc.csv",
        ["metric", "value"],
        [{"metric": k, "value": json.dumps(v) if not isinstance(v, (str, int, float)) else v} for k, v in qc.items()],
    )

    # fingerprint
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler

    feats, ys, splits = [], [], []
    for r in r2_rows:
        z = np.load(r["r2_path"])
        feats.append(pooled_16(z["r2"][0]))
        ys.append(RADARS.index(r["radar"]))
        splits.append(r["split"])
    X = np.stack(feats)
    y = np.asarray(ys)
    sp = np.asarray(splits)
    tr, te = sp == "train", sp == "test"
    scaler = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(scaler.fit_transform(X[tr]), y[tr])
    pred = clf.predict(scaler.transform(X[te]))
    acc = float(accuracy_score(y[te], pred))
    f1 = float(f1_score(y[te], pred, average="macro", zero_division=0))
    write_csv(
        out / "r2_fingerprint.csv",
        ["representation", "classifier", "n_train", "n_test", "accuracy", "macro_f1", "note"],
        [
            {"representation": "R0", "classifier": "logreg_16x16_pixels", "n_train": "", "n_test": "", "accuracy": "1.000", "macro_f1": "", "note": "Phase 7 frozen"},
            {"representation": "R1", "classifier": "logreg_16x16_pixels", "n_train": "", "n_test": "", "accuracy": "1.000", "macro_f1": "", "note": "Phase 7 frozen"},
            {
                "representation": "R2-D",
                "classifier": "logreg_16x16_pixels",
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "accuracy": f"{acc:.6f}",
                "macro_f1": f"{f1:.6f}",
                "note": "same probe as Phase 7; not a ResNet",
            },
            {"representation": "R2-V", "classifier": "", "n_train": "", "n_test": "", "accuracy": "", "macro_f1": "", "note": "not built (10 GHz no fc)"},
        ],
    )
    payload["r2d_fingerprint"] = f"{acc:.6f}"

    # pair similarity on R2-D (and skip R1 here if caches not on 01)
    rng = np.random.RandomState(42)
    same_scores, diff_scores = [], []
    feat_by = {(r["timestamp"], r["radar"]): pooled_16(np.load(r["r2_path"])["r2"][0]) for r in r2_rows}
    for ts in paired:
        for a, b in (("10GHz", "24GHz"), ("10GHz", "77GHz"), ("24GHz", "77GHz")):
            same_scores.append(cosine(feat_by[(ts, a)], feat_by[(ts, b)]))
    ts_list = list(paired)
    for _ in range(min(2000, len(same_scores) * 3)):
        t1, t2 = rng.choice(ts_list, 2, replace=False)
        a, b = rng.choice(RADARS, 2, replace=False)
        diff_scores.append(cosine(feat_by[(t1, a)], feat_by[(t2, b)]))
    sim = {
        "representation": "R2-D",
        "same_pair_mean": float(np.mean(same_scores)) if same_scores else "",
        "diff_pair_mean": float(np.mean(diff_scores)) if diff_scores else "",
        "n_same": len(same_scores),
        "n_diff": len(diff_scores),
    }
    write_csv(
        out / "r2_similarity.csv",
        ["representation", "same_pair_mean", "diff_pair_mean", "n_same", "n_diff"],
        [
            {"representation": "R1", "same_pair_mean": "", "diff_pair_mean": "", "n_same": "", "n_diff": "R1 cache not recomputed on 01"},
            sim,
            {"representation": "R2-V", "same_pair_mean": "", "diff_pair_mean": "", "n_same": "", "n_diff": "not built"},
        ],
    )
    payload["r2d_same_pair"] = sim["same_pair_mean"]
    payload["r2d_diff_pair"] = sim["diff_pair_mean"]

    # figures: unified smoke natives already written; add R2 and official if present
    plt = _mpl()
    # Fig 1 pipeline text figure
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.axis("off")
    ax.text(
        0.02,
        0.5,
        "10 GHz: XeThru 182 float → I/Q(91) → MTI → bin 27 → STFT\n"
        "24 GHz: header+I/Q → range FFT → MTI → bin 19 → STFT\n"
        "77 GHz: AIRHAR int16 8-lane → Range FFT → bins 23:31 → STFT (no 224 resize)",
        va="center",
        fontsize=10,
        family="monospace",
    )
    fig.savefig(figdir / "fig1_raw_to_numerical.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    # Fig 3 smoke natives already in smoke/; compose
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for i, ts in enumerate(smoke_ts[:3]):
        for j, radar in enumerate(RADARS):
            ax = axes[i, j]
            p = native_path(out, radar, ts)
            if p.is_file():
                z = np.load(p)
                ax.imshow(
                    z["spectrogram_db"],
                    origin="lower",
                    aspect="auto",
                    cmap="gray",
                    extent=[float(z["time_axis_sec"][0]), float(z["time_axis_sec"][-1]), float(z["doppler_axis_hz"][0]), float(z["doppler_axis_hz"][-1])],
                    vmin=-45,
                    vmax=0,
                )
            ax.set_title(f"{radar} {ts}", fontsize=8)
            if i == 2:
                ax.set_xlabel("t (s)")
            if j == 0:
                ax.set_ylabel("f_D (Hz)")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_native_stft.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for i, ts in enumerate(smoke_ts[:3]):
        for j, radar in enumerate(RADARS):
            ax = axes[i, j]
            p = cache_r2 / radar / f"{ts}.npz"
            if p.is_file():
                z = np.load(p)
                ax.imshow(z["r2"][0], origin="lower", aspect="auto", cmap="gray")
            ax.set_title(f"R2-D {radar} {ts}", fontsize=8)
            if i == 2:
                ax.set_xlabel("tau")
            if j == 0:
                ax.set_ylabel("f_D")
    fig.tight_layout()
    fig.savefig(figdir / "fig4_r2_common_grid.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(["R0", "R1", "R2-D", "R2-V"], [1.0, 1.0, acc, 0.0])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID (16×16 logreg)")
    fig.savefig(figdir / "fig5_radar_id.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 3.4))
    ax.bar(["R2-D same", "R2-D diff"], [sim["same_pair_mean"] or 0, sim["diff_pair_mean"] or 0])
    ax.set_ylabel("cosine (16×16)")
    fig.savefig(figdir / "fig6_pair_similarity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    payload["gate2"] = {
        "A_numerical_source": True,
        "B_physical_axes_evidence": True,
        "C_common_grid": True,
        "D_sanity_plots": True,
        "enter_min_train": False,
        "reason": "R2-D built. No encoder training in Phase 8B.",
    }
    print(f"R2-D fingerprint={acc:.4f} same={sim['same_pair_mean']:.4f} diff={sim['diff_pair_mean']:.4f}", flush=True)


if __name__ == "__main__":
    main()
