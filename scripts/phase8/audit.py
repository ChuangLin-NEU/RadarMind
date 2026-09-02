#!/usr/bin/env python3
"""Phase 8 physical-data inventory. Does not train. Does not write frozen trees."""
from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"
INV = ROOT / "results" / "data_audit" / "CI4R_file_inventory.csv"
SPLIT = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"
P2_MAN = ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv"
P1_MAN = ROOT / "results" / "phase1_ci4r" / "ci4r_77ghz_manifest.csv"
P1_CACHE = ROOT / "results" / "phase1_ci4r" / "cache" / "ci4r77_airhar_spec_zscore.npz"
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RADARS = ["10GHz", "24GHz", "77GHz"]
TS_RE = re.compile(r"(1[0-9]{9})")
ZIP_01 = os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", ""))
GCS = "https://storage.googleapis.com/cross-frequency_dataset/Raw%20Data/raw_data.zip"

INV_FIELDS = [
    "radar_frequency",
    "acquisition_timestamp",
    "action",
    "split",
    "raw_available",
    "raw_format",
    "numerical_available",
    "numerical_format",
    "png_available",
    "source_path",
    "loader_available",
    "metadata_available",
    "physical_axis_recoverable",
    "status",
    "evidence",
    "archive_raw_member",
    "archive_numerical_member",
    "archive_raw_bytes",
    "local_raw_openable",
    "local_png_path",
]


def write_csv(path: Path, fields: list[str], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def ts_of(name: str) -> str | None:
    m = TS_RE.search(name.replace("\\", "/").split("/")[-1])
    return m.group(1) if m else None


def load_archive_index() -> dict[str, dict[str, list[dict]]]:
    """radar -> timestamp -> list of members."""
    out = {r: defaultdict(list) for r in RADARS}
    for r in read_csv(INV):
        if r.get("is_dir") in ("1", "True", True):
            continue
        mem = r["member"].replace("\\", "/")
        ts = ts_of(mem)
        if not ts:
            continue
        rec = {"member": mem, "suffix": r.get("suffix", ""), "file_size": int(r.get("file_size") or 0)}
        if mem.startswith("Cross-frequency/10ghz/"):
            out["10GHz"][ts].append(rec)
        elif mem.startswith("Cross-frequency/24ghz/"):
            out["24GHz"][ts].append(rec)
        elif mem.startswith("Cross-frequency/77ghz/"):
            out["77GHz"][ts].append(rec)
    return out


def pick_raw(radar: str, items: list[dict]) -> dict | None:
    if radar == "77GHz":
        hits = [x for x in items if x["member"].endswith("_Raw_0.bin") and "/activity_" not in x["member"]]
    else:
        hits = [x for x in items if x["suffix"] == ".dat" or x["member"].endswith(".dat")]
    return hits[0] if hits else None


def pick_num(radar: str, items: list[dict]) -> dict | None:
    if radar != "77GHz":
        mats = [x for x in items if x["suffix"] == ".mat" and "activity_" not in x["member"]]
        return mats[0] if mats else None
    mats = [x for x in items if "activity_spect_matrices" in x["member"] and x["member"].endswith("_matrix.mat")]
    return mats[0] if mats else None


def local_png(man_by: dict, ts: str, radar: str) -> str:
    hit = man_by.get((ts, radar))
    if not hit:
        return ""
    p = ROOT / hit["source_file"]
    return hit["source_file"] if p.is_file() else ""


def probe_gcs() -> dict:
    rec = {"url": GCS, "reachable": False, "content_length": None, "error": ""}
    try:
        import urllib.request

        req = urllib.request.Request(GCS, method="HEAD")
        with urllib.request.urlopen(req, timeout=12) as resp:
            rec["reachable"] = True
            rec["content_length"] = resp.headers.get("Content-Length")
            rec["status"] = resp.status
    except Exception as exc:  # noqa: BLE001
        rec["error"] = str(exc)
    return rec


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    archive = load_archive_index()
    split_rows = read_csv(SPLIT)
    p2 = read_csv(P2_MAN)
    man_by = {(r["timestamp"], r["radar_frequency"]): r for r in p2}
    triples = [r for r in split_rows if r.get("action") in CLASS_NAMES]
    p1_ts = set()
    if P1_MAN.is_file():
        for r in read_csv(P1_MAN):
            p1_ts.add(str(r.get("unix_token", "")))
    p1_cache_ts = set()
    if P1_CACHE.is_file():
        z = np.load(P1_CACHE, allow_pickle=True)
        key = "timestamp" if "timestamp" in z.files else ("unix_token" if "unix_token" in z.files else None)
        if key:
            p1_cache_ts = {str(x) for x in z[key]}
        elif "sample_id" in z.files:
            for s in z["sample_id"]:
                m = TS_RE.search(str(s))
                if m:
                    p1_cache_ts.add(m.group(1))

    zip_here = any(
        Path(p).is_file()
        for p in (
            ROOT / "raw_data.zip",
            Path(ZIP_01) if ZIP_01 else Path(),
        )
        if str(p)
    )
    gcs = probe_gcs()
    (OUT / "remote_zip_probe.json").write_text(json.dumps({"gcs": gcs, "local_zip": zip_here, "linux_01": ZIP_01}, indent=2), encoding="utf-8")

    inv_rows, map_rows = [], []
    counts = {r: {"archive_raw": 0, "archive_num": 0, "local_png": 0, "p1_manifest": 0, "p1_cache": 0} for r in RADARS}

    for acq in triples:
        ts, act, spl = acq["timestamp"], acq["action"], acq["split"]
        for radar in RADARS:
            items = archive[radar].get(ts, [])
            raw = pick_raw(radar, items)
            num = pick_num(radar, items)
            png = local_png(man_by, ts, radar)
            local_raw = False
            if raw and zip_here:
                local_raw = True
            if radar == "10GHz":
                raw_fmt = "XeThru binary .dat (uint32 hdr + 182 float/frame)" if raw else ""
                num_fmt = ""
                loader = True
                meta = "script constants; per-file fc not in header"
                axis = "Doppler Hz from PRF=512 if STFT replayed; velocity needs single fc (UWB: not a tone)"
            elif radar == "24GHz":
                raw_fmt = "Ancortek text .dat (fc,Tsweep,NTS,Bw + I/Q column)" if raw else ""
                num_fmt = ""
                loader = True
                meta = "in-file header if raw opened; PNG has none"
                axis = "Doppler Hz from 1/Tsweep if header read; velocity if fc read; ylim 5.8 GHz is plot only"
            else:
                raw_fmt = "DCA1000 *_Raw_0.bin int16 8-lane" if raw else ""
                num_fmt = "archive MAT sx1 [4096, n_win] in activity_spect_matrices" if num else (
                    "Phase1 AIRHAR 224 z-score cache (not physical grid)" if ts in p1_cache_ts else ""
                )
                loader = True
                meta = "datToImage_77.m / AIRHAR constants"
                axis = "Doppler Hz from PRF=3200; display crop ±500 Hz; v=c f_D/(2 fc) if using script fc"
            if raw and not local_raw:
                status = "UNRESOLVED"
                ev = f"archive lists {raw['member']} ({raw['file_size']} B) in {ZIP_01}; not extracted on this host"
            elif raw and local_raw:
                status = "RAW_READY"
                ev = "local zip openable"
            elif num:
                status = "NUMERICAL_READY" if local_raw else "UNRESOLVED"
                ev = f"numerical member {num['member']} in archive; not on this host"
            elif png:
                status = "PNG_ONLY"
                ev = "official PNG on this host; no archive raw member for this timestamp"
            else:
                status = "MISSING"
                ev = "no PNG and no archive member"
            if ts in p1_ts and radar == "77GHz":
                ev += "; Phase1 77ghz manifest also lists this unix_token"
            inv_rows.append(
                {
                    "radar_frequency": radar,
                    "acquisition_timestamp": ts,
                    "action": act,
                    "split": spl,
                    "raw_available": bool(raw),
                    "raw_format": raw_fmt,
                    "numerical_available": bool(num) or (radar == "77GHz" and ts in p1_cache_ts),
                    "numerical_format": num_fmt,
                    "png_available": bool(png),
                    "source_path": raw["member"] if raw else (png or ""),
                    "loader_available": loader,
                    "metadata_available": bool(raw) or radar == "77GHz",
                    "physical_axis_recoverable": "PARTIAL" if (raw or radar == "77GHz") else "NO",
                    "status": status,
                    "evidence": ev,
                    "archive_raw_member": raw["member"] if raw else "",
                    "archive_numerical_member": num["member"] if num else "",
                    "archive_raw_bytes": raw["file_size"] if raw else "",
                    "local_raw_openable": local_raw,
                    "local_png_path": png,
                }
            )
            map_rows.append(
                {
                    "timestamp": ts,
                    "action": act,
                    "radar": radar,
                    "archive_raw_member": raw["member"] if raw else "",
                    "archive_numerical_member": num["member"] if num else "",
                    "local_png": png,
                    "in_phase1_77_manifest": ts in p1_ts if radar == "77GHz" else "",
                    "in_phase1_airhar_cache": ts in p1_cache_ts if radar == "77GHz" else "",
                }
            )
            if raw:
                counts[radar]["archive_raw"] += 1
            if num:
                counts[radar]["archive_num"] += 1
            if png:
                counts[radar]["local_png"] += 1
            if radar == "77GHz" and ts in p1_ts:
                counts[radar]["p1_manifest"] += 1
            if radar == "77GHz" and ts in p1_cache_ts:
                counts[radar]["p1_cache"] += 1

    write_csv(OUT / "physical_data_inventory.csv", INV_FIELDS, inv_rows)
    write_csv(
        OUT / "raw_mapping.csv",
        [
            "timestamp",
            "action",
            "radar",
            "archive_raw_member",
            "archive_numerical_member",
            "local_png",
            "in_phase1_77_manifest",
            "in_phase1_airhar_cache",
        ],
        map_rows,
    )
    n_ts = len(triples)
    summary = {
        "n_paired_acquisitions": n_ts,
        "n_inventory_rows": len(inv_rows),
        "local_raw_data_zip": zip_here,
        "linux_01_zip": ZIP_01,
        "gcs": gcs,
        "per_radar": counts,
        "status_counts": {},
        "case": "C",
        "r2": "NO",
        "reason": "Archive members exist for 10/24 .dat and 77 Raw_0.bin (01 Linux zip inventory), but this host has no raw_data.zip. Mapping is established. R2 not built. Do not invert PNG.",
    }
    from collections import Counter

    summary["status_counts"] = dict(Counter(r["status"] for r in inv_rows))
    # CASE: all three have archive raw for (almost) all 327?
    if all(counts[r]["archive_raw"] >= n_ts * 0.95 for r in RADARS) and not zip_here:
        summary["case"] = "C"
        summary["case_note"] = "Files exist in archive; blocked by local extraction, not by missing mapping."
    (OUT / "audit_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
