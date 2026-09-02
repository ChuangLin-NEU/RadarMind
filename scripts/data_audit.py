#!/usr/bin/env python3
"""Phase 0 dataset audit. Inventory and sample schema only. No training, no full unzip of 80G packs."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, List

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from parsers import ci4r, fau, milipoint, mmfi, mri, radhar  # noqa: E402
from parsers.common import (  # noqa: E402
    disk_free_bytes,
    extract_small_meta,
    find_incomplete_markers,
    gb,
    open_zip,
    write_csv,
    write_inventory_csv,
    write_json,
    zip_member_stats,
)

STATUS_FIELDS = [
    "dataset",
    "archive",
    "archive_size_GB",
    "readable",
    "incomplete_marker",
    "estimated_uncompressed_GB",
    "status",
    "notes",
]

MANIFEST_FIELDS = [
    "dataset",
    "availability",
    "modality",
    "radar_representation",
    "archive/path",
    "subjects",
    "activities/classes",
    "sessions",
    "sequences",
    "frames",
    "sample_unit",
    "label_source",
    "file_format",
    "example_shape",
    "dtype",
    "official_split",
    "preprocessing_required",
    "estimated_processed_size",
    "evidence",
    "notes",
]


def u(v: Any) -> str:
    if v is None or v == "" or v == []:
        return "UNKNOWN"
    if isinstance(v, (list, tuple, set)):
        return ";".join(str(x) for x in v) if v else "UNKNOWN"
    return str(v)


def ensure_dirs(extract_dir: Path, work_dir: Path, out_dir: Path) -> None:
    extract_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)


def audit_one_zip(archive: Path, dataset: str, out_dir: Path, work_dir: Path, expected_status: str) -> Dict[str, Any]:
    markers = find_incomplete_markers(archive.parent)
    size = archive.stat().st_size if archive.is_file() else 0
    rec: Dict[str, Any] = {
        "dataset": dataset,
        "archive": archive.name if archive.is_file() else "",
        "archive_path": str(archive),
        "archive_size_bytes": size,
        "archive_size_GB": gb(size) if archive.is_file() else 0,
        "readable": False,
        "incomplete_marker": ";".join(markers) if markers else "none",
        "estimated_uncompressed_GB": "UNKNOWN",
        "status": expected_status,
        "notes": "",
        "n_files": "UNKNOWN",
        "zip_error": "",
    }
    if not archive.is_file():
        rec["status"] = "MISSING"
        rec["notes"] = "archive not found"
        return rec
    zf, err = open_zip(archive)
    if zf is None:
        rec["zip_error"] = err
        rec["readable"] = False
        rec["notes"] = f"zip open failed: {err}"
        if expected_status == "READY":
            rec["status"] = "UNREADABLE"
        return rec
    try:
        stats = zip_member_stats(zf)
        rec["readable"] = True
        rec["n_files"] = stats["n_files"]
        rec["n_dirs"] = stats["n_dirs"]
        rec["estimated_uncompressed_GB"] = gb(stats["uncompressed_bytes"])
        rec["compressed_GB"] = gb(stats["compressed_bytes"])
        rec["zip_stats"] = stats
        inv = out_dir / f"{dataset.replace('-', '')}_file_inventory.csv"
        # keep requested names
        name_map = {
            "CI4R": "CI4R_file_inventory.csv",
            "MM-Fi": "MMFi_file_inventory.csv",
            "mRI": "mRI_file_inventory.csv",
            "MiliPoint": "MiliPoint_file_inventory.csv",
            "RadHAR": "RadHAR_file_inventory.csv",
        }
        inv = out_dir / name_map.get(dataset, f"{dataset}_file_inventory.csv")
        n = write_inventory_csv(zf, inv)
        rec["inventory_rows"] = n
        rec["inventory_csv"] = str(inv)
        meta_dir = work_dir / "data_audit_samples" / dataset
        rec["extracted_meta"] = extract_small_meta(zf, meta_dir)
        rec["notes"] = (
            f"central directory OK; {stats['n_files']} files; "
            f"uncompressed {rec['estimated_uncompressed_GB']} GB; full CRC not scanned"
        )
    finally:
        zf.close()
    return rec


def run(data_root: Path, out_dir: Path, work_dir: Path, extract_dir: Path) -> Dict[str, Any]:
    ensure_dirs(extract_dir, work_dir, out_dir)
    facts: Dict[str, Any] = {
        "data_root": str(data_root),
        "disk_free_GB": gb(disk_free_bytes(data_root)),
        "datasets": {},
        "errors": [],
    }

    # ---- FAU: do not open ----
    try:
        facts["datasets"]["FAU"] = {"status_row": fau.audit_status(data_root)}
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "FAU", "error": str(exc), "tb": traceback.format_exc()})

    archives = {
        "CI4R": (data_root / "ci4r" / "raw_data.zip", "READY"),
        "MM-Fi": (data_root / "mmfi" / "all_images.zip", "READY"),
        "mRI": (data_root / "mri" / "doi_10_5061_dryad_9ghx3ffpp__v20231127.zip", "READY"),
        "MiliPoint": (data_root / "milipoint" / "MiliPoint_data.zip", "READY"),
    }
    radhar_zip = data_root / "radhar" / "RadHAR-master.zip"

    for name, (path, expected) in archives.items():
        try:
            print(f"[audit] {name} {path}", flush=True)
            rec = audit_one_zip(path, name, out_dir, work_dir, expected)
            facts["datasets"][name] = {"status_row": rec}
        except Exception as exc:  # noqa: BLE001
            facts["errors"].append({"dataset": name, "error": str(exc), "tb": traceback.format_exc()})
            print(f"[audit] FAIL {name}: {exc}", flush=True)

    # RadHAR archive + extra files
    try:
        print("[audit] RadHAR", flush=True)
        rec = audit_one_zip(radhar_zip, "RadHAR", out_dir, work_dir, "PARTIAL")
        extra = find_incomplete_markers(data_root / "radhar")
        rec["status"] = "PARTIAL"
        rec["notes"] = (rec.get("notes") or "") + "; voxels dir present but expected empty until inspected"
        if extra:
            rec["incomplete_marker"] = ";".join(extra)
        facts["datasets"]["RadHAR"] = {"status_row": rec}
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "RadHAR", "error": str(exc), "tb": traceback.format_exc()})

    # ---- CI4R samples from zip (no full extract) ----
    try:
        zf, err = open_zip(archives["CI4R"][0])
        if zf is None:
            facts["datasets"]["CI4R"]["analyze_error"] = err
        else:
            print("[audit] CI4R member analysis", flush=True)
            facts["datasets"]["CI4R"]["analyze"] = ci4r.analyze_zip(zf)
            dest = work_dir / "data_audit_samples" / "CI4R" / "bins"
            facts["datasets"]["CI4R"]["samples"] = ci4r.sample_bins(zf, dest, k=4)
            zf.close()
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "CI4R-sample", "error": str(exc), "tb": traceback.format_exc()})

    # ---- MM-Fi samples from zip ----
    try:
        zf, err = open_zip(archives["MM-Fi"][0])
        if zf is None:
            facts["datasets"]["MM-Fi"]["analyze_error"] = err
        else:
            print("[audit] MM-Fi member analysis", flush=True)
            facts["datasets"]["MM-Fi"]["analyze"] = mmfi.analyze_zip(zf)
            dest = work_dir / "data_audit_samples" / "MM-Fi" / "images"
            facts["datasets"]["MM-Fi"]["samples"] = mmfi.sample_images(zf, dest, k=8)
            zf.close()
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "MM-Fi-sample", "error": str(exc), "tb": traceback.format_exc()})

    # ---- small zips: extract to data_extracted, keep originals ----
    try:
        print("[audit] extract mRI", flush=True)
        dest = extract_dir / "mRI"
        if not dest.exists():
            mri.extract_all(archives["mRI"][0], dest)
        facts["datasets"]["mRI"]["extracted"] = str(dest)
        facts["datasets"]["mRI"]["inspect"] = mri.inspect_extracted(dest)
        facts["datasets"]["mRI"]["samples"] = mri.read_samples(dest, dest=work_dir / "data_audit_samples" / "mRI")
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "mRI-extract", "error": str(exc), "tb": traceback.format_exc()})

    try:
        print("[audit] extract MiliPoint", flush=True)
        dest = extract_dir / "MiliPoint"
        if not dest.exists():
            milipoint.extract_all(archives["MiliPoint"][0], dest)
        facts["datasets"]["MiliPoint"]["extracted"] = str(dest)
        facts["datasets"]["MiliPoint"]["inspect"] = milipoint.inspect_extracted(dest)
        facts["datasets"]["MiliPoint"]["samples"] = milipoint.read_samples(dest)
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "MiliPoint-extract", "error": str(exc), "tb": traceback.format_exc()})

    try:
        print("[audit] extract RadHAR repo", flush=True)
        dest = extract_dir / "RadHAR"
        if not dest.exists():
            radhar.extract_repo(radhar_zip, dest)
        tree = radhar.inspect_tree(data_root / "radhar", dest)
        facts["datasets"]["RadHAR"]["extracted"] = str(dest)
        facts["datasets"]["RadHAR"]["inspect"] = tree
        facts["datasets"]["RadHAR"]["code"] = radhar.read_readme_and_code(dest)
        facts["datasets"]["RadHAR"]["samples"] = radhar.read_txt_samples(dest)
        if tree.get("n_voxel_files", 0) == 0:
            facts["datasets"]["RadHAR"]["status_row"]["status"] = "PARTIAL"
            facts["datasets"]["RadHAR"]["status_row"]["notes"] += "; voxels has 0 files"
    except Exception as exc:  # noqa: BLE001
        facts["errors"].append({"dataset": "RadHAR-extract", "error": str(exc), "tb": traceback.format_exc()})

    write_json(out_dir / "audit_facts.json", facts)
    write_status_csv(out_dir, facts)
    write_manifest(out_dir, facts)
    write_report(out_dir, facts)
    return facts


def write_status_csv(out_dir: Path, facts: Dict[str, Any]) -> None:
    rows = []
    order = ["CI4R", "MM-Fi", "FAU", "mRI", "MiliPoint", "RadHAR"]
    for name in order:
        ds = facts["datasets"].get(name, {})
        row = ds.get("status_row", {"dataset": name, "status": "UNKNOWN", "notes": "no status_row"})
        rows.append(row)
    write_csv(out_dir / "download_status.csv", rows, STATUS_FIELDS)


def _ci4r_manifest(ds: Dict[str, Any]) -> Dict[str, str]:
    an = ds.get("analyze") or {}
    samples = ds.get("samples") or []
    st = (ds.get("status_row") or {}).get("status", "UNKNOWN")
    shape = "UNKNOWN"
    dtype = "UNKNOWN"
    rep = "UNKNOWN"
    for s in samples:
        if "airhar_nts256" in s and isinstance(s["airhar_nts256"], dict):
            shape = str(s["airhar_nts256"].get("shape"))
            dtype = str(s["airhar_nts256"].get("dtype"))
            rep = s.get("radar_representation_candidate", "UNKNOWN")
            break
        if s.get("radar_representation_candidate"):
            rep = s["radar_representation_candidate"]
    tokens = an.get("path_tokens") or {}
    return {
        "dataset": "CI4R",
        "availability": st,
        "modality": u(tokens.get("modality")),
        "radar_representation": rep if samples else "UNKNOWN",
        "archive/path": (ds.get("status_row") or {}).get("archive_path", "UNKNOWN"),
        "subjects": u(tokens.get("subject")),
        "activities/classes": u(tokens.get("action")),
        "sessions": u(tokens.get("session")),
        "sequences": "UNKNOWN",
        "frames": "UNKNOWN",
        "sample_unit": "one .bin recording if AIRHAR path applies; otherwise UNKNOWN",
        "label_source": "path / class folder in zip if present; AIRHAR uses folder identifier",
        "file_format": u(list((an.get("stats") or {}).get("ext_count", {}).keys())[:8]),
        "example_shape": shape,
        "dtype": dtype,
        "official_split": "AIRHAR last-12 per class as test (code); not verified inside this zip until paths seen",
        "preprocessing_required": "range FFT + STFT to micro-Doppler if using AIRHAR 77GHz .bin path",
        "estimated_processed_size": "UNKNOWN",
        "evidence": an.get("airhar_loader_evidence", "UNKNOWN"),
        "notes": (ds.get("status_row") or {}).get("notes", ""),
    }


def _mmfi_manifest(ds: Dict[str, Any]) -> Dict[str, str]:
    an = ds.get("analyze") or {}
    samples = ds.get("samples") or []
    st = (ds.get("status_row") or {}).get("status", "UNKNOWN")
    img = None
    for s in samples:
        if "image" in s and isinstance(s["image"], dict) and "size" in s["image"]:
            img = s
            break
    n_bin = an.get("n_bin", "UNKNOWN")
    n_img = an.get("n_image", "UNKNOWN")
    if isinstance(n_bin, int) and n_bin == 0 and isinstance(n_img, int) and n_img > 0:
        rep = "not radar in this archive; RGB/image files only (member list)"
        modality = "camera image (from suffix + PIL mode if read)"
    else:
        rep = "UNKNOWN"
        modality = u((an.get("path_tokens") or {}).get("modality"))
    return {
        "dataset": "MM-Fi",
        "availability": st,
        "modality": modality,
        "radar_representation": rep,
        "archive/path": (ds.get("status_row") or {}).get("archive_path", "UNKNOWN"),
        "subjects": u((an.get("path_tokens") or {}).get("subject")),
        "activities/classes": u((an.get("path_tokens") or {}).get("action")),
        "sessions": u((an.get("path_tokens") or {}).get("session")),
        "sequences": "UNKNOWN",
        "frames": n_img if n_img != "UNKNOWN" else "UNKNOWN",
        "sample_unit": "one image file in this zip; official mmwave sample is frame bin (not in this zip if n_bin=0)",
        "label_source": "path tokens and/or MMFi_action_segments.csv if present in zip",
        "file_format": u(list((an.get("stats") or {}).get("ext_count", {}).keys())[:8]),
        "example_shape": str((img or {}).get("image", {}).get("size", "UNKNOWN")),
        "dtype": str((img or {}).get("image", {}).get("mode", "UNKNOWN")),
        "official_split": "UNKNOWN in this archive",
        "preprocessing_required": "if only RGB: cannot do radar HAR from this zip alone",
        "estimated_processed_size": "UNKNOWN",
        "evidence": f"n_image={n_img}; n_bin={n_bin}; " + an.get("official_mmwave_loader_evidence", ""),
        "notes": (ds.get("status_row") or {}).get("notes", ""),
    }


def _small_manifest(name: str, ds: Dict[str, Any], extra: Dict[str, str]) -> Dict[str, str]:
    st = (ds.get("status_row") or {}).get("status", "UNKNOWN")
    ins = ds.get("inspect") or {}
    base = {
        "dataset": name,
        "availability": st,
        "modality": extra.get("modality", "UNKNOWN"),
        "radar_representation": extra.get("radar_representation", "UNKNOWN"),
        "archive/path": (ds.get("status_row") or {}).get("archive_path", "UNKNOWN"),
        "subjects": extra.get("subjects", "UNKNOWN"),
        "activities/classes": extra.get("activities/classes", "UNKNOWN"),
        "sessions": extra.get("sessions", "UNKNOWN"),
        "sequences": extra.get("sequences", "UNKNOWN"),
        "frames": extra.get("frames", "UNKNOWN"),
        "sample_unit": extra.get("sample_unit", "UNKNOWN"),
        "label_source": extra.get("label_source", "UNKNOWN"),
        "file_format": u(list((ins.get("by_ext") or {}).keys())[:10]),
        "example_shape": extra.get("example_shape", "UNKNOWN"),
        "dtype": extra.get("dtype", "UNKNOWN"),
        "official_split": extra.get("official_split", "UNKNOWN"),
        "preprocessing_required": extra.get("preprocessing_required", "UNKNOWN"),
        "estimated_processed_size": extra.get("estimated_processed_size", "UNKNOWN"),
        "evidence": extra.get("evidence", "UNKNOWN"),
        "notes": extra.get("notes", (ds.get("status_row") or {}).get("notes", "")),
    }
    return base


def infer_small_fields(name: str, ds: Dict[str, Any]) -> Dict[str, str]:
    samples = ds.get("samples") or []
    inspect = ds.get("inspect") or {}
    extra = {}
    if name == "mRI":
        extra["evidence"] = inspect.get("dapnet_loader_evidence", "UNKNOWN")
        extra["label_source"] = "JSON under extracted tree if present; DAP-Net uses videolabels/*.json"
        extra["sample_unit"] = "labeled segment window over radar CSV frames (DAP-Net); confirm against extracted files"
        extra["radar_representation"] = "UNKNOWN until a radar file is read as x,y,z,doppler,intensity"
        extra["preprocessing_required"] = "map JSON segments onto radar CSV; sliding window in DAP-Net"
        for s in samples:
            if s.get("kind") == "json" and "keys" in s:
                extra["notes"] = f"json keys={s.get('keys')}"
            if s.get("kind") == "csv" and s.get("header"):
                extra["label_source"] = extra["label_source"] + f"; csv header={s.get('header')[:120]}"
                hdr = (s.get("header") or "").lower()
                if "doppler" in hdr or "x" in hdr:
                    extra["radar_representation"] = "point cloud columns suggested by CSV header; confirm dtype/shape in samples"
    if name == "MiliPoint":
        extra["evidence"] = f"extracted examples={inspect.get('examples', [])[:8]}"
        extra["sample_unit"] = "UNKNOWN"
        extra["radar_representation"] = "UNKNOWN"
        for s in samples:
            if "array" in s:
                extra["example_shape"] = str(s["array"].get("shape"))
                extra["dtype"] = str(s["array"].get("dtype"))
            if "npz_keys" in s:
                extra["notes"] = f"npz_keys={s['npz_keys']}"
            if "mat_keys" in s:
                extra["notes"] = f"mat_keys={s['mat_keys']}"
    if name == "RadHAR":
        extra["evidence"] = inspect.get("dapnet_loader_evidence", "UNKNOWN")
        extra["radar_representation"] = (
            "point cloud .txt (x,y,z,velocity,intensity) if those files exist; "
            "voxels are intermediate if generated by repo scripts"
        )
        extra["sample_unit"] = "sliding window over frames in one .txt (DAP-Net window=60)"
        extra["label_source"] = "action subfolder name (DAP-Net); confirm in extracted repo"
        extra["official_split"] = "train/test folders if present in raw data; repo-only zip may lack them"
        extra["preprocessing_required"] = "parse txt frames; optional voxelize via official scripts"
        extra["notes"] = f"n_voxel_files={inspect.get('n_voxel_files')}; n_txt_in_repo={inspect.get('n_txt_in_repo')}"
        extra["frames"] = "UNKNOWN"
    if name == "FAU":
        extra["radar_representation"] = "UNKNOWN"
        extra["sample_unit"] = "UNKNOWN"
        extra["notes"] = "INCOMPLETE; not opened"
        extra["evidence"] = "aria2 sidecar + size << 61G"
    return extra


def write_manifest(out_dir: Path, facts: Dict[str, Any]) -> None:
    rows = []
    ds = facts["datasets"]
    if "CI4R" in ds:
        rows.append(_ci4r_manifest(ds["CI4R"]))
    if "MM-Fi" in ds:
        rows.append(_mmfi_manifest(ds["MM-Fi"]))
    if "FAU" in ds:
        rows.append(_small_manifest("FAU", ds["FAU"], infer_small_fields("FAU", ds["FAU"])))
    if "mRI" in ds:
        rows.append(_small_manifest("mRI", ds["mRI"], infer_small_fields("mRI", ds["mRI"])))
    if "MiliPoint" in ds:
        rows.append(_small_manifest("MiliPoint", ds["MiliPoint"], infer_small_fields("MiliPoint", ds["MiliPoint"])))
    if "RadHAR" in ds:
        rows.append(_small_manifest("RadHAR", ds["RadHAR"], infer_small_fields("RadHAR", ds["RadHAR"])))
    write_csv(out_dir / "dataset_manifest.csv", rows, MANIFEST_FIELDS)


def _block(title: str, lines: List[str]) -> str:
    return "## " + title + "\n\n" + "\n".join(lines) + "\n"


def write_report(out_dir: Path, facts: Dict[str, Any]) -> None:
    ds = facts["datasets"]
    lines: List[str] = []
    lines.append("# Dataset Audit Report (Phase 0)")
    lines.append("")
    lines.append(f"data_root: `{facts.get('data_root')}`")
    lines.append(f"disk_free_GB: {facts.get('disk_free_GB')}")
    lines.append("")
    lines.append("本报告只写磁盘、压缩包清单、抽样读取和仓库内官方 loader 对得上的内容。没读到的字段标 UNKNOWN。")
    lines.append("")

    # A
    lines.append("## A. Download status")
    lines.append("")
    lines.append("| dataset | status | archive | size_GB | readable | incomplete | uncompressed_GB | notes |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for name in ["CI4R", "MM-Fi", "FAU", "mRI", "MiliPoint", "RadHAR"]:
        row = (ds.get(name) or {}).get("status_row") or {}
        lines.append(
            "| {dataset} | {status} | {archive} | {archive_size_GB} | {readable} | {incomplete_marker} | {estimated_uncompressed_GB} | {notes} |".format(
                dataset=row.get("dataset", name),
                status=row.get("status", "UNKNOWN"),
                archive=row.get("archive", ""),
                archive_size_GB=row.get("archive_size_GB", ""),
                readable=row.get("readable", ""),
                incomplete_marker=str(row.get("incomplete_marker", ""))[:80],
                estimated_uncompressed_GB=row.get("estimated_uncompressed_GB", ""),
                notes=str(row.get("notes", "")).replace("|", "/")[:160],
            )
        )
    lines.append("")

    # B
    lines.append("## B. Dataset structure")
    lines.append("")
    for name in ["CI4R", "MM-Fi", "mRI", "MiliPoint", "RadHAR"]:
        block = ds.get(name) or {}
        lines.append(f"### {name}")
        an = block.get("analyze") or {}
        ins = block.get("inspect") or {}
        st = an.get("stats") or {}
        if st:
            lines.append(f"- zip files: {st.get('n_files')}  dirs: {st.get('n_dirs')}  depth {st.get('depth_min')}-{st.get('depth_max')}")
            lines.append(f"- top_dirs: `{json.dumps(st.get('top_dirs', {}), ensure_ascii=False)[:800]}`")
            lines.append(f"- ext_count: `{json.dumps(st.get('ext_count', {}), ensure_ascii=False)[:800]}`")
            lines.append(f"- largest: `{st.get('largest_file')}` ({st.get('largest_file_bytes')} bytes)")
            lines.append(f"- path_tokens: `{json.dumps(an.get('path_tokens', {}), ensure_ascii=False)[:800]}`")
        if ins:
            lines.append(f"- extracted files: {ins.get('n_files', ins.get('n_repo_files'))}")
            lines.append(f"- by_ext: `{json.dumps(ins.get('by_ext', {}), ensure_ascii=False)[:800]}`")
            if ins.get("readmes"):
                lines.append(f"- readme: {ins.get('readmes')}")
            if "n_voxel_files" in ins:
                lines.append(f"- voxel files: {ins.get('n_voxel_files')}")
                lines.append(f"- repo txt files: {ins.get('n_txt_in_repo')}")
                lines.append(f"- python scripts: {ins.get('python_scripts')}")
        if not st and not ins:
            lines.append("- UNKNOWN (analysis missing)")
        lines.append("")

    # C
    lines.append("## C. Radar modality")
    lines.append("")
    lines.append("只根据文件后缀、抽样数组和 loader 代码写。")
    lines.append("")
    ci = ds.get("CI4R") or {}
    lines.append("### CI4R")
    samples = ci.get("samples") or []
    if samples:
        lines.append(f"- sample records: {len(samples)}")
        for s in samples[:4]:
            lines.append(
                f"- member `{s.get('zip_member')}` size={s.get('file_size')} "
                f"int16_len={s.get('int16_len')} nts={s.get('airhar_nts256')} "
                f"rep={s.get('radar_representation_candidate', 'UNKNOWN')}"
            )
    else:
        lines.append("- UNKNOWN, no sample read")
    lines.append(f"- AIRHAR evidence: {(ci.get('analyze') or {}).get('airhar_loader_evidence', 'UNKNOWN')}")
    lines.append("")

    mm = ds.get("MM-Fi") or {}
    an = mm.get("analyze") or {}
    lines.append("### MM-Fi")
    lines.append(f"- n_image={an.get('n_image')} n_bin={an.get('n_bin')} n_csv={an.get('n_csv')}")
    if an.get("n_bin") == 0:
        lines.append("- this archive has 0 `.bin`. Official mmwave frames are therefore **not in this zip**.")
        lines.append("- files that were sampled are images; they are camera frames unless a radar suffix appears.")
    for s in (mm.get("samples") or [])[:5]:
        lines.append(f"- `{s.get('zip_member')}` image={s.get('image')}")
    lines.append(f"- official mmwave loader (not this zip): {an.get('official_mmwave_loader_evidence', '')}")
    lines.append("")

    for name in ["mRI", "MiliPoint", "RadHAR"]:
        lines.append(f"### {name}")
        block = ds.get(name) or {}
        if name == "RadHAR":
            ins = block.get("inspect") or {}
            lines.append(f"- voxels files: {ins.get('n_voxel_files')}")
            lines.append(f"- txt in repo: {ins.get('n_txt_in_repo')} examples={ins.get('example_txt')}")
            code = block.get("code") or {}
            for ex in (code.get("readme_excerpts") or [])[:3]:
                lines.append(f"- README `{ex.get('path')}`:")
                lines.append("```")
                lines.append((ex.get("head") or "")[:1800])
                lines.append("```")
            for vm in (code.get("voxel_mentions") or [])[:6]:
                lines.append(f"- voxel mention in `{vm.get('path')}`: {vm.get('lines')[:8]}")
        for s in (block.get("samples") or [])[:8]:
            lines.append(f"- sample `{s}`")
        lines.append("")

    lines.append("### FAU")
    lines.append("- not opened. representation UNKNOWN.")
    lines.append("")

    # D E F G H I J
    lines.append("## D. Sample definition")
    lines.append("")
    lines.append("- CI4R: one `.bin` clip if AIRHAR layout holds; label from class folder. Confirm from zip paths.")
    lines.append("- MM-Fi (this zip): one image file. Official radar sample is a mmwave `frameXXX.bin` sequence **if that modality is downloaded later**.")
    lines.append("- mRI: DAP-Net uses one JSON-labeled segment window over a subject radar CSV. Confirm on extracted files.")
    lines.append("- MiliPoint: see extracted file types in section B; if still unclear, UNKNOWN.")
    lines.append("- RadHAR: one action `.txt` is a sequence of point-cloud frames; train sample is a window. Voxels are not present as raw download.")
    lines.append("- FAU: UNKNOWN.")
    lines.append("")

    lines.append("## E. Label structure")
    lines.append("")
    lines.append("See path_tokens, JSON keys, CSV headers and official converters cited above. Fields not present on disk stay UNKNOWN.")
    lines.append("")

    lines.append("## F. Existing preprocessing")
    lines.append("")
    lines.append("- CI4R / AIRHAR: `bin2np` + `RGB_mD` (range FFT, sum range bins 23-31, STFT 4096, resize 224, per-spec z-score). Last 12 files/class = test.")
    lines.append("- MM-Fi / DAP-Net: read mmwave bin as float64 (N,5); segment CSV; **requires mmwave files not shown in all_images.zip if n_bin=0**.")
    lines.append("- mRI / DAP-Net: JSON segments + sliding window T=32 stride=16.")
    lines.append("- RadHAR / DAP-Net: parse txt frames, window 60 stride 10. Repo scripts may voxelize; do not run full generation in Phase 0.")
    lines.append("")

    lines.append("## G. Missing components")
    lines.append("")
    lines.append("- FAU zip incomplete (`.aria2`, ~292M occupied vs ~61G).")
    rad = (ds.get("RadHAR") or {}).get("inspect") or {}
    lines.append(f"- RadHAR voxels: n_files={rad.get('n_voxel_files', 'UNKNOWN')}.")
    lines.append(f"- RadHAR raw train/test txt count in repo zip: {rad.get('n_txt_in_repo', 'UNKNOWN')}. If this is only the GitHub repo, the official capture files may not be in this download.")
    mm_an = (ds.get("MM-Fi") or {}).get("analyze") or {}
    if mm_an.get("n_bin") == 0:
        lines.append("- MM-Fi mmwave / depth / wifi / IR not in `all_images.zip` (0 bin members).")
    if mm_an.get("n_csv") == 0:
        lines.append("- MM-Fi `MMFi_action_segments.csv` not found inside this zip.")
    lines.append("")

    lines.append("## H. Cross-dataset comparability")
    lines.append("")
    lines.append("- Same task (HAR) does not mean same tensor. Do not stack CI4R IQ/micro-Doppler with MM-Fi RGB or RadHAR points without an explicit representation conversion.")
    lines.append("- Point-cloud line (RadHAR / mRI / MM-Fi mmwave) can share a DAP-Net-style window **only after** those radar files exist.")
    lines.append("- CI4R raw is ADC/IQ style (if AIRHAR matches the sampled .bin). That is a different family from point clouds.")
    lines.append("- MM-Fi RGB-only cannot enter a radar-only benchmark.")
    lines.append("")

    lines.append("## I. Recommended preprocessing pipeline (next phase, not run now)")
    lines.append("")
    lines.append("1. CI4R: keep zip; extract 77GHz `.bin` class folders only; run AIRHAR `data_prepare_CI4R.py` to h5; keep official PNG path already in the laptop repo as a second representation.")
    lines.append("2. MM-Fi: do not treat `all_images.zip` as radar. Next download should be mmwave + segment CSV, at night free window, with a 05:55 kill.")
    lines.append("3. mRI: use extracted tree; pair JSON labels with radar CSV; do not invent a split.")
    lines.append("4. MiliPoint: after file types are listed, write a dedicated reader; no train loop.")
    lines.append("5. RadHAR: if repo has no capture txt, do not generate voxels from nothing. If txt exist, voxel script can be dry-run on 1 file.")
    lines.append("6. FAU: resume aria2 after 00:00; do not unzip the stub.")
    lines.append("")

    lines.append("## J. Recommended benchmark design")
    lines.append("")
    lines.append("- within-dataset HAR on CI4R 77GHz micro-Doppler (AIRHAR protocol) — data and loader both exist.")
    lines.append("- cross-subject on CI4R only after subject IDs are confirmed from filenames.")
    lines.append("- point-cloud HAR across RadHAR/mRI/MM-Fi — blocked until MM-Fi mmwave and RadHAR captures are present.")
    lines.append("- multimodal vs radar-only on MM-Fi — RGB is here; radar is not.")
    lines.append("- representation comparison (IQ vs spectrogram vs points) — only CI4R currently has a closed radar path.")
    lines.append("- no model is trained in Phase 0.")
    lines.append("")

    if facts.get("errors"):
        lines.append("## Errors during audit")
        lines.append("")
        for e in facts["errors"]:
            lines.append(f"- {e.get('dataset')}: {e.get('error')}")
        lines.append("")

    (out_dir / "dataset_audit_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", default=os.environ.get("RADARMIND_DATA_ROOT", ""))
    p.add_argument("--out-dir", default=os.environ.get("RADARMIND_DATA_AUDIT_OUT", str(HERE.parent / "results" / "data_audit")))
    p.add_argument("--work-dir", default=os.environ.get("RADARMIND_DATA_AUDIT_WORK", str(HERE.parent / "results" / "data_audit" / "work")))
    p.add_argument("--extract-dir", default=os.environ.get("RADARMIND_DATA_EXTRACTED", ""))
    args = p.parse_args()
    facts = run(Path(args.data_root), Path(args.out_dir), Path(args.work_dir), Path(args.extract_dir))
    print("disk_free_GB", facts.get("disk_free_GB"))
    print("errors", len(facts.get("errors") or []))
    for name, block in facts.get("datasets", {}).items():
        row = block.get("status_row") or {}
        print(name, row.get("status"), row.get("archive_size_GB"), "GB")


if __name__ == "__main__":
    main()
