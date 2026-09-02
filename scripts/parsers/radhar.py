"""RadHAR: repo zip + models; voxels folder may be empty."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List

from .common import array_stats, try_import_numpy

DAPNET_RADHAR = (
    "competitors/dap_net_official/dataset/makecsv/RadHAR2csv.py:"
    "parse_mmwave_txt_frames_xyzdint splits raw .txt on '---', "
    "new frame when point_id==0, columns x,y,z,velocity,intensity. "
    "Directory layout: train/test action subfolders of .txt. "
    "Window 60 / stride 10 in that converter."
)
VOXEL_QUESTION = (
    "Official NESL RadHAR README (to be confirmed from extracted repo) "
    "typically treats voxels as a preprocessing product of point-cloud frames, "
    "not as the primary download. Phase 0 records whether voxel files exist on disk."
)


def extract_repo(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def inspect_tree(radhar_dir: Path, extracted_repo: Path) -> Dict[str, Any]:
    voxels = radhar_dir / "voxels"
    voxel_files = [p for p in voxels.rglob("*") if p.is_file()] if voxels.is_dir() else []
    models = [p.name for p in radhar_dir.iterdir() if p.is_file() and "model" in p.name.lower()]
    repo_files = [p for p in extracted_repo.rglob("*") if p.is_file()] if extracted_repo.exists() else []
    readmes = [str(p.relative_to(extracted_repo)) for p in repo_files if "readme" in p.name.lower()]
    py = [str(p.relative_to(extracted_repo)) for p in repo_files if p.suffix == ".py"]
    txt = [p for p in repo_files if p.suffix.lower() == ".txt"]
    npy = [p for p in repo_files if p.suffix.lower() in {".npy", ".npz"}]
    return {
        "voxel_dir_exists": voxels.is_dir(),
        "n_voxel_files": len(voxel_files),
        "voxel_files": [str(p.relative_to(radhar_dir)) for p in voxel_files[:20]],
        "models_in_root": models,
        "n_repo_files": len(repo_files),
        "readmes": readmes,
        "python_scripts": py[:40],
        "n_txt_in_repo": len(txt),
        "example_txt": [str(p.relative_to(extracted_repo)) for p in txt[:10]],
        "n_npy_in_repo": len(npy),
        "dapnet_loader_evidence": DAPNET_RADHAR,
        "voxel_note": VOXEL_QUESTION,
    }


def read_readme_and_code(extracted_repo: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {"readme_excerpts": [], "voxel_mentions": []}
    if not extracted_repo.exists():
        return out
    for p in extracted_repo.rglob("*"):
        if not p.is_file():
            continue
        name = p.name.lower()
        if name.startswith("readme") or p.suffix.lower() in {".md", ".rst"}:
            text = p.read_text(encoding="utf-8", errors="replace")
            out["readme_excerpts"].append(
                {"path": str(p.relative_to(extracted_repo)), "head": text[:2500]}
            )
        if p.suffix.lower() == ".py":
            text = p.read_text(encoding="utf-8", errors="replace")
            if "voxel" in text.lower():
                lines = [ln.strip() for ln in text.splitlines() if "voxel" in ln.lower()]
                out["voxel_mentions"].append(
                    {"path": str(p.relative_to(extracted_repo)), "lines": lines[:30]}
                )
    return out


def read_txt_samples(extracted_repo: Path, k: int = 4) -> List[Dict[str, Any]]:
    rows = []
    txts = [p for p in extracted_repo.rglob("*.txt") if p.is_file() and p.stat().st_size > 200]
    # prefer files that look like point clouds
    ranked = sorted(txts, key=lambda p: (0 if "train" in str(p).lower() or "data" in str(p).lower() else 1, p.stat().st_size))
    for p in ranked[:k]:
        text = p.read_text(encoding="utf-8", errors="replace")
        rec: Dict[str, Any] = {
            "path": str(p),
            "file_size": p.stat().st_size,
            "head": text[:600],
            "n_sep": text.count("---"),
        }
        if "point_id" in text.lower() or "point id" in text.lower():
            rec["looks_like_radhar_pointcloud"] = True
        rows.append(rec)
    return rows
