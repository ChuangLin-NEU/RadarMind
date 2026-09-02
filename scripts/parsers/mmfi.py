"""MM-Fi all_images.zip inventory and image-sample reads."""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    aggregate_path_tokens,
    extract_members,
    extract_small_meta,
    pick_diverse_members,
    zip_member_stats,
)

DAPNET_MMWAVE = (
    "competitors/dap_net_official/dataset/makecsv/MMFI2csv.py:_load_frames "
    "reads {root}/{scene}/{subject}/{action}/mmwave/frameXXX.bin as float64 reshape (-1,5). "
    "Labels come from MMFi_action_segments.csv columns Environment, Student, Action, Segments. "
    "This evidence is about the official mmwave point-cloud layout, not about all_images.zip."
)


def analyze_zip(zf: zipfile.ZipFile) -> Dict[str, Any]:
    stats = zip_member_stats(zf)
    names = [i.filename.replace("\\", "/") for i in zf.infolist() if not i.is_dir()]
    tokens = aggregate_path_tokens(names)
    img = [n for n in names if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    bins = [n for n in names if n.lower().endswith(".bin")]
    csvs = [n for n in names if n.lower().endswith(".csv")]
    return {
        "stats": stats,
        "path_tokens": tokens,
        "n_image": len(img),
        "n_bin": len(bins),
        "n_csv": len(csvs),
        "example_image": img[:8],
        "example_bin": bins[:8],
        "example_csv": csvs[:8],
        "official_mmwave_loader_evidence": DAPNET_MMWAVE,
        "note": (
            "Archive name is all_images.zip from anonymized_rgb_images. "
            "Whether mmwave bins exist is decided only by member list, not by dataset name."
        ),
    }


def sample_images(zf: zipfile.ZipFile, dest: Path, k: int = 8) -> List[Dict[str, Any]]:
    names = [i.filename.replace("\\", "/") for i in zf.infolist() if not i.is_dir()]
    img = [n for n in names if Path(n).suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}]
    chosen = pick_diverse_members(img, k)
    saved = extract_members(zf, chosen, dest)
    rows = []
    try:
        from PIL import Image  # noqa: WPS433
    except Exception:  # noqa: BLE001
        Image = None
    for src_name, path in zip(chosen, saved):
        p = Path(path)
        rec: Dict[str, Any] = {
            "zip_member": src_name,
            "extracted": str(p),
            "file_size": p.stat().st_size,
            "path_tokens": aggregate_path_tokens([src_name]),
        }
        if Image is None:
            rec["image"] = {"error": "PIL_not_available"}
        else:
            with Image.open(p) as im:
                rec["image"] = {
                    "size": list(im.size),
                    "mode": im.mode,
                    "format": im.format,
                    "bands": list(im.getbands()),
                }
        rows.append(rec)
    return rows
