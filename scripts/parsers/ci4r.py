"""CI4R raw_data.zip inventory and representative member reads."""
from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from .common import (
    aggregate_path_tokens,
    array_stats,
    extract_members,
    extract_small_meta,
    pick_diverse_members,
    try_import_numpy,
    zip_member_stats,
)

# Evidence from local official code (RadarBridge), not from guessing the zip.
AIRHAR_BIN2NP = (
    "third_party/AIRHAR/datasets/data_prepare_CI4R.py:bin2np "
    "reads .bin as int16, reshape (8,-1) Fortran, uses lane 4 as I and lane 8 as Q, "
    "then reshape (256, -1). RGB_mD does range FFT + STFT to a 224x224 micro-Doppler map. "
    "prepare_alabma_dataset labels by class-folder identifier; last 12 files per class are test."
)


def analyze_zip(zf: zipfile.ZipFile) -> Dict[str, Any]:
    stats = zip_member_stats(zf)
    names = [i.filename.replace("\\", "/") for i in zf.infolist() if not i.is_dir()]
    tokens = aggregate_path_tokens(names)
    bins = [n for n in names if n.lower().endswith(".bin")]
    mats = [n for n in names if n.lower().endswith(".mat")]
    pngs = [n for n in names if n.lower().endswith(".png")]
    return {
        "stats": stats,
        "path_tokens": tokens,
        "n_bin": len(bins),
        "n_mat": len(mats),
        "n_png": len(pngs),
        "example_bin": bins[:8],
        "example_mat": mats[:8],
        "example_png": pngs[:8],
        "airhar_loader_evidence": AIRHAR_BIN2NP,
    }


def sample_bins(zf: zipfile.ZipFile, dest: Path, k: int = 4) -> List[Dict[str, Any]]:
    np = try_import_numpy()
    names = [i.filename.replace("\\", "/") for i in zf.infolist() if not i.is_dir()]
    bins = [n for n in names if n.lower().endswith(".bin")]
    chosen = pick_diverse_members(bins, k)
    saved = extract_members(zf, chosen, dest)
    rows = []
    for src_name, path in zip(chosen, saved):
        p = Path(path)
        rec: Dict[str, Any] = {
            "zip_member": src_name,
            "extracted": str(p),
            "file_size": p.stat().st_size,
            "path_tokens": {k: v for k, v in aggregate_path_tokens([src_name]).items()},
        }
        if np is None:
            rec["read"] = {"error": "numpy_not_available"}
            rows.append(rec)
            continue
        raw = np.fromfile(str(p), dtype=np.int16)
        rec["int16_len"] = int(raw.size)
        rec["int16_stats"] = array_stats(raw)
        if raw.size >= 8 and raw.size % 8 == 0:
            lanes = raw.reshape((8, -1), order="F")
            iq = lanes[3, :].astype(np.complex64) + 1j * lanes[7, :].astype(np.complex64)
            rec["airhar_lane4_iq"] = {
                "n_complex": int(iq.size),
                "evidence": AIRHAR_BIN2NP,
            }
            if iq.size % 256 == 0:
                cube = iq.reshape((256, -1), order="F")
                rec["airhar_nts256"] = array_stats(np.abs(cube))
                rec["radar_representation_candidate"] = "complex IQ (AIRHAR bin2np on 77GHz .bin)"
            else:
                rec["airhar_nts256"] = "len not divisible by 256; AIRHAR layout not confirmed for this file"
        else:
            rec["airhar_layout"] = "int16 length not divisible by 8; AIRHAR 8-lane layout not confirmed"
        rows.append(rec)
    return rows
