"""mRI Dryad zip: extract (small), then inspect radar/label files."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from .common import array_stats, try_import_numpy, zip_member_stats

DAPNET_MRI = (
    "competitors/dap_net_official/dataset/makecsv/mRI2csv.py "
    "expects radar CSV at {dataset-root}/{subject}.csv and JSON labels under videolabels. "
    "JSON has subject + action segments. Points written as Frame,X,Y,Z,Doppler,Intensity."
)


def extract_all(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def inspect_extracted(root: Path) -> Dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext: Dict[str, int] = {}
    for p in files:
        by_ext[p.suffix.lower() or "<no_ext>"] = by_ext.get(p.suffix.lower() or "<no_ext>", 0) + 1
    readmes = [str(p.relative_to(root)) for p in files if "readme" in p.name.lower()]
    jsons = [p for p in files if p.suffix.lower() == ".json"]
    csvs = [p for p in files if p.suffix.lower() == ".csv"]
    mats = [p for p in files if p.suffix.lower() == ".mat"]
    npys = [p for p in files if p.suffix.lower() in {".npy", ".npz"}]
    return {
        "n_files": len(files),
        "by_ext": by_ext,
        "readmes": readmes[:20],
        "n_json": len(jsons),
        "n_csv": len(csvs),
        "n_mat": len(mats),
        "n_npy": len(npys),
        "example_json": [str(p.relative_to(root)) for p in jsons[:8]],
        "example_csv": [str(p.relative_to(root)) for p in csvs[:8]],
        "dapnet_loader_evidence": DAPNET_MRI,
    }


def read_samples(root: Path, dest: Path, k: int = 6) -> List[Dict[str, Any]]:
    dest.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    jsons = sorted(p for p in root.rglob("*.json"))[:k]
    for p in jsons:
        rec: Dict[str, Any] = {"path": str(p.relative_to(root)), "kind": "json"}
        try:
            obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
            rec["json_type"] = type(obj).__name__
            if isinstance(obj, dict):
                rec["keys"] = list(obj.keys())[:40]
                for key in ("subject", "action", "label", "labels", "segments"):
                    if key in obj:
                        rec[key] = obj[key] if not isinstance(obj[key], list) else obj[key][:8]
            elif isinstance(obj, list) and obj:
                rec["list_len"] = len(obj)
                rec["first"] = obj[0] if not isinstance(obj[0], (list, dict)) else str(obj[0])[:300]
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)
        rows.append(rec)

    np = try_import_numpy()
    csvs = sorted(p for p in root.rglob("*.csv"))[:k]
    for p in csvs:
        rec = {"path": str(p.relative_to(root)), "kind": "csv", "file_size": p.stat().st_size}
        text = p.read_text(encoding="utf-8", errors="replace").splitlines()
        rec["n_lines"] = len(text)
        rec["header"] = text[0][:400] if text else ""
        rec["line1"] = text[1][:400] if len(text) > 1 else ""
        if np is not None:
            try:
                arr = np.genfromtxt(str(p), delimiter=",", names=True, dtype=None, encoding="utf-8", max_rows=200)
                rec["preview"] = {
                    "dtype": str(arr.dtype),
                    "shape": list(arr.shape) if hasattr(arr, "shape") else [],
                    "names": list(arr.dtype.names) if getattr(arr.dtype, "names", None) else [],
                }
            except Exception as exc:  # noqa: BLE001
                rec["numpy_csv_error"] = str(exc)
        rows.append(rec)
    return rows
