"""MiliPoint_data.zip: extract (small) and inspect arrays / labels."""
from __future__ import annotations

import json
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from .common import array_stats, try_import_numpy


def extract_all(archive: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive, "r") as zf:
        zf.extractall(dest)


def inspect_extracted(root: Path) -> Dict[str, Any]:
    files = [p for p in root.rglob("*") if p.is_file()]
    by_ext: Dict[str, int] = {}
    for p in files:
        ext = p.suffix.lower() or "<no_ext>"
        by_ext[ext] = by_ext.get(ext, 0) + 1
    return {
        "n_files": len(files),
        "by_ext": by_ext,
        "readmes": [str(p.relative_to(root)) for p in files if "readme" in p.name.lower()][:20],
        "examples": [str(p.relative_to(root)) for p in files[:30]],
    }


def read_samples(root: Path, k: int = 8) -> List[Dict[str, Any]]:
    np = try_import_numpy()
    rows: List[Dict[str, Any]] = []
    candidates = []
    for ext in (".npy", ".npz", ".mat", ".pkl", ".pickle", ".h5", ".hdf5", ".csv", ".json", ".txt"):
        candidates.extend(sorted(root.rglob(f"*{ext}"))[:k])
    seen = set()
    for p in candidates:
        if p in seen:
            continue
        seen.add(p)
        if len(rows) >= k + 6:
            break
        rec: Dict[str, Any] = {
            "path": str(p.relative_to(root)),
            "suffix": p.suffix.lower(),
            "file_size": p.stat().st_size,
        }
        try:
            if p.suffix.lower() == ".npy" and np is not None:
                arr = np.load(str(p), allow_pickle=True)
                rec["array"] = array_stats(arr) if getattr(arr, "dtype", None) != object else {"object": True, "shape": list(getattr(arr, "shape", []))}
            elif p.suffix.lower() == ".npz" and np is not None:
                z = np.load(str(p), allow_pickle=True)
                rec["npz_keys"] = list(z.files)
                rec["arrays"] = {k: array_stats(z[k]) for k in list(z.files)[:6]}
            elif p.suffix.lower() == ".json":
                obj = json.loads(p.read_text(encoding="utf-8", errors="replace"))
                rec["json_type"] = type(obj).__name__
                rec["keys"] = list(obj.keys())[:40] if isinstance(obj, dict) else None
                rec["list_len"] = len(obj) if isinstance(obj, list) else None
            elif p.suffix.lower() == ".csv":
                lines = p.read_text(encoding="utf-8", errors="replace").splitlines()
                rec["n_lines"] = len(lines)
                rec["header"] = lines[0][:400] if lines else ""
            elif p.suffix.lower() == ".txt" and p.stat().st_size < 200000:
                rec["head"] = p.read_text(encoding="utf-8", errors="replace")[:800]
            elif p.suffix.lower() in {".pkl", ".pickle"}:
                rec["note"] = "pickle present; not unpickled in Phase 0"
            elif p.suffix.lower() in {".h5", ".hdf5"}:
                rec["note"] = "hdf5 present; keys not opened unless h5py available"
                try:
                    import h5py  # noqa: WPS433

                    with h5py.File(p, "r") as h:
                        rec["h5_keys"] = list(h.keys())[:40]
                except Exception as exc:  # noqa: BLE001
                    rec["h5_error"] = str(exc)
            elif p.suffix.lower() == ".mat":
                try:
                    import scipy.io as sio  # noqa: WPS433

                    m = sio.loadmat(str(p), squeeze_me=True, struct_as_record=False)
                    rec["mat_keys"] = [k for k in m.keys() if not k.startswith("__")][:30]
                except Exception as exc:  # noqa: BLE001
                    rec["mat_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001
            rec["error"] = str(exc)
        rows.append(rec)
    return rows
