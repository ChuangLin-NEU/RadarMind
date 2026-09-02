"""Shared archive audit helpers. Do not extract large zips here."""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import stat
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

INCOMPLETE_SUFFIXES = (".aria2", ".part", ".tmp", ".crdownload")
META_SUFFIXES = {
    ".md",
    ".txt",
    ".csv",
    ".json",
    ".yml",
    ".yaml",
    ".xml",
    ".rst",
    ".ini",
    ".cfg",
}
META_NAME_RE = re.compile(
    r"(readme|license|citation|metadata|label|annot|split|class|mapping|protocol)",
    re.I,
)
SMALL_META_BYTES = 2 * 1024 * 1024


def gb(n: int) -> float:
    return round(n / (1024 ** 3), 4)


def disk_free_bytes(path: Path) -> int:
    usage = shutil.disk_usage(str(path))
    return int(usage.free)


def find_incomplete_markers(directory: Path) -> List[str]:
    hits = []
    if not directory.is_dir():
        return hits
    for p in directory.rglob("*"):
        if p.is_file() and p.suffix.lower() in INCOMPLETE_SUFFIXES:
            hits.append(str(p))
    return hits


def open_zip(archive: Path) -> Tuple[Optional[zipfile.ZipFile], str]:
    try:
        zf = zipfile.ZipFile(archive, "r")
        # Force central-directory read.
        _ = zf.infolist()
        return zf, ""
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}"


def zip_member_stats(zf: zipfile.ZipFile) -> Dict[str, Any]:
    infos = zf.infolist()
    files = [i for i in infos if not i.is_dir()]
    compressed = sum(i.compress_size for i in files)
    uncompressed = sum(i.file_size for i in files)
    ext_count: Counter = Counter()
    ext_bytes: Counter = Counter()
    top: Counter = Counter()
    second: Counter = Counter()
    largest = None
    examples: Dict[str, str] = {}
    for info in files:
        name = info.filename.replace("\\", "/")
        parts = [p for p in name.split("/") if p]
        suffix = Path(name).suffix.lower() or "<no_ext>"
        ext_count[suffix] += 1
        ext_bytes[suffix] += info.file_size
        if parts:
            top[parts[0]] += 1
        if len(parts) >= 2:
            second[parts[1]] += 1
        if largest is None or info.file_size > largest[1]:
            largest = (name, info.file_size)
        if suffix not in examples:
            examples[suffix] = name
    depths = [len([p for p in i.filename.replace("\\", "/").split("/") if p]) for i in files]
    return {
        "n_members": len(infos),
        "n_files": len(files),
        "n_dirs": len(infos) - len(files),
        "compressed_bytes": compressed,
        "uncompressed_bytes": uncompressed,
        "ext_count": dict(ext_count.most_common()),
        "ext_bytes": {k: int(v) for k, v in ext_bytes.most_common()},
        "top_dirs": dict(top.most_common(40)),
        "second_dirs": dict(second.most_common(40)),
        "largest_file": largest[0] if largest else "",
        "largest_file_bytes": largest[1] if largest else 0,
        "example_by_ext": examples,
        "depth_min": min(depths) if depths else 0,
        "depth_max": max(depths) if depths else 0,
        "depth_median": sorted(depths)[len(depths) // 2] if depths else 0,
    }


def write_inventory_csv(zf: zipfile.ZipFile, dest: Path, max_rows: Optional[int] = None) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "member",
                "is_dir",
                "file_size",
                "compress_size",
                "compress_type",
                "date_time",
                "suffix",
                "depth",
                "top",
                "second",
            ]
        )
        for info in zf.infolist():
            name = info.filename.replace("\\", "/")
            parts = [p for p in name.split("/") if p]
            w.writerow(
                [
                    name,
                    int(info.is_dir()),
                    info.file_size,
                    info.compress_size,
                    info.compress_type,
                    "%04d-%02d-%02d %02d:%02d:%02d" % info.date_time,
                    Path(name).suffix.lower(),
                    len(parts),
                    parts[0] if parts else "",
                    parts[1] if len(parts) > 1 else "",
                ]
            )
            n += 1
            if max_rows is not None and n >= max_rows:
                break
    return n


def path_tokens(name: str) -> Dict[str, List[str]]:
    parts = [p for p in name.replace("\\", "/").split("/") if p]
    found: Dict[str, List[str]] = defaultdict(list)
    for p in parts:
        low = p.lower()
        if re.fullmatch(r"(train|test|val|valid|validation)", low):
            found["split"].append(p)
        if re.fullmatch(r"(s|sub|subject|student|p|person)\d{1,4}", low, re.I) or re.fullmatch(
            r"s\d{2,3}", low, re.I
        ):
            found["subject"].append(p)
        if re.fullmatch(r"(e|env|environment|scene)\d{1,3}", low, re.I):
            found["environment"].append(p)
        if re.fullmatch(r"a\d{1,3}", low, re.I):
            found["action"].append(p)
        if low in {
            "mmwave",
            "radar",
            "rgb",
            "image",
            "images",
            "depth",
            "infra",
            "ir",
            "wifi",
            "lidar",
            "skeleton",
            "voxels",
            "pointcloud",
            "point_cloud",
            "iq",
            "bin",
            "adc",
        }:
            found["modality"].append(p)
        if re.search(r"(session|seq|sequence|rep|trial)", low):
            found["session"].append(p)
    return {k: sorted(set(v)) for k, v in found.items()}


def aggregate_path_tokens(names: Iterable[str], limit: int = 200000) -> Dict[str, List[str]]:
    acc: Dict[str, set] = defaultdict(set)
    for i, name in enumerate(names):
        if i >= limit:
            break
        tok = path_tokens(name)
        for k, vals in tok.items():
            acc[k].update(vals)
    return {k: sorted(v) for k, v in acc.items()}


def is_small_meta(info: zipfile.ZipInfo) -> bool:
    if info.is_dir():
        return False
    if info.file_size > SMALL_META_BYTES:
        return False
    name = Path(info.filename).name
    suffix = Path(info.filename).suffix.lower()
    if suffix in META_SUFFIXES:
        return True
    if META_NAME_RE.search(name):
        return True
    return False


def extract_small_meta(zf: zipfile.ZipFile, dest: Path, max_files: int = 40) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for info in zf.infolist():
        if not is_small_meta(info):
            continue
        # skip huge trees of txt point clouds accidentally matching .txt
        if Path(info.filename).suffix.lower() == ".txt" and info.file_size > 256 * 1024:
            continue
        if not META_NAME_RE.search(Path(info.filename).name) and Path(info.filename).suffix.lower() == ".txt":
            continue
        target = dest / Path(info.filename).name
        if target.exists():
            target = dest / (Path(info.filename).as_posix().replace("/", "__"))
        with zf.open(info, "r") as src, target.open("wb") as out:
            out.write(src.read())
        saved.append(str(target))
        if len(saved) >= max_files:
            break
    return saved


def extract_members(zf: zipfile.ZipFile, names: Sequence[str], dest: Path) -> List[str]:
    dest.mkdir(parents=True, exist_ok=True)
    saved = []
    for name in names:
        info = zf.getinfo(name)
        out = dest / Path(name).name
        if out.exists():
            out = dest / Path(name).as_posix().replace("/", "__")
        with zf.open(info, "r") as src, out.open("wb") as fh:
            shutil.copyfileobj(src, fh, 1024 * 1024)
        saved.append(str(out))
    return saved


def try_import_numpy():
    try:
        import numpy as np  # noqa: WPS433

        return np
    except Exception:  # noqa: BLE001
        return None


def array_stats(arr) -> Dict[str, Any]:
    np = try_import_numpy()
    if np is None:
        return {"error": "numpy_not_available"}
    a = np.asarray(arr)
    finite = np.isfinite(a)
    out = {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "nbytes": int(a.nbytes),
        "nan_count": int(np.isnan(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
        "inf_count": int(np.isinf(a).sum()) if np.issubdtype(a.dtype, np.floating) else 0,
        "finite_count": int(finite.sum()),
    }
    if finite.any() and (np.issubdtype(a.dtype, np.number)):
        fa = a[finite] if a.dtype != object else a
        try:
            out["min"] = float(np.min(fa))
            out["max"] = float(np.max(fa))
            out["mean"] = float(np.mean(fa))
        except Exception as exc:  # noqa: BLE001
            out["minmax_error"] = str(exc)
    return out


def write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in fieldnames})


def pick_diverse_members(names: List[str], k: int) -> List[str]:
    if len(names) <= k:
        return list(names)
    # spread across the list; also keep first of several top-level dirs
    by_top: Dict[str, List[str]] = defaultdict(list)
    for n in names:
        top = n.replace("\\", "/").split("/", 1)[0]
        by_top[top].append(n)
    picked = []
    keys = list(by_top.keys())
    i = 0
    while len(picked) < k and keys:
        key = keys[i % len(keys)]
        bucket = by_top[key]
        if bucket:
            picked.append(bucket.pop(0))
        else:
            keys.remove(key)
            continue
        i += 1
    return picked[:k]


def file_magic(path: Path, n: int = 16) -> str:
    with path.open("rb") as f:
        return f.read(n).hex()
