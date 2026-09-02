#!/usr/bin/env python3
"""Phase 8C: freeze R2 cache manifest + split audit. Does not regenerate R2."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"
CACHE = OUT / "cache_r2"
SPLIT = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"
RADARS = ["10GHz", "24GHz", "77GHz"]
MAN_FIELDS = [
    "timestamp",
    "radar",
    "action",
    "split",
    "shape",
    "dtype",
    "min",
    "max",
    "mean",
    "std",
    "checksum",
    "path",
]


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    mapping = read_csv(OUT / "raw_mapping.csv")
    split = {str(r["timestamp"]): r for r in read_csv(SPLIT)}
    by = defaultdict(dict)
    for r in mapping:
        by[str(r["timestamp"])][r["radar"]] = r

    rows = []
    missing = []
    bad = []
    for ts, bands in sorted(by.items()):
        sp = split.get(ts, {})
        for radar in RADARS:
            p = CACHE / radar / f"{ts}.npz"
            rec = {
                "timestamp": ts,
                "radar": radar,
                "action": bands.get(radar, {}).get("action") or sp.get("action", ""),
                "split": sp.get("split", ""),
                "path": str(p),
            }
            if not p.is_file():
                missing.append(f"{radar}/{ts}")
                rec.update({"shape": "", "dtype": "", "min": "", "max": "", "mean": "", "std": "", "checksum": "MISSING"})
                rows.append(rec)
                continue
            try:
                z = np.load(p)
                x = np.asarray(z["r2"])
                rec.update(
                    {
                        "shape": "x".join(str(i) for i in x.shape),
                        "dtype": str(x.dtype),
                        "min": float(x.min()),
                        "max": float(x.max()),
                        "mean": float(x.mean()),
                        "std": float(x.std()),
                        "checksum": sha256_file(p),
                    }
                )
                if x.shape != (1, 224, 224):
                    bad.append(f"{radar}/{ts} shape={x.shape}")
            except Exception as exc:
                bad.append(f"{radar}/{ts} {exc}")
                rec["checksum"] = f"UNREADABLE:{exc}"
            rows.append(rec)
    write_csv(OUT / "r2_cache_manifest.csv", MAN_FIELDS, rows)
    readable = sum(1 for r in rows if r.get("checksum") and not str(r["checksum"]).startswith(("MISSING", "UNREADABLE")))
    print(f"manifest {len(rows)} readable={readable} missing={len(missing)} bad={len(bad)}", flush=True)

    # split audit
    leaks, dups, conflicts, unpaired = [], [], [], []
    seen_pair = set()
    for ts, bands in sorted(by.items()):
        if len(bands) != 3:
            unpaired.append(ts)
        splits = {radar: split.get(ts, {}).get("split", "") for radar in RADARS}
        if len(set(splits.values())) != 1:
            leaks.append({"timestamp": ts, "splits": splits})
        acts = {radar: bands[radar]["action"] for radar in RADARS if radar in bands}
        if len(set(acts.values())) != 1:
            conflicts.append({"timestamp": ts, "actions": acts})
        for radar in bands:
            key = (ts, radar)
            if key in seen_pair:
                dups.append(key)
            seen_pair.add(key)
    n_split = defaultdict(int)
    for ts in by:
        n_split[split.get(ts, {}).get("split", "")] += 1
    lines = [
        "# R2 split audit",
        "",
        f"- paired acquisitions: {len(by)}",
        f"- samples: {len(rows)} (expect 981)",
        f"- readable: {readable}",
        f"- split counts (acquisition): {dict(n_split)}",
        f"- acquisition leakage: {len(leaks)}",
        f"- duplicate (timestamp,radar): {len(dups)}",
        f"- label conflict (same timestamp, different action): {len(conflicts)}",
        f"- unpaired (not 3 radars): {len(unpaired)}",
        f"- Phase 2B 133 conflicts: not in raw_mapping.csv / this 327 set",
        "",
    ]
    if leaks:
        lines.append("## leakage")
        lines.extend(str(x) for x in leaks)
    if dups:
        lines.append("## duplicates")
        lines.extend(str(x) for x in dups)
    if conflicts:
        lines.append("## label conflicts")
        lines.extend(str(x) for x in conflicts)
    if not leaks and not dups and not conflicts:
        lines.append("0 acquisition leakage, 0 duplicate leakage, 0 label conflict.")
    (OUT / "r2_split_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("audit written", flush=True)
    if readable != 981 or leaks or dups or conflicts:
        raise SystemExit("R2 cache/split audit failed")


if __name__ == "__main__":
    main()
