"""FAU Radar_Data_of_Subjects: incomplete archive only. Do not extract."""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .common import find_incomplete_markers, gb


def audit_status(data_root: Path) -> Dict[str, Any]:
    d = data_root / "fau"
    archive = d / "Radar_Data_of_Subjects.zip"
    markers = find_incomplete_markers(d)
    size = archive.stat().st_size if archive.is_file() else 0
    return {
        "dataset": "FAU",
        "archive": archive.name if archive.is_file() else "",
        "archive_path": str(archive) if archive.is_file() else "",
        "archive_size_bytes": size,
        "archive_size_GB": gb(size),
        "readable": False,
        "incomplete_marker": ";".join(markers) if markers else "none",
        "estimated_uncompressed_GB": "UNKNOWN",
        "status": "INCOMPLETE",
        "notes": (
            "Phase 0 skip: .aria2 present and on-disk usage is far below the "
            "expected ~61G package. Zip is not opened or extracted."
        ),
        "dir": str(d),
    }
