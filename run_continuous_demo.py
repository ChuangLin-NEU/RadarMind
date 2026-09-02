#!/usr/bin/env python3
"""One-shot: build Phase 4 artifacts if missing, then start RadarMind on :8502."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results" / "phase4_continuous"
PY = sys.executable


def main() -> int:
    need = [
        OUT / "mri_recording_manifest.csv",
        OUT / "continuous_runs.csv",
        OUT / "demo_registry.json",
        OUT / "phase4_continuous_report.md",
    ]
    if any(not p.is_file() for p in need):
        subprocess.check_call([PY, str(ROOT / "scripts" / "continuous" / "run_all.py")], cwd=str(ROOT))
    print("Continuous Motion page: http://localhost:8502", flush=True)
    return subprocess.call([PY, str(ROOT / "run_radarmind.py")], cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
