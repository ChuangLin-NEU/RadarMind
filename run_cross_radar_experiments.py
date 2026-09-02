#!/usr/bin/env python3
"""Single command: audit → cache → train. Does not overwrite Phase 1."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str) -> None:
    cmd = [PY, str(ROOT / script)]
    print("+", *cmd, flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run("scripts/cross_radar/audit_ci4r.py")
    run("scripts/cross_radar/pack_cache.py")
    run("scripts/cross_radar/train_cross_radar.py")
    run("scripts/cross_radar/analyze_representation.py")


if __name__ == "__main__":
    main()
