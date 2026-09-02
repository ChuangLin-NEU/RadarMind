#!/usr/bin/env python3
"""Phase 3 one-shot: split → domain stats → train → eval → report. Does not overwrite Phase 1/2/v0.1."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [PY, str(ROOT / script)] + (extra or [])
    print("+", *cmd, flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run("scripts/shared_motion/export_split.py")
    run("scripts/shared_motion/domain_stats.py")
    run("scripts/shared_motion/train_shared_motion.py")
    run("scripts/shared_motion/evaluate.py")
    run("scripts/shared_motion/write_report.py")


if __name__ == "__main__":
    main()
