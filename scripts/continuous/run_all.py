#!/usr/bin/env python3
"""Build Phase 4 artifacts. Does not overwrite Phase 1/2/3."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable


def run(rel: str) -> None:
    cmd = [PY, str(ROOT / rel)]
    print("+", *cmd, flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    run("scripts/continuous/build_gt.py")
    run("scripts/continuous/infer.py")
    run("scripts/continuous/metrics.py")
    run("scripts/continuous/pick_demos.py")
    run("scripts/continuous/figures.py")
    run("scripts/continuous/llm_summaries.py")
    run("scripts/continuous/cross_env.py")
    run("scripts/continuous/write_report.py")


if __name__ == "__main__":
    main()
