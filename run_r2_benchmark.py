#!/usr/bin/env python3
"""Repo-root entry for Phase 8C."""
from pathlib import Path
import runpy

runpy.run_path(str(Path(__file__).resolve().parent / "scripts" / "phase8" / "run_r2_benchmark.py"), run_name="__main__")
