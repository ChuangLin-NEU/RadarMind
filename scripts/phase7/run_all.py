#!/usr/bin/env python3
"""Phase 7 driver. Serial. Resume via per-run metrics.json."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PY = sys.executable
OUT = ROOT / "results" / "phase7_radar_invariance"


def run(script: str, extra: list[str] | None = None) -> None:
    cmd = [PY, str(ROOT / "scripts" / "phase7" / script)] + (extra or [])
    print("+", " ".join(cmd), flush=True)
    subprocess.check_call(cmd, cwd=str(ROOT))


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    run("build_r1_cache.py")
    run("domain_stats.py")
    run("forensic_24ghz.py")
    run("train_r1_baseline.py")
    run("compare_fairness.py")
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8"))
    if gate.get("enter_part_d"):
        run("train_m3.py", ["--stage", "all"])
        run("evaluate.py")
    else:
        print("[gate] skip M3:", gate.get("reason"), flush=True)
        (OUT / "M3_SKIPPED.txt").write_text(gate.get("reason", "gate failed") + "\n", encoding="utf-8")
    run("figures.py")
    run("write_report.py")
    print("[phase7] done", flush=True)


if __name__ == "__main__":
    main()
