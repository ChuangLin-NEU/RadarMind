#!/usr/bin/env python3
"""Reproduce tables/figures from frozen outputs. Does not train."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def run(title: str, args: list[str]) -> int:
    print("==>", title)
    return subprocess.call([sys.executable, *args], cwd=str(ROOT))


def main() -> int:
    steps = [
        ("validate metric registry", [str(ROOT / "scripts" / "release" / "validate_metric_registry.py")]),
        ("generate key results snippet", [str(ROOT / "scripts" / "release" / "generate_key_results.py")]),
        ("paper package v2", [str(ROOT / "scripts" / "write_paper_package_v2.py")]),
        ("rebuild paper figures", [str(ROOT / "scripts" / "rebuild_paper_figures.py")]),
        ("validate paper package", [str(ROOT / "validate_paper_package_v2.py")]),
    ]
    failed = []
    for title, args in steps:
        code = run(title, args)
        if code != 0:
            failed.append(title)
            print("FAILED", title, "exit", code)
    if failed:
        print("REPRODUCE FROM FROZEN FAIL:", ", ".join(failed))
        return 1
    print("REPRODUCE FROM FROZEN PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
