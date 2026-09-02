#!/usr/bin/env python3
"""PAPER FIGURE REBUILD: audit is already written; generate Fig. 1–11 and validate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from paper_figures.io import load_snapshot
from paper_figures.rebuild import CAPTIONS, build_all
from paper_figures.validate import run as validate
from paper_figures.write_desktop_word import write_figure_word


def main() -> int:
    snap = load_snapshot()
    metas, book = build_all(snap)
    book.write_markdown(ROOT / "reports" / "figure_metric_provenance.md")
    manifest = {
        "n": len(metas),
        "figures": metas,
        "style": "src/paper/visual_style.py",
        "snapshot": "results/final_delivery/metric_snapshot_full.json",
        "script": "scripts/paper_figures/rebuild.py",
    }
    man_path = ROOT / "figures" / "figure_manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "results" / "final_delivery" / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("generated", len(metas), "figures")
    for m in metas:
        print(" ", m["figure"], m["stem"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
