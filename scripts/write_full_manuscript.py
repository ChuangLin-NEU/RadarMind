#!/usr/bin/env python3
"""Generate RadarMind_Full_Manuscript.docx/md from frozen results. No training."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from full_manuscript.write_docs import write_docx, write_markdown, validate
from paper_v2.snapshot import build_snapshot

OUT = ROOT / "results" / "final_delivery"
FIG = ROOT / "figures" / "main"


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    snap = build_snapshot(ROOT)
    (OUT / "metric_snapshot_full.json").write_text(
        json.dumps(snap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path = OUT / "RadarMind_Full_Manuscript.md"
    info = write_markdown(snap, md_path, FIG)
    write_docx(snap, OUT / "RadarMind_Full_Manuscript.docx", FIG)
    rep = validate(info["md"], info["n_fig"], info["n_tab"], OUT, snap)
    print("FULL MANUSCRIPT")
    print("Status:", rep["status"])
    print("Words:", rep["body_words"])
    print("Abstract words:", rep["abs_words"])
    print("Figures:", rep["n_fig"])
    print("Tables:", rep["n_tab"])
    print("File:", OUT / "RadarMind_Full_Manuscript.docx")
    if rep["status"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
