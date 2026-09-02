"""Validate rebuilt figures and write the Desktop Word plate."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from full_manuscript.narrative import table_specs
from paper_figures.io import load_snapshot
from paper_figures.validate import run
from paper_figures.write_desktop_word import write_figure_word


def main() -> int:
    man = json.loads((ROOT / "figures" / "figure_manifest.json").read_text(encoding="utf-8"))
    ms = (ROOT / "results" / "final_delivery" / "RadarMind_Full_Manuscript.md").read_text(encoding="utf-8")
    snap = load_snapshot()
    tables = table_specs(snap)
    checks = run(man, ms)
    out = ROOT / "reports" / "figure_rebuild_validation.md"
    lines = ["# Figure rebuild validator", "", f"Overall: **{checks['overall']}**", ""]
    for k, v in checks.items():
        if k == "overall" or k.endswith("_detail") or k.endswith("_count"):
            continue
        lines.append(f"- {k}: {v}")
    if checks.get("hard_coded_detail"):
        lines.append(f"- hard-coded literals: {checks['hard_coded_detail']}")
    if checks.get("vector_exports_detail"):
        lines.append(f"- missing exports: {checks['vector_exports_detail']}")
    if checks.get("figure_references_detail"):
        lines.append(f"- references: {checks['figure_references_detail']}")
    if checks.get("protocol_labels_detail"):
        lines.append(f"- protocol flags: {checks['protocol_labels_detail']}")
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    desk = Path.home() / "Desktop" / "RadarMind_Paper_Figures.docx"
    write_figure_word(man, desk, tables=tables)
    print("validator", checks["overall"])
    print("desktop", desk)
    print("report", out)
    return 0 if checks["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
