#!/usr/bin/env python3
"""Rebuild Fig. 2–11, rewrite tables, Word/PDF, visual QA."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from full_manuscript.narrative import table_specs
from full_manuscript.write_docs import validate, write_docx, write_markdown
from paper_figures.finish import main as finish_main
from paper_figures.io import load_snapshot
from paper_figures.rebuild import build_all
from paper_figures.visual_qa_v2 import (
    QA_DIR,
    audit_docx_tables,
    audit_source_colors,
    contact_sheet,
    decide,
    render_pdf_pages,
    word_to_pdf,
    write_report,
)
from paper_v2.snapshot import build_snapshot


def main() -> int:
    snap = load_snapshot()
    metas, book = build_all(snap)
    book.write_markdown(ROOT / "reports" / "figure_metric_provenance.md")
    manifest = {
        "n": len(metas),
        "figures": metas,
        "style": "src/paper/visual_style.py",
        "snapshot": "results/final_delivery/metric_snapshot_full.json",
        "script": "scripts/paper_figures/figures_v2.py",
    }
    (ROOT / "figures" / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (ROOT / "results" / "final_delivery" / "figure_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("figures", len(metas))

    out = ROOT / "results" / "final_delivery"
    fig_dir = ROOT / "figures" / "main"
    md_path = out / "RadarMind_Full_Manuscript.md"
    docx_path = out / "RadarMind_Full_Manuscript.docx"
    info = write_markdown(snap, md_path, fig_dir)
    write_docx(snap, docx_path, fig_dir)
    validate(info["md"], info["n_fig"], info["n_tab"], out, snap)
    print("manuscript", docx_path)

    finish_main()
    desk = Path.home() / "Desktop" / "RadarMind_Paper_Figures.docx"
    shutil.copy2(desk, out / "RadarMind_Paper_Figures.docx")

    QA_DIR.mkdir(parents=True, exist_ok=True)
    color_audit = audit_source_colors()
    table_audit = audit_docx_tables(docx_path)
    table_audit_desk = audit_docx_tables(desk)
    table_audit["vertical"] += table_audit_desk["vertical"]
    table_audit["full_grid"] += table_audit_desk["full_grid"]

    contact = contact_sheet(QA_DIR / "contact_fig02_11.png")
    pdf_ms = out / "RadarMind_Full_Manuscript.pdf"
    pdf_desk = Path.home() / "Desktop" / "RadarMind_Paper_Figures.pdf"
    ok_ms = word_to_pdf(docx_path, pdf_ms)
    ok_desk = word_to_pdf(desk, pdf_desk)
    print("pdf manuscript", ok_ms, pdf_ms if ok_ms else "FAIL")
    print("pdf desktop", ok_desk, pdf_desk if ok_desk else "FAIL")
    shots = []
    if ok_desk:
        shots = render_pdf_pages(pdf_desk, QA_DIR, "desk", max_pages=20)
    elif ok_ms:
        shots = render_pdf_pages(pdf_ms, QA_DIR, "ms", max_pages=30)

    answers = decide(color_audit, table_audit, figs_ok=True)
    answers["artifacts"] = {
        "manuscript_docx": str(docx_path).replace("\\", "/"),
        "manuscript_pdf": str(pdf_ms).replace("\\", "/") if ok_ms else "missing",
        "desktop_docx": str(desk).replace("\\", "/"),
        "desktop_pdf": str(pdf_desk).replace("\\", "/") if ok_desk else "missing",
        "contact_sheet": str(contact).replace("\\", "/"),
        "table_pages": ", ".join(str(p.name) for p in shots[:8]),
        "extra_hex": str(color_audit["extra_hex"]),
        "banned_tokens": str(color_audit["banned_tokens"]),
        "vertical_borders": table_audit["vertical"],
        "full_grid": table_audit["full_grid"],
    }
    report = ROOT / "reports" / "visual_qa_v2.md"
    write_report(report, answers)
    print("qa", answers["overall"], report)
    print("fail_keys", answers.get("_fail_keys"))
    return 0 if answers["overall"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
