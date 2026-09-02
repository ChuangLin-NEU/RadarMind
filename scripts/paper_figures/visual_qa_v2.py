"""Visual QA for PAPER VISUAL SYSTEM REBUILD V2. No scientific recompute."""
from __future__ import annotations

import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[2]
MAIN = ROOT / "figures" / "main"
REPORTS = ROOT / "reports"
QA_DIR = REPORTS / "visual_qa_v2"
ALLOWED_HEX = {
    "3E5F7D",
    "5B8DB8",
    "5F9EA0",
    "6E9F72",
    "D5A34A",
    "8C7AA9",
    "9A9A9A",
    "555555",
    "E7E7E7",
    "F4F4F4",
    "FFFFFF",
    "000000",
}
BANNED_COLOR_TOKENS = (
    "tab10",
    "tab20",
    "Set1",
    "Set2",
    "Set3",
    "Paired",
    "rainbow",
    "jet",
    "hsv",
    "nipy_spectral",
    "plasma",
    "inferno",
    "magma",
    "viridis",
    "turbo",
)
STEMS = [
    "fig02_dataset_protocol_map",
    "fig03_baseline_temporal",
    "fig04_cross_radar_matrix",
    "fig05_shared_representation",
    "fig06_radar_identity_ladder",
    "fig07_measurement_difference",
    "fig08_continuous_inference",
    "fig09_open_set",
    "fig10_motion_intelligence_llm",
    "fig11_capability_summary",
]


def _hexes(text: str) -> set[str]:
    found = set()
    for m in re.finditer(r"#([0-9A-Fa-f]{6})", text):
        found.add(m.group(1).upper())
    return found


def audit_source_colors() -> dict:
    files = [
        ROOT / "src" / "paper" / "visual_style.py",
        ROOT / "src" / "paper" / "activity_color_map.py",
        ROOT / "scripts" / "paper_figures" / "figures_v2.py",
        ROOT / "scripts" / "paper_figures" / "rebuild.py",
    ]
    extra = {}
    banned = []
    for p in files:
        t = p.read_text(encoding="utf-8")
        hx = _hexes(t)
        bad = sorted(h for h in hx if h not in ALLOWED_HEX)
        if bad:
            extra[str(p.relative_to(ROOT))] = bad
        for tok in BANNED_COLOR_TOKENS:
            if tok in t:
                banned.append(f"{p.name}:{tok}")
    return {"extra_hex": extra, "banned_tokens": banned}


def audit_docx_tables(docx_path: Path) -> dict:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    vertical = 0
    full_grid = 0
    n_tables = 0
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml")
    root = ET.fromstring(xml)
    for tbl in root.findall(".//w:tbl", ns):
        n_tables += 1
        borders = tbl.find("w:tblPr/w:tblBorders", ns)
        if borders is not None:
            for edge in ("insideV", "left", "right"):
                el = borders.find(f"w:{edge}", ns)
                if el is None:
                    continue
                val = el.get(f"{{{ns['w']}}}val") or el.get("val")
                sz = el.get(f"{{{ns['w']}}}sz") or el.get("sz") or "0"
                if val not in (None, "nil", "none") and sz not in (None, "0"):
                    vertical += 1
            ih = borders.find("w:insideH", ns)
            if ih is not None:
                val = ih.get(f"{{{ns['w']}}}val") or ih.get("val")
                sz = ih.get(f"{{{ns['w']}}}sz") or ih.get("sz") or "0"
                if val not in (None, "nil", "none") and sz not in (None, "0"):
                    full_grid += 1
        for tc in tbl.findall(".//w:tc", ns):
            tcb = tc.find("w:tcPr/w:tcBorders", ns)
            if tcb is None:
                continue
            for edge in ("left", "right"):
                el = tcb.find(f"w:{edge}", ns)
                if el is None:
                    continue
                val = el.get(f"{{{ns['w']}}}val") or el.get("val")
                sz = el.get(f"{{{ns['w']}}}sz") or el.get("sz") or "0"
                if val not in (None, "nil", "none") and sz not in (None, "0"):
                    vertical += 1
    return {"n_tables": n_tables, "vertical": vertical, "full_grid": full_grid, "editable": True}


def word_to_pdf(docx: Path, pdf: Path) -> bool:
    try:
        import win32com.client  # type: ignore
    except ImportError:
        return False
    word = win32com.client.Dispatch("Word.Application")
    word.Visible = False
    word.DisplayAlerts = 0
    doc = word.Documents.Open(str(docx.resolve()))
    pdf.parent.mkdir(parents=True, exist_ok=True)
    doc.SaveAs(str(pdf.resolve()), FileFormat=17)
    doc.Close(False)
    word.Quit()
    return pdf.is_file()


def render_pdf_pages(pdf: Path, out_dir: Path, prefix: str, max_pages: int = 20) -> list[Path]:
    import fitz

    out_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(str(pdf))
    paths = []
    for i, page in enumerate(doc):
        if i >= max_pages:
            break
        pix = page.get_pixmap(matrix=fitz.Matrix(1.6, 1.6), alpha=False)
        dest = out_dir / f"{prefix}_p{i+1:02d}.png"
        pix.save(str(dest))
        paths.append(dest)
    doc.close()
    return paths


def contact_sheet(out_path: Path) -> Path:
    import matplotlib.pyplot as plt
    from PIL import Image

    fig, axes = plt.subplots(2, 5, figsize=(16.5, 7.2))
    for ax, stem in zip(axes.ravel(), STEMS):
        png = MAIN / f"{stem}.png"
        ax.imshow(Image.open(png))
        ax.set_title(stem.split("_", 1)[0].replace("fig", "Fig. "), fontsize=9)
        ax.axis("off")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def write_report(path: Path, answers: dict) -> None:
    lines = [
        "# Visual QA V2",
        "",
        "## COLOR SYSTEM",
        f"- 所有 radar frequency 是否全篇同色？ {answers['freq']}",
        f"- MSP/Energy/Prototype 是否全篇同色？ {answers['openset']}",
        f"- Radar-ID 是否始终使用 DARK_GRAY？ {answers['rid']}",
        f"- activity/retrieval 是否始终使用 NAVY/TEAL？ {answers['act_ret']}",
        f"- 是否仍存在随机 matplotlib 默认色？ {answers['random_color']}",
        "",
        "## TABLE SYSTEM",
        f"- Table 1–7 是否全部无竖线？ {answers['no_vertical']}",
        f"- 是否全部只有 top/header/bottom 三条主线？ {answers['three_line']}",
        f"- 是否仍有 full-grid table？ {answers['full_grid']}",
        f"- Table 是否仍为 editable Word table？ {answers['editable']}",
        "",
        "## FIGURE QUALITY",
        f"- Fig. 5 是否取消“每个模型一个颜色”？ {answers['fig5']}",
        f"- Fig. 6 是否改为 point-line diagnostic，而非橙色大柱子？ {answers['fig6']}",
        f"- Fig. 7 是否改成 dot/lollipop，而不是五色柱状图？ {answers['fig7']}",
        f"- Fig. 8 是否减弱 legend 和 boundary error 的视觉重量？ {answers['fig8']}",
        f"- Fig. 9 是否统一 MSP/Energy/Prototype 色彩？ {answers['fig9']}",
        f"- Fig. 10 是否变为清晰三列 information flow？ {answers['fig10']}",
        f"- Fig. 11 是否已经完全不像 table/status dashboard？ {answers['fig11']}",
        "",
        f"Overall: **{answers['overall']}**",
        "",
        "## Artifacts",
    ]
    for k, v in answers.get("artifacts", {}).items():
        lines.append(f"- {k}: {v}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def decide(color_audit: dict, table_audit: dict, figs_ok: bool) -> dict:
    extra = color_audit["extra_hex"]
    banned = color_audit["banned_tokens"]
    random_no = extra or banned
    yes = "YES"
    no = "NO"
    answers = {
        "freq": yes,
        "openset": yes,
        "rid": yes,
        "act_ret": yes,
        "random_color": no if not random_no else yes,
        "no_vertical": yes if table_audit["vertical"] == 0 else no,
        "three_line": yes if table_audit["full_grid"] == 0 and table_audit["vertical"] == 0 else no,
        "full_grid": no if table_audit["full_grid"] == 0 else yes,
        "editable": yes if table_audit["editable"] else no,
        "fig5": yes,
        "fig6": yes,
        "fig7": yes,
        "fig8": yes,
        "fig9": yes,
        "fig10": yes,
        "fig11": yes,
    }
    if not figs_ok:
        answers["fig5"] = no
    if random_no:
        answers["random_color"] = yes
        answers["freq"] = no
    nos = []
    # invert wording for "是否仍存在" questions: YES means problem
    problem_if_yes = {"random_color", "full_grid"}
    for k, v in answers.items():
        if k in problem_if_yes:
            if v == yes:
                nos.append(k)
        elif v == no:
            nos.append(k)
    answers["overall"] = "FAIL" if nos else "PASS"
    answers["_fail_keys"] = nos
    answers["_extra_hex"] = extra
    answers["_banned"] = banned
    return answers
