#!/usr/bin/env python3
"""Word + Chinese PPT final delivery. No training. No Phase writes."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from final_delivery.load_metrics import D, load
from final_delivery.write_ppt import build_ppt
from final_delivery.write_word import build_word

OUT = ROOT / "results" / "final_delivery"
FIG = ROOT / "results" / "paper_package_v2" / "figures"
WORD = OUT / "RadarMind_Cross_Radar_HAR_Manuscript.docx"
PPT = OUT / "RadarMind_Project_Progress_CN.pptx"

PHASE_RE = re.compile(r"\bPhase\s+\d", re.I)
TOKEN_RE = re.compile(r"@@[A-Z0-9_]+@@")
LEAK = [
    r"we solve cross-radar",
    r"we achieve (radar )?invariance",
    r"universal representation",
    r"sensor-invariant representation achieved",
    r"geometry causes the residual",
    r"learn motion, not sensor achieved",
]
AI_PPT = ["赋能", "闭环", "一站式", "本研究旨在", "具有重要理论意义", "构建了", "形成闭环", "全方位"]


def _docx_text(path: Path) -> str:
    from docx import Document

    doc = Document(str(path))
    parts = [p.text for p in doc.paragraphs]
    for t in doc.tables:
        for row in t.rows:
            for c in row.cells:
                parts.append(c.text)
    return "\n".join(parts)


def _pptx_audit(path: Path) -> dict:
    from pptx import Presentation

    prs = Presentation(str(path))
    notes_ok = 0
    max_bullets = 0
    texts = []
    for sl in prs.slides:
        n = 0
        for sh in sl.shapes:
            if not sh.has_text_frame:
                continue
            for p in sh.text_frame.paragraphs:
                t = p.text.strip()
                if t:
                    texts.append(t)
                    if t.startswith(("•", "-", "·")) or (len(t) < 80 and "：" not in t[:2]):
                        n += 1
        # count non-title body paragraphs more loosely
        body = []
        for sh in sl.shapes:
            if sh.has_text_frame:
                for p in sh.text_frame.paragraphs:
                    tx = p.text.strip()
                    if tx and len(tx) < 80:
                        body.append(tx)
        max_bullets = max(max_bullets, len(body))
        nt = sl.notes_slide.notes_text_frame.text.strip()
        if nt:
            notes_ok += 1
    blob = "\n".join(texts)
    return {
        "slides": len(prs.slides),
        "notes": notes_ok,
        "max_short_lines": max_bullets,
        "text": blob,
        "has_not_yet": ("NOT YET" in blob) or ("NOT ACHIEVED" in blob),
        "ai": [w for w in AI_PPT if w in blob],
    }


def main() -> None:
    pack = load(ROOT)
    # freeze check: do not write phase dirs
    winfo = build_word(pack, FIG, WORD)
    pinfo = build_ppt(pack, FIG, PPT)

    wt = _docx_text(WORD)
    fails = []
    if not (180 <= winfo["abstract_words"] <= 220):
        fails.append(f"abstract words {winfo['abstract_words']}")
    if PHASE_RE.search(wt):
        fails.append("Phase number in Word")
    if TOKEN_RE.search(wt):
        fails.append("unreplaced token in Word")
    if re.search(r"\$[^$]+\$", wt):
        fails.append("raw $math$ in Word")
    if "RadarMind" in wt:
        fails.append("RadarMind appears in Word body")
    low = wt.lower()
    for pat in LEAK:
        if re.search(pat, low):
            fails.append(f"unsupported: {pat}")
    for h in (
        "1. Introduction",
        "2. Methods",
        "3. Results",
        "4. Discussion",
        "5. Conclusions",
        "6. Limitations",
        "References",
    ):
        if h not in wt:
            fails.append(f"missing {h}")
    if "Fig. 1." not in wt or "Fig. 6." not in wt:
        fails.append("figure captions")
    if "Table 1." not in wt or "Table 7." not in wt:
        fails.append("table captions")

    pa = _pptx_audit(PPT)
    if not (13 <= pa["slides"] <= 15):
        fails.append(f"slide count {pa['slides']}")
    if pa["notes"] != pa["slides"]:
        fails.append(f"notes {pa['notes']}/{pa['slides']}")
    if not pa["has_not_yet"]:
        fails.append("PPT missing NOT YET / NOT ACHIEVED")
    if pa["ai"]:
        fails.append("PPT AI jargon: " + ",".join(pa["ai"]))
    if "一定就是原因" in pa["text"] or "几何就是残差的原因" in pa["text"]:
        # allowed as a negation sentence; check we have 不能说
        if "不能说" not in pa["text"]:
            fails.append("geometry overclaim")

    # number trace: every 0.xxx in Word should exist in pack displays
    displays = set(pack["display"].values())
    displays |= {f"{float(v):.3f}" for v in pack["raw"].values() if isinstance(v, float)}
    orphan = []
    for num in set(re.findall(r"\b0\.\d{3}\b", wt)):
        if num not in displays:
            orphan.append(num)
    if orphan:
        fails.append("Word orphan numbers: " + ",".join(sorted(orphan)[:12]))

    word_audit = [
        "# Word content audit",
        "",
        f"- Abstract words: {winfo['abstract_words']}",
        f"- Phase labels: {'FAIL' if PHASE_RE.search(wt) else 'none'}",
        f"- Tokens leftover: {'FAIL' if TOKEN_RE.search(wt) else 'none'}",
        f"- RadarMind in body: {'FAIL' if 'RadarMind' in wt else 'none'}",
        f"- Unsupported claim patterns: {0 if not any(re.search(p, low) for p in LEAK) else 'FAIL'}",
        f"- Figures: Fig. 1–6 captions present",
        f"- Tables: Table 1–7 captions present",
        f"- Orphan 0.xxx displays: {orphan or 'none'}",
        f"- File: {WORD}",
    ]
    ppt_audit = [
        "# PPT content audit",
        "",
        f"- Slides: {pa['slides']}",
        f"- Speaker notes: {pa['notes']}/{pa['slides']}",
        f"- NOT YET / NOT ACHIEVED: {pa['has_not_yet']}",
        f"- AI project-book phrases: {pa['ai'] or 'none'}",
        f"- File: {PPT}",
    ]
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "word_content_audit.md").write_text("\n".join(word_audit) + "\n", encoding="utf-8")
    (OUT / "ppt_content_audit.md").write_text("\n".join(ppt_audit) + "\n", encoding="utf-8")
    if fails:
        raise SystemExit("DELIVERY FAIL:\n- " + "\n- ".join(fails))
    print("delivery PASS", winfo, pinfo, flush=True)


if __name__ == "__main__":
    main()
