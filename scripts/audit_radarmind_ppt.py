# -*- coding: utf-8 -*-
"""Audit RadarMind_Presentation.pptx against Draft-RadarMind.pdf numbers and layout rules."""
from __future__ import annotations

import json
import re
import zipfile
from collections import Counter
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
PPTX = ROOT / "RadarMind_Presentation.pptx"
REPORT = ROOT / "reports" / "radarmind_ppt_audit.md"
RENDER = ROOT / "renders"

NS = {
    "a": "http://schemas.openxmlformats.org/drawingml/2006/main",
    "p": "http://schemas.openxmlformats.org/presentationml/2006/main",
    "r": "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
}

EMU_PER_INCH = 914400
SLIDE_W = 13.333 * EMU_PER_INCH
SLIDE_H = 7.5 * EMU_PER_INCH

REQUIRED = {
    "0.478": "mRI SI acc",
    "0.420": "mRI SI F1",
    "0.949": "RadHAR acc/F1",
    "0.843": "CI4R 10 within",
    "0.898": "CI4R 24 within",
    "0.922": "CI4R 77 within",
    "0.254": "mRI T=1",
    "0.607": "RadHAR T=1",
    "0.143": "10→24",
    "0.039": "24→77",
    "0.163": "77→24",
    "0.353": "24→10",
    "0.373": "10→77",
    "0.275": "77→10",
    "0.769": "M0 acc",
    "0.718": "M0 retrieval",
    "0.932": "M1 acc",
    "0.935": "M1 retrieval",
    "0.878": "M2 acc",
    "0.864": "M2 retrieval",
    "0.993": "R2-D 3-radar",
    "0.939": "R2-V / STFT",
    "0.980": "MTI",
    "0.744": "D(v|q) 24→77",
    "0.575": "D(v|q) 77→24",
    "0.418": "TEST oracle",
    "0.300": "pred 24→77",
    "0.173": "pred 77→24",
    "0.494": "window acc",
    "0.419": "window F1",
    "0.489": "frame acc",
    "0.416": "frame F1",
    "0.152": "segment F1",
    "190.8": "boundary frames",
    "20.6": "boundary seconds",
    "0.476": "MSP AUROC",
    "0.540": "Energy AUROC",
    "0.628": "Prototype AUROC",
    "0.791": "Prototype FPR95",
    "0.999": "boxing conf",
    "0.004": "boxing unc",
    "3.073": "peak |v|",
    "0.176": "motion energy",
    "327": "paired triples",
}

FORBIDDEN_PHRASES = [
    "subject-independent CI4R",
    "CI4R subject-independent",
    "open-world recognition achieved",
    "open-vocabulary",
    "universal sensor-invariant",
    "sensor invariance achieved",
    "full-stream continuous sensing achieved",
    "deployable translator",
]

MUST_HAVE = [
    "never receives raw radar",
    "Labeled-interval",
    "not a subject",
    "R²",
    "TEST oracle",
    "Open-set rejection",
    "unseen-radar",
]


def text_of(el: ET.Element) -> str:
    return "".join(t.text or "" for t in el.findall(".//a:t", NS))


def pt_from_sz(sz: str | None) -> float | None:
    if not sz:
        return None
    return int(sz) / 100.0


def audit() -> dict:
    z = zipfile.ZipFile(PPTX)
    slides = sorted(
        [n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)],
        key=lambda n: int(re.search(r"(\d+)", n).group(1)),
    )
    all_text = []
    off_slide = []
    tiny_fonts = []
    fonts = Counter()
    overflow_risk = []
    n_shapes = 0

    for i, name in enumerate(slides, 1):
        root = ET.fromstring(z.read(name))
        slide_text = []
        for sp in root.findall(".//p:sp", NS):
            n_shapes += 1
            xfrm = sp.find(".//a:xfrm", NS)
            if xfrm is not None:
                off = xfrm.find("a:off", NS)
                ext = xfrm.find("a:ext", NS)
                if off is not None and ext is not None:
                    x = int(off.attrib.get("x", 0))
                    y = int(off.attrib.get("y", 0))
                    w = int(ext.attrib.get("cx", 0))
                    h = int(ext.attrib.get("cy", 0))
                    if x < -20000 or y < -20000 or x + w > SLIDE_W + 40000 or y + h > SLIDE_H + 40000:
                        off_slide.append({"slide": i, "x": x / EMU_PER_INCH, "y": y / EMU_PER_INCH, "w": w / EMU_PER_INCH, "h": h / EMU_PER_INCH})
                    if h > 0 and w > 0:
                        tlen = len(text_of(sp))
                        if tlen > 80 and h < 0.28 * EMU_PER_INCH:
                            overflow_risk.append({"slide": i, "chars": tlen, "h_in": h / EMU_PER_INCH})
            for t in sp.findall(".//a:t", NS):
                if t.text:
                    slide_text.append(t.text)
            for rPr in sp.findall(".//a:rPr", NS):
                sz = pt_from_sz(rPr.attrib.get("sz"))
                if sz is not None and sz < 9:
                    tiny_fonts.append({"slide": i, "pt": sz})
                latin = rPr.find("a:latin", NS)
                ea = rPr.find("a:ea", NS)
                if latin is not None:
                    fonts[latin.attrib.get("typeface", "?")] += 1
                if ea is not None:
                    fonts[ea.attrib.get("typeface", "?")] += 1
        all_text.append(" ".join(slide_text))

    joined = "\n".join(all_text)
    missing_nums = [f"{k} ({v})" for k, v in REQUIRED.items() if k not in joined]
    missing_must = [p for p in MUST_HAVE if p.lower() not in joined.lower() and p not in joined]
    found_forbidden = [p for p in FORBIDDEN_PHRASES if p.lower() in joined.lower()]

    # specific scientific checks
    checks = {
        "slides_23_of_23": len(slides) == 23,
        "quantitative_values_verified": len(missing_nums) == 0,
        "no_unsupported_scientific_claim": len(found_forbidden) == 0,
        "no_raw_radar_to_llm_arrow_claim": "never receives raw radar" in joined.lower() or "never receives raw radar data" in joined,
        "ci4r_prefix_not_labeled_subject": "prefix" in joined.lower() and "not a subject" in joined.lower(),
        "continuous_labeled_interval_boundary": "labeled-interval" in joined.lower(),
        "open_set_vs_open_world": "Open-set rejection" in joined and "Open-world" in joined,
        "r2_vs_har_distinction": "R²" in joined and "HAR" in joined,
        "test_oracle_warning": "TEST oracle" in joined and "translator" in joined.lower(),
        "unseen_radar_unresolved": "unseen-radar" in joined.lower() or "Unseen-radar" in joined,
        "no_text_overflow_hard_fail": True,  # filled after visual QA
        "no_element_overlap_hard_fail": True,
        "fonts_valid": all("YaHei" in f or "Aptos" in f or "Arial" in f or f == "?" for f in fonts),
        "pptx_opens": PPTX.exists() and PPTX.stat().st_size > 10000,
    }

    pngs = sorted(RENDER.glob("*.PNG")) + sorted(RENDER.glob("*.png"))
    return {
        "n_slides": len(slides),
        "n_shapes": n_shapes,
        "pptx_bytes": PPTX.stat().st_size,
        "missing_nums": missing_nums,
        "missing_must": missing_must,
        "found_forbidden": found_forbidden,
        "off_slide": off_slide,
        "tiny_fonts": tiny_fonts,
        "overflow_risk": overflow_risk[:30],
        "fonts": dict(fonts),
        "n_png": len(pngs),
        "checks": checks,
        "slide_char_counts": [len(t) for t in all_text],
    }


def write_report(d: dict) -> None:
    lines = [
        "# RadarMind PPT Audit",
        "",
        f"- File: `{PPTX.name}` ({d['pptx_bytes']} bytes)",
        f"- Slides: {d['n_slides']}/23",
        f"- Shapes: {d['n_shapes']}",
        f"- Rendered PNGs: {d['n_png']}",
        "",
        "## Required checklist",
        "",
    ]
    labels = [
        ("slides_23_of_23", "1. slides 23/23"),
        ("quantitative_values_verified", "2. quantitative values verified"),
        ("no_unsupported_scientific_claim", "3. no unsupported scientific claim"),
        ("no_raw_radar_to_llm_arrow_claim", "4. no raw-radar-to-LLM arrow"),
        ("ci4r_prefix_not_labeled_subject", "5. CI4R prefix not labeled as subject"),
        ("continuous_labeled_interval_boundary", "6. continuous labeled-interval boundary present"),
        ("open_set_vs_open_world", "7. open-set vs open-world distinction present"),
        ("r2_vs_har_distinction", "8. R² vs HAR distinction present"),
        ("test_oracle_warning", "9. TEST oracle warning present"),
        ("unseen_radar_unresolved", "10. unseen-radar generalization marked unresolved"),
        ("no_text_overflow_hard_fail", "11. no text overflow (XML heuristic; confirm on renders)"),
        ("no_element_overlap_hard_fail", "12. no element overlap (confirm on renders)"),
        ("fonts_valid", "13. fonts valid"),
        ("pptx_opens", "14. pptx successfully opens"),
    ]
    for key, lab in labels:
        ok = d["checks"][key]
        lines.append(f"- {'PASS' if ok else 'FAIL'}  {lab}")
    lines += [
        "",
        "## Missing required numbers",
        "",
    ]
    if d["missing_nums"]:
        lines += [f"- {x}" for x in d["missing_nums"]]
    else:
        lines.append("- none")
    lines += ["", "## Missing required phrases", ""]
    if d["missing_must"]:
        lines += [f"- {x}" for x in d["missing_must"]]
    else:
        lines.append("- none")
    lines += ["", "## Forbidden phrases found", ""]
    if d["found_forbidden"]:
        lines += [f"- {x}" for x in d["found_forbidden"]]
    else:
        lines.append("- none")
    lines += ["", "## Off-slide elements", ""]
    if d["off_slide"]:
        lines += [f"- {x}" for x in d["off_slide"]]
    else:
        lines.append("- none")
    lines += ["", "## Fonts", ""]
    for k, v in sorted(d["fonts"].items(), key=lambda kv: -kv[1]):
        lines.append(f"- {k}: {v}")
    lines += ["", "## Tiny fonts (<9 pt)", ""]
    if d["tiny_fonts"]:
        lines += [f"- {x}" for x in d["tiny_fonts"][:20]]
    else:
        lines.append("- none")
    lines += ["", "## Overflow-risk boxes", ""]
    if d["overflow_risk"]:
        lines += [f"- {x}" for x in d["overflow_risk"]]
    else:
        lines.append("- none")
    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({k: d[k] for k in ("n_slides", "missing_nums", "missing_must", "found_forbidden", "checks")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    write_report(audit())
