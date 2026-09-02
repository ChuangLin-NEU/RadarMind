#!/usr/bin/env python3
"""Validate Paper Package v2. Read-only on Phase 1–10 and old packages."""
from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "scripts"))

from paper_v2.audits import scan_unsupported_leakage
from paper_v2.render import TOKEN_RE
from paper_v2.snapshot import MetricError, build_snapshot

OUT = ROOT / "results" / "paper_package_v2"
PRES = ROOT / "results" / "presentation_package_v2"
SRC_GLOBS = [
    ROOT / "scripts" / "write_paper_package_v2.py",
    *sorted((ROOT / "scripts" / "paper_v2").glob("*.py")),
    *sorted((ROOT / "scripts" / "paper_v2" / "templates").glob("*")),
]
MAGIC = re.compile(r"\b0\.(843|898|922|039|993|939|744|418|510|143|687|782|946)\b")
SECTIONS = [
    "# Cross-Radar Human Activity Recognition",
    "## Abstract",
    "**Keywords:**",
    "## 1. Introduction",
    "## 2. Methods",
    "## 3. Results",
    "## 4. Discussion",
    "## 5. Conclusions",
    "## Data Availability",
    "## Code Availability",
    "## Author Contributions",
    "## Conflict of Interest",
    "## References",
]
PHASE_IN_MS = re.compile(r"\bPhase\s+\d", re.I)
FIGS = [f"fig{i}_" for i in range(1, 10)]


def sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def main() -> int:
    lines = ["# Paper Package v2 validation", ""]
    fails = []

    try:
        snap = build_snapshot(ROOT)
        lines.append(f"- Metric sources: {snap['n_metrics']}/{snap['n_metrics']} resolved")
        metric_ok = snap["n_metrics"]
        metric_n = snap["n_metrics"]
    except MetricError as e:
        lines.append(f"- Metric sources: FAIL\n\n```\n{e}\n```")
        fails.append("metrics")
        metric_ok = 0
        metric_n = 0
        snap = {"metrics": {}}

    ms_path = OUT / "manuscript_v2.md"
    if not ms_path.is_file():
        fails.append("manuscript missing")
        lines.append("- Manuscript: missing")
        write_report(lines, fails)
        return 1
    ms = ms_path.read_text(encoding="utf-8")
    leftover = TOKEN_RE.findall(ms)
    lines.append(f"- Template tokens leftover in manuscript: {len(leftover)}")
    if leftover:
        fails.append("tokens")
        lines.append("  leftover: " + ", ".join(sorted(set(leftover))))

    tok_ok = 0 if leftover else 1
    for p in OUT.glob("*.md"):
        left = TOKEN_RE.findall(p.read_text(encoding="utf-8"))
        if left:
            fails.append(f"tokens:{p.name}")
            tok_ok = 0

    sec_miss = [s for s in SECTIONS if s not in ms]
    lines.append(f"- Manuscript sections: {len(SECTIONS) - len(sec_miss)}/{len(SECTIONS)}")
    if sec_miss:
        fails.append("sections")
        lines.append("  missing: " + "; ".join(sec_miss))

    if PHASE_IN_MS.search(ms):
        fails.append("phase numbers in manuscript")
        lines.append("- Manuscript contains Phase N labels: FAIL")
    else:
        lines.append("- Manuscript Phase labels: none")

    leaks = []
    abs_part = ms.split("## 1. Introduction")[0]
    conc = ms.split("## 5. Conclusions")[1] if "## 5. Conclusions" in ms else ""
    leaks += [f"Abstract:{h}" for h in scan_unsupported_leakage(abs_part)]
    leaks += [f"Conclusions:{h}" for h in scan_unsupported_leakage(conc)]
    for name in ("abstract_long.md", "abstract_short.md"):
        leaks += [f"{name}:{h}" for h in scan_unsupported_leakage((OUT / name).read_text(encoding="utf-8"))]
    lines.append(f"- Unsupported claims in Abstract/Conclusions: {len(leaks)}")
    if leaks:
        fails.append("unsupported leakage")
        lines.append("  " + ", ".join(leaks))

    na = OUT / "number_audit.csv"
    n_rows = list(csv.DictReader(na.open(encoding="utf-8"))) if na.is_file() else []
    unmapped = [r for r in n_rows if not r.get("metric_key")]
    lines.append(f"- Scientific numbers audited: {len(n_rows) - len(unmapped)}/{len(n_rows)}")
    if unmapped:
        fails.append("number audit")
        lines.append("  unmapped: " + ", ".join(r["display_value"] for r in unmapped[:12]))

    ca = OUT / "claims_audit.csv"
    c_rows = list(csv.DictReader(ca.open(encoding="utf-8"))) if ca.is_file() else []
    lines.append(f"- Claims audited: {len(c_rows)}/{len(c_rows) if c_rows else 0}")

    tcons = OUT / "table_consistency_audit.csv"
    t_rows = list(csv.DictReader(tcons.open(encoding="utf-8"))) if tcons.is_file() else []
    t_fail = [r["table"] for r in t_rows if r.get("consistent") != "PASS"]
    lines.append(f"- Tables: {len(t_rows) - len(t_fail)}/{len(t_rows)} PASS")
    if t_fail:
        fails.append("tables")

    fig_ok = 0
    for i in range(1, 10):
        hits = list((OUT / "figures").glob(f"fig{i}_*.png"))
        if hits:
            fig_ok += 1
        else:
            fails.append(f"fig{i}")
    lines.append(f"- Figures: {fig_ok}/9 PASS")
    if not (OUT / "figure_source_audit.csv").is_file():
        fails.append("figure_source_audit")

    snap_path = OUT / "metric_snapshot.json"
    if not snap_path.is_file():
        fails.append("metric_snapshot")
    else:
        lines.append("- metric_snapshot.json: present")

    magic_hits = []
    for p in SRC_GLOBS:
        if not p.is_file():
            continue
        txt = p.read_text(encoding="utf-8")
        if MAGIC.search(txt):
            magic_hits.append(str(p.relative_to(ROOT)))
    lines.append(f"- Magic result numbers in generator source: {len(magic_hits)}")
    if magic_hits:
        fails.append("magic numbers")
        lines.append("  " + ", ".join(magic_hits))

    frozen = json.loads((OUT / "frozen_sha256.json").read_text(encoding="utf-8")) if (OUT / "frozen_sha256.json").is_file() else {}
    drift = []
    for rel, expect in frozen.items():
        p = ROOT / rel
        if not p.is_file():
            drift.append(rel + " missing")
        elif sha(p) != expect:
            drift.append(rel)
    lines.append(f"- Frozen Phase / old-package hashes: {len(frozen) - len(drift)}/{len(frozen)}")
    if drift:
        fails.append("frozen drift")
        lines.append("  changed: " + ", ".join(drift[:12]))

    pres_need = ["presentation_outline.md", "slide_content.md", "demo_notes.md"]
    pres_ok = all((PRES / n).is_file() for n in pres_need)
    lines.append(f"- Presentation files: {'PASS' if pres_ok else 'FAIL'}")
    if not pres_ok:
        fails.append("presentation")

    status = "PASS" if not fails else "FAIL"
    lines = [f"Status: **{status}**", ""] + lines
    if fails:
        lines += ["", "## Failures", ""] + [f"- {x}" for x in fails]
    write_report(lines, fails)
    print(status, "fails=", fails, flush=True)
    return 0 if not fails else 1


def write_report(lines: list[str], fails: list[str]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "package_validation_report.md").write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
