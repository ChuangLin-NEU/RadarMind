#!/usr/bin/env python3
"""Paper Package v2 generator. Read-only on Phase 1–10. No training. No str.format on manuscript."""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from paper_v2.audits import scan_unsupported_leakage, table_consistency, write_claims, write_number_audit
from paper_v2.figures import make_figures
from paper_v2.render import TOKEN_RE, render, render_file
from paper_v2.snapshot import MetricError, build_snapshot, display_map
from paper_v2.tables import build_tables

OUT = ROOT / "results" / "paper_package_v2"
PRES = ROOT / "results" / "presentation_package_v2"
TAB = OUT / "tables"
FIG = OUT / "figures"
PFIG = PRES / "figures"
TPL = Path(__file__).resolve().parent / "paper_v2" / "templates"

FROZEN_GLOBS = [
    "results/phase2_cross_radar/pairing_stats.json",
    "results/phase2_cross_radar/training_runs.csv",
    "results/phase7_radar_invariance/answers.json",
    "results/phase8_physical_representation/answers.json",
    "results/phase9_radarbridge_v2/answers.json",
    "results/phase10_motion_factors/answers.json",
    "results/phase10_sensor_calibration/answers.json",
    "results/phase10_velocity_calibration/answers.json",
    "results/phase10_unified_measurement/answers.json",
    "results/phase10_hardware_measurement/answers.json",
    "results/phase10_motion_conditioned/answers.json",
    "results/phase10_conditional_2d_measurement/answers.json",
    "results/phase10_closure/answers.json",
]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def snapshot_frozen() -> dict[str, str]:
    out = {}
    for rel in FROZEN_GLOBS:
        p = ROOT / rel
        if p.is_file():
            out[rel.replace("\\", "/")] = _sha(p)
    for name in ("paper_package", "presentation_package"):
        d = ROOT / "results" / name
        if not d.is_dir():
            continue
        for f in d.rglob("*"):
            if f.is_file():
                rel = str(f.relative_to(ROOT)).replace("\\", "/")
                out[rel] = _sha(f)
    return out


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def required_sections(ms: str) -> list[str]:
    need = [
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
    return [s for s in need if s not in ms]


def main() -> None:
    frozen_before = snapshot_frozen()
    try:
        snap = build_snapshot(ROOT)
    except MetricError as e:
        raise SystemExit(str(e))
    OUT.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(exist_ok=True)
    FIG.mkdir(exist_ok=True)
    write_text(OUT / "metric_snapshot.json", json.dumps(snap, indent=2, ensure_ascii=False))
    write_text(OUT / "frozen_sha256.json", json.dumps(frozen_before, indent=2))
    displays = display_map(snap)

    rendered = {}
    for name in (
        "manuscript_v2.md",
        "abstract_long.md",
        "abstract_short.md",
        "titles.md",
        "paper_story.md",
        "supported_claims.md",
        "limitations.md",
        "data_availability.md",
        "code_availability.md",
        "supplementary_plan.md",
    ):
        rendered[name] = render_file(TPL / name, displays)
        write_text(OUT / name, rendered[name])

    missing = required_sections(rendered["manuscript_v2.md"])
    if missing:
        raise SystemExit("manuscript missing sections: " + "; ".join(missing))
    if TOKEN_RE.search(rendered["manuscript_v2.md"]):
        raise SystemExit("manuscript still contains @@tokens@@")

    leaks = []
    for loc in ("abstract_long.md", "abstract_short.md", "manuscript_v2.md"):
        text = rendered[loc]
        if loc == "manuscript_v2.md":
            # restricted zones
            low = text.lower()
            abs_part = text.split("## 1. Introduction")[0]
            conc = text.split("## 5. Conclusions")[1] if "## 5. Conclusions" in text else ""
            for part, tag in ((abs_part, "Abstract"), (conc, "Conclusions")):
                hits = scan_unsupported_leakage(part)
                leaks.extend([f"{tag}:{h}" for h in hits])
        else:
            leaks.extend([f"{loc}:{h}" for h in scan_unsupported_leakage(text)])
    if leaks:
        raise SystemExit("UNSUPPORTED claim leakage: " + ", ".join(leaks))

    table_audit = build_tables(snap, TAB)
    write_claims(OUT / "claims_audit.csv")
    num_rows = write_number_audit(
        OUT / "number_audit.csv",
        snap,
        [
            ("manuscript_v2.md", rendered["manuscript_v2.md"]),
            ("abstract_long.md", rendered["abstract_long.md"]),
            ("abstract_short.md", rendered["abstract_short.md"]),
        ],
    )
    # add table md/tex after they exist
    extra_locs = []
    for p in sorted(TAB.glob("*.md")) + sorted(TAB.glob("*.tex")):
        extra_locs.append((p.name, p.read_text(encoding="utf-8")))
    if extra_locs:
        num_rows = write_number_audit(
            OUT / "number_audit.csv",
            snap,
            [
                ("manuscript_v2.md", rendered["manuscript_v2.md"]),
                ("abstract_long.md", rendered["abstract_long.md"]),
                ("abstract_short.md", rendered["abstract_short.md"]),
            ]
            + extra_locs,
        )
    unmapped = [r for r in num_rows if not r["metric_key"]]
    if unmapped:
        raise SystemExit(
            "number audit FAIL, displays not in snapshot: "
            + ", ".join(f"{r['display_value']}@{r['context']}" for r in unmapped[:20])
        )

    cons = table_consistency(TAB)
    if any(r["consistent"] != "PASS" for r in cons):
        raise SystemExit("table consistency FAIL")

    make_figures(snap, FIG, PFIG, "scripts/write_paper_package_v2.py")

    for name in ("presentation_outline.md", "slide_content.md", "demo_notes.md"):
        write_text(PRES / name, render_file(TPL / name, displays))

    frozen_after = snapshot_frozen()
    changed = [k for k, v in frozen_before.items() if frozen_after.get(k) != v]
    extra = [k for k in frozen_after if k not in frozen_before]
    if changed or extra:
        raise SystemExit(f"frozen files changed: {changed} extra={extra}")

    write_text(
        OUT / "package_status.md",
        "Paper Package v2 generated from metric_snapshot.json. Tokens explicit. No Phase results written.\n",
    )
    print(
        f"paper v2 generated metrics={snap['n_metrics']} numbers={len(num_rows)} tables={len(table_audit)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
