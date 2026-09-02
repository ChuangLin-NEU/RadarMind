#!/usr/bin/env python3
"""Path, CI4R-split-reference, and frozen-results inclusion audits. Read-only on metrics."""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WIN = re.compile(r"C:\\\\Users\\\\13919|C:/Users/13919")
LAB = re.compile(r"/home/radar_har")
SKIP_DIRS = {".git", "third_party", "competitors", "build_artifacts", "node_modules", "__pycache__", "data"}

RUNTIME_PY = {
    "scripts/pointcloud/train_voxel.py",
    "scripts/pointcloud/pack_voxels.py",
    "scripts/pointcloud/pack_from_dir.py",
    "scripts/pointcloud/phase1b_build.py",
    "scripts/pointcloud/_probe01.py",
    "scripts/pointcloud/_probe01b.py",
    "scripts/ci4r/phase1a_build.py",
    "scripts/ci4r/phase1a_mat_diverse.py",
    "scripts/phase8/audit.py",
    "scripts/phase8/run_phase8b.py",
    "scripts/phase8/_inspect_24.py",
    "scripts/phase8/_probe_zip.py",
    "scripts/phase8/write_report.py",
    "scripts/phase10/run_phase10d.py",
    "scripts/data_audit_followup.py",
    "scripts/data_audit_peek.py",
    "scripts/data_audit.py",
    "environment.yml",
}


def _iter_text_files():
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.suffix.lower() not in {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".csv", ".sh", ".ps1", ".cff"}:
            continue
        yield p


def write_path_audit() -> None:
    rows = []
    for p in _iter_text_files():
        rel = p.relative_to(ROOT).as_posix()
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        hits = []
        if WIN.search(text):
            hits.append("C:\\Users\\13919")
        if LAB.search(text):
            hits.append("/home/radar_har")
        if not hits:
            continue
        if rel.startswith("scripts/release/") or rel.startswith("reports/"):
            cls = "B"
        elif rel in RUNTIME_PY or (rel.endswith(".py") and ("scripts/" in rel or rel == "environment.yml")):
            cls = "A" if (WIN.search(text) or LAB.search(text)) else "B"
            # After Phase 2 fixes, remaining hits in listed runtime files are FAIL-A
            if rel.endswith(".py") or rel == "environment.yml":
                still = bool(WIN.search(text) or LAB.search(text))
                cls = "A" if still and rel in RUNTIME_PY else ("B" if rel.startswith("reports/") or rel.startswith("results/") else "A")
        elif rel.startswith("results/") or rel.startswith("reports/") or rel.startswith("docs/archive/"):
            cls = "B" if rel.endswith(".md") or rel.endswith(".json") else "C"
            if rel.endswith(".csv"):
                cls = "C"
        else:
            cls = "B"
        rows.append((cls, rel, ", ".join(hits)))

    # Reclassify remaining runtime scripts after fix: if no hit they won't appear.
    lines = [
        "# Path portability audit",
        "",
        "Classes: A = runtime hard-coded path (must be env/relative); B = historical report; C = generated CSV/manifest.",
        "Phase 2 changed Class A scripts to environment variables or repository-relative defaults.",
        "Frozen `answers.json` files may still record the lab path used when the experiment ran. Those files are metric sources and were not rewritten.",
        "",
        "| Class | Path | Pattern |",
        "|---|---|---|",
    ]
    n_a = 0
    for cls, rel, pat in sorted(rows):
        if cls == "A":
            n_a += 1
        lines.append(f"| {cls} | `{rel}` | {pat} |")
    lines += ["", f"Remaining Class A rows: {n_a}", ""]
    (ROOT / "reports" / "path_portability_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_ci4r_refs() -> None:
    needle = "ci4r_subject_independent"
    hits = []
    for p in _iter_text_files():
        rel = p.relative_to(ROOT).as_posix()
        text = p.read_text(encoding="utf-8", errors="replace")
        for i, line in enumerate(text.splitlines(), 1):
            if needle in line:
                hits.append((rel, i, line.strip()[:200]))
    lines = [
        "# CI4R split reference audit",
        "",
        "Legacy filename `data_splits/ci4r_subject_independent.csv` is kept.",
        "Protocol in the file is `prefix_tail4_disjoint`. Not subject-independent.",
        "",
        "| Path | Line | Usage |",
        "|---|---:|---|",
    ]
    for rel, i, line in hits:
        safe = line.replace("|", "\\|")
        lines.append(f"| `{rel}` | {i} | `{safe}` |")
    lines += ["", "Decision this phase: KEEP LEGACY PATH; correct semantics in docs.", ""]
    (ROOT / "reports" / "ci4r_split_reference_audit.md").write_text("\n".join(lines), encoding="utf-8")


def write_frozen_audit() -> None:
    keep_roots = [
        ROOT / "results" / "frozen_2026_08",
        ROOT / "results" / "paper_package_v2",
    ]
    lines = [
        "# Frozen results release audit",
        "",
        "Local files are not deleted. This list is inclusion guidance for git.",
        "",
        "## Keep (metrics / tables / captions)",
        "",
    ]
    keep_ext = {".json", ".csv", ".md", ".tex"}
    skip_name = {"cache_r2.tar"}
    for d in keep_roots:
        if not d.is_dir():
            continue
        lines.append(f"### `{d.relative_to(ROOT).as_posix()}`")
        lines.append("")
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            rel = p.relative_to(ROOT).as_posix()
            mb = p.stat().st_size / (1024 * 1024)
            decision = "KEEP" if p.suffix.lower() in keep_ext and p.name not in skip_name else "EXCLUDE"
            if p.suffix.lower() in {".pt", ".npz", ".npy", ".h5", ".tar"}:
                decision = "EXCLUDE"
            lines.append(f"- {decision} `{rel}` ({mb:.3f} MB)")
        lines.append("")
    extra = [
        "Phase `metrics.json` / `answers.json` / pairing / representation CSV under `results/phase*/` are required for snapshot traceability.",
        "Exclude: `*.pt`, `*.npz`, `*.h5`, `cache_r2.tar`, path-dump CSVs listed in `.gitignore`.",
        "",
    ]
    lines.extend(extra)
    (ROOT / "reports" / "frozen_results_release_audit.md").write_text("\n".join(lines), encoding="utf-8")


def scrub_figure_manifest(path: Path) -> None:
    if not path.is_file():
        return
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    changed = False

    def relify(s: str) -> str:
        nonlocal changed
        if not isinstance(s, str):
            return s
        for prefix in (
            "C:/Users/13919/Desktop/RadarBridge/",
            "C:\\Users\\13919\\Desktop\\RadarBridge\\",
            "C:/Users/13919/Desktop/RadarMind/",
            "C:\\Users\\13919\\Desktop\\RadarMind\\",
        ):
            if s.startswith(prefix):
                changed = True
                return s[len(prefix) :].replace("\\", "/")
        return s.replace("\\", "/") if "RadarBridge" in s or "RadarMind" in s and ":/" in s else s

    if isinstance(data, dict) and "figures" in data:
        for fig in data["figures"]:
            for k in ("file", "pdf", "svg"):
                if k in fig:
                    fig[k] = relify(fig[k])
            if "data_sources" in fig:
                fig["data_sources"] = [relify(x) if isinstance(x, str) else x for x in fig["data_sources"]]
    if changed:
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    write_path_audit()
    write_ci4r_refs()
    write_frozen_audit()
    scrub_figure_manifest(ROOT / "figures" / "figure_manifest.json")
    scrub_figure_manifest(ROOT / "results" / "final_delivery" / "figure_manifest.json")
    print("wrote path / ci4r / frozen audits; scrubbed figure manifests if needed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
