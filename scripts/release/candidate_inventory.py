#!/usr/bin/env python3
"""Simulate the publication file set: include manifest minus gitignore."""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from release.common import ignored, iter_files, load_gitignore  # noqa: E402

INCLUDE = ROOT / "release" / "public_include_manifest.yaml"
FILES_OUT = ROOT / "reports" / "candidate_repository_files.txt"
SUM_OUT = ROOT / "reports" / "candidate_repository_summary.md"


def parse_include(path: Path) -> list[str]:
    items = []
    in_include = False
    for raw in path.read_text(encoding="utf-8").splitlines():
        if raw.startswith("include:"):
            in_include = True
            continue
        if in_include:
            if raw.startswith("not_present") or (raw and not raw.startswith(" ") and not raw.startswith("-")):
                break
            s = raw.strip()
            if s.startswith("- "):
                items.append(s[2:].strip())
    return items


def listed(rel: str, includes: list[str]) -> bool:
    rel = rel.replace("\\", "/")
    for item in includes:
        item = item.rstrip("/")
        if rel == item or rel.startswith(item + "/"):
            return True
    return False


def build_candidate():
    rules = load_gitignore()
    includes = parse_include(INCLUDE)
    rows = []
    for p in iter_files(ROOT):
        rel = p.relative_to(ROOT).as_posix()
        if not listed(rel, includes):
            continue
        if ignored(rel, rules, is_dir=False):
            continue
        rows.append((rel, p, p.stat().st_size))
    rows.sort(key=lambda x: x[0].lower())
    total = sum(s for _, _, s in rows)
    by_ext = Counter()
    for rel, _, s in rows:
        ext = Path(rel).suffix.lower() or "(none)"
        by_ext[ext] += s
    largest = sorted(rows, key=lambda x: x[2], reverse=True)[:50]
    summary = {
        "n_files": len(rows),
        "total_bytes": total,
        "total_mb": round(total / (1024 * 1024), 2),
        "largest": largest[0][0] if largest else "",
        "largest_mb": round(largest[0][2] / (1024 * 1024), 2) if largest else 0,
        "by_ext": by_ext,
        "top50": largest,
        "includes": includes,
    }
    return rows, summary


def write_reports() -> dict:
    rows, summary = build_candidate()
    FILES_OUT.write_text("\n".join(r[0] for r in rows) + ("\n" if rows else ""), encoding="utf-8")
    lines = [
        "# Candidate repository summary",
        "",
        "Selection: `release/public_include_manifest.yaml` minus `.gitignore`.",
        "This is not `git add .` of the whole working tree.",
        "",
        f"- files: {summary['n_files']}",
        f"- size: {summary['total_mb']} MB ({summary['total_bytes']} bytes)",
        f"- largest file: `{summary['largest']}` ({summary['largest_mb']} MB)",
        "",
        "## Size by extension",
        "",
        "| Extension | MB |",
        "|---|---:|",
    ]
    for ext, nbytes in sorted(summary["by_ext"].items(), key=lambda x: -x[1]):
        lines.append(f"| `{ext}` | {nbytes / (1024 * 1024):.2f} |")
    lines += ["", "## Largest 50 included files", "", "| MB | Path |", "|---:|---|"]
    for rel, _, nbytes in summary["top50"]:
        lines.append(f"| {nbytes / (1024 * 1024):.2f} | `{rel}` |")
    if summary["total_bytes"] > 1024 * 1024 * 1024:
        lines += ["", "**BLOCK:** candidate exceeds 1 GB.", ""]
    elif summary["total_bytes"] > 500 * 1024 * 1024:
        lines += ["", "Candidate exceeds 500 MB. Inspect the largest files before git init.", ""]
    else:
        lines += ["", "Candidate is under 500 MB.", ""]
    SUM_OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    summary = write_reports()
    print("files", summary["n_files"], "MB", summary["total_mb"], "largest", summary["largest"])
    if summary["total_bytes"] > 1024 * 1024 * 1024:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
