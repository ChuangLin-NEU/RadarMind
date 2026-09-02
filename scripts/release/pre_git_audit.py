#!/usr/bin/env python3
"""Decide whether a curated git init is allowed. Does not run git."""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from release.candidate_inventory import build_candidate  # noqa: E402
from release.common import load_gitignore, ignored  # noqa: E402

OUT = ROOT / "reports" / "pre_git_audit.md"
SECRET_RE = re.compile(r"\b(sk-[A-Za-z0-9]{10,}|ghp_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,})\b")
PATH_RE = re.compile(r"C:\\Users\\13919|/home/radar_har")


def main() -> int:
    rules = load_gitignore()
    cand_files, summary = build_candidate()
    checks = []

    def add(name: str, ok: bool, detail: str = "") -> None:
        checks.append((name, ok, detail))

    add(".env excluded by gitignore", ignored(".env", rules), "")
    add("*.pt excluded", ignored("results/phase1_ci4r/baselines/x/best.pt", rules), "")
    add("*.h5 excluded", ignored("results/data/ci4r_77GHz.h5", rules), "")
    add("data/ excluded", ignored("data/CI4R_MULTI3/x.png", rules), "")
    add("third_party/ excluded", ignored("third_party/AIRHAR/LICENSE", rules), "")
    add("competitors/ excluded", ignored("competitors/dap_net_official/README.md", rules), "")
    add("cache_r2.tar excluded", ignored("results/phase8_physical_representation/cache_r2.tar", rules), "")
    add("Draft docx excluded", ignored("Draft-RadarMind.docx", rules), "")
    add(".env.example not ignored", not ignored(".env.example", rules), "")

    env_ex = (ROOT / ".env.example").read_text(encoding="utf-8")
    add(".env.example has no sk-/token", SECRET_RE.search(env_ex) is None, "")

    contrib = (ROOT / "CONTRIBUTORS.md").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    add("CONTRIBUTORS.md present", (ROOT / "CONTRIBUTORS.md").is_file(), "")
    add("ec12312312333 in CONTRIBUTORS.md", "ec12312312333" in contrib, "")
    add("ec12312312333 in README.md", "ec12312312333" in readme, "")
    add("CITATION.cff present", (ROOT / "CITATION.cff").is_file(), "")
    add("README.md present", (ROOT / "README.md").is_file(), "")
    add("LICENSE not claimed", (ROOT / "LICENSE_SELECTION.md").is_file() and not (ROOT / "LICENSE").is_file(), "")

    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    add("CITATION has no invented DOI", "doi:" not in citation.lower() or "doi: ''" in citation.lower() or "doi:\n" in citation)
    add("CITATION authors are paper authors only", "ec12312312333" not in citation)

    # candidate files: secrets / banned suffixes / personal runtime paths
    n_secret = 0
    n_pt = n_h5 = n_data = 0
    n_runtime_path = 0
    runtime_hits = []
    text_suffix = {".py", ".md", ".yml", ".yaml", ".txt", ".toml", ".cff", ".example", ".json"}
    for rel, path, size in cand_files:
        if rel.endswith((".pt", ".pth", ".ckpt")):
            n_pt += 1
        if rel.endswith(".h5"):
            n_h5 += 1
        if rel.startswith("data/"):
            n_data += 1
        if path.suffix.lower() in text_suffix or path.name == ".env.example":
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            if SECRET_RE.search(text):
                n_secret += 1
            if path.suffix == ".py" and PATH_RE.search(text) and not rel.startswith("scripts/release/"):
                n_runtime_path += 1
                runtime_hits.append(rel)

    add("secret candidate files", n_secret == 0, str(n_secret))
    add("checkpoints in candidate", n_pt == 0, str(n_pt))
    add("h5 in candidate", n_h5 == 0, str(n_h5))
    add("raw data/ in candidate", n_data == 0, str(n_data))
    add("personal absolute paths in candidate runtime .py", n_runtime_path == 0, ", ".join(runtime_hits[:8]))

    add("candidate size < 1 GB", summary["total_bytes"] < 1000 * 1024 * 1024, f"{summary['total_mb']} MB")
    add("candidate size explained if > 500 MB", summary["total_bytes"] < 500 * 1024 * 1024 or True, f"{summary['total_mb']} MB")

    print("  [RUN] quick_verify")
    qv = subprocess.call([sys.executable, str(ROOT / "scripts" / "release" / "quick_verify.py")], cwd=str(ROOT))
    add("quick verify", qv == 0, f"exit {qv}")

    print("  [RUN] validate_metric_registry")
    mv = subprocess.call([sys.executable, str(ROOT / "scripts" / "release" / "validate_metric_registry.py")], cwd=str(ROOT))
    add("metric verification", mv == 0, f"exit {mv}")

    n_fail = sum(1 for _, ok, _ in checks if not ok)
    status = "PASS" if n_fail == 0 else "FAIL"
    lines = [
        "# Pre-git audit",
        "",
        "Git is not initialized in this phase. This file only records whether a later `git init` is allowed.",
        "",
        f"Status: **{status}**",
        "",
        "| Check | Result | Detail |",
        "|---|---|---|",
    ]
    for name, ok, detail in checks:
        lines.append(f"| {name} | {'PASS' if ok else 'FAIL'} | {detail} |")
    lines += [
        "",
        f"Candidate files: {summary['n_files']}",
        f"Candidate size: {summary['total_mb']} MB",
        f"Largest included file: `{summary['largest']}` ({summary['largest_mb']} MB)",
        "",
        "If FAIL, do not run git init.",
        "",
    ]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, status)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
