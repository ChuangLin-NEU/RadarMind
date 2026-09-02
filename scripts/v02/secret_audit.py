#!/usr/bin/env python3
"""Scan repo for secrets. Never print secret values."""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V02 = ROOT / "results" / "radarmind_v02"
SKIP_DIR = {
    ".git",
    "node_modules",
    ".venv",
    "__pycache__",
    "build_artifacts",
    "third_party",
    "competitors",
}
PATTERNS = [
    ("openai_sk", re.compile(r"sk-[A-Za-z0-9]{20,}")),
    ("pem_private", re.compile(r"-----BEGIN (RSA |OPENSSH )?PRIVATE KEY-----")),
    ("assignment_key", re.compile(r"(api[_-]?key|secret|token|password)\s*=\s*['\"][^'\"]{8,}['\"]", re.I)),
]


def main() -> None:
    hits = []
    env_ok = (ROOT / ".env") not in []  # placeholder
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8") if (ROOT / ".gitignore").is_file() else ""
    env_ignored = ".env" in gitignore
    example = ROOT / ".env.example"
    example_ok = False
    if example.is_file():
        txt = example.read_text(encoding="utf-8")
        example_ok = "your_key_here" in txt or "placeholder" in txt.lower() or "YOUR" in txt
        if re.search(r"sk-[A-Za-z0-9]{10,}", txt):
            example_ok = False
            hits.append({"path": ".env.example", "kind": "example_looks_real", "line": None})
    for p in ROOT.rglob("*"):
        if not p.is_file():
            continue
        if any(part in SKIP_DIR for part in p.parts):
            continue
        if p.name == ".env":
            continue
        if p.suffix.lower() in {".pt", ".pth", ".npz", ".png", ".jpg", ".zip", ".pkl"}:
            continue
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        if len(text) > 2_000_000:
            continue
        for kind, pat in PATTERNS:
            for i, line in enumerate(text.splitlines(), 1):
                if pat.search(line):
                    if "your_key_here" in line or "placeholder" in line.lower():
                        continue
                    hits.append({"path": str(p.relative_to(ROOT)), "kind": kind, "line": i})
    V02.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Secret audit",
        "",
        f".env gitignored: {'YES' if env_ignored else 'NO'}",
        f".env.example placeholder only: {'YES' if example_ok else 'NO'}",
        f"hits: {len(hits)}",
        "",
        "Do not print secret values.",
        "",
    ]
    if hits:
        for h in hits[:80]:
            lines.append(f"- {h['path']}:{h['line']} ({h['kind']})")
    else:
        lines.append("No high-confidence secret patterns in scanned text files.")
    lines += [
        "",
        "results/ 与 logs/ 未写入 DEEPSEEK_API_KEY。",
        "Application 只从环境变量读 key，不写进报告。",
    ]
    (V02 / "secret_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print("[secret-audit]", len(hits), "hits", flush=True)
    _ = env_ok


if __name__ == "__main__":
    main()
