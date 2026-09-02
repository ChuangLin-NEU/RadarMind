"""Shared paths and ignore matching for release scripts."""
from __future__ import annotations

import fnmatch
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

MANUSCRIPT_KEYS = [
    "MRI_SI_ACC",
    "MRI_SI_F1",
    "RADHAR_XYZV_ACC",
    "RADHAR_XYZV_F1",
    "R0_10_WITHIN",
    "R0_24_WITHIN",
    "R0_77_WITHIN",
    "R0_10_TO_24",
    "R0_10_TO_77",
    "R0_24_TO_10",
    "R0_24_TO_77",
    "R0_77_TO_10",
    "R0_77_TO_24",
    "M0_ACC",
    "M1_ACC",
    "M2_ACC",
    "M1_RET",
    "M1_RID",
    "RID_R2D_3CLASS",
    "RID_R2V_2477",
    "CONT_FRAME_ACC",
    "CONT_FRAME_F1",
    "CONT_WIN_ACC",
    "CONT_WIN_F1",
    "CONT_SEG_F1",
    "CONT_BND_SEC",
    "PROTO_AUROC",
    "PROTO_AUROC_STD",
]


def load_gitignore(path: Path | None = None) -> list[str]:
    p = path or (ROOT / ".gitignore")
    lines = []
    for raw in p.read_text(encoding="utf-8").splitlines():
        s = raw.strip()
        if not s or s.startswith("#"):
            continue
        lines.append(s)
    return lines


def _match_part(pattern: str, rel: str, is_dir: bool) -> bool:
    neg = pattern.startswith("!")
    if neg:
        pattern = pattern[1:]
    if pattern.endswith("/"):
        if not is_dir and not rel.endswith("/"):
            # directory rule also matches descendants
            pass
        pattern = pattern.rstrip("/")
    rel = rel.replace("\\", "/").lstrip("./")
    if pattern.startswith("/"):
        pattern = pattern[1:]
        return fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern + "/*") or rel.startswith(pattern + "/")
    if "/" not in pattern.rstrip("/"):
        name = Path(rel).name
        if fnmatch.fnmatch(name, pattern):
            return True
        if fnmatch.fnmatch(rel, pattern):
            return True
        if fnmatch.fnmatch(rel, "*/" + pattern):
            return True
        if is_dir and (fnmatch.fnmatch(rel, pattern) or rel.startswith(pattern + "/")):
            return True
        # descendants of a matching directory name
        parts = rel.split("/")
        return any(fnmatch.fnmatch(part, pattern) for part in parts[:-1])
    return fnmatch.fnmatch(rel, pattern) or rel.startswith(pattern.rstrip("*") ) if pattern.endswith("/**") else (
        fnmatch.fnmatch(rel, pattern) or fnmatch.fnmatch(rel, pattern + "/*") or rel.startswith(pattern.rstrip("/") + "/")
    )


def is_ignored(rel: str, rules: list[str], is_dir: bool = False) -> bool:
    rel = rel.replace("\\", "/").lstrip("./")
    ignored = False
    for rule in rules:
        neg = rule.startswith("!")
        pat = rule[1:] if neg else rule
        if _match_part(pat if not neg else pat, rel, is_dir) or _match_part(rule if not neg else pat, rel, is_dir):
            # re-evaluate with explicit negation
            matched = _match_one(rule, rel, is_dir)
            if matched is True:
                ignored = False if neg else True
    return ignored


def _match_one(rule: str, rel: str, is_dir: bool) -> bool:
    neg = rule.startswith("!")
    pattern = rule[1:] if neg else rule
    dir_only = pattern.endswith("/")
    pattern = pattern.rstrip("/")
    anchored = pattern.startswith("/")
    if anchored:
        pattern = pattern[1:]
    rel_n = rel.replace("\\", "/")
    if dir_only and not (is_dir or "/" in rel_n[len(pattern) :] if rel_n.startswith(pattern) else is_dir):
        if rel_n != pattern and not rel_n.startswith(pattern + "/"):
            if not fnmatch.fnmatch(rel_n, pattern) and not fnmatch.fnmatch(Path(rel_n).name, pattern):
                return False
    if anchored:
        ok = fnmatch.fnmatch(rel_n, pattern) or rel_n.startswith(pattern + "/") or fnmatch.fnmatch(rel_n, pattern + "/**")
    elif "/" not in pattern:
        parts = rel_n.split("/")
        ok = any(fnmatch.fnmatch(p, pattern) for p in parts) or fnmatch.fnmatch(rel_n, pattern)
    else:
        ok = fnmatch.fnmatch(rel_n, pattern) or rel_n.startswith(pattern + "/")
    return ok


def ignored(rel: str, rules: list[str], is_dir: bool = False) -> bool:
    """Last matching gitignore rule wins."""
    state = False
    rel_n = rel.replace("\\", "/").lstrip("./")
    for rule in rules:
        neg = rule.startswith("!")
        if _match_one(rule, rel_n, is_dir):
            state = not neg
    return state


def iter_files(root: Path | None = None):
    root = root or ROOT
    skip_names = {".git"}
    for p in root.rglob("*"):
        if any(part in skip_names for part in p.parts):
            continue
        if p.is_file():
            yield p
