from __future__ import annotations

import re
from pathlib import Path

TOKEN_RE = re.compile(r"@@([A-Z0-9_]+)@@")


class TokenError(RuntimeError):
    pass


def render(template: str, displays: dict[str, str], *, name: str = "template") -> str:
    text = template
    for key, val in displays.items():
        text = text.replace(f"@@{key}@@", val)
    leftover = TOKEN_RE.findall(text)
    if leftover:
        raise TokenError(f"{name}: unreplaced tokens: {sorted(set(leftover))}")
    return text


def render_file(path: Path, displays: dict[str, str]) -> str:
    return render(path.read_text(encoding="utf-8"), displays, name=str(path))
