"""Load .env into os.environ. Never print secret values."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
_SECRET_MARKERS = ("KEY", "SECRET", "TOKEN", "PASSWORD", "AUTHORIZATION")


def load_dotenv(path: Path | None = None) -> None:
    p = path or (ROOT / ".env")
    if p.is_file():
        for raw in p.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and k not in os.environ:
                os.environ[k] = v
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip() and os.environ.get("RADARMIND_LLM_API_KEY", "").strip():
        os.environ["DEEPSEEK_API_KEY"] = os.environ["RADARMIND_LLM_API_KEY"]
    if not os.environ.get("RADARMIND_LLM_PROVIDER", "").strip():
        os.environ.setdefault("RADARMIND_LLM_PROVIDER", "deepseek")


def redact(name: str, value: str) -> str:
    up = name.upper()
    if any(m in up for m in _SECRET_MARKERS):
        return "<redacted>"
    return value
