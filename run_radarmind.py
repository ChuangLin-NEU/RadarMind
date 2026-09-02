#!/usr/bin/env python3
"""Start the RadarMind Replay Application. Consumes existing cache / checkpoints / results."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "radarmind" / "app" / "streamlit_app.py"


def _python() -> str:
    if sys.executable:
        return sys.executable
    return "python"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--offline-demo", action="store_true", help="LLM uses fallback only; models and results stay real")
    ap.add_argument("--port", type=int, default=int(os.environ.get("RADARMIND_PORT", 8502)))
    args = ap.parse_args(argv)
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if args.offline_demo:
        os.environ["RADARMIND_OFFLINE_DEMO"] = "1"
    os.environ["RADARMIND_PORT"] = str(args.port)
    py = _python()
    cmd = [py, "-m", "streamlit", "run", str(APP), "--server.headless", "true", "--server.port", str(args.port)]
    return subprocess.call(cmd, cwd=str(ROOT))


if __name__ == "__main__":
    raise SystemExit(main())
