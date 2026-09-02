#!/usr/bin/env python3
"""Offline demo entry. Predictions stay real; DeepSeek is not called."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_radarmind import main

if __name__ == "__main__":
    raise SystemExit(main(["--offline-demo", *sys.argv[1:]]))
