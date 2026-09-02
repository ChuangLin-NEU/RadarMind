#!/usr/bin/env python3
"""Record the running airhar versions. Do not change conda config."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def ver(mod: str) -> str:
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "present")
    except Exception as e:  # noqa: BLE001
        return f"missing ({e.__class__.__name__})"


def main() -> None:
    py = sys.version.split()[0]
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements_lock.txt").write_text(freeze, encoding="utf-8")
    yml = [
        "name: airhar",
        "channels:",
        "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge",
        "  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main",
        "  - nodefaults",
        "dependencies:",
        f"  - python={py}",
        "  - pip",
        "  - pip:",
        "    - -r requirements_lock.txt",
        "",
        "# airhar is the runtime used for RadarMind v0.2.",
        f"# recorded_from: {sys.executable}",
    ]
    (ROOT / "environment.yml").write_text("\n".join(yml) + "\n", encoding="utf-8")
    rec = {
        "env": "airhar",
        "python": py,
        "executable": sys.executable,
        "pytorch": ver("torch"),
        "torchvision": ver("torchvision"),
        "streamlit": ver("streamlit"),
        "numpy": ver("numpy"),
        "scipy": ver("scipy"),
        "sklearn": ver("sklearn"),
    }
    dest = ROOT / "results" / "radarmind_v02"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "environment_record.json").write_text(json.dumps(rec, indent=2), encoding="utf-8")
    print(json.dumps(rec, indent=2), flush=True)


if __name__ == "__main__":
    main()
