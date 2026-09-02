"""Startup health check. Missing optional artifacts must not crash the app."""
from __future__ import annotations

import os
from pathlib import Path

from radarmind.paths import (
    CI4R_CACHE,
    CI4R_RESNET_PREFIX,
    CI4R_RESNET_RANDOM,
    MRI_CACHE,
    MRI_GRU_SI_XYZ,
    RADHAR_CACHE,
    RADHAR_GRU_XYZV,
    ROOT,
)

PORT_DEFAULT = 8502


def _ok(path: Path) -> bool:
    return path.is_file()


def check() -> dict:
    key = bool(os.environ.get("DEEPSEEK_API_KEY", "").strip())
    offline = os.environ.get("RADARMIND_OFFLINE_DEMO", "").strip() in ("1", "true", "yes")
    items = [
        {"name": "CI4R cache", "path": str(CI4R_CACHE), "ok": _ok(CI4R_CACHE), "required": True},
        {"name": "mRI cache", "path": str(MRI_CACHE), "ok": _ok(MRI_CACHE), "required": True},
        {"name": "RadHAR cache", "path": str(RADHAR_CACHE), "ok": _ok(RADHAR_CACHE), "required": True},
        {"name": "CI4R random checkpoint", "path": str(CI4R_RESNET_RANDOM), "ok": _ok(CI4R_RESNET_RANDOM), "required": True},
        {"name": "CI4R prefix checkpoint", "path": str(CI4R_RESNET_PREFIX), "ok": _ok(CI4R_RESNET_PREFIX), "required": True},
        {"name": "mRI SI checkpoint", "path": str(MRI_GRU_SI_XYZ), "ok": _ok(MRI_GRU_SI_XYZ), "required": True},
        {"name": "RadHAR checkpoint", "path": str(RADHAR_GRU_XYZV), "ok": _ok(RADHAR_GRU_XYZV), "required": True},
        {"name": "Phase 2 training_runs", "path": str(ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"), "ok": _ok(ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"), "required": False},
        {"name": "Phase 3 training_runs", "path": str(ROOT / "results" / "phase3_shared_motion" / "training_runs.csv"), "ok": _ok(ROOT / "results" / "phase3_shared_motion" / "training_runs.csv"), "required": False},
        {"name": "Phase 4 continuous_runs", "path": str(ROOT / "results" / "phase4_continuous" / "continuous_runs.csv"), "ok": _ok(ROOT / "results" / "phase4_continuous" / "continuous_runs.csv"), "required": False},
        {"name": "Phase 5 eval_summary", "path": str(ROOT / "results" / "phase5_open_set" / "eval_summary.json"), "ok": _ok(ROOT / "results" / "phase5_open_set" / "eval_summary.json"), "required": False},
        {"name": "DeepSeek key", "path": "DEEPSEEK_API_KEY", "ok": key, "required": False},
    ]
    required_ok = all(i["ok"] for i in items if i["required"])
    return {
        "required_ok": required_ok,
        "offline_demo": offline,
        "deepseek_key_present": key,
        "port": int(os.environ.get("RADARMIND_PORT", PORT_DEFAULT)),
        "items": items,
    }
