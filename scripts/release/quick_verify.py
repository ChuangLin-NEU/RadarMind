#!/usr/bin/env python3
"""Light publication checks. No training."""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))


def _ok(name: str, pred: bool, detail: str = "") -> bool:
    mark = "PASS" if pred else "FAIL"
    extra = f"  {detail}" if detail else ""
    print(f"  [{mark}] {name}{extra}")
    return pred


def main() -> int:
    t0 = time.perf_counter()
    print("RADARMIND QUICK VERIFY")
    fails = 0

    fails += not _ok("import radarmind", _try_import("radarmind"))
    fails += not _ok("import paper_v2.registry", _try_import("paper_v2.registry"))
    fails += not _ok("import radarmind.intelligence.schema", _try_import("radarmind.intelligence.schema"))
    fails += not _ok("import radarmind.app.streamlit_app", _try_import("radarmind.app.streamlit_app"))

    from radarmind.intelligence.schema import MotionIntelligence, empty_intelligence
    from radarmind.schema import UNKNOWN

    mi = empty_intelligence()
    fails += not _ok("MotionIntelligence schema", isinstance(mi, MotionIntelligence))
    fails += not _ok("UNKNOWN remains UNKNOWN", mi.periodicity.value == UNKNOWN and mi.periodicity.source == "UNKNOWN")

    split = ROOT / "data_splits" / "mri_subject_independent.csv"
    fails += not _ok("mRI SI split present", split.is_file())
    ci4r_legacy = ROOT / "data_splits" / "ci4r_subject_independent.csv"
    fails += not _ok("CI4R prefix split present (legacy name)", ci4r_legacy.is_file())
    if ci4r_legacy.is_file():
        text = ci4r_legacy.read_text(encoding="utf-8")
        fails += not _ok("CI4R split protocol is prefix_tail4_disjoint", "prefix_tail4_disjoint" in text)

    snap_p = ROOT / "results" / "final_delivery" / "metric_snapshot_full.json"
    fails += not _ok("metric snapshot present", snap_p.is_file())
    if snap_p.is_file():
        snap = json.loads(snap_p.read_text(encoding="utf-8"))
        fails += not _ok("snapshot has MRI_SI_ACC", "MRI_SI_ACC" in snap.get("metrics", {}))

    required = [
        ROOT / "README.md",
        ROOT / "CONTRIBUTORS.md",
        ROOT / "CITATION.cff",
        ROOT / "docs" / "DATASETS.md",
        ROOT / "docs" / "KNOWN_NAMING_ISSUES.md",
        ROOT / "results" / "phase2_cross_radar" / "pairing_stats.json",
        ROOT / "figures" / "main" / "fig01_radarmind_framework.svg",
        ROOT / ".env.example",
    ]
    for p in required:
        fails += not _ok(f"required {p.relative_to(ROOT)}", p.is_file())

    env_ex = (ROOT / ".env.example").read_text(encoding="utf-8")
    secretish = any(tok in env_ex.lower() for tok in ("sk-", "bearer ", "ghp_"))
    fails += not _ok(".env.example has no key material", not secretish)

    print("  [RUN] tests/test_manuscript_metrics.py")
    loader = unittest.defaultTestLoader
    suite = loader.discover(str(ROOT / "tests"), pattern="test_manuscript_metrics.py")
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    fails += not _ok("manuscript metric tests", result.wasSuccessful(), f"failures={len(result.failures)+len(result.errors)}")

    elapsed = time.perf_counter() - t0
    status = "PASS" if fails == 0 else "FAIL"
    print(f"RADARMIND QUICK VERIFY {status}  ({elapsed:.1f}s, issues={fails})")
    return 0 if fails == 0 else 1


def _try_import(name: str) -> bool:
    try:
        __import__(name)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"         import error: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    raise SystemExit(main())
