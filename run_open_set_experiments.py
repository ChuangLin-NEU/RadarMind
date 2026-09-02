#!/usr/bin/env python3
"""Phase 5: select 3 LOAO folds, train, eval. If fast, run remaining 10."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = sys.executable
OUT = ROOT / "results" / "phase5_open_set"


def run(args: list[str]) -> None:
    print("+", " ".join(args), flush=True)
    r = subprocess.run(args, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def selected_actions() -> list[str]:
    p = OUT / "selected_open_set_actions.json"
    data = json.loads(p.read_text(encoding="utf-8"))
    return [s["action"] for s in data["selected"]]


def main() -> None:
    run([PY, str(ROOT / "scripts" / "open_set" / "select_actions.py")])
    first = selected_actions()
    t0 = time.time()
    run([PY, str(ROOT / "scripts" / "open_set" / "train_loao.py"), "--actions", *first])
    dt = time.time() - t0
    per = dt / max(len(first), 1)
    print(f"[timing] first {len(first)} folds {dt:.1f}s ({per:.1f}s each)", flush=True)
    from scripts.pointcloud.mri import CLASS_NAMES

    rest = [c for c in CLASS_NAMES if c not in first]
    if per < 240:
        print("[auto] first folds were fast; training remaining LOAO folds", flush=True)
        run([PY, str(ROOT / "scripts" / "open_set" / "train_loao.py"), "--actions", *rest])
        all_acts = first + rest
        (OUT / "full_loao.json").write_text(json.dumps({"full_loao": True, "actions": all_acts}, indent=2), encoding="utf-8")
    else:
        all_acts = first
        (OUT / "full_loao.json").write_text(json.dumps({"full_loao": False, "actions": all_acts, "reason": f"per-fold {per:.1f}s >= 240s"}, indent=2), encoding="utf-8")
    run([PY, str(ROOT / "scripts" / "open_set" / "eval_openset.py"), "--actions", *all_acts])
    run([PY, str(ROOT / "scripts" / "open_set" / "postprocess.py")])
    run([PY, str(ROOT / "scripts" / "open_set" / "figures.py")])
    print("[open-set-pipeline-done]", flush=True)


if __name__ == "__main__":
    main()
