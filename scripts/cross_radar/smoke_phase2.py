#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from radarmind.data.base import DatasetRegistry
from radarmind.llm.provider import LLMService
from radarmind.protocols import ProtocolRegistry
from radarmind.tests.test_llm_guard import run_all


def main() -> None:
    rows = []
    for name in ("ci4r10", "ci4r24", "ci4r77"):
        ds = DatasetRegistry.load_dataset(name)
        sid = ds.sample_ids(split="test")[0]
        s = ds.get(sid)
        assert s.image.shape == (224, 224)
        assert s.subject_id == "UNKNOWN"
        rows.append({"name": f"load {name}", "ok": True, "detail": f"{sid} {s.action_gt}"})
    p = ProtocolRegistry().get("cross_radar")
    rows.append({"name": "protocol cross_radar", "ok": bool(p["available"]), "detail": p.get("note")})
    runs = ROOT / "results/phase2_cross_radar/training_runs.csv"
    rows.append({"name": "training_runs", "ok": runs.is_file(), "detail": str(runs.exists())})
    online = ROOT / "results/radarmind_v01/llm_online_test.json"
    rec = json.loads(online.read_text(encoding="utf-8")) if online.is_file() else {}
    rows.append({"name": "llm online json", "ok": rec.get("n_ok", 0) == 4, "detail": rec.get("ai_mode")})
    key_leak = "sk-" in online.read_text(encoding="utf-8") if online.is_file() else False
    rows.append({"name": "no key in online json", "ok": not key_leak, "detail": "clean"})
    rows.extend(run_all())
    svc = LLMService()
    rows.append({"name": "llm mode", "ok": True, "detail": svc.ui_mode()})
    n_ok = sum(1 for r in rows if r["ok"])
    print(f"passed {n_ok}/{len(rows)}")
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["name"], r["detail"])
    raise SystemExit(0 if n_ok == len(rows) else 1)


if __name__ == "__main__":
    main()
