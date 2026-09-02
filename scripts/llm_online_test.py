#!/usr/bin/env python3
"""Call DeepSeek on Demo 5 JSON only. Never write the API key."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radarmind.llm.envload import load_dotenv
from radarmind.llm.provider import LLMService
from radarmind.paths import V01

load_dotenv()


def _strip_intel(intel: dict) -> dict:
    ev = dict(intel.get("evidence") or {})
    ev.pop("class_probabilities", None)
    out = {k: v for k, v in intel.items() if k != "evidence"}
    out["evidence"] = {k: ev[k] for k in ev if k in ("sample_id", "representation", "action_gt", "model_name", "protocol", "kinematics")}
    return out


def main() -> None:
    tl = json.loads((V01 / "demo_timelines" / "demo_5.json").read_text(encoding="utf-8"))
    examples = json.loads((V01 / "motion_intelligence_examples.json").read_text(encoding="utf-8"))
    intel = None
    for e in examples:
        if e.get("demo_id") == "demo_5":
            intel = _strip_intel(e["motion_intelligence"])
            break
    slim_tl = {
        "dataset": tl.get("dataset"),
        "sequence_id": tl.get("sequence_id"),
        "protocol": tl.get("protocol"),
        "model_name": tl.get("model_name"),
        "events": [
            {
                "start_time": e.get("start_time"),
                "end_time": e.get("end_time"),
                "activity": e.get("activity"),
                "confidence": e.get("confidence"),
                "low_confidence": e.get("low_confidence"),
                "n_windows": e.get("n_windows"),
            }
            for e in tl.get("events", [])
        ],
        "smoothing": tl.get("smoothing"),
    }
    svc = LLMService()
    tasks = [
        ("current", None),
        ("summary", None),
        ("uncertainty", None),
        ("qa", "时间线里有没有弯腰拾取？速度是多少？"),
    ]
    records = []
    for task, q in tasks:
        out = svc.explain(task, intelligence=intel, timeline=slim_tl, question=q)
        records.append(
            {
                "task": task,
                "question": q,
                "input": {"motion_intelligence": intel, "motion_timeline": slim_tl},
                "model": out.get("model"),
                "backend": out.get("backend"),
                "success": bool(out.get("ok")),
                "latency_s": out.get("latency_s"),
                "response": out.get("text"),
                "error": out.get("error"),
            }
        )
        print(task, out.get("backend"), out.get("ok"), flush=True)
    dest = V01 / "llm_online_test.json"
    dest.write_text(
        json.dumps(
            {
                "demo_id": "demo_5",
                "ai_mode": svc.ui_mode(),
                "has_key": svc.has_key(),
                "n": len(records),
                "n_ok": sum(1 for r in records if r["success"] and r["backend"] == "deepseek"),
                "records": records,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("wrote", dest, flush=True)


if __name__ == "__main__":
    main()
