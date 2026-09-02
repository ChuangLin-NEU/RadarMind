#!/usr/bin/env python3
"""LLM summaries from Timeline + series stats only. Never raw radar. pose_* stay as-is."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radarmind.llm.provider import LLMService  # noqa: E402
from scripts.continuous.common import OUT  # noqa: E402


def _series_stats(series: list[dict]) -> dict:
    confs = [float(s["confidence"]) for s in series]
    uncs = [float(s.get("uncertainty") or 0) for s in series]
    vels = [s.get("mean_velocity") for s in series if isinstance(s.get("mean_velocity"), (int, float))]
    peak = None
    peak_t = None
    if vels:
        i = max(range(len(series)), key=lambda k: abs(series[k]["mean_velocity"]) if isinstance(series[k].get("mean_velocity"), (int, float)) else -1)
        peak = series[i].get("mean_velocity")
        peak_t = series[i].get("time")
    low = min(range(len(series)), key=lambda k: series[k]["confidence"]) if series else None
    return {
        "n_windows": len(series),
        "mean_confidence": sum(confs) / len(confs) if confs else None,
        "min_confidence": min(confs) if confs else None,
        "max_uncertainty": max(uncs) if uncs else None,
        "min_confidence_time": series[low]["time"] if low is not None else None,
        "min_confidence_activity": series[low]["activity"] if low is not None else None,
        "peak_abs_velocity": peak,
        "peak_velocity_time": peak_t,
        "note": "summary statistics only; not raw point cloud",
    }


def main() -> None:
    dest = OUT / "llm_summaries"
    dest.mkdir(exist_ok=True)
    svc = LLMService()
    demos = json.loads((OUT / "demo_registry.json").read_text(encoding="utf-8"))["demos"]
    questions = [
        "刚才发生了什么？",
        "动作什么时候发生变化？",
        "哪个阶段模型最不确定？",
        "最大速度出现在哪里？",
        "是否检测到 walk？",
    ]
    for d in demos:
        pred = json.loads((OUT / "predicted_timelines" / f"{d['recording']}.json").read_text(encoding="utf-8"))
        mode = pred["modes"]["confidence_aware"]
        tl = {
            "dataset": "mri",
            "sequence_id": d["recording"],
            "protocol": "subject_independent",
            "model_name": "mri_pointnet_gru",
            "events": mode["events"],
            "smoothing": pred.get("smoothing"),
        }
        intel = {
            "activity": {"value": mode["windows"][0]["activity"] if mode["windows"] else "UNKNOWN", "source": "MODEL"},
            "confidence": {"value": mode["windows"][0]["confidence"] if mode["windows"] else 0, "source": "MODEL"},
            "uncertainty": {"value": mode["windows"][0]["uncertainty"] if mode["windows"] else 1, "source": "MODEL"},
            "direction": {"value": "UNKNOWN", "source": "UNKNOWN"},
            "series_stats": _series_stats(mode["series"]),
        }
        summary = svc.explain("summary", intelligence=intel, timeline=tl)
        qa = {}
        for q in questions:
            qa[q] = svc.explain("qa", intelligence=intel, timeline=tl, question=q)
        rec = {
            "recording": d["recording"],
            "subject": d["subject"],
            "ai_mode": svc.ui_mode(),
            "backend": summary.get("backend"),
            "summary": summary.get("text"),
            "qa": {k: v.get("text") for k, v in qa.items()},
            "guard": {
                "pose_not_paraphrased": True,
                "note": "prompt forbids translating pose_1..pose_10 into other pose names",
            },
        }
        raw = json.dumps(rec, ensure_ascii=False)
        rec["guard"]["no_api_key_in_file"] = "sk-" not in raw
        (dest / f"{d['recording']}.json").write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
        print(d["recording"], svc.ui_mode(), flush=True)


if __name__ == "__main__":
    main()
