#!/usr/bin/env python3
"""Phase 4 continuous tests."""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "phase4_continuous"


def _run(name, fn, rows):
    try:
        detail = fn() or "ok"
        rows.append({"name": name, "ok": True, "detail": str(detail)})
    except Exception as exc:  # noqa: BLE001
        rows.append({"name": name, "ok": False, "detail": f"{exc}\n{traceback.format_exc()}"})


def test_continuous_loading():
    man = OUT / "mri_recording_manifest.csv"
    assert man.is_file(), man
    import csv

    rows = list(csv.DictReader(man.open(encoding="utf-8")))
    assert len(rows) == 20
    test = [r for r in rows if r["split"] == "test"]
    assert {r["subject"] for r in test} == {"subject3", "subject10", "subject20"}
    assert all(int(r["labeled_segments"]) >= 2 for r in test)
    return f"recordings={len(rows)} test={len(test)}"


def test_unlabeled():
    p = OUT / "ground_truth_timelines" / "subject10_0504.json"
    tl = json.loads(p.read_text(encoding="utf-8"))
    ev = tl["events"]
    assert any(e["activity"] == "UNLABELED" for e in ev)
    # unlabeled not glued onto neighbor
    for i, e in enumerate(ev[:-1]):
        if e["activity"] == "UNLABELED":
            assert ev[i + 1]["start_frame"] == e["end_frame"] + 1
    return f"events={len(ev)} unlabeled={sum(e['activity']=='UNLABELED' for e in ev)}"


def test_full_recording_inference():
    p = OUT / "predicted_timelines" / "subject10_0504.json"
    rec = json.loads(p.read_text(encoding="utf-8"))
    assert rec["split"] == "test"
    wins = rec["modes"]["confidence_aware"]["windows"]
    assert wins
    assert wins == sorted(wins, key=lambda w: w["start_frame"])
    assert rec["window"]["T"] == 32
    return f"windows={len(wins)}"


def test_timeline_generation():
    from radarmind.intelligence.timeline import build_timeline

    tl = build_timeline(
        "mri",
        "demo",
        "subject_independent",
        "mri_pointnet_gru",
        [
            {"activity": "pose_1", "confidence": 0.9, "start_time": 0, "end_time": 1, "motion_intelligence": {}},
            {"activity": "walk", "confidence": 0.2, "start_time": 1, "end_time": 2, "motion_intelligence": {}},
            {"activity": "pose_1", "confidence": 0.9, "start_time": 2, "end_time": 3, "motion_intelligence": {}},
        ],
    )
    acts = [e.activity for e in tl.events]
    assert "pose_1" in acts
    return f"events={acts}"


def test_boundary():
    from scripts.continuous.metrics import boundary_errors

    gt = [
        {"activity": "pose_1", "start_frame": 0, "end_frame": 99},
        {"activity": "walk", "start_frame": 100, "end_frame": 199},
    ]
    pred = [
        {"activity": "pose_1", "start_frame": 0, "end_frame": 104},
        {"activity": "walk", "start_frame": 105, "end_frame": 199},
    ]
    b = boundary_errors(gt, pred, fps=10.0)
    assert b["median_frames"] == 5
    assert abs(b["median_seconds"] - 0.5) < 1e-6
    return str(b["median_frames"])


def test_llm_summary():
    from radarmind.llm.provider import LLMService

    svc = LLMService()
    tl = {
        "events": [
            {"activity": "pose_1", "start_time": 0, "end_time": 3, "confidence": 0.91, "low_confidence": False},
            {"activity": "T pose", "start_time": 3, "end_time": 5, "confidence": 0.42, "low_confidence": True},
        ]
    }
    sm = svc.explain("summary", intelligence={"activity": {"value": "pose_1", "source": "MODEL"}, "confidence": {"value": 0.91, "source": "MODEL"}}, timeline=tl)
    assert "pose_1" in sm["text"]
    assert "弯腰" not in sm["text"] and "拾取" not in sm["text"]
    qa = svc.explain("qa", intelligence={"series_stats": {"peak_abs_velocity": 1.2, "peak_velocity_time": 2.0}}, timeline=tl, question="是否检测到 walk？")
    assert "walk" in qa["text"]
    cur = svc.explain("current", intelligence={"activity": {"value": "pose_1", "source": "MODEL"}, "confidence": {"value": 0.42, "source": "MODEL"}, "uncertainty": {"value": 0.8, "source": "MODEL"}})
    assert "低置信度" in cur["text"] or "置信度较低" in cur["text"]
    return f"backend={sm['backend']} ui={svc.ui_mode()}"


def test_replay_state():
    from radarmind.engine import RadarMindEngine

    recs = list((OUT / "predicted_timelines").glob("*.json"))
    assert recs
    rec = json.loads(recs[0].read_text(encoding="utf-8"))
    if rec.get("split") != "test":
        rec = json.loads(next(p for p in recs if json.loads(p.read_text(encoding="utf-8")).get("split") == "test").read_text(encoding="utf-8"))
    eng = RadarMindEngine()
    st = eng.replay_load(mode="continuous", recording=rec["recording"])
    assert st.prediction is not None
    assert eng.replay.mode == "continuous"
    f0 = eng.replay.cursor_frame
    eng.replay_step("next")
    assert eng.replay.cursor_frame != f0 or len(eng.replay.pred_windows) == 1
    return f"rec={rec['recording']} windows={st.n_windows} cursor={eng.replay.cursor_frame}"


def test_app_backend():
    from radarmind.engine import RadarMindEngine

    eng = RadarMindEngine()
    assert callable(eng.llm.ui_mode)
    assert eng.llm.ui_mode() in ("DeepSeek Online", "Template Fallback")
    proto = eng.protocols.get("cross_environment")
    assert proto["available"] is False
    return f"ui={eng.llm.ui_mode()} cross_env={proto['reason']}"


def main() -> list[dict]:
    rows = []
    _run("continuous loading", test_continuous_loading, rows)
    _run("UNLABELED handling", test_unlabeled, rows)
    _run("full recording inference", test_full_recording_inference, rows)
    _run("timeline generation", test_timeline_generation, rows)
    _run("boundary calculation", test_boundary, rows)
    _run("LLM summary / Q&A / hedge", test_llm_summary, rows)
    _run("Replay state continuous", test_replay_state, rows)
    _run("Streamlit backend / protocol", test_app_backend, rows)
    n_ok = sum(1 for r in rows if r["ok"])
    dest = OUT / "phase4_smoke.md"
    OUT.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "# Phase 4 smoke\n\n" + f"passed {n_ok}/{len(rows)}\n\n" + "\n".join(f"- {'PASS' if r['ok'] else 'FAIL'} {r['name']}: {r['detail'][:400]}" for r in rows),
        encoding="utf-8",
    )
    return rows


if __name__ == "__main__":
    rows = main()
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["name"])
        if not r["ok"]:
            print(r["detail"][:800])
    raise SystemExit(0 if all(r["ok"] for r in rows) else 1)
