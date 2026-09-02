#!/usr/bin/env python3
"""RadarMind v0.2 smoke. Writes results/radarmind_v02/smoke_test_report.md."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

V02 = ROOT / "results" / "radarmind_v02"


def _run(name, fn, rows):
    try:
        detail = fn() or "ok"
        rows.append({"name": name, "ok": True, "detail": str(detail)})
    except Exception as exc:  # noqa: BLE001
        rows.append({"name": name, "ok": False, "detail": f"{exc}\n{traceback.format_exc()}"})


def test_ci4r():
    from radarmind.tests.smoke import test_ci4r_load

    return test_ci4r_load()


def test_mri():
    from radarmind.tests.smoke import test_mri_load

    return test_mri_load()


def test_radhar():
    from radarmind.tests.smoke import test_radhar_load

    return test_radhar_load()


def test_registry():
    from radarmind.models.registry import ModelRegistry

    reg = ModelRegistry()
    names = reg.names()
    assert "mri_pointnet_gru" in names
    return ",".join(names)


def test_mi():
    from radarmind.tests.smoke import test_motion_intelligence

    return test_motion_intelligence()


def test_timeline():
    from radarmind.tests.smoke import test_timeline

    return test_timeline()


def test_llm():
    from radarmind.tests.smoke import test_llm_fallback
    from radarmind.tests.test_llm_guard import test_unknown_motion_not_called_walk

    a = test_llm_fallback()
    b = test_unknown_motion_not_called_walk()
    return f"{a} | {b}"


def test_cross_radar_page():
    from radarmind.app.streamlit_app import page_cross_radar

    assert callable(page_cross_radar)
    p = ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"
    assert p.is_file()
    return str(p)


def test_shared_page():
    from radarmind.app.streamlit_app import page_shared_motion

    assert callable(page_shared_motion)
    p = ROOT / "results" / "phase3_shared_motion" / "training_runs.csv"
    assert p.is_file()
    return str(p)


def test_continuous_page():
    from radarmind.app.streamlit_app import page_continuous

    assert callable(page_continuous)
    p = ROOT / "results" / "phase4_continuous" / "phase4_continuous_report.md"
    assert p.is_file()
    txt = p.read_text(encoding="utf-8")
    assert "full-stream inference over entire recording" not in txt
    assert "labeled-interval" in txt or "labeled intervals" in txt
    return "terminology ok"


def test_openset_page():
    from radarmind.app.streamlit_app import page_open_set

    assert callable(page_open_set)
    p = ROOT / "results" / "phase5_open_set" / "eval_summary.json"
    assert p.is_file(), "phase5 eval missing"
    return str(p)


def test_offline_llm():
    from radarmind.llm.provider import LLMService

    svc = LLMService(force_offline=True)
    out = svc.explain("current", intelligence={"activity": {"value": "walk", "source": "MODEL"}, "confidence": {"value": 0.9}})
    assert out["backend"] == "fallback"
    assert svc.ui_mode() == "Template Fallback"
    return "offline fallback"


def test_health():
    from radarmind.health import check

    h = check()
    assert h["required_ok"], h
    return f"port={h['port']} items={len(h['items'])}"


def test_all_pages():
    from radarmind.app import streamlit_app as app

    names = [
        "page_home",
        "page_replay",
        "page_continuous",
        "page_open_set",
        "page_generalization",
        "page_ablation",
        "page_cross_radar",
        "page_shared_motion",
        "page_pipeline",
        "page_status",
    ]
    missing = [n for n in names if not callable(getattr(app, n, None))]
    assert not missing, missing
    return ",".join(names)


def test_freeze_files():
    root = ROOT / "results" / "frozen_2026_08"
    need = [
        "artifact_manifest.csv",
        "master_results.csv",
        "scientific_findings.json",
        "negative_results.md",
        "limitations.md",
        "project_status.md",
        "research_backlog.md",
    ]
    miss = [n for n in need if not (root / n).is_file()]
    assert not miss, miss
    return f"{len(need)} files"


def test_app_startup():
    from radarmind.engine import RadarMindEngine
    from radarmind.results_io import research_status

    eng = RadarMindEngine()
    assert callable(eng.llm.ui_mode)
    off = RadarMindEngine(offline_demo=True)
    assert off.llm.force_offline
    rs = research_status()
    assert "DONE" in rs and "PARTIAL" in rs and "PENDING" in rs
    return f"ui_mode={eng.llm.ui_mode()} pages_done={len(rs['DONE'])}"


def main() -> list[dict]:
    rows = []
    _run("CI4R loading", test_ci4r, rows)
    _run("mRI loading", test_mri, rows)
    _run("RadHAR loading", test_radhar, rows)
    _run("model registry", test_registry, rows)
    _run("Motion Intelligence", test_mi, rows)
    _run("Timeline", test_timeline, rows)
    _run("DeepSeek/fallback", test_llm, rows)
    _run("Cross-Radar page", test_cross_radar_page, rows)
    _run("Shared Motion page", test_shared_page, rows)
    _run("Continuous page", test_continuous_page, rows)
    _run("Open-set page", test_openset_page, rows)
    _run("Offline LLM", test_offline_llm, rows)
    _run("Health check", test_health, rows)
    _run("All pages registered", test_all_pages, rows)
    _run("Freeze files", test_freeze_files, rows)
    _run("Application startup", test_app_startup, rows)
    V02.mkdir(parents=True, exist_ok=True)
    n_ok = sum(1 for r in rows if r["ok"])
    lines = ["# RadarMind v0.2 smoke test", "", f"passed {n_ok}/{len(rows)}", ""]
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"## {mark}  {r['name']}")
        lines.append("")
        lines.append("```")
        lines.append(r["detail"][:4000])
        lines.append("```")
        lines.append("")
    (V02 / "smoke_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = main()
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["name"])
        if not r["ok"]:
            print(r["detail"][:800])
    raise SystemExit(0 if all(r["ok"] for r in rows) else 1)
