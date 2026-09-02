"""RadarMind v0.1 smoke tests. Writes results/radarmind_v01/smoke_test_report.md."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_SCRIPTS = str(ROOT / "scripts")
if _SCRIPTS not in sys.path:
    sys.path.insert(0, _SCRIPTS)

from radarmind.paths import V01


def _run(name, fn, rows):
    try:
        detail = fn() or "ok"
        rows.append({"name": name, "ok": True, "detail": str(detail)})
    except Exception as exc:  # noqa: BLE001
        rows.append({"name": name, "ok": False, "detail": f"{exc}\n{traceback.format_exc()}"})


def test_ci4r_load():
    from radarmind.data.base import DatasetRegistry

    ds = DatasetRegistry.load_dataset("ci4r")
    ids = ds.sample_ids(split="test", protocol="within_distribution")
    s = ds.get(ids[0])
    assert s.image is not None and s.image.shape == (224, 224)
    assert s.representation == "micro_doppler"
    return f"{s.sample_id} {s.image.shape} gt={s.action_gt}"


def test_mri_load():
    from radarmind.data.base import DatasetRegistry

    ds = DatasetRegistry.load_dataset("mri")
    ids = ds.sample_ids(split="test", protocol="subject_independent")
    s = ds.get(ids[0])
    assert s.points is not None and s.points.ndim == 3
    assert s.representation == "point_cloud"
    return f"{s.sample_id} {s.points.shape} subject={s.subject_id}"


def test_radhar_load():
    from radarmind.data.base import DatasetRegistry

    ds = DatasetRegistry.load_dataset("radhar")
    ids = ds.sample_ids(split="test", protocol="official_split")
    s = ds.get(ids[0])
    assert s.points is not None
    assert s.subject_id == "UNKNOWN"
    return f"{s.sample_id} {s.points.shape}"


def test_empty_frame():
    import torch
    from models.point_encoder import PointNetGRU

    net = PointNetGRU(5, in_ch=4)
    x = torch.zeros(1, 4, 8, 4)
    mask = torch.zeros(1, 4, 8, dtype=torch.bool)
    y = net(x, mask)
    assert torch.isfinite(y).all()
    return f"logits={tuple(y.shape)} finite"


def test_checkpoint_and_infer():
    from radarmind.data.base import DatasetRegistry
    from radarmind.models.registry import ModelRegistry

    reg = ModelRegistry()
    loaded = []
    ds = DatasetRegistry.load_dataset("ci4r")
    s = ds.get(ds.sample_ids(split="test", protocol="within_distribution")[0])
    p = reg.predict(s, model_name="ci4r_resnet18")
    assert p.activity
    assert isinstance(p.prediction_entropy, float)
    loaded.append(f"ci4r_resnet18 → {p.activity} p={p.confidence:.3f}")

    ds = DatasetRegistry.load_dataset("mri")
    s = ds.get(ds.sample_ids(split="test", protocol="subject_independent")[0])
    p = reg.predict(s, model_name="mri_pointnet_gru")
    loaded.append(f"mri_pointnet_gru → {p.activity} p={p.confidence:.3f}")

    ds = DatasetRegistry.load_dataset("radhar")
    s = ds.get(ds.sample_ids(split="test", protocol="official_split")[0])
    p = reg.predict(s, model_name="radhar_pointnet_gru")
    loaded.append(f"radhar_pointnet_gru → {p.activity} p={p.confidence:.3f}")
    return "; ".join(loaded)


def test_motion_intelligence():
    from radarmind.data.base import DatasetRegistry
    from radarmind.engine import RadarMindEngine

    eng = RadarMindEngine()
    ds = DatasetRegistry.load_dataset("radhar")
    s = ds.get(ds.sample_ids(split="test", protocol="official_split")[0])
    mi = eng.intelligence(s)
    d = mi.to_dict()
    assert d["activity"]["source"] in ("MODEL", "UNKNOWN")
    assert d["mean_velocity"]["source"] in ("PHYSICS_DERIVED", "UNKNOWN")
    assert d["direction"]["value"] == "UNKNOWN"
    assert d["subject"]["value"] == "UNKNOWN"
    return f"vel={d['mean_velocity']['value']} dir={d['direction']['value']}"


def test_timeline():
    from radarmind.intelligence.timeline import build_timeline

    rows = [
        {"sample_id": "a", "activity": "walk", "confidence": 0.9, "start_time": 0, "end_time": 1, "motion_intelligence": {}},
        {"sample_id": "b", "activity": "walk", "confidence": 0.8, "start_time": 1, "end_time": 2, "motion_intelligence": {}},
        {"sample_id": "c", "activity": "walk", "confidence": 0.7, "start_time": 2, "end_time": 3, "motion_intelligence": {}},
        {"sample_id": "d", "activity": "stop", "confidence": 0.6, "start_time": 3, "end_time": 4, "motion_intelligence": {}},
        {"sample_id": "e", "activity": "stop", "confidence": 0.65, "start_time": 4, "end_time": 5, "motion_intelligence": {}},
        {"sample_id": "f", "activity": "bend", "confidence": 0.7, "start_time": 5, "end_time": 6, "motion_intelligence": {}},
    ]
    tl = build_timeline("demo", "seq", "official_split", "test", rows)
    acts = [e.activity for e in tl.events]
    assert acts == ["walk", "stop", "bend"], acts
    assert abs(tl.events[0].end_time - 3) < 1e-6
    return f"events={acts}"


def test_llm_fallback():
    from radarmind.llm.provider import LLMService

    svc = LLMService()
    intel = {
        "activity": {"value": "walk", "source": "MODEL"},
        "confidence": {"value": 0.32, "source": "MODEL"},
        "uncertainty": {"value": 0.8, "source": "MODEL"},
        "direction": {"value": "UNKNOWN", "source": "UNKNOWN"},
    }
    tl = {"events": [{"start_time": 0, "end_time": 3, "activity": "walk", "confidence": 0.32, "low_confidence": True}]}
    out = svc.explain("current", intelligence=intel, timeline=tl)
    assert "置信度较低" in out["text"] or "可能" in out["text"]
    assert out["backend"] in ("fallback", "deepseek")
    mode = svc.ui_mode()
    assert mode in ("DeepSeek Online", "Template Fallback"), mode
    return f"backend={out['backend']} ui_mode={mode}"


def test_app_backend():
    from radarmind.engine import RadarMindEngine

    eng = RadarMindEngine()
    ds = eng.load_dataset("ci4r")
    sid = ds.sample_ids(split="test", protocol="within_distribution")[0]
    sample = eng.load_sample("ci4r", sid)
    pred = eng.infer(sample)
    mi = eng.intelligence(sample, pred)
    st = eng.replay_load(dataset="ci4r", sample_id=sid, protocol="within_distribution")
    assert st.prediction is not None
    text = eng.explain(timeline=st.timeline, intelligence=st.intelligence, task="summary")
    assert text["text"]
    return f"replay windows={st.n_windows} pred={pred.activity} mi_src={mi.activity.source}"


def main() -> list[dict]:
    rows = []
    _run("CI4R sample loading", test_ci4r_load, rows)
    _run("mRI sample loading", test_mri_load, rows)
    _run("RadHAR sample loading", test_radhar_load, rows)
    _run("PointNet empty-frame handling", test_empty_frame, rows)
    _run("model checkpoint loading + infer", test_checkpoint_and_infer, rows)
    _run("MotionIntelligence generation", test_motion_intelligence, rows)
    _run("Timeline generation", test_timeline, rows)
    _run("LLM fallback", test_llm_fallback, rows)
    _run("Application backend", test_app_backend, rows)
    V01.mkdir(parents=True, exist_ok=True)
    n_ok = sum(1 for r in rows if r["ok"])
    lines = ["# RadarMind v0.1 smoke test", "", f"passed {n_ok}/{len(rows)}", ""]
    for r in rows:
        mark = "PASS" if r["ok"] else "FAIL"
        lines.append(f"## {mark}  {r['name']}")
        lines.append("")
        lines.append("```")
        lines.append(r["detail"][:4000])
        lines.append("```")
        lines.append("")
    (V01 / "smoke_test_report.md").write_text("\n".join(lines), encoding="utf-8")
    return rows


if __name__ == "__main__":
    rows = main()
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["name"])
        if not r["ok"]:
            print(r["detail"][:800])
    raise SystemExit(0 if all(r["ok"] for r in rows) else 1)
