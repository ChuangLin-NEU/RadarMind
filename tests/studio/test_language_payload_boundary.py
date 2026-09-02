import json

from services.demo_registry import default_demo
from services.intelligence_service import get_intelligence
from services.language_service import _contains_forbidden, build_payload, explain
from services.runtime import session_clear


def test_language_payload_excludes_raw_radar():
    session_clear()
    spec = default_demo()
    packed = get_intelligence(spec["sample_id"])
    payload = build_payload(packed["fields"], task="current")
    provided = payload["provided"]
    assert provided["motion_timeline"] is None
    assert "motion_intelligence" in provided
    hits = _contains_forbidden(provided)
    assert hits == []
    blob = json.dumps(provided)
    assert "embedding" not in blob
    assert "logits" not in blob
    assert "xyz" not in provided["motion_intelligence"]
    assert "excluded" in payload
    assert "raw point cloud" in payload["excluded"]
    assert "hidden neural features" in payload["excluded"]


def test_explain_uses_structured_payload():
    session_clear()
    spec = default_demo()
    out = explain(spec["sample_id"], task="current")
    assert out["text"]
    assert out["language_mode"] in ("ONLINE", "TEMPLATE_FALLBACK", "OFFLINE")
    assert out["payload"]["provided"]["motion_timeline"] is None
    assert _contains_forbidden(out["payload"]["provided"]) == []
