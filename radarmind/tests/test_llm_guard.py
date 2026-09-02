"""Hallucination guards: LLM / fallback must not invent facts absent from JSON."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from radarmind.llm.fallback import render
from radarmind.llm.prompt import SYSTEM_PROMPT, build_user_payload

BANNED_IF_ABSENT = ("弯腰", "拾取", "kneel", "crawl", "sit", "坐下", "爬")


def _walk_stop():
    intel = {
        "activity": {"value": "walk", "source": "MODEL"},
        "confidence": {"value": 0.82, "source": "MODEL"},
        "uncertainty": {"value": 0.3, "source": "MODEL"},
        "direction": {"value": "UNKNOWN", "source": "UNKNOWN"},
        "mean_velocity": {"value": "UNKNOWN", "source": "UNKNOWN"},
        "subject": {"value": "UNKNOWN", "source": "UNKNOWN"},
    }
    tl = {
        "events": [
            {"start_time": 0, "end_time": 3, "activity": "walk", "confidence": 0.82, "low_confidence": False},
            {"start_time": 3, "end_time": 5, "activity": "stop", "confidence": 0.71, "low_confidence": False},
        ]
    }
    return intel, tl


def test_summary_does_not_invent_bend():
    intel, tl = _walk_stop()
    text = render("summary", intel, tl)
    for w in ("弯腰", "拾取", "bend", "pick"):
        assert w not in text.lower() and w not in text, text
    assert "walk" in text and "stop" in text
    return "no invented bend/pick"


def test_low_confidence_hedge():
    intel = {
        "activity": {"value": "boxing", "source": "MODEL"},
        "confidence": {"value": 0.28, "source": "MODEL"},
        "uncertainty": {"value": 0.81, "source": "MODEL"},
    }
    text = render("current", intel, None)
    assert ("可能" in text) or ("置信度较低" in text), text
    return text[:80]


def test_unknown_velocity_not_invented():
    intel = {
        "activity": {"value": "walk", "source": "MODEL"},
        "confidence": {"value": 0.9, "source": "MODEL"},
        "uncertainty": {"value": 0.2, "source": "MODEL"},
        "mean_velocity": {"value": "UNKNOWN", "source": "UNKNOWN"},
    }
    text = render("uncertainty", intel, None)
    assert "UNKNOWN" in text or "没有" in text
    assert "1.2 m/s" not in text
    return "velocity stays unknown"


def test_payload_has_no_raw_radar():
    intel, tl = _walk_stop()
    user = build_user_payload("summary", intel, tl)
    for k in ("raw IQ", "point_cloud", "spectrogram image", "tensor"):
        assert k not in user
    assert "motion_intelligence" in user and "motion_timeline" in user
    return "payload is JSON-only"


def test_system_prompt_forbids_extra_actions():
    assert "不得描述 JSON 里没有出现的动作" in SYSTEM_PROMPT
    assert "不得把 raw IQ" in SYSTEM_PROMPT
    return "system prompt present"


def test_unknown_motion_not_called_walk():
    intel = {
        "activity": {"value": "UNKNOWN_MOTION", "source": "MODEL"},
        "confidence": {"value": 0.31, "source": "MODEL"},
        "uncertainty": {"value": 0.4, "source": "MODEL"},
        "evidence": {"nearest_known_class": "walk"},
        "mean_velocity": {"value": 0.4, "source": "PHYSICS_DERIVED"},
    }
    text = render("current", intel, None)
    assert "正在走路" not in text
    assert "UNKNOWN" in text or "无法可靠归入" in text or "无法归入" in text
    assert "walk" in text
    assert "走路" not in text
    return text[:120]


def test_qa_subject_unknown():
    intel, tl = _walk_stop()
    text = render("qa", intel, tl, "这个人是谁？朝哪个方向走？")
    assert "没有" in text or "UNKNOWN" in text
    return text[:100]


def run_all() -> list[dict]:
    rows = []
    for fn in (
        test_summary_does_not_invent_bend,
        test_low_confidence_hedge,
        test_unknown_velocity_not_invented,
        test_payload_has_no_raw_radar,
        test_system_prompt_forbids_extra_actions,
        test_unknown_motion_not_called_walk,
        test_qa_subject_unknown,
    ):
        try:
            rows.append({"name": fn.__name__, "ok": True, "detail": fn()})
        except Exception as exc:  # noqa: BLE001
            rows.append({"name": fn.__name__, "ok": False, "detail": str(exc)})
    return rows


if __name__ == "__main__":
    rows = run_all()
    for r in rows:
        print(("PASS" if r["ok"] else "FAIL"), r["name"], r["detail"])
    raise SystemExit(0 if all(r["ok"] for r in rows) else 1)
