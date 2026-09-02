"""Studio-1 gate: real sample → inference → MI. Stop here if this fails."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from services.demo_registry import default_demo
from services.health_service import check
from services.inference_service import run_inference
from services.intelligence_service import get_intelligence
from services.language_service import build_payload, language_mode
from services.replay_service import get_sample_frame, get_sample_metadata
from services.result_service import radhar_context
from services.runtime import session_clear


def main() -> int:
    session_clear()
    spec = default_demo()
    sid = spec["sample_id"]
    print("demo", sid)
    health = check()
    print("health", json.dumps(health, ensure_ascii=False))
    if not health["ready"]:
        print("FAIL health")
        return 1
    meta = get_sample_metadata(sid)
    print("meta", meta["shape"], meta["ground_truth"], meta["channel_code"], meta["n_frames"])
    fr = get_sample_frame(sid, 0)
    print("frame0", fr["n_points"], "axes", fr["axes"])
    if fr["n_points"] <= 0:
        print("FAIL empty first frame after mask")
        return 1
    pred = run_inference(sid)
    print("pred", pred["activity"], pred["confidence"], pred["uncertainty"], pred["correct"])
    intel = get_intelligence(sid)
    groups = {k: [(x["key"], x["value"], x["source"]) for x in v] for k, v in intel["groups"].items()}
    print("mi_groups", json.dumps(groups, ensure_ascii=False, default=str))
    payload = build_payload(intel["fields"], task="current")
    dumped = json.dumps(payload["provided"])
    leaked = any(x in dumped.lower() for x in ("point_cloud", "embedding", "logits", '"xyz"'))
    print("language_mode", language_mode(), "leaked", leaked)
    ctx = radhar_context()
    print("radhar_acc", ctx["accuracy"]["display"], ctx["macro_f1"]["display"])
    if pred["activity"] != "boxing":
        print("FAIL prediction", pred["activity"])
        return 1
    peak = next(x for x in intel["groups"]["PHYSICS_DERIVED"] if x["key"] == "peak_velocity")
    if peak["value"] == "UNKNOWN":
        print("FAIL peak velocity UNKNOWN")
        return 1
    unk = [x["key"] for x in intel["groups"]["UNKNOWN"]]
    if "direction" not in unk:
        print("FAIL direction not UNKNOWN")
        return 1
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
