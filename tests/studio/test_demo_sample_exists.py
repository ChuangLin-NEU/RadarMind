from services.demo_registry import default_demo
from services.replay_service import get_sample_frame, get_sample_metadata


def test_demo_sample_registered():
    spec = default_demo()
    assert spec["sample_id"] == "radhar___20_boxing_6_w0000"
    assert spec["dataset"] == "radhar"
    assert spec["expected_activity"] == "boxing"
    assert "confidence" not in spec
    assert "peak_velocity" not in spec


def test_demo_sample_loads_from_cache():
    spec = default_demo()
    meta = get_sample_metadata(spec["sample_id"])
    assert meta["sample_id"] == spec["sample_id"]
    assert meta["ground_truth"] == "boxing"
    assert meta["representation"] == "point_cloud"
    assert meta["n_frames"] > 0
    assert meta["n_channels"] >= 4
    frame = get_sample_frame(spec["sample_id"], 0)
    assert frame["n_points"] > 0
    assert len(frame["xyz"]) == frame["n_points"]
    assert all(len(p) == 3 for p in frame["xyz"])
