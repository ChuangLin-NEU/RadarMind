from services.demo_registry import default_demo
from services.inference_service import run_inference
from services.runtime import session_clear


def test_live_demo_inference_boxing():
    session_clear()
    spec = default_demo()
    pred = run_inference(spec["sample_id"])
    assert pred["sample_id"] == spec["sample_id"]
    assert pred["activity"] == "boxing"
    assert pred["ground_truth"] == "boxing"
    assert pred["correct"] is True
    assert pred["execution_mode"] == "LIVE MODEL INFERENCE"
    assert isinstance(pred["confidence"], float)
    assert pred["confidence"] > 0.9
    assert isinstance(pred["uncertainty"], float)
    assert pred["model_name"] == "radhar_pointnet_gru"
    assert pred["checkpoint_alias"] == "radhar_pointnet_gru"
    assert "embedding" not in pred
    assert "logits" not in pred
    assert pred["top_probabilities"][0]["activity"] == "boxing"
    assert abs(pred["top_probabilities"][0]["probability"] - pred["confidence"]) < 1e-9
