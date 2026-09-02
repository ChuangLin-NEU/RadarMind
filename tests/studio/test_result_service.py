from services.result_service import SNAPSHOT, get_metric, radhar_context, snapshot_ready


def test_full_snapshot_is_211():
    assert SNAPSHOT.is_file()
    assert snapshot_ready()
    acc = get_metric("RADHAR_XYZV_ACC")
    f1 = get_metric("RADHAR_XYZV_F1")
    assert acc["scope"] == "dataset_level"
    assert acc["display"] == "0.949"
    assert f1["display"] == "0.949"
    assert abs(float(acc["value"]) - 0.9492286115007013) < 1e-9
    assert "phase1_pointcloud" in acc["source_file"]
    ctx = radhar_context()
    assert ctx["label"] == "Dataset-level metric"
    assert "Not this sample" in ctx["note"]
