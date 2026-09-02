from radarmind.schema import UNKNOWN
from services.demo_registry import default_demo
from services.intelligence_service import get_intelligence
from services.runtime import session_clear


def test_motion_intelligence_provenance():
    session_clear()
    spec = default_demo()
    packed = get_intelligence(spec["sample_id"])
    fields = packed["fields"]
    assert fields["activity"]["value"] == "boxing"
    assert fields["activity"]["source"] == "MODEL"
    assert fields["confidence"]["source"] == "MODEL"
    assert fields["uncertainty"]["source"] == "MODEL"
    assert fields["mean_velocity"]["source"] == "PHYSICS_DERIVED"
    assert fields["peak_velocity"]["source"] == "PHYSICS_DERIVED"
    assert fields["motion_intensity"]["source"] == "PHYSICS_DERIVED"
    assert fields["direction"]["value"] == UNKNOWN
    assert fields["direction"]["source"] == "UNKNOWN"
    assert fields["periodicity"]["value"] == UNKNOWN
    assert fields["subject"]["value"] == UNKNOWN
    assert fields["environment"]["value"] == UNKNOWN
    unk_keys = {x["key"] for x in packed["groups"]["UNKNOWN"]}
    assert "direction" in unk_keys
    assert fields["direction"]["value"] != 0
    assert fields["periodicity"]["value"] != 0
    assert isinstance(fields["peak_velocity"]["value"], float)
    assert fields["peak_velocity"]["unit"] == "m/s"
