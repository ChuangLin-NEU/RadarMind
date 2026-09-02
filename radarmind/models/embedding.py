"""MotionEmbedding returned by CI4R encoders."""
from __future__ import annotations

from dataclasses import dataclass

from radarmind.schema import UNKNOWN, jsonable


@dataclass
class MotionEmbedding:
    vector: list
    radar_frequency: str
    action: str
    acquisition_id: str
    split: str
    model: str
    normalization: str = "zscore_per_image"

    def to_dict(self) -> dict:
        return jsonable(self)
