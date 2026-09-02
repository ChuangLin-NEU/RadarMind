"""MotionIntelligence. Every field is tagged MODEL / PHYSICS_DERIVED / METADATA / UNKNOWN."""
from __future__ import annotations

from dataclasses import dataclass, field

from radarmind.schema import Tagged, jsonable, unknown


@dataclass
class MotionIntelligence:
    activity: Tagged
    confidence: Tagged
    direction: Tagged
    mean_velocity: Tagged
    peak_velocity: Tagged
    motion_intensity: Tagged
    periodicity: Tagged
    duration: Tagged
    motion_state: Tagged
    uncertainty: Tagged
    dataset: Tagged
    radar: Tagged
    subject: Tagged
    environment: Tagged
    evidence: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return jsonable(self)


def empty_intelligence() -> MotionIntelligence:
    return MotionIntelligence(
        activity=unknown("no prediction"),
        confidence=unknown(),
        direction=unknown("toward/away not established"),
        mean_velocity=unknown(),
        peak_velocity=unknown(),
        motion_intensity=unknown(),
        periodicity=unknown("no period estimator in v0.1"),
        duration=unknown(),
        motion_state=unknown(),
        uncertainty=unknown(),
        dataset=unknown(),
        radar=unknown(),
        subject=unknown(),
        environment=unknown("no validated environment metadata"),
        evidence={},
    )
