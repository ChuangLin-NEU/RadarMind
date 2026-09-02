"""Fixed muted activity colors drawn only from visual_style palette."""
from __future__ import annotations

from paper.visual_style import AMBER, BLUE, DARK_GRAY, GREEN, NAVY, PURPLE, TEAL

ACTIVITY_COLOR = {
    "T pose": NAVY,
    "pose_1": BLUE,
    "pose_2": TEAL,
    "pose_3": GREEN,
    "pose_4": AMBER,
    "pose_5": PURPLE,
    "pose_6": DARK_GRAY,
    "walk": TEAL,
    "boxing": NAVY,
    "jack": BLUE,
    "jump": GREEN,
    "squat": AMBER,
    "wave": PURPLE,
}

FALLBACK = [NAVY, BLUE, TEAL, GREEN, AMBER, PURPLE, DARK_GRAY]


def map_activities(names: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    extra_i = 0
    for name in names:
        if not name or name == "UNLABELED":
            continue
        if name in ACTIVITY_COLOR:
            out[name] = ACTIVITY_COLOR[name]
        elif name not in out:
            out[name] = FALLBACK[extra_i % len(FALLBACK)]
            extra_i += 1
    return out
