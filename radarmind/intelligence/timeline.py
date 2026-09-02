"""Window-level predictions → merged MotionEvents. No learned segmenter."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from radarmind.intelligence.schema import MotionIntelligence
from radarmind.schema import jsonable


@dataclass
class MotionEvent:
    start_time: float
    end_time: float
    activity: str
    confidence: float
    motion_intelligence: dict = field(default_factory=dict)
    n_windows: int = 1
    low_confidence: bool = False

    def to_dict(self) -> dict:
        return jsonable(self)


@dataclass
class MotionTimeline:
    dataset: str
    sequence_id: str
    protocol: str
    model_name: str
    events: list
    windows: list
    smoothing: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "protocol": self.protocol,
            "model_name": self.model_name,
            "events": [e.to_dict() if hasattr(e, "to_dict") else e for e in self.events],
            "windows": self.windows,
            "smoothing": self.smoothing,
        }


def _majority(labels: list[str]) -> str:
    counts: dict[str, int] = {}
    for x in labels:
        counts[x] = counts.get(x, 0) + 1
    return max(counts, key=counts.get)


def smooth_labels(labels: list[str], confs: list[float], window: int = 3, conf_min: float = 0.35) -> list[str]:
    """Confidence hold + isolated-flip majority. Do not majority-vote away a real trailing change."""
    n = len(labels)
    if n == 0:
        return []
    held = list(labels)
    for i in range(n):
        if confs[i] < conf_min and i > 0:
            held[i] = held[i - 1]
    out = list(held)
    if window >= 3:
        for i in range(1, n - 1):
            if held[i] != held[i - 1] and held[i] != held[i + 1] and held[i - 1] == held[i + 1]:
                out[i] = held[i - 1]
    return out


def merge_adjacent(times: list[tuple[float, float]], labels: list[str], confs: list[float], intelligences: list[dict]) -> list[MotionEvent]:
    events: list[MotionEvent] = []
    if not labels:
        return events
    cur_lab, t0, t1 = labels[0], times[0][0], times[0][1]
    acc_c, n, last_mi = [confs[0]], 1, intelligences[0]
    for i in range(1, len(labels)):
        if labels[i] == cur_lab:
            t1 = times[i][1]
            acc_c.append(confs[i])
            n += 1
            last_mi = intelligences[i]
        else:
            c = float(sorted(acc_c)[len(acc_c) // 2])
            events.append(MotionEvent(t0, t1, cur_lab, c, last_mi, n, c < 0.35))
            cur_lab, t0, t1 = labels[i], times[i][0], times[i][1]
            acc_c, n, last_mi = [confs[i]], 1, intelligences[i]
    c = float(sorted(acc_c)[len(acc_c) // 2])
    events.append(MotionEvent(t0, t1, cur_lab, c, last_mi, n, c < 0.35))
    return events


def build_timeline(
    dataset: str,
    sequence_id: str,
    protocol: str,
    model_name: str,
    window_rows: Iterable[dict],
    majority_k: int = 3,
    conf_min: float = 0.35,
) -> MotionTimeline:
    rows = list(window_rows)
    labels = [r["activity"] for r in rows]
    confs = [float(r["confidence"]) for r in rows]
    times = [(float(r["start_time"]), float(r["end_time"])) for r in rows]
    intel = [r.get("motion_intelligence", {}) for r in rows]
    smoothed = smooth_labels(labels, confs, window=majority_k, conf_min=conf_min)
    events = merge_adjacent(times, smoothed, confs, intel)
    return MotionTimeline(
        dataset=dataset,
        sequence_id=sequence_id,
        protocol=protocol,
        model_name=model_name,
        events=events,
        windows=rows,
        smoothing={"majority_k": majority_k, "confidence_threshold": conf_min, "merge": "adjacent_same_activity"},
    )
