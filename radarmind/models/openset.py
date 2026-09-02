"""Post-hoc open-set scores. Thresholds come from validation known only (stored in pack)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from radarmind.models.prediction import MotionPrediction
from radarmind.schema import UNKNOWN

STATUS_KNOWN = "KNOWN"
STATUS_UNKNOWN = "UNKNOWN"
STATUS_UNCERTAIN = "UNCERTAIN"


def energy_known_score(logits, T: float = 1.0) -> float:
    z = np.asarray(logits, dtype=np.float64).reshape(-1) / T
    m = float(z.max())
    return float(T * (m + np.log(np.exp(z - m).sum())))


def msp_score(logits) -> float:
    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - z.max()
    p = np.exp(z)
    p = p / p.sum()
    return float(p.max())


def prototype_score(embedding, prototypes) -> float:
    e = np.asarray(embedding, dtype=np.float64).reshape(-1)
    proto = np.asarray(prototypes, dtype=np.float64)
    d = ((proto - e) ** 2).sum(axis=1)
    return float(-d.min())


def decide_status(score: float, threshold: float, confidence: float) -> str:
    if score < threshold:
        return STATUS_UNKNOWN
    if confidence < 0.5:
        return STATUS_UNCERTAIN
    return STATUS_KNOWN


def apply_open_set(pred: MotionPrediction, pack: dict) -> MotionPrediction:
    method = str(pack.get("open_set_method") or pack.get("method") or "Energy")
    thr = float(pack["threshold"])
    T = float(pack.get("energy_temperature") or 1.0)
    logits = pred.logits if pred.logits != UNKNOWN else None
    emb = pred.embedding if pred.embedding != UNKNOWN else None
    if method == "MSP":
        score = msp_score(logits)
    elif method == "Prototype":
        score = prototype_score(emb, pack["prototypes"])
    else:
        score = energy_known_score(logits, T)
        method = "Energy"
    pred.open_set_method = method
    pred.open_set_score = score
    pred.threshold = thr
    pred.open_set_status = decide_status(score, thr, float(pred.confidence))
    pred.nearest_known_class = pred.activity
    return pred


def display_activity(pred: MotionPrediction) -> str:
    if pred.open_set_status == STATUS_UNKNOWN:
        return "UNKNOWN_MOTION"
    return pred.activity


def load_app_pack(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))
