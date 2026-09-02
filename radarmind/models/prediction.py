"""Unified model output. Entropy is computed from softmax when available."""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from radarmind.schema import UNKNOWN, jsonable


@dataclass
class MotionPrediction:
    activity: str
    confidence: float
    class_probabilities: dict
    embedding: list | str
    model_name: str
    dataset: str
    protocol: str
    prediction_entropy: float | str = UNKNOWN
    uncertainty: float | str = UNKNOWN
    logits: list | str = UNKNOWN
    sample_id: str = UNKNOWN
    open_set_status: str = ""
    open_set_score: float | str = UNKNOWN
    open_set_method: str = ""
    threshold: float | str = UNKNOWN
    nearest_known_class: str = UNKNOWN

    def to_dict(self) -> dict:
        return jsonable(self)


def from_logits(logits, class_names: list[str], model_name: str, dataset: str, protocol: str, sample_id: str, embedding=None) -> MotionPrediction:
    import numpy as np

    z = np.asarray(logits, dtype=np.float64).reshape(-1)
    z = z - z.max()
    exp = np.exp(z)
    prob = exp / exp.sum()
    k = int(prob.argmax())
    conf = float(prob[k])
    entropy = float(-(prob * np.log(np.clip(prob, 1e-12, 1.0))).sum())
    n = max(len(class_names), 1)
    unc = float(entropy / math.log(n)) if n > 1 else 0.0
    names = list(class_names)
    if len(names) < len(prob):
        names = names + [f"class_{i}" for i in range(len(names), len(prob))]
    probs = {names[i]: float(prob[i]) for i in range(len(prob))}
    emb: list | str = UNKNOWN
    if embedding is not None:
        emb = [float(x) for x in np.asarray(embedding, dtype=np.float64).reshape(-1).tolist()]
    return MotionPrediction(
        activity=names[k] if 0 <= k < len(names) else UNKNOWN,
        confidence=conf,
        class_probabilities=probs,
        embedding=emb,
        model_name=model_name,
        dataset=dataset,
        protocol=protocol,
        prediction_entropy=entropy,
        uncertainty=unc,
        logits=[float(x) for x in np.asarray(logits, dtype=np.float64).reshape(-1).tolist()],
        sample_id=sample_id,
    )
