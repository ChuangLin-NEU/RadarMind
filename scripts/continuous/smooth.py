#!/usr/bin/env python3
"""Three smoothing versions. Parameters chosen on validation subjects only."""
from __future__ import annotations


def raw(labels: list[str], confs: list[float]) -> list[str]:
    return list(labels)


def isolated_flip(labels: list[str], confs: list[float]) -> list[str]:
    out = list(labels)
    for i in range(1, len(out) - 1):
        if out[i] != out[i - 1] and out[i] != out[i + 1] and out[i - 1] == out[i + 1]:
            out[i] = out[i - 1]
    return out


def confidence_aware(labels: list[str], confs: list[float], conf_min: float = 0.35) -> list[str]:
    n = len(labels)
    if n == 0:
        return []
    held = list(labels)
    for i in range(n):
        if confs[i] < conf_min:
            if 0 < i < n - 1 and labels[i - 1] == labels[i + 1]:
                held[i] = labels[i - 1]
            elif i > 0:
                held[i] = held[i - 1]
    return isolated_flip(held, confs)


def apply(name: str, labels: list[str], confs: list[float], conf_min: float = 0.35) -> list[str]:
    if name == "raw":
        return raw(labels, confs)
    if name == "isolated_flip":
        return isolated_flip(labels, confs)
    return confidence_aware(labels, confs, conf_min=conf_min)
