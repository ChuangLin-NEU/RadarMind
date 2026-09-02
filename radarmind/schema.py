"""Shared field tagging. Never invent a value to look complete."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

UNKNOWN = "UNKNOWN"
Source = Literal["MODEL", "PHYSICS_DERIVED", "METADATA", "UNKNOWN"]


@dataclass
class Tagged:
    value: Any = UNKNOWN
    source: Source = "UNKNOWN"
    unit: str = UNKNOWN
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def tagged(value: Any, source: Source, unit: str = UNKNOWN, note: str = "") -> Tagged:
    if value is None:
        return Tagged(UNKNOWN, "UNKNOWN", unit, note)
    return Tagged(value, source, unit, note)


def unknown(note: str = "", unit: str = UNKNOWN) -> Tagged:
    return Tagged(UNKNOWN, "UNKNOWN", unit, note)


def jsonable(obj: Any) -> Any:
    if isinstance(obj, Tagged):
        return obj.to_dict()
    if dataclasses_is(obj):
        return {k: jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [jsonable(v) for v in obj]
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:
            return str(obj)
    return obj


def dataclasses_is(obj: Any) -> bool:
    return hasattr(obj, "__dataclass_fields__") and not isinstance(obj, type)
