"""Load snapshot metrics and record figure provenance. No hand-filled scalars."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SNAP_PATH = ROOT / "results" / "final_delivery" / "metric_snapshot_full.json"


class MetricAccessError(RuntimeError):
    pass


def load_snapshot(path: Path | None = None) -> dict:
    p = path or SNAP_PATH
    if not p.is_file():
        raise MetricAccessError(f"missing snapshot {p}")
    return json.loads(p.read_text(encoding="utf-8"))


class MetricBook:
    """Every plotted scalar is recorded: figure -> item -> key -> source file/field."""

    def __init__(self, snap: dict):
        self.snap = snap
        self.rows: list[dict[str, Any]] = []

    def _entry(self, key: str) -> dict:
        mets = self.snap.get("metrics") or self.snap
        if key not in mets:
            raise MetricAccessError(f"metric {key} not in snapshot")
        return mets[key]

    def value(self, key: str, figure: str, item: str) -> float:
        e = self._entry(key)
        v = float(e["value"])
        self.rows.append(
            {
                "figure": figure,
                "plotted_item": item,
                "metric_key": key,
                "value": v,
                "display": e.get("display", f"{v:.3f}"),
                "source_file": e.get("source_file", ""),
                "source_field": e.get("source_field") or e.get("source_row", ""),
                "protocol": e.get("protocol", ""),
                "kind": "snapshot",
            }
        )
        return v

    def display(self, key: str, figure: str, item: str) -> str:
        e = self._entry(key)
        self.value(key, figure, item)
        return str(e.get("display", f"{float(e['value']):.3f}"))

    def record_file(self, figure: str, item: str, source_file: str, source_field: str, note: str = "") -> None:
        self.rows.append(
            {
                "figure": figure,
                "plotted_item": item,
                "metric_key": "",
                "value": "",
                "display": note,
                "source_file": source_file,
                "source_field": source_field,
                "protocol": note,
                "kind": "frozen_file",
            }
        )

    def write_markdown(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "# Figure metric provenance",
            "",
            "Every plotted scalar traces `figure → plotted item → metric key → source file → source field`.",
            "Schematic geometry (box positions, arrows) is not a scientific metric.",
            "Hard-coded scientific metrics in plotting code: forbidden.",
            "",
            "| figure | plotted item | metric key | value | display | source file | source field | protocol | kind |",
            "| ------ | ------------ | ---------- | ----- | ------- | ----------- | ------------ | -------- | ---- |",
        ]
        for r in self.rows:
            val = r["value"]
            if val == "":
                vs = ""
            else:
                vs = f"{float(val):.6g}"
            lines.append(
                "| {figure} | {plotted_item} | {metric_key} | {vs} | {display} | {source_file} | {source_field} | {protocol} | {kind} |".format(
                    vs=vs, **{k: str(r.get(k, "")).replace("|", "/") for k in (
                        "figure", "plotted_item", "metric_key", "display",
                        "source_file", "source_field", "protocol", "kind",
                    )}
                )
            )
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
