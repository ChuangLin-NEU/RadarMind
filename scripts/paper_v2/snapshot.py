"""Build metric_snapshot.json from the registry. Fail if any source is missing."""
from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path

from .registry import METRICS, by_key


class MetricError(RuntimeError):
    pass


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _walk(obj, field: str):
    cur = obj
    if not field:
        return cur
    for part in field.split("."):
        if isinstance(cur, list):
            try:
                cur = cur[int(part)]
            except (ValueError, IndexError) as e:
                raise MetricError(f"cannot index list with {part!r}") from e
        elif isinstance(cur, dict):
            if part not in cur:
                raise MetricError(f"missing field {part!r} in {list(cur)[:12]}")
            cur = cur[part]
        else:
            raise MetricError(f"cannot walk {part!r} on {type(cur)}")
    return cur


def _format(value, rule: str) -> str:
    if rule == "int":
        return str(int(value))
    if rule == "str":
        return str(value)
    if rule == "sci":
        return f"{float(value):.6e}"
    if rule == "f2":
        return f"{float(value):.2f}"
    if rule == "f3":
        return f"{float(value):.3f}"
    if rule == "f1":
        return f"{float(value):.1f}"
    raise MetricError(f"unknown format_rule {rule}")


def _csv_row(path: Path, match: dict, field: str):
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in match.items())]
    if not hits:
        raise MetricError(f"no CSV row matching {match} in {path}")
    if len(hits) > 1:
        raise MetricError(f"ambiguous CSV match {match} in {path}: {len(hits)} rows")
    if field not in hits[0]:
        raise MetricError(f"column {field} missing in {path}")
    return hits[0][field]


def _csv_count(path: Path, field: str, value: str) -> int:
    with path.open(encoding="utf-8", newline="") as f:
        return sum(1 for r in csv.DictReader(f) if r.get(field) == value)


def _csv_rows(path: Path, match: dict) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    hits = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in match.items())]
    if not hits:
        raise MetricError(f"no CSV rows matching {match} in {path}")
    return hits


def _csv_mean(path: Path, match: dict, field: str) -> float:
    hits = _csv_rows(path, match)
    if field not in hits[0]:
        raise MetricError(f"column {field} missing in {path}")
    return sum(float(r[field]) for r in hits) / len(hits)


def _csv_sum(path: Path, match: dict, field: str) -> float:
    hits = _csv_rows(path, match)
    if field not in hits[0]:
        raise MetricError(f"column {field} missing in {path}")
    return sum(float(r[field]) for r in hits)


def _json_list_field(path: Path, field: str) -> list[float]:
    obj = _read_json(path)
    if not isinstance(obj, list):
        raise MetricError(f"{path} is not a JSON list")
    out = []
    for item in obj:
        if field not in item:
            raise MetricError(f"field {field} missing in list item of {path}")
        out.append(float(item[field]))
    if not out:
        raise MetricError(f"empty JSON list in {path}")
    return out


def _mean(xs: list[float]) -> float:
    return sum(xs) / len(xs)


def _std_sample(xs: list[float]) -> float:
    if len(xs) < 2:
        raise MetricError("need >=2 values for sample std")
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / (len(xs) - 1)
    return var ** 0.5


def _md_pass_count(path: Path) -> int:
    n = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("## PASS") or s.startswith("- PASS"):
            n += 1
    if n == 0:
        raise MetricError(f"no PASS lines in {path}")
    return n


def build_snapshot(root: Path) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    raw: dict[str, dict] = {}
    errors: list[str] = []
    pending = []
    for spec in METRICS:
        key = spec["key"]
        if spec["kind"] in ("derived_sub", "derived_div"):
            pending.append(spec)
            continue
        rel = spec["source_file"]
        path = root / rel
        if not path.is_file():
            errors.append(f"{key}: missing source {rel}")
            continue
        try:
            kind = spec["kind"]
            if kind == "json":
                value = _walk(_read_json(path), spec["source_field"])
            elif kind == "json_len":
                value = len(_walk(_read_json(path), spec["source_field"]))
            elif kind == "json_mean":
                value = _mean(_json_list_field(path, spec["source_field"]))
            elif kind == "json_std":
                value = _std_sample(_json_list_field(path, spec["source_field"]))
            elif kind == "csv_row":
                value = _csv_row(path, spec["match"], spec["source_field"])
            elif kind == "csv_count":
                value = _csv_count(path, spec["source_field"], spec["match"]["value"])
            elif kind == "csv_mean":
                value = _csv_mean(path, spec["match"], spec["source_field"])
            elif kind == "csv_sum":
                value = _csv_sum(path, spec["match"], spec["source_field"])
            elif kind == "md_pass_count":
                value = _md_pass_count(path)
            else:
                raise MetricError(f"unknown kind {kind}")
            if spec["format_rule"] != "str":
                value = float(value) if spec["format_rule"] != "int" else int(float(value))
            raw[key] = {
                "key": key,
                "value": value,
                "display": _format(value, spec["format_rule"]),
                "source_file": rel,
                "source_field": spec["source_field"],
                "source_row": json.dumps(spec["match"], ensure_ascii=False) if spec["match"] else spec["source_field"],
                "format_rule": spec["format_rule"],
                "protocol": spec["protocol"],
                "compare_group": spec["compare_group"],
                "note": spec["note"],
                "read_timestamp": now,
            }
        except Exception as e:
            errors.append(f"{key}: {e}")
    lookup = by_key()
    for spec in pending:
        key = spec["key"]
        try:
            av = raw[spec["a"]]["value"]
            if spec["kind"] == "derived_sub":
                value = float(av) - float(raw[spec["b"]]["value"])
                src = f"derived: {spec['a']} - {spec['b']}"
            elif spec["kind"] == "derived_div":
                scale = spec["scale"]
                if scale is None:
                    value = float(av) / float(raw[spec["b"]]["value"])
                    src = f"derived: {spec['a']} / {spec['b']}"
                else:
                    value = float(av) / float(scale)
                    src = f"derived: {spec['a']} / {scale}"
            else:
                raise MetricError(spec["kind"])
            raw[key] = {
                "key": key,
                "value": value,
                "display": _format(value, spec["format_rule"]),
                "source_file": src,
                "source_field": spec["a"],
                "source_row": spec["b"] or str(spec["scale"]),
                "format_rule": spec["format_rule"],
                "protocol": spec["protocol"] or lookup[key]["protocol"],
                "compare_group": spec["compare_group"],
                "note": spec["note"],
                "read_timestamp": now,
            }
        except Exception as e:
            errors.append(f"{key}: {e}")
    if errors:
        raise MetricError("metric source audit FAILED:\n" + "\n".join(errors))
    missing = [m["key"] for m in METRICS if m["key"] not in raw]
    if missing:
        raise MetricError("unresolved metrics: " + ", ".join(missing))
    return {"generated_at": now, "n_metrics": len(raw), "metrics": raw}


def display_map(snap: dict) -> dict[str, str]:
    return {k: v["display"] for k, v in snap["metrics"].items()}


def value_map(snap: dict) -> dict[str, float | int | str]:
    return {k: v["value"] for k, v in snap["metrics"].items()}
