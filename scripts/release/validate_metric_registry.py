#!/usr/bin/env python3
"""Trace manuscript snapshot keys back to source result files. Read-only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

from paper_v2.registry import by_key  # noqa: E402
from paper_v2.snapshot import MetricError, build_snapshot  # noqa: E402
from release.common import MANUSCRIPT_KEYS  # noqa: E402

SNAP = ROOT / "results" / "final_delivery" / "metric_snapshot_full.json"
OUT = ROOT / "reports" / "metric_registry_trace.md"


def main() -> int:
    stored = json.loads(SNAP.read_text(encoding="utf-8"))
    live = build_snapshot(ROOT)
    specs = by_key()
    rows = []
    n_fail = 0
    for key in MANUSCRIPT_KEYS:
        spec = specs[key]
        stored_m = stored["metrics"][key]
        live_m = live["metrics"][key]
        src = spec["source_file"] or "(derived)"
        src_ok = True if not spec["source_file"] else (ROOT / spec["source_file"]).is_file()
        display_ok = str(stored_m["display"]) == str(live_m["display"])
        value_ok = stored_m["value"] == live_m["value"]
        ok = src_ok and display_ok and value_ok
        if not ok:
            n_fail += 1
        rows.append(
            {
                "key": key,
                "source_file": src,
                "source_exists": src_ok,
                "stored_display": stored_m["display"],
                "live_display": live_m["display"],
                "ok": ok,
            }
        )

    lines = [
        "# Metric registry trace",
        "",
        "Stored snapshot: `results/final_delivery/metric_snapshot_full.json`",
        "Live rebuild: `scripts/paper_v2/snapshot.py` + `scripts/paper_v2/registry.py`",
        "",
        f"Manuscript keys checked: {len(rows)}. Failures: {n_fail}.",
        "",
        "| Key | Source file | Source exists | Stored display | Live display | Match |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['key']}` | `{r['source_file']}` | {r['source_exists']} | {r['stored_display']} | {r['live_display']} | {'PASS' if r['ok'] else 'FAIL'} |"
        )
    lines += ["", f"Status: {'PASS' if n_fail == 0 else 'FAIL'}", ""]
    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT, "fail", n_fail)
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except MetricError as exc:
        print("MetricError", exc)
        raise SystemExit(1)
