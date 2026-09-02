"""Collect finished run metrics into comparison tables."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "tables"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    rows = []
    for metrics_path in sorted(RUNS.glob("*/metrics.json")):
        row = json.loads(metrics_path.read_text(encoding="utf-8"))
        row["run"] = metrics_path.parent.name
        rows.append(row)
    if not rows:
        raise SystemExit("no runs")
    keys = ["run", "model", "accuracy", "macro_f1", "n_params", "train_minutes", "seed"]
    with (OUT / "all_runs.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
