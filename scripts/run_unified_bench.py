"""Run the Level-2 model comparison on the same h5 / official split."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_local.py"

MODELS = [
    ("simple_cnn", 1e-3, 64),
    ("resnet", 1e-4, 12),
    ("resnet18", 1e-3, 64),
    ("cnnlstm", 1e-4, 16),
    ("radmamba", 5e-3, 80),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--h5", default=str(ROOT / "third_party" / "AIRHAR" / "datasets" / "CI4R" / "ci4r_data.h5"))
    args = parser.parse_args()

    rows = []
    for name, lr, dim in MODELS:
        tag = f"bench_{name}_seed{args.seed}"
        cmd = [
            PY,
            str(TRAIN),
            "--model",
            name,
            "--h5",
            args.h5,
            "--epochs",
            str(args.epochs),
            "--lr",
            str(lr),
            "--dim",
            str(dim),
            "--seed",
            str(args.seed),
            "--tag",
            tag,
        ]
        print("RUN", " ".join(cmd))
        subprocess.check_call(cmd)
        metrics = json.loads((ROOT / "results" / "runs" / tag / "metrics.json").read_text(encoding="utf-8"))
        rows.append(metrics)

    out = ROOT / "results" / "tables" / "unified_bench.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "accuracy", "macro_f1", "n_params", "train_minutes", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
