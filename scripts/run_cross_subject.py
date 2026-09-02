"""Train one model per LOSO fold and write mean ± std."""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_local.py"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="77GHz")
    parser.add_argument("--model", default="simple_cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    fold_dir = ROOT / "results" / "splits" / f"loso_{args.freq}"
    folds = json.loads((fold_dir / "folds.json").read_text(encoding="utf-8"))["folds"]
    rows = []
    for fold in folds:
        held = fold["held_out"]
        tag = f"loso_{args.freq}_{args.model}_held{held}_seed{args.seed}"
        cmd = [
            PY,
            str(TRAIN),
            "--model",
            args.model,
            "--h5",
            fold["h5"],
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--tag",
            tag,
        ]
        print("RUN", " ".join(cmd))
        subprocess.check_call(cmd)
        metrics = json.loads((ROOT / "results" / "runs" / tag / "metrics.json").read_text(encoding="utf-8"))
        metrics["held_out"] = held
        rows.append(metrics)

    acc = np.array([r["accuracy"] for r in rows], dtype=float)
    f1 = np.array([r["macro_f1"] for r in rows], dtype=float)
    summary = {
        "model": args.model,
        "freq": args.freq,
        "accuracy_mean": float(acc.mean()),
        "accuracy_std": float(acc.std(ddof=1) if len(acc) > 1 else 0.0),
        "macro_f1_mean": float(f1.mean()),
        "macro_f1_std": float(f1.std(ddof=1) if len(f1) > 1 else 0.0),
        "folds": rows,
    }
    out = ROOT / "results" / "tables" / f"cross_subject_{args.freq}_{args.model}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    csv_path = out.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out", "accuracy", "macro_f1", "n_params", "seed"])
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in ["held_out", "accuracy", "macro_f1", "n_params", "seed"]})
    print(json.dumps({k: summary[k] for k in summary if k != "folds"}, indent=2))


if __name__ == "__main__":
    main()
