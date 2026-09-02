"""Train on one radar frequency and test on another. Official last-12 split per band."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
PY = sys.executable
TRAIN = ROOT / "scripts" / "train_local.py"
PREPARE = ROOT / "scripts" / "prepare_ci4r_from_png.py"


def ensure_h5(freq: str) -> Path:
    mapping = {"77GHz": "77GHz", "24GHz": "24GHz", "10GHz": "10GHz"}
    path = ROOT / "results" / "data" / f"ci4r_{mapping[freq]}.h5"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            [
                PY,
                str(PREPARE),
                "--freq",
                freq,
                "--out-h5",
                str(path),
                "--split-csv",
                str(ROOT / "results" / "splits" / f"ci4r_{freq}_last12.csv"),
            ]
        )
    return path


def mix_h5(train_h5: Path, test_h5: Path, out_h5: Path) -> None:
    with h5py.File(train_h5, "r") as src, h5py.File(test_h5, "r") as tgt, h5py.File(out_h5, "w") as dst:
        dst.create_group("train").create_dataset("spectrograms", data=src["train/spectrograms"][:])
        dst["train"].create_dataset("labels", data=src["train/labels"][:])
        dst.create_group("test").create_dataset("spectrograms", data=tgt["test/spectrograms"][:])
        dst["test"].create_dataset("labels", data=tgt["test/labels"][:])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-freq", default="24GHz")
    parser.add_argument("--test-freq", default="77GHz")
    parser.add_argument("--model", default="simple_cnn")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    train_h5 = ensure_h5(args.train_freq)
    test_h5 = ensure_h5(args.test_freq)
    mixed = ROOT / "results" / "data" / f"cross_{args.train_freq}_to_{args.test_freq}.h5"
    mix_h5(train_h5, test_h5, mixed)
    tag = f"cross_{args.train_freq}_to_{args.test_freq}_{args.model}_seed{args.seed}"
    subprocess.check_call(
        [
            PY,
            str(TRAIN),
            "--model",
            args.model,
            "--h5",
            str(mixed),
            "--epochs",
            str(args.epochs),
            "--seed",
            str(args.seed),
            "--tag",
            tag,
        ]
    )
    metrics = json.loads((ROOT / "results" / "runs" / tag / "metrics.json").read_text(encoding="utf-8"))
    metrics.update({"train_freq": args.train_freq, "test_freq": args.test_freq})
    out = ROOT / "results" / "tables" / f"{tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
