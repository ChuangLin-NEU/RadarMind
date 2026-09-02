"""Build AIRHAR-compatible CI4R h5 from official spectrogram PNGs.

This is the Level-1 path when raw_data.zip (93 GB) is not fully downloaded.
STFT is already done by CI4R-MULTI3; we only resize to 224x224 and z-score,
matching AIRHAR's last-12-per-class split.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ci4r_meta import CLASS_NAMES, official_last12_split, list_samples  # noqa: E402


def load_gray_224(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    return image


def zscore(spec: np.ndarray) -> np.ndarray:
    std = spec.std()
    if std < 1e-8:
        return spec - spec.mean()
    return (spec - spec.mean()) / std


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="77GHz", choices=["77GHz", "24GHz", "10GHz"])
    parser.add_argument(
        "--spec-root",
        default=str(ROOT / "data" / "CI4R_MULTI3" / "spectrograms" / "11_class_activity_data"),
    )
    parser.add_argument(
        "--out-h5",
        default=str(ROOT / "third_party" / "AIRHAR" / "datasets" / "CI4R" / "ci4r_data.h5"),
    )
    parser.add_argument(
        "--split-csv",
        default=str(ROOT / "results" / "splits" / "ci4r_official_last12.csv"),
    )
    args = parser.parse_args()

    samples = list_samples(Path(args.spec_root), args.freq)
    if not samples:
        raise SystemExit(f"no PNG found for {args.freq} under {args.spec_root}")

    split_rows = official_last12_split(samples)
    train_x, train_y, test_x, test_y = [], [], [], []
    table = []
    for sample, split in split_rows:
        spec = zscore(load_gray_224(sample.path))
        onehot = np.zeros(len(CLASS_NAMES), dtype=np.float32)
        onehot[sample.class_index] = 1.0
        if split == "train":
            train_x.append(spec)
            train_y.append(onehot)
        else:
            test_x.append(spec)
            test_y.append(onehot)
        table.append(
            {
                "filename": sample.filename,
                "freq": sample.freq,
                "class": sample.class_name,
                "class_index": sample.class_index,
                "subject": sample.subject,
                "split": split,
                "path": str(sample.path),
            }
        )

    out_h5 = Path(args.out_h5)
    out_h5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_h5, "w") as f:
        train = f.create_group("train")
        test = f.create_group("test")
        train.create_dataset("spectrograms", data=np.stack(train_x), dtype="float32")
        train.create_dataset("labels", data=np.stack(train_y), dtype="float32")
        test.create_dataset("spectrograms", data=np.stack(test_x), dtype="float32")
        test.create_dataset("labels", data=np.stack(test_y), dtype="float32")

    split_csv = Path(args.split_csv)
    split_csv.parent.mkdir(parents=True, exist_ok=True)
    with split_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
        writer.writeheader()
        writer.writerows(table)

    subjects_train = sorted({row["subject"] for row in table if row["split"] == "train"})
    subjects_test = sorted({row["subject"] for row in table if row["split"] == "test"})
    leaked = sorted(set(subjects_train) & set(subjects_test))
    summary = {
        "freq": args.freq,
        "n_train": len(train_x),
        "n_test": len(test_x),
        "shape": [224, 224],
        "subjects_train": subjects_train,
        "subjects_test": subjects_test,
        "subjects_in_both": leaked,
        "h5": str(out_h5),
        "split_csv": str(split_csv),
    }
    summary_path = split_csv.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
