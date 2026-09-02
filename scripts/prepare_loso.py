"""Write leave-one-subject-out h5 files. Keep official last-12 numbers in a separate folder."""
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
from ci4r_meta import CLASS_NAMES, list_samples  # noqa: E402


def load_gray_224(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    std = image.std()
    return (image - image.mean()) / std if std > 1e-8 else image - image.mean()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="77GHz")
    parser.add_argument(
        "--spec-root",
        default=str(ROOT / "data" / "CI4R_MULTI3" / "spectrograms" / "11_class_activity_data"),
    )
    parser.add_argument("--min-subject-count", type=int, default=20)
    args = parser.parse_args()

    samples = list_samples(Path(args.spec_root), args.freq)
    counts: dict[str, int] = {}
    for sample in samples:
        counts[sample.subject] = counts.get(sample.subject, 0) + 1
    subjects = sorted(s for s, n in counts.items() if n >= args.min_subject_count)
    if len(subjects) < 2:
        raise SystemExit(f"not enough subjects: {counts}")

    out_root = ROOT / "results" / "splits" / f"loso_{args.freq}"
    out_root.mkdir(parents=True, exist_ok=True)
    fold_info = []
    for held in subjects:
        train_x, train_y, test_x, test_y, table = [], [], [], [], []
        for sample in samples:
            spec = load_gray_224(sample.path)
            onehot = np.zeros(len(CLASS_NAMES), dtype=np.float32)
            onehot[sample.class_index] = 1.0
            split = "test" if sample.subject == held else "train"
            if split == "train":
                train_x.append(spec)
                train_y.append(onehot)
            else:
                test_x.append(spec)
                test_y.append(onehot)
            table.append(
                {
                    "filename": sample.filename,
                    "subject": sample.subject,
                    "class": sample.class_name,
                    "split": split,
                    "held_out": held,
                }
            )
        h5_path = out_root / f"held_{held}.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_group("train").create_dataset("spectrograms", data=np.stack(train_x), dtype="float32")
            f["train"].create_dataset("labels", data=np.stack(train_y), dtype="float32")
            f.create_group("test").create_dataset("spectrograms", data=np.stack(test_x), dtype="float32")
            f["test"].create_dataset("labels", data=np.stack(test_y), dtype="float32")
        csv_path = out_root / f"held_{held}.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(table[0].keys()))
            writer.writeheader()
            writer.writerows(table)
        fold_info.append({"held_out": held, "n_train": len(train_x), "n_test": len(test_x), "h5": str(h5_path)})
    (out_root / "folds.json").write_text(json.dumps({"freq": args.freq, "folds": fold_info}, indent=2), encoding="utf-8")
    print(json.dumps(fold_info, indent=2))


if __name__ == "__main__":
    main()
