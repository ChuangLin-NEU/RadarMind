#!/usr/bin/env python3
"""Train VoxelCNNGRU on packed official RadHAR voxels."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]


class VoxelU8(Dataset):
    def __init__(self, X, y, keep):
        self.X = X
        self.idx = np.flatnonzero(keep)
        self.y = y.astype(np.int64)

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        return torch.from_numpy(np.array(self.X[j], dtype=np.float32)), torch.tensor(int(self.y[j]))
sys.path.insert(0, str(ROOT / "scripts"))
from models.point_encoder import VoxelCNNGRU  # noqa: E402
from pointcloud.common import SEED, append_csv  # noqa: E402
from pointcloud.phase1b_train import RUN_FIELDS, metrics, set_seed, train_net  # noqa: E402
from pointcloud.radhar import CLASS_NAMES  # noqa: E402


def main() -> None:
    pack = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "results" / "phase1_pointcloud" / "cache" / "radhar_voxels.npz")
    out = Path(sys.argv[2] if len(sys.argv) > 2 else ROOT / "results" / "phase1_pointcloud" / "baselines" / "radhar_official_voxel_cnn_gru")
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(pack, allow_pickle=True, mmap_mode="r")
    X, y, role = z["X"], z["y"], np.array([str(s) for s in z["split"]])
    tr, va, te = role == "train", role == "val", role == "test"
    print("shapes", X.shape, "tr/va/te", int(tr.sum()), int(va.sum()), int(te.sum()), device, flush=True)
    net = VoxelCNNGRU(5).to(device)
    out.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(VoxelU8(X, y, tr), batch_size=8, shuffle=True)
    vl = DataLoader(VoxelU8(X, y, va), batch_size=8)
    el = DataLoader(VoxelU8(X, y, te), batch_size=8)
    net, yt, pt, best, hist = train_net(net, tl, vl, el, device, out, epochs=20)
    m = metrics(yt, pt, CLASS_NAMES)
    m.update({"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "best_val_acc": float(best)})
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.savetxt(out / "cm.csv", confusion_matrix(yt, pt, labels=list(range(5))), fmt="%d", delimiter=",")
    row = {
        "dataset": "RadHAR",
        "split": "official_train_test",
        "representation": "official_voxel_10x32x32",
        "features": "occupancy",
        "temporal_length": 60,
        "model": "voxel_cnn_gru",
        "seed": SEED,
        "accuracy": m["accuracy"],
        "macro_f1": m["macro_f1"],
        "weighted_f1": m["weighted_f1"],
        "n_train": m["n_train"],
        "n_val": m["n_val"],
        "n_test": m["n_test"],
        "best_val_acc": m["best_val_acc"],
    }
    append_csv(ROOT / "results/phase1_pointcloud/training_runs.csv", row, RUN_FIELDS)
    append_csv(ROOT / "results/phase1_pointcloud/baselines/training_runs.csv", row, RUN_FIELDS)
    print(json.dumps(row, indent=2), flush=True)


if __name__ == "__main__":
    main()
