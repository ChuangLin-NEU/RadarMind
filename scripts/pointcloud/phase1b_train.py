#!/usr/bin/env python3
"""Phase 1B baselines. Fixed seed. Test once after val selection. No search."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from models.point_encoder import PointNetGRU, PointNetPool, VoxelCNNGRU  # noqa: E402
from pointcloud.common import FEAT_SLICES, SEED, append_csv  # noqa: E402
from pointcloud.mri import CLASS_NAMES as MRI_CLASSES  # noqa: E402
from pointcloud.radhar import CLASS_NAMES as RAD_CLASSES  # noqa: E402
from pointcloud.voxels_official import voxel_windows  # noqa: E402
from pointcloud.radhar import parse_mmwave_txt  # noqa: E402

RUN_FIELDS = [
    "dataset",
    "split",
    "representation",
    "features",
    "temporal_length",
    "model",
    "seed",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "n_train",
    "n_val",
    "n_test",
    "best_val_acc",
]


class WindowDS(Dataset):
    def __init__(self, X, mask, y, n_ch: int, t_keep: int | None):
        if t_keep is not None and t_keep == 1:
            mid = X.shape[1] // 2
            X = X[:, mid : mid + 1]
            mask = mask[:, mid : mid + 1]
        self.X = torch.from_numpy(X[..., :n_ch].astype(np.float32))
        self.mask = torch.from_numpy(mask.astype(np.bool_))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.mask[i], self.y[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def metrics(y_true, y_pred, class_names):
    n = len(class_names)
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [{"class": class_names[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])} for i in range(n)],
    }


@torch.no_grad()
def eval_loader(net, loader, device):
    net.eval()
    ys, ps = [], []
    for batch in loader:
        if len(batch) == 3:
            x, mask, y = batch
            logits = net(x.to(device), mask.to(device))
        else:
            x, y = batch
            logits = net(x.to(device))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_net(net, train_loader, val_loader, test_loader, device, out: Path, epochs=20, lr=1e-3):
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best, best_state, bad = -1.0, None, 0
    hist = []
    for epoch in range(epochs):
        net.train()
        losses = []
        for batch in train_loader:
            opt.zero_grad()
            if len(batch) == 3:
                x, mask, y = batch
                logits = net(x.to(device), mask.to(device))
                yb = y.to(device)
            else:
                x, y = batch
                logits = net(x.to(device))
                yb = y.to(device)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        yv, pv = eval_loader(net, val_loader, device)
        acc = float(accuracy_score(yv, pv))
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": acc})
        print(out.name, hist[-1], flush=True)
        if acc > best:
            best, bad = acc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    yt, pt = eval_loader(net, test_loader, device)
    return net, yt, pt, best, hist


def apply_split_mri(z, split_csv: Path):
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    mp = {r["action_segment_id"]: r["split"] for r in rows}
    segs = [str(s) for s in z["action_segment_id"]]
    assign = np.array([mp.get(s, "heldout") for s in segs])
    return assign


def apply_split_radhar(z, split_csv: Path):
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    mp = {r["source_file"]: r["split_role"] for r in rows}
    src = [str(s) for s in z["source_file"]]
    return np.array([mp.get(s, "heldout") for s in src])


def run_point_pack(name, z, assign, class_names, features, model_name, t_keep, device, out, epochs):
    dest = out / name
    if (dest / "metrics.json").exists():
        print("[skip-exists]", dest, flush=True)
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    n_ch = FEAT_SLICES[features]
    X, mask, y = z["X"], z["mask"], z["y"]
    tr, va, te = assign == "train", assign == "val", assign == "test"
    if va.sum() == 0 or tr.sum() == 0 or te.sum() == 0:
        return {"error": f"empty split {name} tr={tr.sum()} va={va.sum()} te={te.sum()}"}
    ncls = len(class_names)
    factory = {
        "pointnet_pool": lambda: PointNetPool(ncls, n_ch),
        "pointnet_gru": lambda: PointNetGRU(ncls, n_ch),
    }[model_name]
    net = factory().to(device)
    tr_ds = WindowDS(X[tr], mask[tr], y[tr], n_ch, t_keep)
    va_ds = WindowDS(X[va], mask[va], y[va], n_ch, t_keep)
    te_ds = WindowDS(X[te], mask[te], y[te], n_ch, t_keep)
    tl = DataLoader(tr_ds, batch_size=16, shuffle=True)
    vl = DataLoader(va_ds, batch_size=32)
    el = DataLoader(te_ds, batch_size=32)
    dest.mkdir(parents=True, exist_ok=True)
    net, yt, pt, best, hist = train_net(net, tl, vl, el, device, dest, epochs=epochs)
    m = metrics(yt, pt, class_names)
    m.update({"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "best_val_acc": float(best)})
    (dest / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.savetxt(dest / "cm.csv", confusion_matrix(yt, pt, labels=list(range(len(class_names)))), fmt="%d", delimiter=",")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    torch.save(net.state_dict(), dest / "best.pt")
    return m


def cm_fig(cm_path: Path, dest: Path, names, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cm = np.loadtxt(cm_path, delimiter=",")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


class VoxelDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def run_voxel_radhar(rad_root: Path, split_csv: Path, cache_npz: Path, device, out: Path, epochs: int):
    dest = out / "radhar_official_voxel_cnn_gru"
    if (dest / "metrics.json").exists():
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    xs, ys, roles = [], [], []
    for r in rows:
        fp = rad_root / r["source_file"]
        if not fp.exists():
            continue
        frames = parse_mmwave_txt(fp)
        wins = voxel_windows(frames, 60, 10)
        if wins.shape[0] == 0:
            continue
        xs.append(wins)
        ys.append(np.full((wins.shape[0],), RAD_CLASSES.index(r["action"]), np.int64))
        roles.append(np.array([r["split_role"]] * wins.shape[0]))
        print("[voxel]", r["source_file"], wins.shape[0], flush=True)
    X = np.concatenate(xs, 0)
    y = np.concatenate(ys, 0)
    role = np.concatenate(roles, 0)
    tr, va, te = role == "train", role == "val", role == "test"
    net = VoxelCNNGRU(5).to(device)
    dest.mkdir(parents=True, exist_ok=True)
    tl = DataLoader(VoxelDS(X[tr], y[tr]), batch_size=8, shuffle=True)
    vl = DataLoader(VoxelDS(X[va], y[va]), batch_size=8)
    el = DataLoader(VoxelDS(X[te], y[te]), batch_size=8)
    net, yt, pt, best, hist = train_net(net, tl, vl, el, device, dest, epochs=epochs)
    m = metrics(yt, pt, RAD_CLASSES)
    m.update({"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "best_val_acc": float(best)})
    (dest / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.savetxt(dest / "cm.csv", confusion_matrix(yt, pt, labels=list(range(5))), fmt="%d", delimiter=",")
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mri-pack", default="")
    ap.add_argument("--radhar-pack", default="")
    ap.add_argument("--mri-random-split", default="")
    ap.add_argument("--mri-si-split", default="")
    ap.add_argument("--radhar-split", default="")
    ap.add_argument("--radhar-root", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--figures-dir", default="")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--skip-voxel", action="store_true")
    args = ap.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    figdir = Path(args.figures_dir) if args.figures_dir else out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    runs_csv = out / "training_runs.csv"
    summary = []

    def log_run(row, extra_metrics=None):
        append_csv(runs_csv, row, RUN_FIELDS)
        if extra_metrics:
            summary.append({**row, **{k: extra_metrics.get(k) for k in ("per_class",)}})

    if args.mri_pack:
        z = np.load(args.mri_pack, allow_pickle=True)
        jobs = []
        if args.mri_si_split:
            a = apply_split_mri(z, Path(args.mri_si_split))
            for feat in ("XYZ", "XYZV", "XYZVI"):
                jobs.append(("mri_si_gru_" + feat.lower(), a, feat, "pointnet_gru", None, "subject_independent", 32))
            jobs.append(("mri_si_pool_t1_xyzvi", a, "XYZVI", "pointnet_pool", 1, "subject_independent", 1))
        if args.mri_random_split:
            a = apply_split_mri(z, Path(args.mri_random_split))
            jobs.append(("mri_random_gru_xyzvi", a, "XYZVI", "pointnet_gru", None, "random_segment", 32))
        for name, assign, feat, model, tkeep, proto, tlen in jobs:
            print("====", name, flush=True)
            m = run_point_pack(name, z, assign, MRI_CLASSES, feat, model, tkeep, device, out, args.epochs)
            row = {
                "dataset": "mRI",
                "split": proto,
                "representation": "sampled_TxPxC",
                "features": feat,
                "temporal_length": tlen,
                "model": model,
                "seed": SEED,
                "accuracy": m.get("accuracy", ""),
                "macro_f1": m.get("macro_f1", ""),
                "weighted_f1": m.get("weighted_f1", ""),
                "n_train": m.get("n_train", ""),
                "n_val": m.get("n_val", ""),
                "n_test": m.get("n_test", ""),
                "best_val_acc": m.get("best_val_acc", ""),
            }
            log_run(row, m)
            cm = out / name / "cm.csv"
            if cm.exists() and name == "mri_si_gru_xyzvi":
                cm_fig(cm, figdir / "fig4_mri_si_cm.png", MRI_CLASSES, "mRI subject-independent PointNet-GRU XYZVI")

    if args.radhar_pack:
        z = np.load(args.radhar_pack, allow_pickle=True)
        a = apply_split_radhar(z, Path(args.radhar_split))
        jobs = [
            ("radhar_official_gru_xyz", a, "XYZ", "pointnet_gru", None, 60),
            ("radhar_official_gru_xyzv", a, "XYZV", "pointnet_gru", None, 60),
            ("radhar_official_gru_xyzvi", a, "XYZVI", "pointnet_gru", None, 60),
            ("radhar_official_pool_t1_xyzvi", a, "XYZVI", "pointnet_pool", 1, 1),
        ]
        for name, assign, feat, model, tkeep, tlen in jobs:
            print("====", name, flush=True)
            m = run_point_pack(name, z, assign, RAD_CLASSES, feat, model, tkeep, device, out, args.epochs)
            row = {
                "dataset": "RadHAR",
                "split": "official_train_test",
                "representation": "sampled_TxPxC",
                "features": feat,
                "temporal_length": tlen,
                "model": model,
                "seed": SEED,
                "accuracy": m.get("accuracy", ""),
                "macro_f1": m.get("macro_f1", ""),
                "weighted_f1": m.get("weighted_f1", ""),
                "n_train": m.get("n_train", ""),
                "n_val": m.get("n_val", ""),
                "n_test": m.get("n_test", ""),
                "best_val_acc": m.get("best_val_acc", ""),
            }
            log_run(row, m)
            cm = out / name / "cm.csv"
            if cm.exists() and name == "radhar_official_gru_xyzvi":
                cm_fig(cm, figdir / "fig7_radhar_official_cm.png", RAD_CLASSES, "RadHAR official split PointNet-GRU XYZVI")

    if args.radhar_root and args.radhar_split and not args.skip_voxel:
        print("==== voxel", flush=True)
        m = run_voxel_radhar(Path(args.radhar_root), Path(args.radhar_split), Path(args.radhar_pack or ""), device, out, args.epochs)
        log_run(
            {
                "dataset": "RadHAR",
                "split": "official_train_test",
                "representation": "official_voxel_10x32x32",
                "features": "occupancy",
                "temporal_length": 60,
                "model": "voxel_cnn_gru",
                "seed": SEED,
                "accuracy": m.get("accuracy", ""),
                "macro_f1": m.get("macro_f1", ""),
                "weighted_f1": m.get("weighted_f1", ""),
                "n_train": m.get("n_train", ""),
                "n_val": m.get("n_val", ""),
                "n_test": m.get("n_test", ""),
                "best_val_acc": m.get("best_val_acc", ""),
            },
            m,
        )

    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("[train-done]", flush=True)


if __name__ == "__main__":
    main()
