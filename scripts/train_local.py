"""Train a local baseline or AIRHAR backbone on an existing CI4R h5.

Saves config, seed, checkpoint, history CSV, and test metrics.
Does not change official AIRHAR training code.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset
import h5py

ROOT = Path(__file__).resolve().parents[1]
AIRHAR = ROOT / "third_party" / "AIRHAR"
sys.path.insert(0, str(AIRHAR))
sys.path.insert(0, str(ROOT / "scripts"))
from ci4r_meta import CLASS_NAMES  # noqa: E402
from models.resnet18_spec import ResNet18Spec  # noqa: E402
from models.simple_cnn import SimpleCNN  # noqa: E402


class H5SpectrogramDataset(Dataset):
    def __init__(self, h5_path: Path, split: str):
        with h5py.File(h5_path, "r") as f:
            self.x = torch.from_numpy(f[f"{split}/spectrograms"][:])
            self.y = torch.from_numpy(f[f"{split}/labels"][:])

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(name: str, num_classes: int, dim: int):
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return ResNet18Spec(num_classes=num_classes)
    if name in {"radmamba", "resnet", "cnnlstm", "cnngru", "bilstm"}:
        import importlib.util

        if str(AIRHAR) not in sys.path:
            sys.path.insert(0, str(AIRHAR))
        spec = importlib.util.spec_from_file_location("airhar_models", AIRHAR / "models.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.CoreModel(
            hidden_size=dim,
            num_layers=1,
            backbone_type=name,
            dim=dim,
            dt_rank=0 if name == "radmamba" else 4,
            d_state=4,
            image_height=224,
            image_width=224,
            num_classes=num_classes,
            channels=1,
            dropout=0.0,
            optional_avg_pool=False,
            channel_confusion_layer=1,
            channel_confusion_out_channels=1,
            time_downsample_factor=4,
        )
    raise ValueError(name)


@torch.no_grad()
def evaluate(net, loader, device):
    net.eval()
    ys, ps, losses = [], [], []
    criterion = nn.CrossEntropyLoss()
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        logits = net(x)
        target = y.argmax(dim=1)
        losses.append(criterion(logits, target).item())
        ys.append(target.cpu().numpy())
        ps.append(logits.argmax(dim=1).cpu().numpy())
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "loss": float(np.mean(losses)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--h5", default=str(AIRHAR / "datasets" / "CI4R" / "ci4r_data.h5"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--out", default="")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tag = args.tag or f"{args.model}_seed{args.seed}"
    out_dir = Path(args.out) if args.out else ROOT / "results" / "runs" / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ds = H5SpectrogramDataset(Path(args.h5), "train")
    test_ds = H5SpectrogramDataset(Path(args.h5), "test")
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    net = build_model(args.model, num_classes=len(CLASS_NAMES), dim=args.dim).to(device)
    n_params = int(sum(p.numel() for p in net.parameters()))
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    config = {**vars(args), "device": str(device), "n_params": n_params, "n_train": len(train_ds), "n_test": len(test_ds)}
    (out_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    history = []
    best_acc = -1.0
    best_pred = None
    t0 = time.time()
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad()
            logits = net(x)
            loss = criterion(logits, y.argmax(dim=1))
            loss.backward()
            opt.step()
            losses.append(loss.item())
        ev = evaluate(net, test_loader, device)
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "test_loss": ev["loss"],
            "test_accuracy": ev["accuracy"],
            "test_macro_f1": ev["macro_f1"],
            "elapsed_min": (time.time() - t0) / 60.0,
        }
        history.append(row)
        print(row)
        if ev["accuracy"] > best_acc:
            best_acc = ev["accuracy"]
            best_pred = ev
            torch.save(net.state_dict(), out_dir / "best.pt")

    with (out_dir / "history.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0].keys()))
        writer.writeheader()
        writer.writerows(history)

    metrics = {
        "accuracy": best_pred["accuracy"],
        "macro_f1": best_pred["macro_f1"],
        "n_params": n_params,
        "train_minutes": (time.time() - t0) / 60.0,
        "model": args.model,
        "seed": args.seed,
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    cm = confusion_matrix(best_pred["y_true"], best_pred["y_pred"], labels=list(range(len(CLASS_NAMES))))
    np.savetxt(out_dir / "cm.csv", cm, fmt="%d", delimiter=",")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
