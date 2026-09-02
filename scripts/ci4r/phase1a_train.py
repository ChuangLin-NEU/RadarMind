#!/usr/bin/env python3
"""Phase 1A baselines on cached AIRHAR 224x224 spectrograms. No search."""
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
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from models.resnet18_spec import ResNet18Spec  # noqa: E402
from models.simple_cnn import SimpleCNN  # noqa: E402

CLASS_NAMES = ["Towards", "Away", "Pick", "Bend", "Sit", "Kneel", "Crawl", "Toes", "Limp", "SStep", "Scissor"]
SEED = 0


class SpecDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


class CNNGRU(nn.Module):
    """Freq-axis conv, then GRU along the remaining time axis of the 224x224 map."""

    def __init__(self, num_classes: int = 11):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((32, 224)),
        )
        self.gru = nn.GRU(32, 64, batch_first=True)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = self.conv(x)  # B,32,32,224
        seq = h.mean(dim=2).transpose(1, 2)  # B,224,32
        out, _ = self.gru(seq)
        return self.fc(out[:, -1])


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_split(split_csv: Path, protocol_keep=None):
    rows = list(csv.DictReader(split_csv.open(encoding="utf-8")))
    if protocol_keep:
        rows = [r for r in rows if r["split"] in protocol_keep]
    return {r["sample_id"]: r["split"] for r in rows}


def metrics(y_true, y_pred):
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=list(range(11)), zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "per_class": [
            {"class": CLASS_NAMES[i], "precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i])}
            for i in range(11)
        ],
    }


@torch.no_grad()
def eval_loader(net, loader, device):
    net.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device)
        logits = net(x)
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(ys), np.concatenate(ps)


def train_one(name, net, X, y, ids, assign, device, out, epochs=25, batch=16, lr=1e-3):
    tr = np.array([assign.get(i) == "train" for i in ids])
    va = np.array([assign.get(i) == "val" for i in ids])
    te = np.array([assign.get(i) == "test" for i in ids])
    if va.sum() == 0:
        # prefix split has val; random has val. If missing, peel 15% from train.
        tr_idx = np.where(tr)[0]
        rng = np.random.RandomState(SEED)
        rng.shuffle(tr_idx)
        n_va = max(1, int(0.15 * len(tr_idx)))
        va[tr_idx[:n_va]] = True
        tr[tr_idx[:n_va]] = False
    if te.sum() == 0 or tr.sum() == 0:
        return {"error": f"empty split train={tr.sum()} test={te.sum()}"}

    train_loader = DataLoader(SpecDS(X[tr], y[tr]), batch_size=batch, shuffle=True)
    val_loader = DataLoader(SpecDS(X[va], y[va]), batch_size=batch)
    test_loader = DataLoader(SpecDS(X[te], y[te]), batch_size=batch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    best = -1.0
    best_state = None
    patience, bad = 6, 0
    hist = []
    for epoch in range(epochs):
        net.train()
        losses = []
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = crit(net(xb), yb)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        yv, pv = eval_loader(net, val_loader, device)
        acc = accuracy_score(yv, pv)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": float(acc)})
        print(name, hist[-1], flush=True)
        if acc > best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    yt, pt = eval_loader(net, test_loader, device)
    m = metrics(yt, pt)
    m.update({"n_train": int(tr.sum()), "n_val": int(va.sum()), "n_test": int(te.sum()), "best_val_acc": float(best), "model": name})
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps(m, indent=2), encoding="utf-8")
    np.savetxt(out / "cm.csv", confusion_matrix(yt, pt, labels=list(range(11))), fmt="%d", delimiter=",")
    torch.save(best_state, out / "best.pt")
    with (out / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(hist[0].keys()))
        w.writeheader()
        w.writerows(hist)
    return m


def cm_fig(cm_path: Path, dest: Path, title: str):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    cm = np.loadtxt(cm_path, delimiter=",")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(11), CLASS_NAMES, rotation=45, ha="right", fontsize=7)
    ax.set_yticks(range(11), CLASS_NAMES, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    dest.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--random-split", required=True)
    ap.add_argument("--si-split", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--figures-dir", default="")
    args = ap.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    z = np.load(args.cache, allow_pickle=True)
    X, y, ids = z["X"], z["y"], [str(s) for s in z["sample_id"]]
    out = Path(args.out)

    models = {
        "simple_cnn": lambda: SimpleCNN(11, 1),
        "resnet18": lambda: ResNet18Spec(11, 1),
        "cnn_gru": lambda: CNNGRU(11),
    }
    summary = []
    for split_name, split_csv in [("random_record", args.random_split), ("prefix_tail4_disjoint", args.si_split)]:
        assign = load_split(Path(split_csv), protocol_keep=("train", "val", "test"))
        for mname, factory in models.items():
            net = factory().to(device)
            dest = out / split_name / mname
            print("====", split_name, mname, flush=True)
            m = train_one(mname, net, X, y, ids, assign, device, dest, epochs=args.epochs)
            m["split"] = split_name
            summary.append(m)
            if (dest / "cm.csv").exists():
                cm_fig(dest / "cm.csv", dest / "cm.png", f"{split_name} / {mname}")
    (out / "baseline_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    figdir = Path(args.figures_dir) if args.figures_dir else out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        plt = None
    if plt is not None:
        ok = [s for s in summary if "accuracy" in s]
        # Fig.5: copy CMs for the two splits, prefer resnet18 then first available
        for split_name in ("random_record", "prefix_tail4_disjoint"):
            pick = None
            for prefer in ("resnet18", "simple_cnn", "cnn_gru"):
                p = out / split_name / prefer / "cm.csv"
                if p.exists():
                    pick = (prefer, p)
                    break
            if pick:
                cm_fig(pick[1], figdir / f"fig5_cm_{split_name}.png", f"{split_name} / {pick[0]}")
        labels, accs, f1s = [], [], []
        for s in ok:
            labels.append(f"{s['split']}\n{s['model']}")
            accs.append(s["accuracy"])
            f1s.append(s["macro_f1"])
        fig, ax = plt.subplots(figsize=(9, 3.8))
        x = np.arange(len(labels))
        ax.bar(x - 0.18, accs, 0.36, label="Accuracy")
        ax.bar(x + 0.18, f1s, 0.36, label="Macro-F1")
        ax.set_xticks(x, labels, fontsize=7)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_ylabel("test")
        ax.set_title("Fig.6 random-record vs prefix-tail4-disjoint")
        fig.tight_layout()
        fig.savefig(figdir / "fig6_split_comparison.png", dpi=140)
        plt.close(fig)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
