#!/usr/bin/env python3
"""Within-radar, zero-shot cross-radar, and CORAL UDA. seed=42. Resume if metrics exist."""
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
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from models.resnet18_spec import ResNet18Spec  # noqa: E402

OUT = ROOT / "results" / "phase2_cross_radar"
CACHE = OUT / "cache"
SEED = 42
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RUN_FIELDS = [
    "run_id",
    "protocol",
    "train_radar",
    "test_radar",
    "method",
    "n_classes",
    "seed",
    "accuracy",
    "macro_f1",
    "n_train",
    "n_val",
    "n_test",
    "best_val_acc",
    "note",
]


class SpecDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_band(freq: str):
    key = freq.replace("GHz", "")
    z = np.load(CACHE / f"ci4r_{key}ghz_png224.npz", allow_pickle=True)
    return {
        "X": z["X"],
        "y": z["y"],
        "split": np.array([str(s) for s in z["split"]]),
        "timestamp": np.array([str(s) for s in z["timestamp"]]),
        "sample_id": np.array([str(s) for s in z["sample_id"]]),
        "action": np.array([str(s) for s in z["action"]]),
    }


def coral_loss(hs, ht):
    def cov(h):
        h = h - h.mean(0, keepdim=True)
        n = max(h.size(0) - 1, 1)
        return h.t() @ h / n

    d = hs.size(1)
    return ((cov(hs) - cov(ht)) ** 2).sum() / (4.0 * d * d)


class EmbNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.backbone = ResNet18Spec(num_classes, 1)

    def forward(self, x, return_emb=False):
        bb = self.backbone.backbone
        if x.ndim == 3:
            x = x.unsqueeze(1)
        h = bb.conv1(x)
        h = bb.bn1(h)
        h = bb.relu(h)
        h = bb.maxpool(h)
        h = bb.layer1(h)
        h = bb.layer2(h)
        h = bb.layer3(h)
        h = bb.layer4(h)
        h = bb.avgpool(h)
        emb = torch.flatten(h, 1)
        logits = bb.fc(emb)
        if return_emb:
            return logits, emb
        return logits


@torch.no_grad()
def eval_xy(net, X, y, device, batch=32):
    net.eval()
    ps = []
    for i in range(0, len(y), batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        ps.append(net(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(ps) if ps else np.zeros((0,), np.int64)
    return {
        "accuracy": float(accuracy_score(y, pred)) if len(y) else 0.0,
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)) if len(y) else 0.0,
        "pred": pred,
    }


def append_run(row: dict) -> None:
    path = OUT / "training_runs.csv"
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in RUN_FIELDS})


def train_run(run_id, src, tgt, method, device, epochs=20, coral_w=1.0, force=False):
    dest = OUT / "runs" / run_id
    if (dest / "metrics.json").is_file() and not force:
        print("[skip]", run_id, flush=True)
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    dest.mkdir(parents=True, exist_ok=True)
    set_seed(SEED)
    S, T = load_band(src), load_band(tgt)
    tr, va, te = S["split"] == "train", S["split"] == "val", T["split"] == "test"
    net = EmbNet(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    tgt_tr = T["split"] == "train"
    best, best_state, bad = -1.0, None, 0
    hist = []
    Xs, ys = S["X"][tr], S["y"][tr]
    Xt = T["X"][tgt_tr] if method == "coral_uda" else None
    loader = DataLoader(SpecDS(Xs, ys), batch_size=16, shuffle=True)
    for epoch in range(epochs):
        net.train()
        losses = []
        tgt_i = 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                logits, emb = net(xb, return_emb=True)
                loss = crit(logits, yb)
                if method == "coral_uda" and Xt is not None and len(Xt):
                    j = np.random.randint(0, len(Xt), size=min(16, len(Xt)))
                    xt = torch.from_numpy(Xt[j]).float().to(device)
                    _, embt = net(xt, return_emb=True)
                    loss = loss + coral_w * coral_loss(emb, embt)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        vm = eval_xy(net, S["X"][va], S["y"][va], device)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": vm["accuracy"]})
        print(run_id, hist[-1], flush=True)
        if vm["accuracy"] > best:
            best = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                break
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_xy(net, T["X"][te], T["y"][te], device)
    torch.save(best_state, dest / "best.pt")
    np.savetxt(dest / "cm.csv", confusion_matrix(T["y"][te], tm["pred"], labels=list(range(7))), fmt="%d", delimiter=",")
    row = {
        "run_id": run_id,
        "protocol": "acquisition_timestamp_disjoint",
        "train_radar": src,
        "test_radar": tgt,
        "method": method,
        "n_classes": 7,
        "seed": SEED,
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "best_val_acc": float(best),
        "note": "UDA uses target train unlabeled only" if method == "coral_uda" else "source-only; test radar unseen in training",
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    append_run(row)
    print("[done]", run_id, row["accuracy"], row["macro_f1"], flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    jobs = [
        ("within_10", "10GHz", "10GHz", "source_only"),
        ("within_24", "24GHz", "24GHz", "source_only"),
        ("within_77", "77GHz", "77GHz", "source_only"),
        ("cross_77_to_10", "77GHz", "10GHz", "source_only"),
        ("cross_10_to_77", "10GHz", "77GHz", "source_only"),
        ("cross_77_to_24", "77GHz", "24GHz", "source_only"),
        ("cross_24_to_77", "24GHz", "77GHz", "source_only"),
        ("cross_10_to_24", "10GHz", "24GHz", "source_only"),
        ("cross_24_to_10", "24GHz", "10GHz", "source_only"),
        ("coral_uda_77_to_10", "77GHz", "10GHz", "coral_uda"),
        ("coral_uda_77_to_24", "77GHz", "24GHz", "coral_uda"),
    ]
    for rid, src, tgt, method in jobs:
        train_run(rid, src, tgt, method, device, force=args.force)


if __name__ == "__main__":
    main()
