#!/usr/bin/env python3
"""R1 ResNet18 within + single-source transfer. seed=42. Same capacity/rules as Phase 2. Writes Phase 7 only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import CLASS_NAMES, EmbNet, OUT, SEED, append_csv, load_band, set_seed  # noqa: E402

RUN_FIELDS = [
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
# keep run_id first for readability
RUN_FIELDS = [
    "run_id",
    "model",
    "protocol",
    "train_radar",
    "test_radar",
    "method",
    "representation",
] + RUN_FIELDS


class SpecDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X.astype(np.float32))
        self.y = torch.from_numpy(y.astype(np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


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


def train_run(run_id, src, tgt, device, epochs=20, force=False):
    dest = OUT / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").is_file() and not force:
        print("[skip]", run_id, flush=True)
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
    set_seed(SEED)
    S, T = load_band(src, "R1"), load_band(tgt, "R1")
    tr, va, te = S["split"] == "train", S["split"] == "val", T["split"] == "test"
    net = EmbNet(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, bad = -1.0, None, 0
    hist = []
    loader = DataLoader(SpecDS(S["X"][tr], S["y"][tr]), batch_size=16, shuffle=True)
    for epoch in range(epochs):
        net.train()
        losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                loss = crit(net(xb), yb)
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
            torch.save(best_state, dest / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= 6:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint {run_id}")
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_xy(net, T["X"][te], T["y"][te], device)
    np.savetxt(dest / "cm.csv", confusion_matrix(T["y"][te], tm["pred"], labels=list(range(7))), fmt="%d", delimiter=",")
    row = {
        "run_id": run_id,
        "model": "R1_source_only",
        "protocol": "acquisition_timestamp_disjoint",
        "train_radar": src,
        "test_radar": tgt,
        "method": "source_only",
        "representation": "R1",
        "n_classes": 7,
        "seed": SEED,
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "best_val_acc": float(best),
        "note": "R1 fair PNG; source-only; test radar unseen in training",
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    append_csv(OUT / "training_runs.csv", RUN_FIELDS, row)
    print("[done]", run_id, row["accuracy"], row["macro_f1"], flush=True)
    return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "runs").mkdir(exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    jobs = [
        ("r1_within_10", "10GHz", "10GHz"),
        ("r1_within_24", "24GHz", "24GHz"),
        ("r1_within_77", "77GHz", "77GHz"),
        ("r1_cross_77_to_10", "77GHz", "10GHz"),
        ("r1_cross_10_to_77", "10GHz", "77GHz"),
        ("r1_cross_77_to_24", "77GHz", "24GHz"),
        ("r1_cross_24_to_77", "24GHz", "77GHz"),
        ("r1_cross_10_to_24", "10GHz", "24GHz"),
        ("r1_cross_24_to_10", "24GHz", "10GHz"),
    ]
    for rid, src, tgt in jobs:
        try:
            train_run(rid, src, tgt, device, force=args.force)
        except Exception as exc:  # noqa: BLE001
            dest = OUT / "runs" / rid
            dest.mkdir(parents=True, exist_ok=True)
            (dest / "failure.json").write_text(json.dumps({"run_id": rid, "error": str(exc)}, indent=2), encoding="utf-8")
            print("[fail]", rid, exc, flush=True)
            raise


if __name__ == "__main__":
    main()
