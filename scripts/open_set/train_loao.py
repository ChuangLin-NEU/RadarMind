#!/usr/bin/env python3
"""Retrain 12-class PointNet-GRU per LOAO fold. Do not reuse 13-class checkpoint."""
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from models.point_encoder import PointNetGRU  # noqa: E402
from scripts.open_set.protocol import known_classes, loao_masks, remap_y, subject_assign  # noqa: E402
from scripts.pointcloud.phase1b_train import WindowDS  # noqa: E402

SEED = 42
N_CH = 3
EPOCHS = 20
LR = 1e-3
PATIENCE = 6


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def eval_acc(net, loader, device):
    net.eval()
    ys, ps = [], []
    for x, mask, y in loader:
        logits = net(x.to(device), mask.to(device))
        ys.append(y.numpy())
        ps.append(logits.argmax(1).cpu().numpy())
    yt, pt = np.concatenate(ys), np.concatenate(ps)
    return float(accuracy_score(yt, pt)), float(f1_score(yt, pt, average="macro", zero_division=0))


def train_one(X, mask, y, tr, va, te, ncls, device, dest: Path, batch: int, epochs: int) -> dict:
    dest.mkdir(parents=True, exist_ok=True)
    net = PointNetGRU(ncls, in_ch=N_CH).to(device)
    tl = DataLoader(WindowDS(X[tr], mask[tr], y[tr], N_CH, None), batch_size=batch, shuffle=True)
    vl = DataLoader(WindowDS(X[va], mask[va], y[va], N_CH, None), batch_size=32)
    el = DataLoader(WindowDS(X[te], mask[te], y[te], N_CH, None), batch_size=32)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    best, best_state, bad = -1.0, None, 0
    hist = []
    t0 = time.time()
    for epoch in range(epochs):
        net.train()
        losses = []
        for xb, mb, yb in tl:
            opt.zero_grad(set_to_none=True)
            xb, mb, yb = xb.to(device), mb.to(device), yb.to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = crit(net(xb, mb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.item()))
        vacc, vf1 = eval_acc(net, vl, device)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": vacc, "val_macro_f1": vf1})
        print(dest.name, hist[-1], flush=True)
        if vacc > best:
            best, bad = vacc, 0
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    net.load_state_dict(best_state)
    tacc, tf1 = eval_acc(net, el, device)
    torch.save(net.state_dict(), dest / "best.pt")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc", "val_macro_f1"])
        w.writeheader()
        w.writerows(hist)
    return {
        "best_val_acc": float(best),
        "known_test_acc": tacc,
        "known_test_macro_f1": tf1,
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test_known": int(te.sum()),
        "seconds": time.time() - t0,
        "epochs_ran": len(hist),
        "batch": batch,
        "seed": SEED,
        "features": "XYZ",
        "T": 32,
        "P": 64,
    }


def run_fold(z, assign, held_out: str, out_root: Path, device, epochs: int) -> dict:
    dest = out_root / "folds" / held_out.replace(" ", "_")
    if (dest / "best.pt").is_file() and (dest / "train_metrics.json").is_file():
        print("[skip-exists]", dest, flush=True)
        return json.loads((dest / "train_metrics.json").read_text(encoding="utf-8"))
    actions = np.array([str(a) for a in z["action"]])
    known = known_classes(held_out)
    masks = loao_masks(actions, assign, held_out)
    y = remap_y(z["y"], actions, known)
    tr, va, te = masks["train"], masks["val"], masks["test_known"]
    assert (y[tr] >= 0).all() and (y[va] >= 0).all() and (y[te] >= 0).all()
    assert masks["blocked_unseen_non_test"].sum() >= 0
    meta = {
        "held_out": held_out,
        "known_classes": known,
        "n_blocked_unseen_non_test": int(masks["blocked_unseen_non_test"].sum()),
        "n_test_unknown": int(masks["test_unknown"].sum()),
        "protocol": "LOAO + subject_independent",
        "note": "unseen action excluded from train, val, and later threshold fitting",
    }
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "fold_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    batch = 16
    last_err = None
    for b in (16, 8):
        try:
            m = train_one(z["X"], z["mask"], y, tr, va, te, len(known), device, dest, b, epochs)
            m.update(meta)
            (dest / "train_metrics.json").write_text(json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8")
            return m
        except RuntimeError as e:
            last_err = e
            if "out of memory" not in str(e).lower():
                raise
            print("[oom-retry]", b, flush=True)
            torch.cuda.empty_cache()
            batch = b
    raise last_err


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase5_open_set"))
    args = ap.parse_args()
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    if device.type != "cuda":
        print("[warn] CUDA missing; training on CPU", flush=True)
    z = np.load(ROOT / "results" / "phase1_pointcloud" / "cache" / "mri_windows.npz", allow_pickle=True)
    assign = subject_assign(z, ROOT / "data_splits" / "mri_subject_independent.csv")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for act in args.actions:
        print("==== fold", act, flush=True)
        m = run_fold(z, assign, act, out, device, args.epochs)
        rows.append(m)
    (out / "train_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[train-done]", flush=True)


if __name__ == "__main__":
    main()
