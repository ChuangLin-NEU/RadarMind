#!/usr/bin/env python3
"""Phase 8C: R2-D ResNet18 CE benchmark. No CORAL/M3/contrastive. Resume on metrics.json."""
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

OUT = ROOT / "results" / "phase8_physical_representation"
CACHE = OUT / "cache_r2"
CKPT = OUT / "checkpoints_r2"
SPLIT = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"
SEED = 42
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RADARS = ["10GHz", "24GHz", "77GHz"]
LR = 1e-3
BATCH = 16
EPOCHS = 20
PATIENCE = 6
RUN_FIELDS = [
    "run_id",
    "protocol",
    "train_radar",
    "test_radar",
    "method",
    "representation",
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


class EmbNet(nn.Module):
    def __init__(self, num_classes: int = 7):
        super().__init__()
        self.backbone = ResNet18Spec(num_classes, 1)

    def forward(self, x, return_emb=False):
        bb = self.backbone.backbone
        if x.ndim == 3:
            x = x.unsqueeze(1)
        if x.ndim == 4 and x.size(1) != 1:
            x = x[:, :1]
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


class SpecDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(np.asarray(X, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.int64))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_run(row: dict) -> None:
    path = OUT / "r2_training_runs.csv"
    rows = read_csv(path) if path.is_file() else []
    rows = [r for r in rows if r.get("run_id") != row["run_id"]]
    rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in RUN_FIELDS})


def load_r2_pack() -> dict:
    mapping = read_csv(OUT / "raw_mapping.csv")
    split = {str(r["timestamp"]): r for r in read_csv(SPLIT)}
    Xs, ys, splits, tss, acts, rads = [], [], [], [], [], []
    for r in mapping:
        ts, radar = str(r["timestamp"]), r["radar"]
        z = np.load(CACHE / radar / f"{ts}.npz")
        x = np.asarray(z["r2"], np.float32)
        if x.ndim == 3:
            x = x[0]
        Xs.append(x)
        ys.append(CLASS_NAMES.index(r["action"]))
        splits.append(split[ts]["split"])
        tss.append(ts)
        acts.append(r["action"])
        rads.append(radar)
    return {
        "X": np.stack(Xs),
        "y": np.asarray(ys, np.int64),
        "split": np.array(splits),
        "timestamp": np.array(tss),
        "action": np.array(acts),
        "radar": np.array(rads),
    }


def mask_for(pack, radars, split_name):
    rad = np.isin(pack["radar"], list(radars))
    return rad & (pack["split"] == split_name)


@torch.no_grad()
def eval_xy(net, X, y, device, batch=32):
    net.eval()
    if len(y) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "pred": np.zeros((0,), np.int64)}
    ps = []
    for i in range(0, len(y), batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        ps.append(net(xb).argmax(1).cpu().numpy())
    pred = np.concatenate(ps)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred": pred,
    }


def train_run(run_id, pack, train_radars, test_radar, protocol, note, device, force=False):
    dest = CKPT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").is_file() and not force:
        row = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        append_run(row)
        print("[skip]", run_id, row.get("accuracy"), flush=True)
        return row
    set_seed(SEED)
    tr = mask_for(pack, train_radars, "train")
    va = mask_for(pack, train_radars, "val")
    te = mask_for(pack, [test_radar] if isinstance(test_radar, str) else test_radar, "test")
    if int(tr.sum()) == 0 or int(va.sum()) == 0:
        row = {
            "run_id": run_id,
            "protocol": protocol,
            "train_radar": "+".join(train_radars),
            "test_radar": test_radar if isinstance(test_radar, str) else "+".join(test_radar),
            "method": "source_only_ce",
            "representation": "R2-D",
            "n_classes": 7,
            "seed": SEED,
            "accuracy": "",
            "macro_f1": "",
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "n_test": int(te.sum()),
            "best_val_acc": "",
            "note": "FAILED empty split",
        }
        append_run(row)
        print("[fail]", run_id, "empty split", flush=True)
        return row
    net = EmbNet(7).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    best, best_state, bad = -1.0, None, 0
    hist = []
    loader = DataLoader(SpecDS(pack["X"][tr], pack["y"][tr]), batch_size=BATCH, shuffle=True)
    for epoch in range(EPOCHS):
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
        vm = eval_xy(net, pack["X"][va], pack["y"][va], device)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": vm["accuracy"]})
        print(run_id, hist[-1], flush=True)
        if vm["accuracy"] > best:
            best = vm["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
            torch.save(best_state, dest / "best.pt")
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:
        raise RuntimeError(f"no checkpoint {run_id}")
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_xy(net, pack["X"][te], pack["y"][te], device)
    tgt = test_radar if isinstance(test_radar, str) else test_radar[0]
    np.savetxt(
        dest / "cm.csv",
        confusion_matrix(pack["y"][te], tm["pred"], labels=list(range(7))),
        fmt="%d",
        delimiter=",",
    )
    # per-sample correctness for clipping analysis
    np.savez(
        dest / "test_pred.npz",
        timestamp=pack["timestamp"][te],
        radar=pack["radar"][te],
        y=pack["y"][te],
        pred=tm["pred"],
        correct=(pack["y"][te] == tm["pred"]).astype(np.int8),
    )
    row = {
        "run_id": run_id,
        "protocol": protocol,
        "train_radar": "+".join(train_radars),
        "test_radar": test_radar if isinstance(test_radar, str) else "+".join(test_radar),
        "method": "source_only_ce",
        "representation": "R2-D",
        "n_classes": 7,
        "seed": SEED,
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "best_val_acc": float(best),
        "note": note,
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    append_run(row)
    print("[done]", run_id, f"{row['accuracy']:.4f}", f"{row['macro_f1']:.4f}", flush=True)
    return row


def jobs():
    out = []
    for r in RADARS:
        out.append((f"r2_within_{r[:2]}", [r], r, "within_radar", "R2-D CE within; same HP as Phase 7 source-only"))
    pairs = [("10GHz", "24GHz"), ("10GHz", "77GHz"), ("24GHz", "10GHz"), ("24GHz", "77GHz"), ("77GHz", "10GHz"), ("77GHz", "24GHz")]
    for s, t in pairs:
        out.append((f"r2_cross_{s[:2]}_to_{t[:2]}", [s], t, "single_source_cross_radar", "R2-D CE; target unused in train/val"))
    loro = [(["10GHz", "24GHz"], "77GHz"), (["10GHz", "77GHz"], "24GHz"), (["24GHz", "77GHz"], "10GHz")]
    for srcs, tgt in loro:
        rid = "r2_loro_" + "_".join(x[:2] for x in srcs) + f"_to_{tgt[:2]}"
        out.append((rid, srcs, tgt, "leave_one_radar_out", "R2-D CE LORO; held-out radar unused in train/val"))
    out.append(("r2_shared_ce", RADARS, "all", "multi_radar_shared", "R2-D CE shared; probes/retrieval only, not a new loss"))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)
    pack = load_r2_pack()
    print("loaded R2", pack["X"].shape, flush=True)
    for run_id, srcs, tgt, protocol, note in jobs():
        try:
            test = RADARS if tgt == "all" else tgt
            train_run(run_id, pack, srcs, test, protocol, note, device, force=args.force)
        except Exception as exc:
            print("[fail]", run_id, type(exc).__name__, exc, flush=True)
            append_run(
                {
                    "run_id": run_id,
                    "protocol": protocol,
                    "train_radar": "+".join(srcs),
                    "test_radar": tgt,
                    "method": "source_only_ce",
                    "representation": "R2-D",
                    "n_classes": 7,
                    "seed": SEED,
                    "accuracy": "",
                    "macro_f1": "",
                    "n_train": "",
                    "n_val": "",
                    "n_test": "",
                    "best_val_acc": "",
                    "note": f"FAILED {type(exc).__name__}: {exc}",
                }
            )


if __name__ == "__main__":
    main()
