#!/usr/bin/env python3
"""Phase 9: RadarBridge v2 known-radar adapters + shared encoder. R2-D only."""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase8.run_r2_benchmark import CLASS_NAMES, RADARS, load_r2_pack  # noqa: E402
from phase9.models import RadarBridgeV2, exact_pair_loss, pairwise_cosine_distance  # noqa: E402

OUT = ROOT / "results" / "phase9_radarbridge_v2"
CKPT = OUT / "checkpoints"
SEED = 42
LR = 1e-3
BATCH = 16
PAIR_BATCH = 8
EPOCHS = 20
PATIENCE = 6
LAMBDAS = [0.0, 0.1, 0.5, 1.0]
RUN_FIELDS = [
    "run_id",
    "model",
    "protocol",
    "lambda_pair",
    "train_radar",
    "test_radar",
    "accuracy",
    "macro_f1",
    "acc_10",
    "acc_24",
    "acc_77",
    "n_train",
    "n_val",
    "n_test",
    "best_val_acc",
    "adapter_params",
    "encoder_params",
    "seed",
    "note",
]


class SampleDS(Dataset):
    def __init__(self, X, y, radar):
        self.X = torch.from_numpy(np.asarray(X, np.float32))
        self.y = torch.from_numpy(np.asarray(y, np.int64))
        self.radar = list(radar)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i], self.radar[i]


class PairedDS(Dataset):
    def __init__(self, by_ts, timestamps, radars):
        self.by_ts = by_ts
        self.timestamps = list(timestamps)
        self.radars = list(radars)

    def __len__(self):
        return len(self.timestamps)

    def __getitem__(self, i):
        rec = self.by_ts[self.timestamps[i]]
        xs = [torch.from_numpy(rec[r]["x"]) for r in self.radars]
        y = int(rec[self.radars[0]]["y"])
        return xs, y, self.timestamps[i]


def set_seed(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def append_run(row: dict) -> None:
    path = OUT / "training_runs.csv"
    rows = [r for r in read_csv(path) if r.get("run_id") != row["run_id"]]
    rows.append({k: row.get(k, "") for k in RUN_FIELDS})
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=RUN_FIELDS)
        w.writeheader()
        w.writerows(rows)


def pack_index(pack) -> dict:
    by_ts = defaultdict(dict)
    for i in range(len(pack["y"])):
        ts = str(pack["timestamp"][i])
        r = pack["radar"][i]
        by_ts[ts][r] = {
            "x": np.asarray(pack["X"][i], np.float32),
            "y": int(pack["y"][i]),
            "split": pack["split"][i],
            "action": pack["action"][i],
        }
    return by_ts


def split_timestamps(by_ts, split_name, radars):
    out = []
    for ts, rec in by_ts.items():
        if all(r in rec and rec[r]["split"] == split_name for r in radars):
            out.append(ts)
    return sorted(out)


def mask_samples(pack, radars, split_name):
    return np.isin(pack["radar"], list(radars)) & (pack["split"] == split_name)


@torch.no_grad()
def eval_samples(net, X, y, radars, device, override=None, batch=32):
    net.eval()
    if len(y) == 0:
        return {"accuracy": 0.0, "macro_f1": 0.0, "pred": np.zeros((0,), np.int64), "emb": np.zeros((0, 512), np.float32)}
    preds, embs = [], []
    for i in range(0, len(y), batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        rb = list(radars[i : i + batch])
        logits, emb = net(xb, rb, override=override)
        preds.append(logits.argmax(1).cpu().numpy())
        embs.append(emb.cpu().numpy())
    pred = np.concatenate(preds)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "pred": pred,
        "emb": np.concatenate(embs),
    }


def per_radar_acc(pack, mask, pred):
    out = {}
    y = pack["y"][mask]
    rad = pack["radar"][mask]
    for r in RADARS:
        m = rad == r
        out[r] = float(accuracy_score(y[m], pred[m])) if m.any() else ""
    return out


def save_ckpt(net, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), dest / "best.pt")


def build_net(use_adapters, allowed, device):
    net = RadarBridgeV2(use_adapters=use_adapters, allowed_radars=allowed).to(device)
    return net


def train_samplewise(run_id, pack, train_radars, use_adapters, device, force, note, model_name):
    dest = CKPT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").is_file() and not force:
        row = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        append_run(row)
        print("[skip]", run_id, row.get("accuracy"), flush=True)
        return row
    set_seed(SEED)
    tr = mask_samples(pack, train_radars, "train")
    va = mask_samples(pack, train_radars, "val")
    te = mask_samples(pack, RADARS if set(train_radars) == set(RADARS) else train_radars, "test")
    if set(train_radars) == set(RADARS):
        te = mask_samples(pack, RADARS, "test")
    net = build_net(use_adapters, train_radars if use_adapters else RADARS, device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(SampleDS(pack["X"][tr], pack["y"][tr], pack["radar"][tr]), batch_size=BATCH, shuffle=True)
    best, best_state, bad, hist = -1.0, None, 0, []
    for epoch in range(EPOCHS):
        net.train()
        losses = []
        for xb, yb, rb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits, _ = net(xb, list(rb))
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
        vm = eval_samples(net, pack["X"][va], pack["y"][va], pack["radar"][va], device)
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
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_samples(net, pack["X"][te], pack["y"][te], pack["radar"][te], device)
    pra = per_radar_acc(pack, te, tm["pred"])
    row = {
        "run_id": run_id,
        "model": model_name,
        "protocol": "known_radar_joint" if set(train_radars) == set(RADARS) else "source_only",
        "lambda_pair": 0.0,
        "train_radar": "+".join(train_radars),
        "test_radar": "10GHz+24GHz+77GHz" if set(train_radars) == set(RADARS) else "+".join(train_radars),
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "acc_10": pra.get("10GHz", ""),
        "acc_24": pra.get("24GHz", ""),
        "acc_77": pra.get("77GHz", ""),
        "n_train": int(tr.sum()),
        "n_val": int(va.sum()),
        "n_test": int(te.sum()),
        "best_val_acc": float(best),
        "adapter_params": net.adapter_params() if use_adapters else 0,
        "encoder_params": net.encoder_params(),
        "seed": SEED,
        "note": note,
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    np.savez(dest / "test_pred.npz", timestamp=pack["timestamp"][te], radar=pack["radar"][te], y=pack["y"][te], pred=tm["pred"])
    append_run(row)
    print("[done]", run_id, f"{row['accuracy']:.4f}", flush=True)
    return row


def train_paired(run_id, by_ts, pack, train_radars, lambda_pair, device, force, note, model_name, protocol="known_radar_joint"):
    dest = CKPT / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").is_file() and not force:
        row = json.loads((dest / "metrics.json").read_text(encoding="utf-8"))
        append_run(row)
        print("[skip]", run_id, row.get("accuracy"), flush=True)
        return row
    set_seed(SEED)
    tr_ts = split_timestamps(by_ts, "train", train_radars)
    va_ts = split_timestamps(by_ts, "val", train_radars)
    test_radars = RADARS if set(train_radars) == set(RADARS) else train_radars
    te = mask_samples(pack, test_radars if protocol != "leave_one_radar_out" else train_radars, "test")
    if protocol == "known_radar_joint":
        te = mask_samples(pack, RADARS, "test")
    net = build_net(True, train_radars, device)
    opt = torch.optim.AdamW(net.parameters(), lr=LR)
    crit = nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    loader = DataLoader(PairedDS(by_ts, tr_ts, train_radars), batch_size=PAIR_BATCH, shuffle=True, collate_fn=_pair_collate)
    va_x, va_y, va_r = _flat_from_ts(by_ts, va_ts, train_radars)
    best, best_state, bad, hist = -1.0, None, 0, []
    for epoch in range(EPOCHS):
        net.train()
        losses, pair_losses = [], []
        for xs, yb, _ts in loader:
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp):
                embs, logits_all, ce = [], [], 0.0
                for r, xr in zip(train_radars, xs):
                    xr = xr.to(device)
                    logits, emb = net(xr, r)
                    ce = ce + crit(logits, yb)
                    embs.append(emb)
                    logits_all.append(logits)
                ce = ce / len(train_radars)
                if len(embs) == 3:
                    lp = exact_pair_loss(*embs)
                else:
                    lp = pairwise_cosine_distance(embs[0], embs[1])
                loss = ce + float(lambda_pair) * lp
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(float(loss.detach().cpu()))
            pair_losses.append(float(lp.detach().cpu()))
        vm = eval_samples(net, va_x, va_y, va_r, device)
        hist.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(losses)),
                "pair_loss": float(np.mean(pair_losses)),
                "val_acc": vm["accuracy"],
            }
        )
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
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_samples(net, pack["X"][te], pack["y"][te], pack["radar"][te], device)
    pra = per_radar_acc(pack, te, tm["pred"])
    row = {
        "run_id": run_id,
        "model": model_name,
        "protocol": protocol,
        "lambda_pair": float(lambda_pair),
        "train_radar": "+".join(train_radars),
        "test_radar": "10GHz+24GHz+77GHz" if protocol == "known_radar_joint" else "+".join(train_radars),
        "accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "acc_10": pra.get("10GHz", ""),
        "acc_24": pra.get("24GHz", ""),
        "acc_77": pra.get("77GHz", ""),
        "n_train": len(tr_ts) * len(train_radars),
        "n_val": len(va_ts) * len(train_radars),
        "n_test": int(te.sum()),
        "best_val_acc": float(best),
        "adapter_params": net.adapter_params(),
        "encoder_params": net.encoder_params(),
        "seed": SEED,
        "note": note,
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "pair_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    np.savez(dest / "test_pred.npz", timestamp=pack["timestamp"][te], radar=pack["radar"][te], y=pack["y"][te], pred=tm["pred"])
    append_run(row)
    print("[done]", run_id, f"{row['accuracy']:.4f}", "val", best, flush=True)
    return row


def _pair_collate(batch):
    xs_lists = list(zip(*[b[0] for b in batch]))
    xs = [torch.stack(list(t), 0) for t in xs_lists]
    y = torch.tensor([b[1] for b in batch], dtype=torch.long)
    ts = [b[2] for b in batch]
    return xs, y, ts


def _flat_from_ts(by_ts, timestamps, radars):
    X, y, r = [], [], []
    for ts in timestamps:
        for radar in radars:
            X.append(by_ts[ts][radar]["x"])
            y.append(by_ts[ts][radar]["y"])
            r.append(radar)
    return np.stack(X), np.asarray(y, np.int64), np.array(r)


def select_lambda(rows):
    best = max(rows, key=lambda r: (float(r["best_val_acc"]), -float(r["lambda_pair"])))
    return best


def train_loro(pack, by_ts, srcs, tgt, lambda_pair, device, force):
    rid = "loro_" + "_".join(x[:2] for x in srcs) + f"_to_{tgt[:2]}"
    note = f"LORO adapters {srcs}; held-out {tgt} never trained; lambda={lambda_pair}"
    row = train_paired(
        rid,
        by_ts,
        pack,
        srcs,
        lambda_pair,
        device,
        force,
        note,
        "A2_LORO" if lambda_pair > 0 else "A1_LORO",
        protocol="leave_one_radar_out",
    )
    dest = CKPT / rid
    net = build_net(True, srcs, device)
    net.load_state_dict(torch.load(dest / "best.pt", map_location="cpu"))
    net.to(device).eval()
    te = mask_samples(pack, [tgt], "test")
    X, y, rad = pack["X"][te], pack["y"][te], pack["radar"][te]
    variants = [("identity", "identity")]
    for s in srcs:
        variants.append((s, s))
    recs = []
    for name, override in variants:
        ev = eval_samples(net, X, y, rad, device, override=override)
        recs.append(
            {
                "run_id": rid,
                "train_radar": "+".join(srcs),
                "held_out": tgt,
                "test_adapter": name,
                "accuracy": ev["accuracy"],
                "macro_f1": ev["macro_f1"],
                "n_test": int(te.sum()),
                "lambda_pair": lambda_pair,
                "note": "no target adapter; no target unlabeled fit",
            }
        )
        print(f"[loro] {rid} adapter={name} acc={ev['accuracy']:.4f}", flush=True)
    return row, recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    CKPT.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, torch.cuda.get_device_name(0) if device.type == "cuda" else "", flush=True)
    pack = load_r2_pack()
    by_ts = pack_index(pack)
    print("loaded R2", pack["X"].shape, "acquisitions", len(by_ts), flush=True)

    dummy = RadarBridgeV2(True, RADARS)
    param_info = {
        "adapter_each": sum(p.numel() for p in dummy.adapters["10GHz"].parameters()),
        "adapter_all_three": dummy.adapter_params(),
        "shared_encoder": dummy.encoder_params(),
        "representation": "R2-D",
        "r2v": "NO",
        "physics_gate": "EFFECTIVE_FC_UNRESOLVED",
    }
    (OUT / "model_params.json").write_text(json.dumps(param_info, indent=2), encoding="utf-8")
    print("params", param_info, flush=True)

    train_samplewise("A0", pack, RADARS, False, device, args.force, "shared encoder only, CE, no radar-specific adapter", "A0")
    train_samplewise("A1", pack, RADARS, True, device, args.force, "radar-specific adapters + shared encoder + CE", "A1")

    lam_rows = []
    for lam in LAMBDAS:
        tag = str(lam).replace(".", "p")
        row = train_paired(
            f"A2_lam{tag}",
            by_ts,
            pack,
            RADARS,
            lam,
            device,
            args.force,
            f"adapters + CE + exact-pair; lambda={lam}",
            "A2_search",
        )
        lam_rows.append(row)
    chosen = select_lambda(lam_rows)
    chosen_lam = float(chosen["lambda_pair"])
    (OUT / "lambda_selection.json").write_text(
        json.dumps(
            {
                "candidates": [{k: r[k] for k in ("run_id", "lambda_pair", "best_val_acc", "accuracy")} for r in lam_rows],
                "selected_run_id": chosen["run_id"],
                "selected_lambda": chosen_lam,
                "rule": "max acquisition-split val_acc; tie → smaller lambda",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    # copy selected as A2
    src = CKPT / chosen["run_id"]
    dst = CKPT / "A2"
    dst.mkdir(parents=True, exist_ok=True)
    for name in ("best.pt", "metrics.json", "history.csv", "test_pred.npz"):
        p = src / name
        if p.is_file():
            (dst / name).write_bytes(p.read_bytes())
    a2 = dict(chosen)
    a2["run_id"] = "A2"
    a2["model"] = "A2"
    a2["note"] = f"selected lambda={chosen_lam} by val_acc from {chosen['run_id']}"
    (dst / "metrics.json").write_text(json.dumps(a2, indent=2), encoding="utf-8")
    append_run(a2)
    print("[A2 selected]", chosen["run_id"], "lambda", chosen_lam, flush=True)

    loro_rows = []
    for srcs, tgt in ((["10GHz", "24GHz"], "77GHz"), (["10GHz", "77GHz"], "24GHz"), (["24GHz", "77GHz"], "10GHz")):
        _, recs = train_loro(pack, by_ts, srcs, tgt, chosen_lam, device, args.force)
        loro_rows.extend(recs)
    fields = ["run_id", "train_radar", "held_out", "test_adapter", "accuracy", "macro_f1", "n_test", "lambda_pair", "note"]
    with (OUT / "loro_adapter_transfer.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(loro_rows)
    print("phase9 training done", flush=True)


if __name__ == "__main__":
    main()
