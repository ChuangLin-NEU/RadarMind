#!/usr/bin/env python3
"""Phase 3 training. Acquisition-level split. Held-out radar never seen in LORO."""
from __future__ import annotations

import argparse
import csv
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from shared_motion.common import (  # noqa: E402
    AUG_SPEC,
    CLASS_NAMES,
    EmbNet,
    NanLossError,
    OUT,
    RADARS,
    SEED,
    infonce,
    load_packs,
    physics_aug,
    set_seed,
    views_for,
)

RUN_FIELDS = [
    "run_id",
    "model",
    "protocol",
    "train_radars",
    "test_radar",
    "lambda_contrast",
    "physics_aug",
    "seed",
    "n_train_acq",
    "n_val_acq",
    "n_test_acq",
    "n_train_views",
    "best_val_acc",
    "activity_accuracy",
    "macro_f1",
    "status",
    "note",
]


def _append(path: Path, fields: list[str], row: dict) -> None:
    new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in fields})


def _batch_from_views(views, timestamps, train_radars, rng, aug: bool):
    xs, ys, pair_a, pair_b = [], [], [], []
    for ts in timestamps:
        rads = [r for r in train_radars if r in views.get(ts, {})]
        if not rads:
            continue
        start = len(xs)
        for r in rads:
            x, y, _ = views[ts][r]
            xs.append(physics_aug(x, rng) if aug else x)
            ys.append(y)
        if len(rads) >= 2:
            a, b = rng.choice(len(rads), size=2, replace=False)
            pair_a.append(start + int(a))
            pair_b.append(start + int(b))
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, np.int64), pair_a, pair_b


@torch.no_grad()
def eval_views(net, views, radars, device, batch=32):
    net.eval()
    ys, ps = [], []
    xs, ybuf = [], []
    for ts, slot in views.items():
        for r in radars:
            if r not in slot:
                continue
            x, y, _ = slot[r]
            xs.append(x)
            ybuf.append(y)
            if len(xs) >= batch:
                xb = torch.from_numpy(np.stack(xs)).float().to(device)
                ps.append(net(xb).argmax(1).cpu().numpy())
                ys.extend(ybuf)
                xs, ybuf = [], []
    if xs:
        xb = torch.from_numpy(np.stack(xs)).float().to(device)
        ps.append(net(xb).argmax(1).cpu().numpy())
        ys.extend(ybuf)
    if not ys:
        return {"accuracy": 0.0, "macro_f1": 0.0, "n": 0}
    pred = np.concatenate(ps)
    y = np.asarray(ys)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n": int(len(y)),
    }


def train_one(job: dict, packs, device, epochs=20, batch_size=16, force=False) -> dict:
    run_id = job["run_id"]
    dest = OUT / "runs" / run_id
    dest.mkdir(parents=True, exist_ok=True)
    if (dest / "metrics.json").is_file() and not force:
        print("[skip]", run_id, flush=True)
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))

    set_seed(SEED)
    train_radars = job["train_radars"]
    test_radar = job["test_radar"]
    lam = float(job["lambda_contrast"])
    aug = bool(job["physics_aug"])
    tr_v = views_for(packs, train_radars, "train")
    va_v = views_for(packs, train_radars, "val")
    te_radars = [test_radar] if test_radar != "all" else list(RADARS)
    te_v = views_for(packs, te_radars, "test")

    net = EmbNet(len(CLASS_NAMES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3)
    crit = torch.nn.CrossEntropyLoss()
    amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    rng = np.random.RandomState(SEED)
    tss = list(tr_v.keys())
    best, best_state, bad = -1.0, None, 0
    hist = []

    for epoch in range(epochs):
        net.train()
        rng.shuffle(tss)
        losses = []
        for i in range(0, len(tss), batch_size):
            packed = _batch_from_views(tr_v, tss[i : i + batch_size], train_radars, rng, aug)
            if packed is None:
                continue
            X, y, pa, pb = packed
            xb = torch.from_numpy(X).float().to(device)
            yb = torch.from_numpy(y).to(device)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", enabled=amp):
                    logits, emb = net(xb, return_emb=True)
                    loss = crit(logits, yb)
                    if lam > 0 and len(pa) >= 2:
                        loss = loss + lam * infonce(emb[pa], emb[pb])
                if not torch.isfinite(loss):
                    info = {
                        "run_id": run_id,
                        "epoch": epoch,
                        "batch_i": i,
                        "loss": str(loss.detach().float().cpu()),
                        "n": int(len(y)),
                    }
                    (dest / "nan_batch.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                    raise NanLossError(f"NaN loss at epoch {epoch} batch {i}")
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise
                raise
            losses.append(float(loss.detach().float().cpu()))
        vm = eval_views(net, va_v, train_radars, device)
        hist.append({"epoch": epoch, "train_loss": float(np.mean(losses) if losses else 0), "val_acc": vm["accuracy"]})
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
        raise RuntimeError("no finite checkpoint")
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_views(net, te_v, te_radars, device)
    row = {
        "run_id": run_id,
        "model": job["model"],
        "protocol": job["protocol"],
        "train_radars": "+".join(train_radars),
        "test_radar": test_radar,
        "lambda_contrast": lam,
        "physics_aug": aug,
        "seed": SEED,
        "n_train_acq": len(tr_v),
        "n_val_acq": len(va_v),
        "n_test_acq": len(te_v),
        "n_train_views": sum(len(v) for v in tr_v.values()),
        "best_val_acc": float(best),
        "activity_accuracy": tm["accuracy"],
        "macro_f1": tm["macro_f1"],
        "status": "ok",
        "note": job.get("note", ""),
    }
    (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
        w.writeheader()
        w.writerows(hist)
    _append(OUT / "training_runs.csv", RUN_FIELDS, row)
    print("[done]", run_id, row["activity_accuracy"], row["macro_f1"], flush=True)
    return row


def jobs_stage_a() -> list[dict]:
    note = "Task A multi-radar shared representation; train 10+24+77 train acq only"
    return [
        {
            "run_id": "M0_shared",
            "model": "M0",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.0,
            "physics_aug": False,
            "note": note + "; CE only",
        },
        {
            "run_id": "M1_l01",
            "model": "M1",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "physics_aug": False,
            "note": note + "; CE+InfoNCE lambda=0.1",
        },
        {
            "run_id": "M1_l05",
            "model": "M1",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.5,
            "physics_aug": False,
            "note": note + "; CE+InfoNCE lambda=0.5",
        },
        {
            "run_id": "M1_l10",
            "model": "M1",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 1.0,
            "physics_aug": False,
            "note": note + "; CE+InfoNCE lambda=1.0",
        },
    ]


def pick_lambda() -> float:
    path = OUT / "lambda_choice.json"
    if path.is_file():
        return float(json.loads(path.read_text(encoding="utf-8"))["lambda_contrast"])
    best_l, best_v = 0.5, -1.0
    chosen = "M1_l05"
    for rid, lam in (("M1_l01", 0.1), ("M1_l05", 0.5), ("M1_l10", 1.0)):
        m = OUT / "runs" / rid / "metrics.json"
        if not m.is_file():
            continue
        rec = json.loads(m.read_text(encoding="utf-8"))
        v = float(rec.get("best_val_acc") or -1)
        if v > best_v:
            best_v, best_l, chosen = v, lam, rid
    rec = {
        "lambda_contrast": best_l,
        "selected_run": chosen,
        "criterion": "best source-radar val accuracy among {0.1, 0.5, 1.0}",
    }
    path.write_text(json.dumps(rec, indent=2), encoding="utf-8")
    src = OUT / "runs" / chosen / "best.pt"
    dst_dir = OUT / "runs" / "M1_shared"
    dst_dir.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        import shutil

        shutil.copy2(src, dst_dir / "best.pt")
        src_m = OUT / "runs" / chosen / "metrics.json"
        if src_m.is_file():
            row = json.loads(src_m.read_text(encoding="utf-8"))
            row["run_id"] = "M1_shared"
            row["note"] = f"selected {chosen} by val acc; " + row.get("note", "")
            (dst_dir / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
    return best_l


def jobs_stage_b(lam: float) -> list[dict]:
    loro = [
        (["10GHz", "24GHz"], "77GHz"),
        (["10GHz", "77GHz"], "24GHz"),
        (["24GHz", "77GHz"], "10GHz"),
    ]
    jobs = [
        {
            "run_id": "M2_shared",
            "model": "M2",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": lam,
            "physics_aug": True,
            "note": f"Task A Shared Motion v0.1; CE+InfoNCE+physics aug lambda={lam}",
        }
    ]
    for src, tgt in loro:
        tag = "_".join(s.replace("GHz", "") for s in src) + f"_to_{tgt.replace('GHz', '')}"
        jobs.append(
            {
                "run_id": f"M0_loro_{tag}",
                "model": "M0",
                "protocol": "leave_one_radar_out",
                "train_radars": src,
                "test_radar": tgt,
                "lambda_contrast": 0.0,
                "physics_aug": False,
                "note": f"Task B LORO source-only; held-out {tgt} unused in train/val/norm/contrastive",
            }
        )
        jobs.append(
            {
                "run_id": f"M2_loro_{tag}",
                "model": "M2",
                "protocol": "leave_one_radar_out",
                "train_radars": src,
                "test_radar": tgt,
                "lambda_contrast": lam,
                "physics_aug": True,
                "note": f"Task B LORO Shared Motion; pairs only among {src}; held-out {tgt} unused",
            }
        )
    return jobs


def run_job(job, packs, device, force=False) -> dict:
    dest = OUT / "runs" / job["run_id"]
    dest.mkdir(parents=True, exist_ok=True)
    batch = 16
    try:
        try:
            return train_one(job, packs, device, batch_size=batch, force=force)
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            print("[oom-retry]", job["run_id"], "batch 8", flush=True)
            torch.cuda.empty_cache()
            return train_one(job, packs, device, batch_size=8, force=True)
    except Exception as exc:  # noqa: BLE001
        row = {
            "run_id": job["run_id"],
            "model": job["model"],
            "protocol": job["protocol"],
            "train_radars": "+".join(job["train_radars"]),
            "test_radar": job["test_radar"],
            "lambda_contrast": job["lambda_contrast"],
            "physics_aug": job["physics_aug"],
            "seed": SEED,
            "status": "failed",
            "note": f"{type(exc).__name__}: {exc}",
            "activity_accuracy": "",
            "macro_f1": "",
        }
        (dest / "failed.json").write_text(
            json.dumps({"error": str(exc), "trace": traceback.format_exc()}, indent=2),
            encoding="utf-8",
        )
        _append(OUT / "failed_runs.csv", RUN_FIELDS, row)
        print("[failed]", job["run_id"], exc, flush=True)
        return row


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--only", default="", help="comma run_ids, empty=all remaining")
    ap.add_argument("--stage", default="all", choices=["a", "b", "all"])
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "augmentation_spec.json").write_text(json.dumps(AUG_SPEC, indent=2), encoding="utf-8")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    packs = load_packs()
    only = {x.strip() for x in args.only.split(",") if x.strip()}
    if args.stage in ("a", "all"):
        for job in jobs_stage_a():
            if only and job["run_id"] not in only:
                continue
            run_job(job, packs, device, force=args.force)
    lam = pick_lambda() if args.stage in ("b", "all") else 0.5
    if args.stage in ("b", "all"):
        for job in jobs_stage_b(lam):
            if only and job["run_id"] not in only:
                continue
            run_job(job, packs, device, force=args.force)


if __name__ == "__main__":
    main()
