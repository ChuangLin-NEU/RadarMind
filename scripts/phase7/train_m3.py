#!/usr/bin/env python3
"""R1 M0/M1/M3. M3 = CE + paired contrastive + GRL radar head. Phase 7 only."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import (  # noqa: E402
    CLASS_NAMES,
    EmbNet,
    M3Net,
    OUT,
    RADARS,
    SEED,
    append_csv,
    infonce,
    load_packs,
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
    "lambda_adv",
    "representation",
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


def _batch_from_views(views, timestamps, train_radars, rng):
    xs, ys, rs, pair_a, pair_b = [], [], [], [], []
    rmap = {r: i for i, r in enumerate(train_radars)}
    for ts in timestamps:
        rads = [r for r in train_radars if r in views.get(ts, {})]
        if not rads:
            continue
        start = len(xs)
        for r in rads:
            x, y, _ = views[ts][r]
            xs.append(x)
            ys.append(y)
            rs.append(rmap[r])
        if len(rads) >= 2:
            a, b = rng.choice(len(rads), size=2, replace=False)
            pair_a.append(start + int(a))
            pair_b.append(start + int(b))
    if not xs:
        return None
    return np.stack(xs), np.asarray(ys, np.int64), np.asarray(rs, np.int64), pair_a, pair_b


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
                out = net.encoder(xb) if isinstance(net, M3Net) else net(xb)
                ps.append(out.argmax(1).cpu().numpy())
                ys.extend(ybuf)
                xs, ybuf = [], []
    if xs:
        xb = torch.from_numpy(np.stack(xs)).float().to(device)
        out = net.encoder(xb) if isinstance(net, M3Net) else net(xb)
        ps.append(out.argmax(1).cpu().numpy())
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
    fail_p = dest / "failure.json"
    if (dest / "metrics.json").is_file() and not force:
        print("[skip]", run_id, flush=True)
        return json.loads((dest / "metrics.json").read_text(encoding="utf-8"))

    set_seed(SEED)
    train_radars = job["train_radars"]
    test_radar = job["test_radar"]
    lam_c = float(job["lambda_contrast"])
    lam_a = float(job.get("lambda_adv", 0.0))
    use_m3 = job["model"] == "M3"
    tr_v = views_for(packs, train_radars, "train")
    va_v = views_for(packs, train_radars, "val")
    te_radars = [test_radar] if test_radar != "all" else list(RADARS)
    te_v = views_for(packs, te_radars, "test")

    if use_m3:
        net = M3Net(len(CLASS_NAMES), len(train_radars)).to(device)
    else:
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
            packed = _batch_from_views(tr_v, tss[i : i + batch_size], train_radars, rng)
            if packed is None:
                continue
            X, y, rlab, pa, pb = packed
            xb = torch.from_numpy(X).float().to(device)
            yb = torch.from_numpy(y).to(device)
            rb = torch.from_numpy(rlab).to(device)
            opt.zero_grad(set_to_none=True)
            try:
                with torch.amp.autocast("cuda", enabled=amp):
                    if use_m3:
                        logits, emb, rlog = net(xb, lambd_adv=lam_a, return_emb=True)
                        loss = crit(logits, yb) + crit(rlog, rb)
                    else:
                        logits, emb = net(xb, return_emb=True)
                        loss = crit(logits, yb)
                    if lam_c > 0 and len(pa) >= 2:
                        loss = loss + lam_c * infonce(emb[pa], emb[pb])
                if not torch.isfinite(loss):
                    info = {"run_id": run_id, "epoch": epoch, "batch_i": i, "loss": str(loss.detach().float().cpu())}
                    (dest / "nan_batch.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
                    raise RuntimeError(f"NaN loss {run_id} epoch {epoch}")
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            except RuntimeError as exc:
                fail_p.write_text(json.dumps({"run_id": run_id, "error": str(exc)}, indent=2), encoding="utf-8")
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
        raise RuntimeError(f"no finite checkpoint {run_id}")
    net.load_state_dict(best_state)
    net.to(device)
    tm = eval_views(net, te_v, te_radars, device)
    row = {
        "run_id": run_id,
        "model": job["model"],
        "protocol": job["protocol"],
        "train_radars": "+".join(train_radars),
        "test_radar": test_radar,
        "lambda_contrast": lam_c,
        "lambda_adv": lam_a,
        "representation": "R1",
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
    append_csv(OUT / "training_runs.csv", RUN_FIELDS, row)
    print("[done]", run_id, row["activity_accuracy"], row["macro_f1"], flush=True)
    return row


def jobs_task_a() -> list[dict]:
    note = "Task A on R1; train 10+24+77 train acq only"
    return [
        {
            "run_id": "R1_M0_shared",
            "model": "M0",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.0,
            "lambda_adv": 0.0,
            "note": note + "; CE only",
        },
        {
            "run_id": "R1_M1_shared",
            "model": "M1",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "lambda_adv": 0.0,
            "note": note + "; CE+InfoNCE lambda=0.1",
        },
        {
            "run_id": "R1_M3_adv001",
            "model": "M3",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "lambda_adv": 0.01,
            "note": note + "; CE+contrast+GRL adv=0.01",
        },
        {
            "run_id": "R1_M3_adv005",
            "model": "M3",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "lambda_adv": 0.05,
            "note": note + "; CE+contrast+GRL adv=0.05",
        },
        {
            "run_id": "R1_M3_adv010",
            "model": "M3",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "lambda_adv": 0.1,
            "note": note + "; CE+contrast+GRL adv=0.1",
        },
        {
            "run_id": "R1_M3_adv050",
            "model": "M3",
            "protocol": "multi_radar_shared",
            "train_radars": list(RADARS),
            "test_radar": "all",
            "lambda_contrast": 0.1,
            "lambda_adv": 0.5,
            "note": note + "; CE+contrast+GRL adv=0.5",
        },
    ]


def jobs_loro(lam_adv: float | None) -> list[dict]:
    folds = [
        ("10_24_to_77", ["10GHz", "24GHz"], "77GHz"),
        ("10_77_to_24", ["10GHz", "77GHz"], "24GHz"),
        ("24_77_to_10", ["24GHz", "77GHz"], "10GHz"),
    ]
    out = []
    for tag, src, tgt in folds:
        out.append(
            {
                "run_id": f"R1_M0_loro_{tag}",
                "model": "M0",
                "protocol": "leave_one_radar_out",
                "train_radars": src,
                "test_radar": tgt,
                "lambda_contrast": 0.0,
                "lambda_adv": 0.0,
                "note": f"LORO R1 CE; {tgt} unused in train/val/contrastive/domain/HP",
            }
        )
        if lam_adv is None:
            for a in (0.01, 0.05, 0.1, 0.5):
                aid = f"{a:.2f}".replace(".", "")
                out.append(
                    {
                        "run_id": f"R1_M3_loro_{tag}_adv{aid}",
                        "model": "M3",
                        "protocol": "leave_one_radar_out",
                        "train_radars": src,
                        "test_radar": tgt,
                        "lambda_contrast": 0.1,
                        "lambda_adv": a,
                        "note": f"LORO R1 M3; domain head sees only {src}; {tgt} unused",
                    }
                )
        else:
            out.append(
                {
                    "run_id": f"R1_M3_loro_{tag}",
                    "model": "M3",
                    "protocol": "leave_one_radar_out",
                    "train_radars": src,
                    "test_radar": tgt,
                    "lambda_contrast": 0.1,
                    "lambda_adv": lam_adv,
                    "note": f"LORO R1 M3; domain head sees only {src}; {tgt} unused",
                }
            )
    return out


def pick_m3_shared() -> dict:
    path = OUT / "m3_lambda_choice.json"
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    best = None
    for rid, lam in (
        ("R1_M3_adv001", 0.01),
        ("R1_M3_adv005", 0.05),
        ("R1_M3_adv010", 0.1),
        ("R1_M3_adv050", 0.5),
    ):
        m = OUT / "runs" / rid / "metrics.json"
        if not m.is_file():
            continue
        row = json.loads(m.read_text(encoding="utf-8"))
        val = float(row["best_val_acc"])
        if best is None or val > best["best_val_acc"]:
            best = {"run_id": rid, "lambda_adv": lam, "best_val_acc": val, "selection": "source_val_activity"}
    if best is None:
        raise RuntimeError("no M3 shared runs")
    # copy chosen weights to R1_M3_shared
    src = OUT / "runs" / best["run_id"] / "best.pt"
    dest = OUT / "runs" / "R1_M3_shared"
    dest.mkdir(parents=True, exist_ok=True)
    import shutil

    shutil.copy2(src, dest / "best.pt")
    chosen = json.loads((OUT / "runs" / best["run_id"] / "metrics.json").read_text(encoding="utf-8"))
    chosen = dict(chosen)
    chosen["run_id"] = "R1_M3_shared"
    chosen["note"] = chosen.get("note", "") + f"; selected {best['run_id']} by source val"
    (dest / "metrics.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    path.write_text(json.dumps(best, indent=2), encoding="utf-8")
    return best


def pick_loro_m3() -> list[dict]:
    """Per-fold source-val selection. Held-out radar never used."""
    folds = [
        "10_24_to_77",
        "10_77_to_24",
        "24_77_to_10",
    ]
    chosen = []
    for tag in folds:
        best = None
        for a in (0.01, 0.05, 0.1, 0.5):
            aid = f"{a:.2f}".replace(".", "")
            rid = f"R1_M3_loro_{tag}_adv{aid}"
            m = OUT / "runs" / rid / "metrics.json"
            if not m.is_file():
                continue
            row = json.loads(m.read_text(encoding="utf-8"))
            val = float(row["best_val_acc"])
            if best is None or val > best["best_val_acc"]:
                best = {"fold": tag, "run_id": rid, "lambda_adv": a, "best_val_acc": val}
        if best:
            import shutil

            dest = OUT / "runs" / f"R1_M3_loro_{tag}"
            dest.mkdir(parents=True, exist_ok=True)
            shutil.copy2(OUT / "runs" / best["run_id"] / "best.pt", dest / "best.pt")
            row = json.loads((OUT / "runs" / best["run_id"] / "metrics.json").read_text(encoding="utf-8"))
            row = dict(row)
            row["run_id"] = f"R1_M3_loro_{tag}"
            (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
            chosen.append(best)
    (OUT / "m3_loro_lambda_choice.json").write_text(json.dumps(chosen, indent=2), encoding="utf-8")
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["task_a", "loro", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device", device, flush=True)
    packs = load_packs("R1")
    if args.stage in ("task_a", "all"):
        for job in jobs_task_a():
            train_one(job, packs, device, force=args.force)
        print("[pick]", pick_m3_shared(), flush=True)
    if args.stage in ("loro", "all"):
        for job in jobs_loro(None):
            train_one(job, packs, device, force=args.force)
        print("[pick-loro]", pick_loro_m3(), flush=True)


if __name__ == "__main__":
    main()
