#!/usr/bin/env python3
"""MSP / Energy / Prototype Distance. Thresholds from validation known only."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from models.point_encoder import PointNetGRU  # noqa: E402
from scripts.open_set.protocol import known_classes, loao_masks, remap_y, subject_assign  # noqa: E402

N_CH = 3
ENERGY_T = 1.0
KNOWN_KEEP_Q = 0.05  # 5th percentile of known scores → ~5% known-val reject


@torch.no_grad()
def forward_pack(net, X, mask, idx, device, batch=32):
    net.eval()
    logits_l, emb_l = [], []
    for i in range(0, len(idx), batch):
        sl = idx[i : i + batch]
        x = torch.from_numpy(X[sl][..., :N_CH].astype(np.float32)).to(device)
        m = torch.from_numpy(mask[sl].astype(np.bool_)).to(device)
        h = net.enc(x, m)
        out, _ = net.gru(h)
        emb = out[:, -1]
        logits = net.fc(emb)
        logits_l.append(logits.float().cpu().numpy())
        emb_l.append(emb.float().cpu().numpy())
    return np.concatenate(logits_l, 0), np.concatenate(emb_l, 0)


def softmax(z):
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def energy_known_score(logits, T=ENERGY_T):
    z = logits / T
    m = z.max(axis=1, keepdims=True)
    return T * (m[:, 0] + np.log(np.exp(z - m).sum(axis=1)))


def fpr95(y_true, ood_score):
    """Unknown is positive. ood_score high = more unknown."""
    fpr, tpr, _ = roc_curve(y_true, ood_score)
    if (tpr >= 0.95).any():
        return float(fpr[np.argmax(tpr >= 0.95)])
    return float(fpr[-1])


def detect_metrics(y_unknown, ood_score):
    return {
        "AUROC": float(roc_auc_score(y_unknown, ood_score)),
        "AUPR": float(average_precision_score(y_unknown, ood_score)),
        "FPR95": fpr95(y_unknown, ood_score),
    }


def threshold_from_val(known_score_val: np.ndarray) -> float:
    return float(np.quantile(known_score_val, KNOWN_KEEP_Q))


def open_set_pred(pred_cls, known_score, thr, known_names):
    out = []
    for c, s in zip(pred_cls, known_score):
        if s < thr:
            out.append("UNKNOWN")
        else:
            out.append(known_names[int(c)])
    return np.array(out)


def eval_method(name, known_score_val, known_score_te, unk_score_te, pred_cls, y_true_names, known_names):
    thr = threshold_from_val(known_score_val)
    ood_te = np.concatenate([-known_score_te, -unk_score_te])
    y_u = np.concatenate([np.zeros(len(known_score_te), dtype=np.int64), np.ones(len(unk_score_te), dtype=np.int64)])
    det = detect_metrics(y_u, ood_te)
    labels_k = open_set_pred(pred_cls, known_score_te, thr, known_names)
    labels_u = open_set_pred(np.zeros(len(unk_score_te), dtype=int), unk_score_te, thr, known_names)
    pred_all = np.concatenate([labels_k, labels_u])
    gt_all = np.concatenate([y_true_names, np.array(["UNKNOWN"] * len(unk_score_te))])
    names = known_names + ["UNKNOWN"]
    osa = float(accuracy_score(gt_all, pred_all))
    cm = confusion_matrix(gt_all, pred_all, labels=names)
    pred_closed = [known_names[int(c)] for c in pred_cls]
    known_acc = float(accuracy_score(y_true_names, pred_closed))
    known_f1 = float(f1_score(y_true_names, pred_closed, average="macro", zero_division=0))
    return {
        "method": name,
        "threshold": thr,
        "threshold_rule": f"quantile_{KNOWN_KEEP_Q}_of_val_known_score",
        "known_accuracy": known_acc,
        "known_macro_f1": known_f1,
        "open_set_accuracy": osa,
        **det,
        "n_known_test": int(len(known_score_te)),
        "n_unknown_test": int(len(unk_score_te)),
        "cm_labels": names,
        "cm": cm.tolist(),
        "score_known_test": known_score_te.tolist(),
        "score_unknown_test": unk_score_te.tolist(),
        "pred_open_known": labels_k.tolist(),
        "pred_open_unknown": labels_u.tolist(),
        "gt_known": y_true_names.tolist(),
    }


def eval_fold(z, assign, held_out: str, ckpt: Path, device) -> dict:
    known = known_classes(held_out)
    actions = np.array([str(a) for a in z["action"]])
    masks = loao_masks(actions, assign, held_out)
    y = remap_y(z["y"], actions, known)
    net = PointNetGRU(len(known), in_ch=N_CH).to(device)
    try:
        state = torch.load(ckpt, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(ckpt, map_location=device)
    net.load_state_dict(state)
    X, mask = z["X"], z["mask"]
    idx_tr = np.where(masks["train"])[0]
    idx_va = np.where(masks["val"])[0]
    idx_tk = np.where(masks["test_known"])[0]
    idx_tu = np.where(masks["test_unknown"])[0]
    lg_tr, em_tr = forward_pack(net, X, mask, idx_tr, device)
    lg_va, em_va = forward_pack(net, X, mask, idx_va, device)
    lg_tk, em_tk = forward_pack(net, X, mask, idx_tk, device)
    lg_tu, em_tu = forward_pack(net, X, mask, idx_tu, device)
    y_tk = y[idx_tk]
    pred_tk = lg_tk.argmax(1)
    sm_va = softmax(lg_va).max(1)
    sm_tk = softmax(lg_tk).max(1)
    sm_tu = softmax(lg_tu).max(1)
    en_va = energy_known_score(lg_va)
    en_tk = energy_known_score(lg_tk)
    en_tu = energy_known_score(lg_tu)
    proto = []
    for c in range(len(known)):
        sel = y[idx_tr] == c
        proto.append(em_tr[sel].mean(0) if sel.any() else np.zeros(em_tr.shape[1], dtype=np.float64))
    proto = np.stack(proto, 0)
    def nearest_dist(emb):
        d = ((emb[:, None, :] - proto[None, :, :]) ** 2).sum(-1)
        return d.min(1), d.argmin(1)

    d_va, _ = nearest_dist(em_va)
    d_tk, _ = nearest_dist(em_tk)
    d_tu, _ = nearest_dist(em_tu)
    proto_va = -d_va
    proto_tk = -d_tk
    proto_tu = -d_tu
    y_names = np.array([known[int(i)] for i in y_tk])
    methods = [
        eval_method("MSP", sm_va, sm_tk, sm_tu, pred_tk, y_names, known),
        eval_method("Energy", en_va, en_tk, en_tu, pred_tk, y_names, known),
        eval_method("Prototype", proto_va, proto_tk, proto_tu, pred_tk, y_names, known),
    ]
    # drop huge lists from summary copy; keep them in per-method files
    slim = []
    for m in methods:
        slim.append({k: v for k, v in m.items() if k not in ("score_known_test", "score_unknown_test", "pred_open_known", "pred_open_unknown", "gt_known")})
    return {
        "held_out": held_out,
        "known_classes": known,
        "n_train": int(masks["train"].sum()),
        "n_val": int(masks["val"].sum()),
        "n_test_known": int(masks["test_known"].sum()),
        "n_test_unknown": int(masks["test_unknown"].sum()),
        "n_blocked_unseen_non_test": int(masks["blocked_unseen_non_test"].sum()),
        "methods": slim,
        "full_methods": methods,
        "embeddings": {
            "known_test": em_tk.tolist(),
            "unknown_test": em_tu.tolist(),
            "prototypes": proto.tolist(),
            "known_test_labels": y_names.tolist(),
        },
        "sample_ids": {
            "test_known": [str(z["sample_id"][i]) for i in idx_tk],
            "test_unknown": [str(z["sample_id"][i]) for i in idx_tu],
        },
        "actions_test_known": [str(actions[i]) for i in idx_tk],
        "actions_test_unknown": [str(actions[i]) for i in idx_tu],
        "energy_temperature": ENERGY_T,
        "note": "thresholds from validation known only; unseen never used for fitting",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--actions", nargs="+", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "phase5_open_set"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = np.load(ROOT / "results" / "phase1_pointcloud" / "cache" / "mri_windows.npz", allow_pickle=True)
    assign = subject_assign(z, ROOT / "data_splits" / "mri_subject_independent.csv")
    out = Path(args.out)
    rows = []
    for act in args.actions:
        dest = out / "folds" / act.replace(" ", "_")
        ckpt = dest / "best.pt"
        if not ckpt.is_file():
            raise FileNotFoundError(ckpt)
        print("==== eval", act, flush=True)
        pack = eval_fold(z, assign, act, ckpt, device)
        full = pack.pop("full_methods")
        emb = pack.pop("embeddings")
        (dest / "eval.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
        for m in full:
            name = m["method"].lower()
            (dest / f"scores_{name}.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
        np.savez_compressed(
            dest / "embeddings.npz",
            known_test=np.asarray(emb["known_test"]),
            unknown_test=np.asarray(emb["unknown_test"]),
            prototypes=np.asarray(emb["prototypes"]),
            known_test_labels=np.asarray(emb["known_test_labels"]),
        )
        rows.append(pack)
    (out / "eval_summary.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print("[eval-done]", flush=True)


if __name__ == "__main__":
    main()
