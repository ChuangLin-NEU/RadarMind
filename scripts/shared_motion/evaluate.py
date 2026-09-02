#!/usr/bin/env python3
"""Probes, retrieval, UMAP/t-SNE. Writes only under results/phase3_shared_motion/."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from shared_motion.common import CLASS_NAMES, EmbNet, OUT, RADARS, load_packs, views_for  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
METRIC_FIELDS = [
    "model",
    "protocol",
    "train_radars",
    "test_radar",
    "activity_accuracy",
    "macro_f1",
    "activity_probe_accuracy",
    "activity_probe_macro_f1",
    "radar_probe_accuracy",
    "radar_probe_macro_f1",
    "retrieval_exact_top1",
    "retrieval_exact_top5",
    "retrieval_action_top1",
    "retrieval_action_top5",
    "note",
]


def load_net(run_id: str) -> EmbNet:
    net = EmbNet(len(CLASS_NAMES))
    ckpt = OUT / "runs" / run_id / "best.pt"
    state = torch.load(ckpt, map_location="cpu", weights_only=True)
    net.load_state_dict(state)
    net.to(DEVICE)
    net.eval()
    return net


@torch.no_grad()
def embed_views(net, views, radars):
    xs, ys, acts, rads, tss = [], [], [], [], []
    for ts, slot in views.items():
        for r in radars:
            if r not in slot:
                continue
            x, y, a = slot[r]
            xs.append(x)
            ys.append(y)
            acts.append(a)
            rads.append(r)
            tss.append(ts)
    if not xs:
        return {
            "emb": np.zeros((0, 512), np.float32),
            "y": np.zeros((0,), np.int64),
            "action": np.array([], dtype=object),
            "radar": np.array([], dtype=object),
            "timestamp": np.array([], dtype=object),
        }
    embs = []
    for i in range(0, len(xs), 32):
        xb = torch.from_numpy(np.stack(xs[i : i + 32])).float().to(DEVICE)
        _, e = net(xb, return_emb=True)
        embs.append(e.cpu().numpy())
    return {
        "emb": np.concatenate(embs, axis=0),
        "y": np.asarray(ys, np.int64),
        "action": np.array(acts),
        "radar": np.array(rads),
        "timestamp": np.array(tss),
    }


def probe_train_test(Etr, ytr, Ete, yte):
    if len(set(ytr)) < 2 or len(ytr) < 8 or len(yte) < 1:
        return {"accuracy": None, "macro_f1": None}
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(Etr)
    Xte = scaler.transform(Ete)
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def retrieval_pair(qa, ga, k=5):
    if len(qa["emb"]) == 0 or len(ga["emb"]) == 0:
        return {"exact_top1": None, "exact_top5": None, "action_top1": None, "action_top5": None, "n": 0}
    a = qa["emb"] / (np.linalg.norm(qa["emb"], axis=1, keepdims=True) + 1e-8)
    b = ga["emb"] / (np.linalg.norm(ga["emb"], axis=1, keepdims=True) + 1e-8)
    sim = a @ b.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    exact1, exact5, act1, act5 = [], [], [], []
    for i in range(len(qa["timestamp"])):
        hits_ts = ga["timestamp"][nn[i]]
        hits_act = ga["action"][nn[i]]
        exact1.append(hits_ts[0] == qa["timestamp"][i])
        exact5.append(qa["timestamp"][i] in hits_ts)
        act1.append(hits_act[0] == qa["action"][i])
        act5.append(qa["action"][i] in hits_act)
    return {
        "exact_top1": float(np.mean(exact1)),
        "exact_top5": float(np.mean(exact5)),
        "action_top1": float(np.mean(act1)),
        "action_top5": float(np.mean(act5)),
        "n": int(len(exact1)),
    }


def project_2d(E):
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=42).fit_transform(E), "umap"
    except Exception:
        perp = min(30, max(5, len(E) // 4))
        return TSNE(n_components=2, random_state=42, init="pca", perplexity=perp).fit_transform(E), "tsne"


def save_scatter(Z, labels, title, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 4.8))
    for u in sorted(set(labels)):
        m = labels == u
        ax.scatter(Z[m, 0], Z[m, 1], s=12, label=str(u), alpha=0.75)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def mean_ret(rows):
    keys = ("exact_top1", "exact_top5", "action_top1", "action_top5")
    out = {}
    for k in keys:
        vs = [r[k] for r in rows if r.get(k) is not None]
        out[k] = float(np.mean(vs)) if vs else None
    return out


def eval_task_a(run_id: str, model: str, packs) -> list[dict]:
    ckpt = OUT / "runs" / run_id / "best.pt"
    if not ckpt.is_file():
        return []
    net = load_net(run_id)
    tr = embed_views(net, views_for(packs, RADARS, "train"), RADARS)
    te = embed_views(net, views_for(packs, RADARS, "test"), RADARS)
    act = probe_train_test(tr["emb"], tr["action"], te["emb"], te["action"])
    rad = probe_train_test(tr["emb"], tr["radar"], te["emb"], te["radar"])
    by = {r: embed_views(net, views_for(packs, [r], "test"), [r]) for r in RADARS}
    rets = []
    detail = []
    for a in RADARS:
        for b in RADARS:
            if a == b:
                continue
            rec = retrieval_pair(by[a], by[b])
            rec.update({"query": a, "gallery": b})
            rets.append(rec)
            detail.append(rec)
    dest = OUT / "cache"
    dest.mkdir(exist_ok=True)
    np.savez_compressed(
        dest / f"emb_{run_id}_test.npz",
        embedding=te["emb"],
        action=te["action"],
        radar=te["radar"],
        timestamp=te["timestamp"],
    )
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    if len(te["emb"]) >= 10:
        Z, kind = project_2d(te["emb"])
        save_scatter(Z, te["action"], f"{model} {kind} by activity", figdir / f"{kind}_{run_id}_activity.png")
        save_scatter(Z, te["radar"], f"{model} {kind} by radar", figdir / f"{kind}_{run_id}_radar.png")
    (OUT / f"retrieval_{run_id}.json").write_text(json.dumps(detail, indent=2), encoding="utf-8")
    metrics = json.loads((OUT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8")) if (OUT / "runs" / run_id / "metrics.json").is_file() else {}
    avg = mean_ret(rets)
    return [
        {
            "model": model,
            "protocol": "multi_radar_shared",
            "train_radars": "10+24+77",
            "test_radar": "all",
            "activity_accuracy": metrics.get("activity_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "activity_probe_accuracy": act["accuracy"],
            "activity_probe_macro_f1": act["macro_f1"],
            "radar_probe_accuracy": rad["accuracy"],
            "radar_probe_macro_f1": rad["macro_f1"],
            "retrieval_exact_top1": avg["exact_top1"],
            "retrieval_exact_top5": avg["exact_top5"],
            "retrieval_action_top1": avg["action_top1"],
            "retrieval_action_top5": avg["action_top5"],
            "note": "probes: linear, fit train embeddings, eval test; retrieval mean over 6 directed pairs",
        }
    ]


def eval_loro(run_id: str, model: str, train_radars: list[str], test_radar: str, packs) -> list[dict]:
    ckpt = OUT / "runs" / run_id / "best.pt"
    if not ckpt.is_file():
        return []
    net = load_net(run_id)
    te = embed_views(net, views_for(packs, [test_radar], "test"), [test_radar])
    metrics = json.loads((OUT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8")) if (OUT / "runs" / run_id / "metrics.json").is_file() else {}
    return [
        {
            "model": model,
            "protocol": "leave_one_radar_out",
            "train_radars": "+".join(r.replace("GHz", "") for r in train_radars),
            "test_radar": test_radar,
            "activity_accuracy": metrics.get("activity_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "activity_probe_accuracy": "",
            "activity_probe_macro_f1": "",
            "radar_probe_accuracy": "",
            "radar_probe_macro_f1": "",
            "retrieval_exact_top1": "",
            "retrieval_exact_top5": "",
            "retrieval_action_top1": "",
            "retrieval_action_top5": "",
            "note": f"LORO; {test_radar} unused in train/val/norm/contrastive; n_test={len(te['y'])}",
        }
    ]


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    packs = load_packs()
    rows = []
    rows += eval_task_a("M0_shared", "M0", packs)
    rows += eval_task_a("M1_shared", "M1", packs)
    rows += eval_task_a("M2_shared", "M2", packs)
    loro = [
        ("M0_loro_10_24_to_77", "M0", ["10GHz", "24GHz"], "77GHz"),
        ("M2_loro_10_24_to_77", "M2", ["10GHz", "24GHz"], "77GHz"),
        ("M0_loro_10_77_to_24", "M0", ["10GHz", "77GHz"], "24GHz"),
        ("M2_loro_10_77_to_24", "M2", ["10GHz", "77GHz"], "24GHz"),
        ("M0_loro_24_77_to_10", "M0", ["24GHz", "77GHz"], "10GHz"),
        ("M2_loro_24_77_to_10", "M2", ["24GHz", "77GHz"], "10GHz"),
    ]
    for rid, model, src, tgt in loro:
        rows += eval_loro(rid, model, src, tgt, packs)
    dest = OUT / "representation_metrics.csv"
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=METRIC_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print("wrote", dest, "n", len(rows), flush=True)


if __name__ == "__main__":
    main()
