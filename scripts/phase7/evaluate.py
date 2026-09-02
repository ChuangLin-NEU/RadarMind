#!/usr/bin/env python3
"""Probes, retrieval, LORO tables for Phase 7 R1 models."""
from __future__ import annotations

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
from phase7.common import CLASS_NAMES, EmbNet, M3Net, OUT, RADARS, load_packs, views_for, write_csv  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PROBE_FIELDS = [
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
    "activity_preservation",
    "radar_suppression",
    "generalization_gain",
    "note",
]
RET_FIELDS = [
    "model",
    "query",
    "gallery",
    "exact_top1",
    "exact_top5",
    "action_top1",
    "action_top5",
    "n",
]
LORO_FIELDS = [
    "model",
    "protocol",
    "train_radars",
    "test_radar",
    "activity_accuracy",
    "macro_f1",
    "n_test",
    "note",
]


def load_net(run_id: str):
    dest = OUT / "runs" / run_id / "best.pt"
    state = torch.load(dest, map_location="cpu", weights_only=True)
    if any(k.startswith("radar_head") or k.startswith("encoder.") for k in state):
        # infer n_radar from head
        n_r = state["radar_head.weight"].shape[0] if "radar_head.weight" in state else 3
        net = M3Net(len(CLASS_NAMES), n_r)
        net.load_state_dict(state)
        enc = net.encoder
    else:
        enc = EmbNet(len(CLASS_NAMES))
        enc.load_state_dict(state)
    enc.to(DEVICE)
    enc.eval()
    return enc


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
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(scaler.fit_transform(Etr), ytr)
    pred = clf.predict(scaler.transform(Ete))
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


def eval_task_a(run_id: str, model: str, packs, baseline_act=None, baseline_rad=None) -> tuple[list[dict], list[dict]]:
    ckpt = OUT / "runs" / run_id / "best.pt"
    if not ckpt.is_file():
        return [], []
    net = load_net(run_id)
    tr = embed_views(net, views_for(packs, RADARS, "train"), RADARS)
    te = embed_views(net, views_for(packs, RADARS, "test"), RADARS)
    act = probe_train_test(tr["emb"], tr["action"], te["emb"], te["action"])
    rad = probe_train_test(tr["emb"], tr["radar"], te["emb"], te["radar"])
    by = {r: embed_views(net, views_for(packs, [r], "test"), [r]) for r in RADARS}
    rets = []
    for a in RADARS:
        for b in RADARS:
            if a == b:
                continue
            rec = retrieval_pair(by[a], by[b])
            rec.update({"model": model, "query": a, "gallery": b})
            rets.append(rec)
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
        save_scatter(Z, te["action"], f"{model} {kind} by activity", figdir / f"figH_{kind}_{run_id}_activity.png")
        save_scatter(Z, te["radar"], f"{model} {kind} by radar", figdir / f"figH_{kind}_{run_id}_radar.png")
    metrics = json.loads((OUT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    avg = mean_ret(rets)
    ap = act["accuracy"]
    rp = rad["accuracy"]
    pres = None if baseline_act is None or ap is None else float(ap / (baseline_act + 1e-12))
    supp = None if baseline_rad is None or rp is None else float((baseline_rad - rp) / (baseline_rad + 1e-12))
    return [
        {
            "model": model,
            "protocol": "multi_radar_shared",
            "train_radars": "10+24+77",
            "test_radar": "all",
            "activity_accuracy": metrics.get("activity_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "activity_probe_accuracy": ap,
            "activity_probe_macro_f1": act["macro_f1"],
            "radar_probe_accuracy": rp,
            "radar_probe_macro_f1": rad["macro_f1"],
            "activity_preservation": pres,
            "radar_suppression": supp,
            "generalization_gain": "",
            "note": "probes: linear, fit train embeddings, eval test",
        }
    ], rets


def eval_loro(run_id: str, model: str, train_radars: list[str], test_radar: str, packs) -> list[dict]:
    ckpt = OUT / "runs" / run_id / "best.pt"
    if not ckpt.is_file():
        return []
    net = load_net(run_id)
    te = embed_views(net, views_for(packs, [test_radar], "test"), [test_radar])
    metrics = json.loads((OUT / "runs" / run_id / "metrics.json").read_text(encoding="utf-8"))
    return [
        {
            "model": model,
            "protocol": "leave_one_radar_out",
            "train_radars": "+".join(r.replace("GHz", "") for r in train_radars),
            "test_radar": test_radar,
            "activity_accuracy": metrics.get("activity_accuracy"),
            "macro_f1": metrics.get("macro_f1"),
            "n_test": len(te["y"]),
            "note": f"LORO; {test_radar} unused in train/val/norm/contrastive/domain/HP",
        }
    ]


def main() -> None:
    packs = load_packs("R1")
    probes, rets = [], []
    # first pass M0 to get baselines
    m0p, m0r = eval_task_a("R1_M0_shared", "M0", packs)
    m0_act = m0p[0]["activity_probe_accuracy"] if m0p else None
    m0_rad = m0p[0]["radar_probe_accuracy"] if m0p else None
    probes += m0p
    rets += m0r
    for rid, model in (("R1_M1_shared", "M1"), ("R1_M3_shared", "M3")):
        p, r = eval_task_a(rid, model, packs, baseline_act=m0_act, baseline_rad=m0_rad)
        probes += p
        rets += r
    loro = []
    for rid, model, src, tgt in (
        ("R1_M0_loro_10_24_to_77", "M0", ["10GHz", "24GHz"], "77GHz"),
        ("R1_M3_loro_10_24_to_77", "M3", ["10GHz", "24GHz"], "77GHz"),
        ("R1_M0_loro_10_77_to_24", "M0", ["10GHz", "77GHz"], "24GHz"),
        ("R1_M3_loro_10_77_to_24", "M3", ["10GHz", "77GHz"], "24GHz"),
        ("R1_M0_loro_24_77_to_10", "M0", ["24GHz", "77GHz"], "10GHz"),
        ("R1_M3_loro_24_77_to_10", "M3", ["24GHz", "77GHz"], "10GHz"),
    ):
        loro += eval_loro(rid, model, src, tgt, packs)
    # generalization gain: mean LORO M3 - M0
    by_fold = {}
    for row in loro:
        by_fold.setdefault(row["test_radar"], {})[row["model"]] = row.get("activity_accuracy")
    gains = []
    for tgt, mm in by_fold.items():
        if "M0" in mm and "M3" in mm and mm["M0"] not in ("", None) and mm["M3"] not in ("", None):
            gains.append(float(mm["M3"]) - float(mm["M0"]))
    gmean = float(np.mean(gains)) if gains else None
    for row in probes:
        if row["model"] == "M3":
            row["generalization_gain"] = gmean
    write_csv(OUT / "probe_results.csv", PROBE_FIELDS, probes)
    write_csv(OUT / "retrieval_results.csv", RET_FIELDS, rets)
    write_csv(OUT / "loro_results.csv", LORO_FIELDS, loro)
    print("probes", len(probes), "retrieval", len(rets), "loro", len(loro), flush=True)


if __name__ == "__main__":
    main()
