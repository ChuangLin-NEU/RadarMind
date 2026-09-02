#!/usr/bin/env python3
"""UMAP/t-SNE + linear probes on motion embeddings. seed=42."""
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models.resnet18_spec import ResNet18Spec  # noqa: E402
from train_cross_radar import CLASS_NAMES, EmbNet, load_band  # noqa: E402

OUT = ROOT / "results" / "phase2_cross_radar"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_net(run_id: str) -> EmbNet:
    net = EmbNet(len(CLASS_NAMES))
    state = torch.load(OUT / "runs" / run_id / "best.pt", map_location="cpu")
    net.load_state_dict(state)
    net.to(DEVICE)
    net.eval()
    return net


@torch.no_grad()
def embed(net, X):
    embs = []
    for i in range(0, len(X), 32):
        xb = torch.from_numpy(X[i : i + 32]).float().to(DEVICE)
        _, e = net(xb, return_emb=True)
        embs.append(e.cpu().numpy())
    return np.concatenate(embs, axis=0)


def probe(X, y):
    if len(set(y)) < 2 or len(y) < 8:
        return {"accuracy": None, "macro_f1": None, "n": int(len(y))}
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    clf = LogisticRegression(max_iter=400, random_state=42)
    clf.fit(Xs, y)
    pred = clf.predict(Xs)
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "n": int(len(y)),
        "note": "fit on same split (diagnostic, not a held-out claim)",
    }


def main() -> None:
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    rows = []
    packs = {f: load_band(f) for f in ("10GHz", "24GHz", "77GHz")}
    for run_id, label in (("within_77", "source77"), ("coral_uda_77_to_10", "coral77to10")):
        ckpt = OUT / "runs" / run_id / "best.pt"
        if not ckpt.is_file():
            continue
        net = load_net(run_id)
        E, acts, rads, splits = [], [], [], []
        for freq, d in packs.items():
            te = d["split"] == "test"
            e = embed(net, d["X"][te])
            E.append(e)
            acts.extend(d["action"][te].tolist())
            rads.extend([freq] * int(te.sum()))
            splits.extend(["test"] * int(te.sum()))
            np.savez_compressed(
                OUT / "cache" / f"emb_{run_id}_{freq}.npz",
                embedding=e,
                action=d["action"][te],
                radar=np.array([freq] * int(te.sum())),
                timestamp=d["timestamp"][te],
            )
        E = np.concatenate(E, axis=0)
        acts = np.array(acts)
        rads = np.array(rads)
        act_p = probe(E, acts)
        rad_p = probe(E, rads)
        rows.append({"encoder": run_id, "probe": "activity", **act_p})
        rows.append({"encoder": run_id, "probe": "radar_id", **rad_p})
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            Z = TSNE(n_components=2, random_state=42, init="pca", perplexity=min(30, max(5, len(E) // 4))).fit_transform(E)
            for color_by, vals, title in (
                ("action", acts, f"{label} t-SNE by action"),
                ("radar", rads, f"{label} t-SNE by radar"),
            ):
                fig, ax = plt.subplots(figsize=(6.2, 4.8))
                for u in sorted(set(vals)):
                    m = vals == u
                    ax.scatter(Z[m, 0], Z[m, 1], s=12, label=u, alpha=0.75)
                ax.legend(fontsize=7, loc="best")
                ax.set_title(title)
                fig.tight_layout()
                fig.savefig(figdir / f"tsne_{run_id}_{color_by}.png", dpi=140)
                plt.close(fig)
        except Exception as exc:  # noqa: BLE001
            print("[tsne-skip]", exc, flush=True)
    dest = OUT / "representation_probe.csv"
    fields = ["encoder", "probe", "accuracy", "macro_f1", "n", "note"]
    with dest.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    main()
