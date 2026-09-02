#!/usr/bin/env python3
"""Fig.4–6 from training_runs.csv after Phase 1B training."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    root = Path(__file__).resolve().parents[2]
    runs = list(csv.DictReader((root / "results/phase1_pointcloud/baselines/training_runs.csv").open(encoding="utf-8")))
    figdir = root / "results/phase1_pointcloud/figures"
    figdir.mkdir(parents=True, exist_ok=True)

    def acc(pred):
        hits = [r for r in runs if pred(r) and r.get("accuracy") not in ("", None)]
        return hits[0] if hits else None

    r_rand = acc(lambda r: r["dataset"] == "mRI" and r["split"] == "random_segment" and r["features"] == "XYZVI" and r["model"] == "pointnet_gru")
    r_si = acc(lambda r: r["dataset"] == "mRI" and r["split"] == "subject_independent" and r["features"] == "XYZVI" and r["model"] == "pointnet_gru")
    if r_rand and r_si:
        fig, ax = plt.subplots(figsize=(5.5, 3.2))
        labs = ["random-segment", "subject-independent"]
        a = [float(r_rand["accuracy"]), float(r_si["accuracy"])]
        f = [float(r_rand["macro_f1"]), float(r_si["macro_f1"])]
        x = np.arange(2)
        ax.bar(x - 0.18, a, 0.36, label="Accuracy")
        ax.bar(x + 0.18, f, 0.36, label="Macro-F1")
        ax.set_xticks(x, labs)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Fig.4 mRI random-segment vs subject-independent")
        fig.tight_layout()
        fig.savefig(figdir / "fig4_mri_random_vs_si.png", dpi=140)
        plt.close(fig)

    def grab(ds, feat, split=None):
        for r in runs:
            if r["dataset"] != ds or r["features"] != feat or r["model"] != "pointnet_gru":
                continue
            if split and r["split"] != split:
                continue
            if r.get("accuracy") in ("", None):
                continue
            return float(r["accuracy"]), float(r["macro_f1"])
        return None

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.2))
    for ax, ds, split in ((axes[0], "mRI", "subject_independent"), (axes[1], "RadHAR", "official_train_test")):
        feats = ["XYZ", "XYZV", "XYZVI"]
        accs, f1s = [], []
        for ft in feats:
            g = grab(ds, ft, split)
            accs.append(g[0] if g else 0)
            f1s.append(g[1] if g else 0)
        x = np.arange(3)
        ax.bar(x - 0.18, accs, 0.36, label="Accuracy")
        ax.bar(x + 0.18, f1s, 0.36, label="Macro-F1")
        ax.set_xticks(x, feats)
        ax.set_ylim(0, 1)
        ax.set_title(ds)
        ax.legend(fontsize=7)
    fig.suptitle("Fig.5 feature ablation (PointNet-GRU)")
    fig.tight_layout()
    fig.savefig(figdir / "fig5_feature_ablation.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.2))
    items = []
    for ds, split, t1_model in (
        ("mRI", "subject_independent", "pointnet_pool"),
        ("RadHAR", "official_train_test", "pointnet_pool"),
    ):
        t1 = next((r for r in runs if r["dataset"] == ds and r["model"] == t1_model and str(r["temporal_length"]) == "1"), None)
        tn = next((r for r in runs if r["dataset"] == ds and r["model"] == "pointnet_gru" and r["features"] == "XYZVI"), None)
        if t1 and tn:
            items.append((f"{ds}\nT=1", float(t1["accuracy"]), float(t1["macro_f1"])))
            items.append((f"{ds}\nT>1 GRU", float(tn["accuracy"]), float(tn["macro_f1"])))
    if items:
        x = np.arange(len(items))
        ax.bar(x - 0.18, [i[1] for i in items], 0.36, label="Accuracy")
        ax.bar(x + 0.18, [i[2] for i in items], 0.36, label="Macro-F1")
        ax.set_xticks(x, [i[0] for i in items], fontsize=8)
        ax.set_ylim(0, 1)
        ax.legend()
        ax.set_title("Fig.6 single-frame vs temporal")
        fig.tight_layout()
        fig.savefig(figdir / "fig6_temporal.png", dpi=140)
        plt.close(fig)
    print("wrote figures")


if __name__ == "__main__":
    main()
