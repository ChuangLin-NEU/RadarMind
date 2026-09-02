#!/usr/bin/env python3
"""Phase 5 figures. Read eval artifacts only."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase5_open_set"
FIG = OUT / "figures"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig1_protocol(plt):
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.axis("off")
    ax.set_title("Fig. 1  Open-set evaluation protocol (LOAO + subject-independent)")
    lines = [
        "Train subjects / Val subjects / Test subjects remain disjoint.",
        "Held-out action: all windows removed from train, val, and threshold fitting.",
        "Test known = test-subject windows of the 12 known actions.",
        "Test unknown = test-subject windows of the unseen action.",
        "Threshold = 5th percentile of validation known scores.",
        "This is open-set recognition, not open-vocabulary, not open-world.",
    ]
    for i, t in enumerate(lines):
        ax.text(0.02, 0.82 - i * 0.13, t, fontsize=10, transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG / "fig1_protocol.png", dpi=140)
    plt.close(fig)


def fig2_scores(plt, method: dict, held: str, dest: Path):
    kn = np.asarray(method["score_known_test"], dtype=float)
    un = np.asarray(method["score_unknown_test"], dtype=float)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.hist(kn, bins=30, alpha=0.6, label="known test", density=True)
    ax.hist(un, bins=30, alpha=0.6, label="unknown test", density=True)
    ax.axvline(method["threshold"], color="k", ls="--", label=f"thr={method['threshold']:.3f}")
    ax.set_xlabel(f"{method['method']} known-score")
    ax.set_ylabel("density")
    ax.set_title(f"Fig. 2  Known vs Unknown  held-out={held}  {method['method']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig3_roc(plt, method: dict, held: str, dest: Path):
    from sklearn.metrics import roc_curve

    kn = np.asarray(method["score_known_test"], dtype=float)
    un = np.asarray(method["score_unknown_test"], dtype=float)
    y = np.concatenate([np.zeros(len(kn)), np.ones(len(un))])
    s = np.concatenate([-kn, -un])
    fpr, tpr, _ = roc_curve(y, s)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(fpr, tpr, label=f"AUROC={method['AUROC']:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("FPR (known as unknown)")
    ax.set_ylabel("TPR (unknown detected)")
    ax.set_title(f"Fig. 3  ROC  held-out={held}  {method['method']}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig4_cm(plt, method: dict, held: str, dest: Path):
    names = method["cm_labels"]
    cm = np.asarray(method["cm"], dtype=float)
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(names)), names, rotation=60, ha="right", fontsize=7)
    ax.set_yticks(range(len(names)), names, fontsize=7)
    ax.set_title(f"Fig. 4  Open-set CM  held-out={held}  {method['method']}")
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig5_proto(plt, fold: Path, held: str, dest: Path):
    z = np.load(fold / "embeddings.npz", allow_pickle=True)
    kn = np.asarray(z["known_test"])
    un = np.asarray(z["unknown_test"])
    proto = np.asarray(z["prototypes"])
    labs = np.asarray(z["known_test_labels"])
    from sklearn.decomposition import PCA

    X = np.concatenate([kn, un, proto], 0)
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    n_k, n_u = len(kn), len(un)
    fig, ax = plt.subplots(figsize=(6.4, 5.0))
    for lab in sorted(set(labs.tolist())):
        m = labs == lab
        ax.scatter(xy[:n_k][m, 0], xy[:n_k][m, 1], s=8, alpha=0.5, label=lab)
    ax.scatter(xy[n_k : n_k + n_u, 0], xy[n_k : n_k + n_u, 1], s=18, c="k", marker="x", label=f"unknown ({held})")
    ax.scatter(xy[n_k + n_u :, 0], xy[n_k + n_u :, 1], s=40, c="red", marker="*", label="prototype")
    ax.set_title(f"Fig. 5  Embedding / prototype PCA  held-out={held}")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def fig6_replay(plt, demo: dict, dest: Path):
    fig, ax = plt.subplots(figsize=(9, 2.6))
    ax.set_title("Fig. 6  RadarMind open-set replay case  (composite replay for open-set demonstration)")
    colors = {"UNKNOWN_MOTION": "#c0392b", "UNKNOWN": "#c0392b"}
    for i, e in enumerate(demo.get("events") or []):
        act = e["activity"]
        ax.barh(0, e["end"] - e["start"], left=e["start"], height=0.45, color=colors.get(act, "#2980b9"), alpha=0.85)
        ax.text((e["start"] + e["end"]) / 2, 0, act, ha="center", va="center", color="white", fontsize=9)
    ax.set_yticks([])
    ax.set_xlabel("composite window index")
    ax.set_xlim(0, max((e["end"] for e in demo.get("events") or [{"end": 1}]), default=1))
    fig.tight_layout()
    fig.savefig(dest, dpi=140)
    plt.close(fig)


def pick_best(methods: list[dict]) -> dict:
    return max(methods, key=lambda m: (m["AUROC"], -m["FPR95"]))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig1_protocol(plt)
    summary = json.loads((OUT / "eval_summary.json").read_text(encoding="utf-8"))
    demo_p = OUT / "composite_demo.json"
    demo = json.loads(demo_p.read_text(encoding="utf-8")) if demo_p.is_file() else {"events": []}
    # default figures from first selected / pose_4 if present
    prefer = next((r for r in summary if r["held_out"] == "pose_4"), summary[0])
    fold = OUT / "folds" / prefer["held_out"].replace(" ", "_")
    methods = []
    for name in ("msp", "energy", "prototype"):
        p = fold / f"scores_{name}.json"
        if p.is_file():
            methods.append(json.loads(p.read_text(encoding="utf-8")))
    best = pick_best(methods) if methods else prefer["methods"][0]
    if "score_known_test" in best:
        fig2_scores(plt, best, prefer["held_out"], FIG / "fig2_score_distribution.png")
        fig3_roc(plt, best, prefer["held_out"], FIG / "fig3_roc.png")
        fig4_cm(plt, best, prefer["held_out"], FIG / "fig4_openset_cm.png")
    if (fold / "embeddings.npz").is_file():
        fig5_proto(plt, fold, prefer["held_out"], FIG / "fig5_prototype.png")
    fig6_replay(plt, demo, FIG / "fig6_replay_case.png")
    print("[figures-done]", FIG, flush=True)


if __name__ == "__main__":
    main()
