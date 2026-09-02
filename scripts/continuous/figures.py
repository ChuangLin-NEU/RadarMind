#!/usr/bin/env python3
"""Phase 4 figures. Written only under results/phase4_continuous/figures/."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuous.common import CLASS_NAMES, OUT  # noqa: E402


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def fig_pipeline(figdir: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10.5, 2.2))
    ax.axis("off")
    boxes = [
        "recording\n(mRI / Frame #)",
        "window T=32\nstride=16",
        "PointNet-GRU\n(frozen)",
        "temporal\nsmoothing",
        "action\nboundary",
        "Timeline\n+ LLM",
    ]
    for i, t in enumerate(boxes):
        ax.add_patch(plt.Rectangle((i * 1.7, 0.3), 1.5, 1.2, fill=False, lw=1.2))
        ax.text(i * 1.7 + 0.75, 0.9, t, ha="center", va="center", fontsize=8)
        if i < len(boxes) - 1:
            ax.annotate("", xy=((i + 1) * 1.7, 0.9), xytext=(i * 1.7 + 1.5, 0.9), arrowprops=dict(arrowstyle="->"))
    ax.set_xlim(-0.1, 10.3)
    ax.set_ylim(0, 1.8)
    ax.set_title("Continuous inference pipeline (MVP)")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_pipeline.png", dpi=140)
    plt.close(fig)


def _track(ax, events, y, title, fps):
    colors = {}
    cmap = _mpl().cm.tab20
    for i, name in enumerate(CLASS_NAMES + ["UNLABELED"]):
        colors[name] = cmap(i % 20)
    for e in events:
        a, b = e["start_frame"] / fps, (e["end_frame"] + 1) / fps
        ax.barh(y, b - a, left=a, height=0.35, color=colors.get(e["activity"], "0.7"), edgecolor="k", lw=0.3)
        if b - a > 4:
            ax.text((a + b) / 2, y, e["activity"], ha="center", va="center", fontsize=6)
    ax.set_title(title, loc="left", fontsize=9)


def fig_timeline(figdir: Path, rec: str, tag: str) -> None:
    plt = _mpl()
    gt = json.loads((OUT / "ground_truth_timelines" / f"{rec}.json").read_text(encoding="utf-8"))
    pred = json.loads((OUT / "predicted_timelines" / f"{rec}.json").read_text(encoding="utf-8"))
    fps = float(pred["frame_rate"])
    fig, axes = plt.subplots(3, 1, figsize=(11, 4.8), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1.2]})
    _track(axes[0], gt["events"], 0, "GT", fps)
    axes[0].set_yticks([])
    _track(axes[1], pred["modes"]["confidence_aware"]["events"], 0, "Prediction (confidence-aware)", fps)
    axes[1].set_yticks([])
    ser = pred["modes"]["confidence_aware"]["series"]
    if ser:
        t = [s["time"] for s in ser]
        axes[2].plot(t, [s["confidence"] for s in ser], lw=1.0, label="confidence")
        vel = [s.get("mean_velocity") for s in ser]
        if any(isinstance(v, (int, float)) for v in vel):
            vv = [v if isinstance(v, (int, float)) else np.nan for v in vel]
            ax2 = axes[2].twinx()
            ax2.plot(t, vv, lw=0.9, color="C1", label="mean_velocity")
            ax2.set_ylabel("velocity")
        axes[2].set_ylabel("confidence")
        axes[2].set_xlabel("time (s)")
        axes[2].set_ylim(0, 1.05)
    fig.suptitle(f"{rec}  subject={pred['subject']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / tag, dpi=140)
    plt.close(fig)


def fig_confusion(figdir: Path) -> None:
    plt = _mpl()
    path = figdir / "continuous_confusion.csv"
    if not path.is_file():
        return
    cm = np.loadtxt(path, delimiter=",")
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=75, fontsize=7)
    ax.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax.set_xlabel("prediction")
    ax.set_ylabel("window GT (labeled only)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(figdir / "fig4_confusion.png", dpi=140)
    plt.close(fig)


def fig_boundary(figdir: Path) -> None:
    plt = _mpl()
    cache = OUT / "cache" / "test_window_labels.npz"
    if not cache.is_file():
        return
    z = np.load(cache, allow_pickle=True)
    err = np.asarray(z["errors"])
    if err.size == 0:
        return
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.hist(err, bins=20, color="0.35")
    ax.set_xlabel("boundary error (frames)")
    ax.set_ylabel("count")
    ax.set_title(f"n={len(err)}  median={np.median(err):.1f} frames")
    fig.tight_layout()
    fig.savefig(figdir / "fig5_boundary_error.png", dpi=140)
    plt.close(fig)


def fig_series(figdir: Path, rec: str) -> None:
    plt = _mpl()
    pred = json.loads((OUT / "predicted_timelines" / f"{rec}.json").read_text(encoding="utf-8"))
    ser = pred["modes"]["confidence_aware"]["series"]
    t = [s["time"] for s in ser]
    fig, axes = plt.subplots(3, 1, figsize=(10, 5.2), sharex=True)
    axes[0].plot(t, [s["confidence"] for s in ser], lw=1)
    axes[0].set_ylabel("confidence")
    axes[1].plot(t, [s.get("mean_velocity") if isinstance(s.get("mean_velocity"), (int, float)) else np.nan for s in ser], lw=1)
    axes[1].set_ylabel("mean_velocity")
    axes[2].plot(t, [s.get("motion_energy") if isinstance(s.get("motion_energy"), (int, float)) else np.nan for s in ser], lw=1)
    axes[2].set_ylabel("motion_energy")
    axes[2].set_xlabel("time (s)")
    fig.suptitle(f"{rec} MotionIntelligenceSeries")
    fig.tight_layout()
    fig.savefig(figdir / "fig3_series.png", dpi=140)
    plt.close(fig)


def main() -> None:
    figdir = OUT / "figures"
    figdir.mkdir(exist_ok=True)
    fig_pipeline(figdir)
    demos = []
    demo_p = OUT / "demo_registry.json"
    if demo_p.is_file():
        demos = json.loads(demo_p.read_text(encoding="utf-8")).get("demos", [])
    recs = [d["recording"] for d in demos] if demos else []
    if not recs:
        recs = [p.stem for p in (OUT / "predicted_timelines").glob("*.json")]
        recs = [r for r in recs if json.loads((OUT / "predicted_timelines" / f"{r}.json").read_text(encoding="utf-8")).get("split") == "test"]
    if recs:
        fig_timeline(figdir, recs[0], "fig2_gt_vs_pred.png")
        fig_series(figdir, recs[0])
        for i, r in enumerate(recs[:3], 1):
            fig_timeline(figdir, r, f"fig6_demo_{i}_{r}.png")
    fig_confusion(figdir)
    fig_boundary(figdir)
    print("figures", figdir, flush=True)


if __name__ == "__main__":
    main()
