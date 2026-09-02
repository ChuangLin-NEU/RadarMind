#!/usr/bin/env python3
"""Paper figures from frozen result files. No new experiments, no invented points."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "results" / "paper_package" / "figures"


def _plt():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "figure.dpi": 140, "savefig.dpi": 200, "axes.grid": False})
    return plt


def save(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name)
    import matplotlib.pyplot as plt

    plt.close(fig)


def fig1(plt):
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis("off")
    ax.set_title("Fig. 1  RadarMind architecture (as implemented)")
    blocks = [
        (0.2, 4.2, "Radar Data\nCI4R / mRI / RadHAR", "DONE"),
        (2.4, 4.2, "Sensor Adapter\nspec / point cloud", "DONE"),
        (4.6, 4.2, "Motion Encoder\nResNet / GRU / Shared", "DONE"),
        (6.8, 4.2, "Evaluation\nHAR / XR / Open-set", "DONE"),
        (9.0, 4.2, "MI / Timeline\nDeepSeek / App", "DONE"),
        (0.2, 1.4, "Motion Rep.\nnot invariant", "PARTIAL"),
        (2.4, 1.4, "Full-stream\nno raw CSV", "PARTIAL"),
        (4.6, 1.4, "Open-world\nopen-vocab", "PARTIAL"),
        (6.8, 1.4, "Cross-env / Live\nMM-Fi / FAU", "PENDING"),
        (9.0, 1.4, "Open-set MVP\n≠ open-world", "DONE"),
    ]
    color = {"DONE": "#2b8cbe", "PARTIAL": "#e8b84a", "PENDING": "#bdbdbd"}
    for x, y, t, stt in blocks:
        ax.add_patch(plt.Rectangle((x, y), 2.0, 1.5, fc=color[stt], ec="k", lw=0.6, alpha=0.9))
        ax.text(x + 1.0, y + 0.75, t + f"\n[{stt}]", ha="center", va="center", fontsize=7)
    ax.annotate("", xy=(2.35, 4.95), xytext=(2.2, 4.95), arrowprops=dict(arrowstyle="->"))
    (FIG / "fig1_framework.png").parent.mkdir(parents=True, exist_ok=True)
    save(fig, "fig1_framework.png")


def fig2(plt):
    fig, ax = plt.subplots(figsize=(8.4, 3.6))
    ax.axis("off")
    ax.set_title("Fig. 2  Dataset / representation overview")
    ax.table(
        cellText=[
            ["CI4R 77 GHz", "micro-Doppler", "random / prefix-group"],
            ["CI4R 10/24/77 PNG", "official PNG", "within / zero-shot / shared"],
            ["mRI", "point cloud T=32 P=64", "SI / LOAO / labeled-interval"],
            ["RadHAR", "point cloud T=60", "official Train/Test"],
        ],
        colLabels=["Dataset", "Representation", "Protocols in this freeze"],
        loc="center",
    )
    save(fig, "fig2_datasets.png")


def fig3(plt):
    p1a_r = json.loads((ROOT / "results/phase1_ci4r/baselines/random_record/resnet18/metrics.json").read_text())["accuracy"]
    p1a_p = json.loads((ROOT / "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json").read_text())["accuracy"]
    runs = list(csv.DictReader((ROOT / "results/phase1_pointcloud/training_runs.csv").open(encoding="utf-8")))
    mri_si = next(float(r["accuracy"]) for r in runs if r["dataset"] == "mRI" and r["split"] == "subject_independent" and r["features"] == "XYZ" and r["temporal_length"] == "32")
    xr = list(csv.DictReader((ROOT / "results/phase2_cross_radar/training_runs.csv").open(encoding="utf-8")))
    hard = min(float(r["accuracy"]) for r in xr if r["method"] == "source_only" and r["train_radar"] != r["test_radar"])
    ev = json.loads((ROOT / "results/phase5_open_set/eval_summary.json").read_text())
    proto = float(np.mean([next(m["AUROC"] for m in r["methods"] if m["method"] == "Prototype") for r in ev]))
    labels = [
        "L1 within\nCI4R random Acc",
        "L2 group/SI\nCI4R prefix Acc\nmRI SI Acc",
        "L3 cross-radar\nmin zero-shot Acc",
        "L4 open-set\nPrototype AUROC",
    ]
    vals = [p1a_r, p1a_p, hard, proto]
    fig, ax = plt.subplots(figsize=(8.8, 4.2))
    ax.bar(range(4), vals, color=["#2b8cbe", "#7bccc4", "#f03b20", "#fdae61"])
    ax.set_xticks(range(4), labels, fontsize=8)
    ax.set_ylim(0, 1)
    ax.set_ylabel("reported metric (not interchangeable)")
    ax.set_title("Fig. 3  Generalization ladder — difficulty increases; metrics are not equivalent")
    ax.text(1, p1a_p + 0.03, f"prefix {p1a_p:.3f}\nSI {mri_si:.3f}", ha="center", fontsize=8)
    for i, v in enumerate(vals):
        if i != 1:
            ax.text(i, v + 0.02, f"{v:.3f}", ha="center", fontsize=8)
    save(fig, "fig3_generalization_ladder.png")


def fig4(plt):
    runs = list(csv.DictReader((ROOT / "results/phase1_pointcloud/training_runs.csv").open(encoding="utf-8")))
    pairs = [
        ("mRI T=1", next(float(r["accuracy"]) for r in runs if r["dataset"] == "mRI" and r["temporal_length"] == "1")),
        ("mRI T=32 XYZVI", next(float(r["accuracy"]) for r in runs if r["dataset"] == "mRI" and r["split"] == "subject_independent" and r["features"] == "XYZVI" and r["temporal_length"] == "32")),
        ("RadHAR T=1", next(float(r["accuracy"]) for r in runs if r["dataset"] == "RadHAR" and r["temporal_length"] == "1")),
        ("RadHAR T=60 XYZV", next(float(r["accuracy"]) for r in runs if r["dataset"] == "RadHAR" and r["features"] == "XYZV" and r["temporal_length"] == "60")),
    ]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    ax.bar([p[0] for p in pairs], [p[1] for p in pairs], color="#2b8cbe")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Fig. 4  Temporal contribution (T=1 uses PointNetPool)")
    save(fig, "fig4_temporal.png")


def fig5(plt):
    xr = list(csv.DictReader((ROOT / "results/phase2_cross_radar/training_runs.csv").open(encoding="utf-8")))
    bands = ["10GHz", "24GHz", "77GHz"]
    m = np.full((3, 3), np.nan)
    for r in xr:
        if r["method"] != "source_only":
            continue
        i, j = bands.index(r["train_radar"]), bands.index(r["test_radar"])
        m[i, j] = float(r["accuracy"])
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(m, cmap="RdYlGn", vmin=0, vmax=1)
    ax.set_xticks(range(3), ["→10", "→24", "→77"])
    ax.set_yticks(range(3), ["10→", "24→", "77→"])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, f"{m[i, j]:.3f}", ha="center", va="center", fontsize=9)
    ax.set_title("Fig. 5  Cross-radar source-only Accuracy")
    fig.colorbar(im, ax=ax, fraction=0.046)
    save(fig, "fig5_cross_radar_heatmap.png")


def fig6(plt):
    demo = ROOT / "results" / "phase2_cross_radar" / "demo_registry.json"
    man = ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv"
    ts = None
    if demo.is_file():
        d = json.loads(demo.read_text(encoding="utf-8"))
        if d.get("demos"):
            ts = d["demos"][0].get("timestamp")
    if man.is_file() and ts:
        rows = [r for r in csv.DictReader(man.open(encoding="utf-8")) if r.get("timestamp") == ts]
        fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
        shown = 0
        for ax, freq in zip(axes, ("10GHz", "24GHz", "77GHz")):
            hit = next((r for r in rows if r.get("radar_frequency") == freq), None)
            ax.set_title(freq)
            if hit and (ROOT / hit["source_file"]).is_file():
                img = plt.imread(str(ROOT / hit["source_file"]))
                ax.imshow(img)
                shown += 1
            else:
                ax.text(0.5, 0.5, "PNG not on disk", ha="center")
            ax.axis("off")
        fig.suptitle(f"Fig. 6  Same acquisition {ts}  (real official PNG)")
        save(fig, "fig6_same_acquisition.png")
        return
    fig, ax = plt.subplots(figsize=(7, 2.4))
    ax.axis("off")
    ax.set_title("Fig. 6  Same-acquisition 10/24/77 PNG not copied; see Cross-Radar page")
    save(fig, "fig6_same_acquisition.png")


def fig7(plt):
    src = list((ROOT / "results" / "phase3_shared_motion" / "figures").glob("tsne*.png")) if (ROOT / "results" / "phase3_shared_motion" / "figures").is_dir() else []
    if src:
        for i, p in enumerate(sorted(src)[:2]):
            shutil.copy2(p, FIG / f"fig7_tsne_src{i+1}_{p.name}")
        fig, ax = plt.subplots(figsize=(7.2, 3.2))
        ax.axis("off")
        ax.set_title("Fig. 7  t-SNE copied from Phase 3 figures (not recomputed)")
        ax.text(0.5, 0.5, "\n".join(p.name for p in sorted(src)[:4]), ha="center")
        save(fig, "fig7_shared_tsne.png")
        return
    mets = list(csv.DictReader((ROOT / "results/phase3_shared_motion/representation_metrics.csv").open(encoding="utf-8")))
    mets = [r for r in mets if r["protocol"] == "multi_radar_shared"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(mets))
    ax.bar(x - 0.15, [float(r["retrieval_action_top1"]) for r in mets], 0.3, label="retrieval action top-1")
    ax.bar(x + 0.15, [float(r["activity_accuracy"]) for r in mets], 0.3, label="Task A Acc")
    ax.set_xticks(x, [r["model"] for r in mets])
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Fig. 7  Shared-task / retrieval (t-SNE files not present; no invented embedding plot)")
    save(fig, "fig7_shared_tsne.png")


def fig8(plt):
    mets = [r for r in csv.DictReader((ROOT / "results/phase3_shared_motion/representation_metrics.csv").open(encoding="utf-8")) if r["protocol"] == "multi_radar_shared"]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(len(mets))
    ax.bar(x - 0.15, [float(r["activity_probe_accuracy"]) for r in mets], 0.3, label="Activity probe")
    ax.bar(x + 0.15, [float(r["radar_probe_accuracy"]) for r in mets], 0.3, label="Radar-ID probe")
    ax.set_xticks(x, [r["model"] for r in mets])
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, ls="--", c="k", lw=0.6)
    ax.set_title("Fig. 8  Activity probe vs Radar-ID probe")
    ax.legend()
    save(fig, "fig8_probes.png")


def fig9(plt):
    rec = "subject20_0513"
    gt_p = ROOT / "results/phase4_continuous/ground_truth_timelines" / f"{rec}.json"
    pr_p = ROOT / "results/phase4_continuous/predicted_timelines" / f"{rec}.json"
    fig, ax = plt.subplots(figsize=(9.2, 2.8))
    if gt_p.is_file() and pr_p.is_file():
        gt = json.loads(gt_p.read_text(encoding="utf-8"))
        pr = json.loads(pr_p.read_text(encoding="utf-8"))
        ev = pr.get("modes", {}).get("confidence_aware", {}).get("events") or pr.get("events") or []
        for i, e in enumerate(gt.get("events") or []):
            ax.barh(1, float(e.get("end_frame", 0)) - float(e.get("start_frame", 0)), left=float(e.get("start_frame", 0)), height=0.35, color="#2b8cbe", alpha=0.7)
        for e in ev:
            a0 = float(e.get("start_frame", e.get("start_time", 0)))
            a1 = float(e.get("end_frame", e.get("end_time", 0)))
            ax.barh(0, a1 - a0, left=a0, height=0.35, color="#e34a33", alpha=0.7)
        ax.set_yticks([0, 1], ["prediction", "GT"])
        ax.set_title(f"Fig. 9  Labeled-interval timeline  {rec}  (not full-stream)")
    else:
        ax.set_title("Fig. 9  timeline files missing")
    save(fig, "fig9_continuous_timeline.png")


def fig10(plt):
    sc = ROOT / "results/phase5_open_set/folds/pose_4/scores_prototype.json"
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    if sc.is_file():
        d = json.loads(sc.read_text(encoding="utf-8"))
        kn, un = np.asarray(d["score_known_test"]), np.asarray(d["score_unknown_test"])
        axes[0].hist(kn, bins=24, alpha=0.6, label="known", density=True)
        axes[0].hist(un, bins=24, alpha=0.6, label="unknown", density=True)
        axes[0].axvline(d["threshold"], c="k", ls="--")
        axes[0].set_title("score  held-out=pose_4 Prototype")
        axes[0].legend()
        from sklearn.metrics import roc_curve

        y = np.r_[np.zeros(len(kn)), np.ones(len(un))]
        s = np.r_[-kn, -un]
        fpr, tpr, _ = roc_curve(y, s)
        axes[1].plot(fpr, tpr, label=f"AUROC={d['AUROC']:.3f}")
        axes[1].plot([0, 1], [0, 1], "k--", lw=0.7)
        axes[1].set_title("ROC")
        axes[1].legend()
    fig.suptitle("Fig. 10  Open-set scores / ROC from saved Phase 5 files")
    save(fig, "fig10_openset.png")


def fig11(plt):
    fig, ax = plt.subplots(figsize=(8.8, 3.2))
    ax.axis("off")
    ax.set_title("Fig. 11  Application workflow (real pages)")
    ax.text(
        0.5,
        0.5,
        "Home → Replay → Generalization → Cross-Radar → Shared Motion\n→ Continuous (labeled-interval) → Open-Set → Research Status\nOffline Demo: fallback text; predictions remain real.",
        ha="center",
        va="center",
    )
    save(fig, "fig11_application.png")


def main() -> None:
    plt = _plt()
    fig1(plt)
    fig2(plt)
    fig3(plt)
    fig4(plt)
    fig5(plt)
    fig6(plt)
    fig7(plt)
    fig8(plt)
    fig9(plt)
    fig10(plt)
    fig11(plt)
    print("[figures]", FIG, flush=True)


if __name__ == "__main__":
    main()
