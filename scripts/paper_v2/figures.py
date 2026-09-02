"""Figures from metric_snapshot only. No hand-written bar heights."""
from __future__ import annotations

import csv
from pathlib import Path


def D(snap, key: str) -> str:
    return snap["metrics"][key]["display"]


def V(snap, key: str) -> float:
    return float(snap["metrics"][key]["value"])


def make_figures(snap: dict, fig_dir: Path, pres_dir: Path, script_name: str) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    pres_dir.mkdir(parents=True, exist_ok=True)
    audit = []

    def save(fig, name, panels, sources):
        fig.savefig(fig_dir / name, dpi=160, bbox_inches="tight")
        fig.savefig(pres_dir / name, dpi=140, bbox_inches="tight")
        plt.close(fig)
        if isinstance(panels, str):
            panels = [panels]
        if isinstance(sources, str):
            sources = [sources]
        for panel, src in zip(panels, sources):
            audit.append(
                {
                    "figure": name,
                    "panel": panel,
                    "source_file": "results/paper_package_v2/metric_snapshot.json",
                    "metric": src,
                    "generation_script": script_name,
                }
            )

    # Fig 1
    fig, ax = plt.subplots(figsize=(7.4, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.text(5, 3.3, f"{D(snap,'N_PAIRED_ACQ')} paired acquisitions", ha="center", fontsize=12)
    ax.text(5, 2.5, "10 / 24 / 77 GHz   ·   same timestamp", ha="center")
    ax.text(5, 1.7, f"{D(snap,'N_ACTIONS')} actions   ·   split {D(snap,'SPLIT_TRAIN')}/{D(snap,'SPLIT_VAL')}/{D(snap,'SPLIT_TEST')}", ha="center")
    ax.text(5, 0.8, "split by acquisition id  ·  filename prefix is not subject", ha="center", fontsize=9)
    ax.set_title("Fig. 1  Paired three-radar benchmark")
    save(fig, "fig1_benchmark.png", ["schematic"], ["N_PAIRED_ACQ,N_ACTIONS,SPLIT_TRAIN,SPLIT_VAL,SPLIT_TEST"])

    # Fig 2 matrix
    labels = ["10", "24", "77"]
    keys = [
        ["R0_10_WITHIN", "R0_10_TO_24", "R0_10_TO_77"],
        ["R0_24_TO_10", "R0_24_WITHIN", "R0_24_TO_77"],
        ["R0_77_TO_10", "R0_77_TO_24", "R0_77_WITHIN"],
    ]
    mat = np.array([[V(snap, k) for k in row] for row in keys], dtype=float)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrBr")
    ax.set_xticks([0, 1, 2], labels)
    ax.set_yticks([0, 1, 2], labels)
    ax.set_xlabel("Test radar (GHz)")
    ax.set_ylabel("Train radar (GHz)")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, D(snap, keys[i][j]), ha="center", va="center", color="black", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="7-class HAR accuracy")
    ax.set_title("Fig. 2  Official-representation within vs cross-radar")
    save(fig, "fig2_within_cross.png", ["matrix"], [",".join(k for row in keys for k in row)])

    # Fig 3 representation evolution (3-radar only)
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    xs = ["Official\nPNG", "Cropped\nPNG", "Numerical\nDoppler"]
    ys = [V(snap, "RID_R0_3CLASS"), V(snap, "RID_R1_3CLASS"), V(snap, "RID_R2D_3CLASS")]
    ax.plot(xs, ys, "o-")
    ax.set_ylim(0.90, 1.02)
    ax.set_ylabel("3-radar Radar-ID")
    ax.set_title("Fig. 3  Three-radar Radar-ID along representation stages")
    save(fig, "fig3_representation.png", ["line"], ["RID_R0_3CLASS,RID_R1_3CLASS,RID_R2D_3CLASS"])

    # Fig 4 split protocol
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].bar(["Official", "Cropped", "Numerical\nDoppler"], [V(snap, "RID_R0_3CLASS"), V(snap, "RID_R1_3CLASS"), V(snap, "RID_R2D_3CLASS")])
    axes[0].set_ylim(0, 1.05)
    axes[0].set_ylabel("Radar-ID")
    axes[0].set_title("A. 3-radar probe (10/24/77)")
    axes[1].bar(["R2-D 24/77", "Velocity\n24/77", "Unified\nSTFT"], [V(snap, "RID_R2D_2477"), V(snap, "RID_R2V_2477"), V(snap, "U1_RID")])
    axes[1].set_ylim(0, 1.05)
    axes[1].set_title("B. 2-radar probe (24/77 only)")
    fig.suptitle("Fig. 4  Radar-ID persists — protocols are not pooled")
    fig.tight_layout()
    save(
        fig,
        "fig4_radar_id.png",
        ["A", "B"],
        ["RID_R0_3CLASS,RID_R1_3CLASS,RID_R2D_3CLASS", "RID_R2D_2477,RID_R2V_2477,U1_RID"],
    )

    # Fig 5 processing / calibration
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.5))
    axes[0].bar(["U0", "U1", "U2", "M2"], [V(snap, "U0_RID"), V(snap, "U1_RID"), V(snap, "U2_RID"), V(snap, "M2_RID")])
    axes[0].set_ylim(0.90, 1.02)
    axes[0].set_ylabel("24/77 Radar-ID")
    axes[0].set_title("A. Harmonization (2-radar)")
    axes[1].bar(["C0", "C-best"], [V(snap, "C0_RID"), V(snap, "CBEST_RID")])
    axes[1].set_ylim(0.90, 1.02)
    axes[1].set_title("B. Known-radar calibration (3-radar)")
    fig.suptitle("Fig. 5  Signal-processing and calibration ablation")
    fig.tight_layout()
    save(fig, "fig5_processing.png", ["A", "B"], ["U0_RID,U1_RID,U2_RID,M2_RID", "C0_RID,CBEST_RID"])

    # Fig 6 split by probe family
    fig, axes = plt.subplots(1, 3, figsize=(10.4, 3.6), sharey=False)
    groups = [
        (
            "A. Representation probe",
            [("R2-D", "RID_R2D_3CLASS", "ACT_R2D_REP"), ("R1", "RID_R1_3CLASS", "ACT_R1_REP")],
            (0.90, 1.02),
            (0.80, 1.02),
        ),
        (
            "B. Known-radar encoder",
            [("A0", "RID_A0_INPUT", "A0_ACC"), ("A1", "RID_A1_ADAPTER", "A1_ACC"), ("A2", "RID_A2_EMBED", "A2_ACC")],
            (0.90, 1.02),
            (0.70, 1.02),
        ),
        (
            "C. Motion-factor probe",
            [("All", "FAC_RID", "FAC_ACT"), ("Stable", "FAC_RID_STABLE", "FAC_ACT_STABLE")],
            (0.90, 1.02),
            (0.40, 0.80),
        ),
    ]
    src_panels = []
    for ax, (title, pts, xlim, ylim) in zip(axes, groups):
        keys = []
        for name, xk, yk in pts:
            ax.scatter(V(snap, xk), V(snap, yk))
            ax.annotate(name, (V(snap, xk), V(snap, yk)), textcoords="offset points", xytext=(4, 4), fontsize=8)
            keys.append(f"{xk}/{yk}")
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("Radar-ID")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        src_panels.append(",".join(keys))
    axes[0].set_ylabel("Activity / HAR (protocol-specific)")
    fig.suptitle("Fig. 6  Activity information vs radar information (separate probe families)")
    fig.tight_layout()
    save(fig, "fig6_tradeoff.png", ["A", "B", "C"], src_panels)

    # Fig 7
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    x = np.array([0, 1])
    ax.bar(x - 0.18, [V(snap, "D2477_B0_R2"), V(snap, "D7724_B0_R2")], 0.34, label="B0 mean D(v)")
    ax.bar(x + 0.18, [V(snap, "D2477_B1_R2"), V(snap, "D7724_B1_R2")], 0.34, label="B1 D(v|q)")
    ax.set_xticks(x, ["24→77", "77→24"])
    ax.set_ylabel("R² of velocity-marginal transfer")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.set_title("Fig. 7  Motion-conditioned D(v|q)  — not HAR accuracy")
    save(fig, "fig7_dvq.png", ["bars"], ["D2477_B0_R2,D2477_B1_R2,D7724_B0_R2,D7724_B1_R2"])

    # Fig 8
    fig, ax = plt.subplots(figsize=(5.8, 3.5))
    ax.bar([-0.18], [V(snap, "ORACLE_R2_2D")], 0.34, label="TEST oracle")
    ax.bar([0.18], [V(snap, "B2_R2_2477")], 0.34, label="B2 24→77")
    ax.bar([1.18], [V(snap, "B2_R2_7724")], 0.34, label="B2 77→24")
    ax.axhline(V(snap, "K20_TRAIN_VAR"), ls="--", lw=1, label="TRAIN explained var")
    ax.set_xticks([0, 1], ["24→77", "77→24"])
    ax.set_ylabel("2D Δ R²")
    ax.set_ylim(0, 0.6)
    ax.legend(fontsize=8)
    ax.set_title("Fig. 8  Low-rank 2D ceiling (oracle is not deployment)")
    save(fig, "fig8_lowrank.png", ["bars"], ["ORACLE_R2_2D,B2_R2_2477,B2_R2_7724,K20_TRAIN_VAR"])

    # Fig 9
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    steps = [
        ("High within-radar accuracy", "#E8EEF6"),
        ("Cross-radar collapse", "#F3E6D8"),
        ("Plotting?  partial confound", "#F7F1E4"),
        ("Numerical Doppler: radar identity remains", "#F7F1E4"),
        ("Velocity alignment (24/77): weak effect", "#F7F1E4"),
        ("Unified processing: insufficient", "#F7F1E4"),
        ("Calibration / adapter: insufficient", "#F7F1E4"),
        ("Motion-conditioned response: partly predictable", "#E8F0EA"),
        ("Missing spatial / acquisition variables", "#E8EEF6"),
    ]
    for i, (t, c) in enumerate(steps):
        y = 0.93 - i * 0.105
        ax.add_patch(plt.Rectangle((0.10, y - 0.038), 0.80, 0.076, facecolor=c, edgecolor="#334155", linewidth=1.0))
        ax.text(0.5, y, t, ha="center", va="center", fontsize=10)
        if i < len(steps) - 1:
            ax.annotate("", xy=(0.5, y - 0.066), xytext=(0.5, y - 0.038), arrowprops=dict(arrowstyle="->", color="#334155"))
    ax.set_title("Fig. 9  Progressive diagnosis")
    save(fig, "fig9_ladder.png", ["schematic"], ["narrative"])

    captions = """# Figure captions (paper numbering)

**Fig. 1.** Paired three-radar CI4R subset: one timestamp, three spectrograms, acquisition-level split.

**Fig. 2.** Official-representation 7-class HAR matrix. Diagonal: same-sensor. Off-diagonal: single-source transfer.

**Fig. 3.** Three-radar Radar-ID along official PNG → cropped PNG → numerical Doppler.

**Fig. 4.** Radar-ID by protocol. Panel A: 3-radar probe. Panel B: 24/77 two-radar probe. The panels are not one series.

**Fig. 5.** Harmonization (24/77 Radar-ID) and known-radar calibration (3-radar Radar-ID).

**Fig. 6.** Activity / HAR information versus Radar-ID, split by probe family. Axes are not pooled across panels.

**Fig. 7.** Motion-conditioned velocity-marginal transfer $R^2$. Not a HAR score.

**Fig. 8.** Low-rank 2D paired-difference ceiling. The TEST oracle is an upper bound.

**Fig. 9.** Progressive diagnosis from high within-radar accuracy to missing spatial / acquisition variables.
"""
    (fig_dir / "figure_captions.md").write_text(captions, encoding="utf-8")
    with (fig_dir.parent / "figure_source_audit.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["figure", "panel", "source_file", "metric", "generation_script"])
        w.writeheader()
        for r in audit:
            w.writerow(r)
    return audit
