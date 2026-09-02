"""Twelve main-text figures from the metric snapshot and existing result CSVs."""
from __future__ import annotations

from pathlib import Path


def D(snap, key: str) -> str:
    return snap["metrics"][key]["display"]


def V(snap, key: str) -> float:
    return float(snap["metrics"][key]["value"])


def make_figures(snap: dict, fig_dir: Path, script_name: str) -> list[dict]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig_dir.mkdir(parents=True, exist_ok=True)
    audit: list[dict] = []

    def save(fig, name, panel, sources):
        fig.savefig(fig_dir / name, dpi=160, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        audit.append(
            {
                "figure": name,
                "panel": panel,
                "source_file": "results/final_delivery/metric_snapshot_full.json",
                "metric": sources,
                "generation_script": script_name,
            }
        )

    # Fig 1 architecture
    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6.4)
    ax.axis("off")
    layers = [
        (5.0, 5.4, "Layer I  Radar recognition", "spectrogram · point cloud · HAR\ncross-subject · cross-radar"),
        (5.0, 3.5, "Layer II  Motion understanding", "shared representation · continuous inference\nopen-set · Motion Intelligence · Timeline"),
        (5.0, 1.6, "Layer III  Semantic interaction", "language layer · explanation · QA · application"),
    ]
    for x, y, title, body in layers:
        ax.add_patch(plt.Rectangle((0.4, y - 0.75), 6.4, 1.55, fill=True, facecolor="#f4f1e8", edgecolor="#333", lw=1.1))
        ax.text(3.6, y + 0.45, title, ha="center", va="center", fontsize=10, fontweight="bold")
        ax.text(3.6, y - 0.15, body, ha="center", va="center", fontsize=8)
    ax.add_patch(plt.Rectangle((7.05, 0.35), 2.7, 5.7, fill=True, facecolor="#f7ecec", edgecolor="#7a3030", lw=1.1))
    ax.text(8.4, 5.5, "Current boundary", ha="center", fontsize=9, fontweight="bold", color="#7a3030")
    ax.text(8.4, 3.9, "Cross-environment\nNOT YET\n\nSensor invariance\nNOT YET\n\nOpen-world\nNOT YET", ha="center", va="center", fontsize=8)
    ax.set_title("Fig. 1  RadarMind overview")
    save(fig, "fig1_overview.png", "layers", "architecture schematic")

    # Fig 2 datasets
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.axis("off")
    boxes = [
        (0.3, 0.4, "CI4R", f"{D(snap,'N_PAIRED_ACQ')} paired acq.\n10 / 24 / 77 GHz\nspectrogram / raw\ncross-radar"),
        (3.0, 0.4, "mRI", f"{D(snap,'MRI_N_SUBJECTS')} subjects\n{D(snap,'MRI_N_ACTIONS')} actions\npoint cloud\nSI / continuous / LOAO"),
        (5.7, 0.4, "RadHAR", f"{D(snap,'RADHAR_N_ACTIONS')} actions\nofficial train/test\npoint cloud\ntemporal / feature"),
    ]
    ax.set_xlim(0, 8.6)
    ax.set_ylim(0, 3.4)
    for x, y, t, b in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.5, 2.6, fill=True, facecolor="#eef2f6", edgecolor="#333"))
        ax.text(x + 1.25, 2.65, t, ha="center", fontsize=11, fontweight="bold")
        ax.text(x + 1.25, 1.45, b, ha="center", va="center", fontsize=8)
    ax.set_title("Fig. 2  Datasets and representations (not a single unified benchmark)")
    save(fig, "fig2_datasets.png", "three boxes", "N_PAIRED_ACQ,MRI_N_SUBJECTS,MRI_N_ACTIONS,RADHAR_N_ACTIONS")

    # Fig 3 baselines + temporal
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    ax = axes[0]
    names = ["mRI SI", "mRI random", "RadHAR XYZV", "RadHAR voxel"]
    vals = [V(snap, "MRI_SI_ACC"), V(snap, "MRI_RAND_ACC"), V(snap, "RADHAR_XYZV_ACC"), V(snap, "RADHAR_VOXEL_ACC")]
    disp = [D(snap, "MRI_SI_ACC"), D(snap, "MRI_RAND_ACC"), D(snap, "RADHAR_XYZV_ACC"), D(snap, "RADHAR_VOXEL_ACC")]
    ax.bar(range(4), vals, color=["#3d5a80", "#98c1d9", "#ee6c4d", "#e0a458"])
    ax.set_xticks(range(4), names, rotation=20, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    for i, d in enumerate(disp):
        ax.text(i, vals[i] + 0.02, d, ha="center", fontsize=8)
    ax.set_title("Baseline recognition")
    ax = axes[1]
    xs = [1, 32, 1, 60]
    ys = [V(snap, "MRI_T1_ACC"), V(snap, "MRI_SI_ACC"), V(snap, "RADHAR_T1_ACC"), V(snap, "RADHAR_XYZV_ACC")]
    ax.plot([1, 32], [ys[0], ys[1]], "o-", label="mRI SI", color="#3d5a80")
    ax.plot([1, 60], [ys[2], ys[3]], "s-", label="RadHAR official", color="#ee6c4d")
    ax.set_xlabel("Temporal length T")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.set_title("Temporal context")
    fig.tight_layout()
    save(fig, "fig3_baseline_temporal.png", "two panels", "MRI_SI_ACC,MRI_T1_ACC,RADHAR_XYZV_ACC,RADHAR_T1_ACC")

    # Fig 4 matrix
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
            ax.text(j, i, D(snap, keys[i][j]), ha="center", va="center", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, label="7-class HAR accuracy")
    ax.set_title("Fig. 4  Within-radar vs cross-radar (official plotted)")
    save(fig, "fig4_cross_radar.png", "matrix", ",".join(k for row in keys for k in row))

    # Fig 5 R0→R2-V
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    labs = ["R0 3-radar\nRadar-ID", "R2-D 3-radar\nRadar-ID", "R2-D 24/77\nRadar-ID", "R2-V 24/77\nRadar-ID"]
    vs = [V(snap, "RID_R0_3CLASS"), V(snap, "RID_R2D_3CLASS"), V(snap, "RID_R2D_2477"), V(snap, "RID_R2V_2477")]
    ds = [D(snap, "RID_R0_3CLASS"), D(snap, "RID_R2D_3CLASS"), D(snap, "RID_R2D_2477"), D(snap, "RID_R2V_2477")]
    colors = ["#6b705c", "#6b705c", "#bc6c25", "#bc6c25"]
    ax.bar(range(4), vs, color=colors)
    ax.set_xticks(range(4), labs, fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Radar-ID accuracy")
    ax.axhline(1.0, color="#999", ls="--", lw=0.8)
    for i, d in enumerate(ds):
        ax.text(i, vs[i] + 0.03, d, ha="center", fontsize=8)
    ax.text(0.5, 1.08, "3-radar probe", ha="center", fontsize=8, color="#6b705c")
    ax.text(2.5, 1.08, "24/77 probe (not the same protocol)", ha="center", fontsize=8, color="#bc6c25")
    ax.set_title("Fig. 5  Representation reconstruction does not erase radar identity")
    save(fig, "fig5_reconstruction.png", "bars", "RID_R0_3CLASS,RID_R2D_3CLASS,RID_R2D_2477,RID_R2V_2477")

    # Fig 6 Radar-ID vs activity
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    pts = [
        ("R0", V(snap, "RID_R0_3CLASS"), V(snap, "R0_24_WITHIN"), "3-radar / HAR"),
        ("R2-D", V(snap, "RID_R2D_3CLASS"), V(snap, "ACT_R2D_REP"), "3-radar probes"),
        ("R2-V", V(snap, "RID_R2V_2477"), V(snap, "ACT_R2V_2477"), "24/77 probes"),
        ("M1", V(snap, "M1_RID"), V(snap, "M1_ACTP"), "shared embed."),
        ("A2", V(snap, "RID_A2_EMBED"), V(snap, "A2_ACC"), "known-radar adapter"),
    ]
    for name, x, y, _fam in pts:
        ax.scatter([x], [y], s=60)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.set_xlabel("Radar-ID (protocol differs by marker family)")
    ax.set_ylabel("Activity information")
    ax.set_xlim(0.85, 1.02)
    ax.set_ylim(0.7, 1.02)
    ax.set_title("Fig. 6  Activity information vs residual radar identity")
    save(fig, "fig6_tradeoff.png", "scatter", "RID_R0_3CLASS,RID_R2D_3CLASS,RID_R2V_2477,M1_RID,RID_A2_EMBED")

    # Fig 7 shared / retrieval / LORO
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.5))
    ax = axes[0]
    labs = ["M0 CE", "M1 +InfoNCE", "M2 +phys. aug"]
    acc = [V(snap, "M0_ACC"), V(snap, "M1_ACC"), V(snap, "M2_ACC")]
    ret = [V(snap, "M0_RET"), V(snap, "M1_RET"), V(snap, "M2_RET")]
    x = np.arange(3)
    ax.bar(x - 0.18, acc, 0.36, label="known-radar Acc")
    ax.bar(x + 0.18, ret, 0.36, label="same-action top-1")
    ax.set_xticks(x, labs, fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=7)
    ax.set_title("Paired learning (known radars)")
    ax = axes[1]
    ax.bar([0], [V(snap, "M1_RID")], color="#c1121f")
    ax.text(0, V(snap, "M1_RID") + 0.02, D(snap, "M1_RID"), ha="center")
    ax.set_xticks([0], ["M1 Radar-ID"])
    ax.set_ylim(0, 1.15)
    ax.set_title("Sensor identity remains")
    fig.tight_layout()
    save(fig, "fig7_shared.png", "two panels", "M0_ACC,M1_ACC,M2_ACC,M0_RET,M1_RET,M2_RET,M1_RID")

    # Fig 8 continuous
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labs = ["Frame Acc", "Frame F1", "Window Acc", "Segment F1"]
    vs = [V(snap, "CONT_FRAME_ACC"), V(snap, "CONT_FRAME_F1"), V(snap, "CONT_WIN_ACC"), V(snap, "CONT_SEG_F1")]
    ds = [D(snap, "CONT_FRAME_ACC"), D(snap, "CONT_FRAME_F1"), D(snap, "CONT_WIN_ACC"), D(snap, "CONT_SEG_F1")]
    ax.bar(range(4), vs, color="#3d5a80")
    ax.set_xticks(range(4), labs)
    ax.set_ylim(0, 0.7)
    for i, d in enumerate(ds):
        ax.text(i, vs[i] + 0.02, d, ha="center", fontsize=9)
    ax.set_ylabel("Score")
    ax.set_title(f"Fig. 8  Labeled-interval continuous replay  (n_windows={D(snap,'CONT_N_WINDOWS')})")
    save(fig, "fig8_continuous.png", "bars", "CONT_FRAME_ACC,CONT_FRAME_F1,CONT_WIN_ACC,CONT_SEG_F1,CONT_N_WINDOWS")

    # Fig 9 open-set
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    labs = ["MSP", "Energy", "Prototype"]
    means = [V(snap, "MSP_AUROC"), V(snap, "ENERGY_AUROC"), V(snap, "PROTO_AUROC")]
    stds = [V(snap, "MSP_AUROC_STD"), V(snap, "ENERGY_AUROC_STD"), V(snap, "PROTO_AUROC_STD")]
    ax.bar(range(3), means, yerr=stds, capsize=4, color=["#adb5bd", "#6c757d", "#3d5a80"])
    ax.axhline(0.5, color="#999", ls="--", lw=0.8, label="chance")
    ax.set_xticks(range(3), labs)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("AUROC (13-fold LOAO)")
    for i, k in enumerate(["MSP_AUROC", "ENERGY_AUROC", "PROTO_AUROC"]):
        ax.text(i, means[i] + stds[i] + 0.03, D(snap, k), ha="center", fontsize=8)
    ax.set_title("Fig. 9  Open-set unknown-motion detection")
    save(fig, "fig9_openset.png", "bars", "MSP_AUROC,ENERGY_AUROC,PROTO_AUROC")

    # Fig 10 MI → LLM
    fig, ax = plt.subplots(figsize=(8.2, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    boxes = [
        (0.2, 1.0, "Model\nprediction", f"boxing\nconf {D(snap,'BOX_CONF')}"),
        (2.6, 1.0, "Motion\nIntelligence", f"|v| peak {D(snap,'BOX_PEAK_V')}\nenergy {D(snap,'BOX_ENERGY')}"),
        (5.0, 1.0, "Timeline", f"{D(snap,'BOX_NWIN')} windows\nmerged event"),
        (7.4, 1.0, "Language\nlayer", "explanation\nQA · UNKNOWN"),
    ]
    for x, y, t, b in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.1, 1.7, fill=True, facecolor="#e8f1e8", edgecolor="#333"))
        ax.text(x + 1.05, 2.4, t, ha="center", fontsize=8, fontweight="bold")
        ax.text(x + 1.05, 1.5, b, ha="center", fontsize=8)
    ax.annotate("", xy=(2.55, 1.85), xytext=(2.3, 1.85), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(4.95, 1.85), xytext=(4.7, 1.85), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.35, 1.85), xytext=(7.1, 1.85), arrowprops=dict(arrowstyle="->"))
    ax.text(5, 0.35, "LLM never receives raw IQ, spectrogram, or point cloud.", ha="center", fontsize=8)
    ax.set_title("Fig. 10  Structured Motion Intelligence to language")
    save(fig, "fig10_llm.png", "pipeline", "BOX_CONF,BOX_PEAK_V,BOX_ENERGY,BOX_NWIN")

    # Fig 11 measurement
    fig, ax = plt.subplots(figsize=(6.8, 3.6))
    labs = ["D(v|q)\n24→77 R2", "D(v|q)\n77→24 R2", "2D oracle R2", "2D B2 R2"]
    vs = [V(snap, "D2477_B1_R2"), V(snap, "D7724_B1_R2"), V(snap, "ORACLE_R2_2D"), V(snap, "B2_R2_2477")]
    ds = [D(snap, "D2477_B1_R2"), D(snap, "D7724_B1_R2"), D(snap, "ORACLE_R2_2D"), D(snap, "B2_R2_2477")]
    ax.bar(range(4), vs, color="#8d6a50")
    ax.set_xticks(range(4), labs, fontsize=8)
    ax.set_ylabel("R2")
    ax.set_ylim(min(0, min(vs) - 0.1), max(vs) + 0.15)
    for i, d in enumerate(ds):
        ax.text(i, vs[i] + 0.02, d, ha="center", fontsize=8)
    ax.set_title("Fig. 11  Measurement-response diagnosis (not a translator)")
    save(fig, "fig11_measurement.png", "bars", "D2477_B1_R2,D7724_B1_R2,ORACLE_R2_2D,B2_R2_2477")

    # Fig 12 app + roadmap
    fig, ax = plt.subplots(figsize=(8.2, 3.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.add_patch(plt.Rectangle((0.3, 0.4), 4.6, 3.2, fill=True, facecolor="#eef2f6", edgecolor="#333"))
    ax.text(2.6, 3.25, "Implemented now", ha="center", fontsize=10, fontweight="bold")
    ax.text(
        2.6,
        1.8,
        "multi-rep HAR · SI eval\ncross-radar benchmark\nlabeled-interval continuous\nopen-set LOAO\nMotion Intelligence · LLM\napplication smoke "
        f"{D(snap,'SMOKE_V02')}/{D(snap,'SMOKE_V02')}",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.add_patch(plt.Rectangle((5.2, 0.4), 4.5, 3.2, fill=True, facecolor="#f7ecec", edgecolor="#7a3030"))
    ax.text(7.45, 3.25, "Not yet", ha="center", fontsize=10, fontweight="bold", color="#7a3030")
    ax.text(
        7.45,
        1.8,
        "sensor-invariant tokens\nrobust unseen-radar DG\nfull-stream continuous\nstrong unknown detection\ncross-environment\nopen-world / open-vocab",
        ha="center",
        va="center",
        fontsize=8,
    )
    ax.set_title("Fig. 12  RadarMind status and next acquisition protocol")
    save(fig, "fig12_roadmap.png", "two boxes", "SMOKE_V02")

    return audit
