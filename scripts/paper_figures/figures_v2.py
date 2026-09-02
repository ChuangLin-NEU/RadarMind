"""Fig. 2–11 redrawn from visual_style only. Scientific values from snapshot/frozen files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np

from paper.activity_color_map import map_activities
from paper.visual_style import (
    AMBER,
    BLUE,
    DARK_GRAY,
    DATASET,
    GHZ,
    GRAY,
    GREEN,
    LIGHT_GRAY,
    MM_DOUBLE,
    NAVY,
    OPENSET,
    PALE_GRAY,
    PURPLE,
    RADAR_ID,
    RETRIEVAL,
    STATUS,
    TEAL,
    WHITE,
    ACTIVITY,
    contrast_text,
    despine,
    fmt3,
    navy_pale,
    pale,
    panel_label,
    save_figure,
)
from paper_figures.timeline import extract_mode_events, extract_raw_windows, load_gt_events

MAIN = ROOT / "figures" / "main"


def fig02(book) -> dict:
    n_paired = book.display("N_PAIRED_ACQ", "Fig. 2", "paired acquisitions")
    n_subj = book.display("MRI_N_SUBJECTS", "Fig. 2", "mRI subjects")
    n_act = book.display("MRI_N_ACTIONS", "Fig. 2", "mRI actions")
    n_rh = book.display("RADHAR_N_ACTIONS", "Fig. 2", "RadHAR actions")
    book.display("CI4R77_RAND_ACC", "Fig. 2", "CI4R-77 random")
    book.display("CI4R77_PREFIX_ACC", "Fig. 2", "CI4R-77 prefix")

    rows = [
        f"CI4R paired 10/24/77\n{n_paired} agreed triples",
        "CI4R 77 GHz 11-class\nrandom vs prefix-group",
        f"mRI point cloud\n{n_subj} subjects, {n_act} actions",
        f"RadHAR official\n{n_rh} actions",
    ]
    cols = [
        "Cross-radar\npaired",
        "Random vs\nprefix-group",
        "Subject-\nindependent",
        "Labeled-interval\ncontinuous",
        "LOAO\nopen-set",
        "Official\nsplit",
        "Temporal /\nfeature ablation",
    ]
    grid = np.array(
        [
            [2, 0, 0, 0, 0, 1, 0],
            [0, 2, 0, 0, 0, 1, 0],
            [0, 0, 2, 2, 2, 1, 1],
            [0, 0, 0, 0, 0, 2, 2],
        ]
    )
    colors = {0: WHITE, 1: navy_pale(), 2: NAVY}
    labels = {0: "—", 1: "used", 2: "primary"}
    fig, ax = plt.subplots(figsize=(MM_DOUBLE, 3.35))
    for i in range(4):
        for j in range(7):
            v = int(grid[i, j])
            ax.add_patch(Rectangle((j, 3 - i), 1, 1, facecolor=colors[v], edgecolor=WHITE, lw=1.4))
            tc = WHITE if v == 2 else DARK_GRAY
            ax.text(j + 0.5, 3 - i + 0.5, labels[v], ha="center", va="center", fontsize=7, color=tc)
    ax.set_xticks([x + 0.5 for x in range(7)], cols, fontsize=6.8, color=DARK_GRAY)
    ax.set_yticks([y + 0.5 for y in range(4)], list(reversed(rows)), fontsize=7, color=DARK_GRAY)
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 4)
    ax.tick_params(length=0, colors=DARK_GRAY)
    for side in ("top", "right", "bottom", "left"):
        ax.spines[side].set_visible(True)
        ax.spines[side].set_linewidth(0.6)
        ax.spines[side].set_color(LIGHT_GRAY)
    paths = save_figure(fig, "fig02_dataset_protocol_map", MAIN)
    return paths


def fig03(book) -> dict:
    items = [
        ("mRI SI\nXYZ T=32", "MRI_SI_ACC", DATASET["mRI"]),
        ("RadHAR\nofficial XYZV", "RADHAR_XYZV_ACC", DATASET["RadHAR"]),
        ("CI4R 10 GHz\nwithin", "R0_10_WITHIN", GHZ["10"]),
        ("CI4R 24 GHz\nwithin", "R0_24_WITHIN", GHZ["24"]),
        ("CI4R 77 GHz\nwithin", "R0_77_WITHIN", GHZ["77"]),
    ]
    vals = [book.value(k, "Fig. 3", lab.replace("\n", " ")) for lab, k, _ in items]
    fig, axes = plt.subplots(1, 2, figsize=(MM_DOUBLE, 3.15), gridspec_kw={"width_ratios": [1.28, 1.0]})
    ax = axes[0]
    xs = np.array([0.0, 1.0, 2.85, 3.85, 4.85])
    ax.bar(xs, vals, width=0.72, color=[c for _, _, c in items], edgecolor=DARK_GRAY, linewidth=0.4)
    ax.set_xticks(xs, [lab for lab, _, _ in items], fontsize=6.6, color=DARK_GRAY)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Accuracy")
    for x, v in zip(xs, vals):
        ax.text(x, v + 0.02, fmt3(v), ha="center", va="bottom", fontsize=7, color=DARK_GRAY)
    despine(ax)
    panel_label(ax, "a")

    ax = axes[1]
    pairs = [
        ("mRI", "MRI_T1_ACC", "MRI_SI_ACC", "T=1", "T=32", pale(TEAL, 0.45), TEAL),
        ("RadHAR", "RADHAR_T1_ACC", "RADHAR_XYZV_ACC", "T=1", "T=60", pale(PURPLE, 0.45), PURPLE),
    ]
    for i, (name, k1, k2, l1, l2, c1, c2) in enumerate(pairs):
        v1 = book.value(k1, "Fig. 3", f"{name} {l1}")
        v2 = book.value(k2, "Fig. 3", f"{name} {l2}")
        x0 = i * 2.3
        ax.bar([x0, x0 + 0.85], [v1, v2], width=0.62, color=[c1, c2], edgecolor=DARK_GRAY, linewidth=0.4)
        ax.text(x0, v1 + 0.02, fmt3(v1), ha="center", fontsize=7, color=DARK_GRAY)
        ax.text(x0 + 0.85, v2 + 0.02, fmt3(v2), ha="center", fontsize=7, color=DARK_GRAY)
        ax.text(x0, -0.09, l1, ha="center", va="top", fontsize=6.8, color=DARK_GRAY, transform=ax.get_xaxis_transform())
        ax.text(x0 + 0.85, -0.09, l2, ha="center", va="top", fontsize=6.8, color=DARK_GRAY, transform=ax.get_xaxis_transform())
        ax.text(x0 + 0.42, 1.02, name, ha="center", fontsize=7.5, color=DARK_GRAY)
    ax.set_ylim(0, 1.08)
    ax.set_xlim(-0.55, 3.7)
    ax.set_xticks([])
    ax.set_ylabel("Accuracy")
    despine(ax)
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.2)
    return save_figure(fig, "fig03_baseline_temporal", MAIN)


def fig04(book) -> dict:
    keys = [
        ["R0_10_WITHIN", "R0_10_TO_24", "R0_10_TO_77"],
        ["R0_24_TO_10", "R0_24_WITHIN", "R0_24_TO_77"],
        ["R0_77_TO_10", "R0_77_TO_24", "R0_77_WITHIN"],
    ]
    mat = np.array([[book.value(k, "Fig. 4", k) for k in row] for row in keys], dtype=float)
    fig, ax = plt.subplots(figsize=(MM_DOUBLE * 0.52, 3.45))
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="cividis")
    ax.set_xticks([0, 1, 2], ["10 GHz", "24 GHz", "77 GHz"])
    ax.set_yticks([0, 1, 2], ["10 GHz", "24 GHz", "77 GHz"])
    ax.set_xlabel("Test radar")
    ax.set_ylabel("Train radar")
    cmap = mpl_cividis()
    for i in range(3):
        for j in range(3):
            rgba = cmap(mat[i, j])
            ax.text(j, i, fmt3(mat[i, j]), ha="center", va="center", fontsize=9, color=contrast_text(rgba))
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Accuracy")
    cb.ax.tick_params(labelsize=7, colors=DARK_GRAY)
    cb.outline.set_linewidth(0.6)
    despine(ax, heatmap=True)
    fig.tight_layout()
    return save_figure(fig, "fig04_cross_radar_matrix", MAIN)


def mpl_cividis():
    return plt.get_cmap("cividis")


def fig05(book) -> dict:
    models = ["M0", "M1", "M2"]
    acc = [book.value(k, "Fig. 5", f"{m} activity") for m, k in zip(models, ["M0_ACC", "M1_ACC", "M2_ACC"])]
    ret = [book.value(k, "Fig. 5", f"{m} retrieval") for m, k in zip(models, ["M0_RET", "M1_RET", "M2_RET"])]
    rid = [book.value(k, "Fig. 5", f"{m} Radar-ID") for m, k in zip(models, ["M0_RID", "M1_RID", "M2_RID_SHARE"])]
    fig, axes = plt.subplots(1, 2, figsize=(MM_DOUBLE, 3.15))
    ax = axes[0]
    x = np.arange(3)
    w = 0.36
    ax.bar(x - w / 2, acc, w, color=ACTIVITY, edgecolor=DARK_GRAY, lw=0.4, label="Activity accuracy")
    ax.bar(x + w / 2, ret, w, color=RETRIEVAL, edgecolor=DARK_GRAY, lw=0.4, label="Same-action retrieval")
    for i in range(3):
        ax.text(x[i] - w / 2, acc[i] + 0.018, fmt3(acc[i]), ha="center", fontsize=7, color=DARK_GRAY)
        ax.text(x[i] + w / 2, ret[i] + 0.018, fmt3(ret[i]), ha="center", fontsize=7, color=DARK_GRAY)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Score")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2, fontsize=7.5)
    despine(ax)
    panel_label(ax, "a")

    ax = axes[1]
    ax.bar(x, rid, color=RADAR_ID, edgecolor=DARK_GRAY, lw=0.4)
    for i, v in enumerate(rid):
        ax.text(i, v + 0.015, fmt3(v), ha="center", fontsize=7, color=DARK_GRAY)
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel("Radar-ID accuracy")
    ax.axhline(1.0, color=LIGHT_GRAY, lw=0.6, linestyle=(0, (3, 2)), zorder=0)
    despine(ax)
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.4)
    return save_figure(fig, "fig05_shared_representation", MAIN)


def fig06(book) -> dict:
    a_items = [
        ("R0", "RID_R0_3CLASS", None),
        ("R1", "RID_R1_3CLASS", "ACT_R1_REP"),
        ("R2-D", "RID_R2D_3CLASS", "ACT_R2D_REP"),
    ]
    b_items = [
        ("R2-D", "RID_R2D_2477", "ACT_R2D_2477"),
        ("R2-V", "RID_R2V_2477", "ACT_R2V_2477"),
        ("U1 STFT", "U1_RID", "U1_ACT"),
        ("U2 MTI", "U2_RID", "U2_ACT"),
        ("D(v|q) cal.", "CAL_RID_AFTER", "CAL_ACT_AFTER"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(MM_DOUBLE, 3.25), gridspec_kw={"width_ratios": [0.82, 1.18]})

    def panel(ax, items, title, letter):
        xs = np.arange(len(items))
        rid = [book.value(k, "Fig. 6", f"{lab} Radar-ID") for lab, k, _ in items]
        act_x, act_y = [], []
        for i, (lab, _k, ak) in enumerate(items):
            if not ak:
                continue
            try:
                v = book.value(ak, "Fig. 6", f"{lab} activity")
            except Exception:
                continue
            act_x.append(i)
            act_y.append(v)
        ax.plot(xs, rid, color=RADAR_ID, lw=1.2, zorder=2)
        ax.scatter(xs, rid, s=36, color=RADAR_ID, zorder=3, label="Radar-ID")
        if act_x:
            ax.plot(act_x, act_y, color=ACTIVITY, lw=1.15, ls=(0, (3, 2)), zorder=2)
            ax.scatter(act_x, act_y, s=36, facecolors=WHITE, edgecolors=ACTIVITY, linewidths=1.15, zorder=3, label="Activity probe")
        for i, v in enumerate(rid):
            ax.text(i, min(v + 0.012, 1.015), fmt3(v), ha="center", fontsize=6.6, color=DARK_GRAY)
        for i, v in zip(act_x, act_y):
            ax.text(i, v - 0.018, fmt3(v), ha="center", va="top", fontsize=6.6, color=NAVY)
        ax.set_xticks(xs, [lab for lab, _, _ in items], fontsize=7)
        ax.set_ylim(0.80, 1.02)
        ax.set_ylabel("Probe accuracy")
        ax.set_title(title, fontsize=8.5, color=DARK_GRAY, pad=4)
        ax.legend(loc="lower left", fontsize=7)
        despine(ax)
        panel_label(ax, letter)

    panel(axes[0], a_items, "3-radar probe", "a")
    panel(axes[1], b_items, "24/77 probe", "b")
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, "fig06_radar_identity_ladder", MAIN)


def fig07(book) -> dict:
    rows = [
        ("D(v|q), 24→77", "D2477_B1_R2", BLUE, "o", True),
        ("D(v|q), 77→24", "D7724_B1_R2", GREEN, "o", True),
        ("2D TEST oracle", "ORACLE_R2_2D", DARK_GRAY, "D", False),
        ("B2 predicted, 24→77", "B2_R2_2477", BLUE, "o", False),
        ("B2 predicted, 77→24", "B2_R2_7724", GREEN, "o", False),
    ]
    vals = [book.value(k, "Fig. 7", lab) for lab, k, *_ in rows]
    fig, ax = plt.subplots(figsize=(MM_DOUBLE, 3.15))
    ys = np.arange(len(rows))[::-1]
    for y, v, (lab, _k, col, mk, filled) in zip(ys, vals, rows):
        ax.hlines(y, 0, v, color=col, lw=1.15)
        fc = col if filled else WHITE
        ax.scatter([v], [y], s=42, marker=mk, facecolors=fc, edgecolors=col, linewidths=1.15, zorder=3)
        ax.text(v + 0.012, y, fmt3(v), va="center", fontsize=7, color=DARK_GRAY)
    ax.text(vals[2] + 0.095, ys[2], "upper bound", va="center", fontsize=6.5, color=GRAY)
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlabel("R²")
    ax.set_xlim(0, 0.80)
    ax.set_ylim(-0.6, 4.6)
    despine(ax)
    fig.tight_layout()
    return save_figure(fig, "fig07_measurement_difference", MAIN)


def fig08(book) -> dict:
    from matplotlib.patches import Patch

    pred_path = ROOT / "results" / "phase4_continuous" / "predicted_timelines" / "subject20_0513.json"
    gt_path = ROOT / "results" / "phase4_continuous" / "ground_truth_timelines" / "subject20_0513.json"
    t_max = 320.0
    gt = [e for e in load_gt_events(gt_path) if e["start_time"] < t_max]
    wins = extract_raw_windows(pred_path, t_max)
    evs = extract_mode_events(pred_path)
    sm_ev = [e for e in evs["confidence_aware"] if e["start_time"] < t_max]
    book.record_file("Fig. 8", "GT", str(gt_path.relative_to(ROOT)).replace("\\", "/"), "events", "")
    book.record_file("Fig. 8", "raw windows", str(pred_path.relative_to(ROOT)).replace("\\", "/"), "modes.raw.windows", "")
    book.record_file("Fig. 8", "smoothed", str(pred_path.relative_to(ROOT)).replace("\\", "/"), "modes.confidence_aware.events", "")

    acts = []
    for src in (gt, wins, sm_ev):
        for e in src:
            a = e["activity"]
            if a and a != "UNLABELED" and a not in acts:
                acts.append(a)
    cmap = map_activities(acts)

    fig = plt.figure(figsize=(MM_DOUBLE, 4.55))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.55, 1.0], width_ratios=[1.4, 0.6], hspace=0.38, wspace=0.32)
    ax = fig.add_subplot(gs[0, :])
    tracks = [(2.35, gt, True, "Ground truth"), (1.35, wins, False, "Raw windows"), (0.35, sm_ev, False, "Smoothed events")]
    for y, seq, allow_unlab, name in tracks:
        ax.text(-10, y + 0.22, name, ha="right", va="center", fontsize=7, color=DARK_GRAY)
        for e in seq:
            a = e["activity"]
            x0, x1 = e["start_time"], min(e["end_time"], t_max)
            if x1 <= 0:
                continue
            if a == "UNLABELED":
                if not allow_unlab:
                    continue
                ax.add_patch(Rectangle((x0, y), x1 - x0, 0.52, facecolor=LIGHT_GRAY, edgecolor="none", hatch="////"))
            else:
                ax.add_patch(Rectangle((x0, y), max(x1 - x0, 0.12), 0.52, facecolor=cmap.get(a, GRAY), edgecolor="none"))
    ax.set_xlim(0, t_max)
    ax.set_ylim(0.1, 3.15)
    ax.set_xlabel("Time (s)")
    ax.set_yticks([])
    despine(ax)
    panel_label(ax, "a", x=-0.10, y=1.02)
    handles = [Patch(facecolor=cmap[a], edgecolor="none", label=a) for a in acts[:8]]
    handles.append(Patch(facecolor=LIGHT_GRAY, hatch="////", edgecolor=GRAY, label="Unlabeled (not scored)"))
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=6.2, frameon=False, bbox_to_anchor=(0.5, -0.01))

    ax = fig.add_subplot(gs[1, 0])
    keys = ["CONT_SEG_F1_RAW", "CONT_SEG_F1_FLIP", "CONT_SEG_F1"]
    labs = ["Raw", "Isolated flip", "Confidence-aware"]
    vals = [book.value(k, "Fig. 8", lab) for k, lab in zip(keys, labs)]
    cols = [pale(NAVY, 0.65), pale(NAVY, 0.28), NAVY]
    ax.bar(range(3), vals, color=cols, edgecolor=DARK_GRAY, lw=0.4)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.012, fmt3(v), ha="center", fontsize=7, color=DARK_GRAY)
    ax.set_xticks(range(3), labs, fontsize=7)
    ax.set_ylabel("Segment F1")
    ax.set_ylim(0, 1.05)
    nwin = book.display("CONT_N_WINDOWS", "Fig. 8", "n windows")
    ax.text(0.02, 0.90, f"n = {nwin}", transform=ax.transAxes, fontsize=6.5, color=GRAY)
    despine(ax)
    panel_label(ax, "b", x=-0.12, y=1.06)

    ax = fig.add_subplot(gs[1, 1])
    bnd = book.value("CONT_BND_SEC", "Fig. 8", "boundary error s")
    ax.plot([0, 0], [0, bnd], color=RADAR_ID, lw=1.2)
    ax.scatter([0], [bnd], s=36, color=RADAR_ID, zorder=3)
    ax.text(0.08, bnd, fmt3(bnd) + " s", va="center", fontsize=7, color=DARK_GRAY)
    ax.set_xlim(-0.8, 1.4)
    ax.set_ylim(0, max(bnd * 1.25, 1))
    ax.set_xticks([])
    ax.set_ylabel("Median boundary error (s)")
    despine(ax)
    panel_label(ax, "c", x=-0.18, y=1.06)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    return save_figure(fig, "fig08_continuous_inference", MAIN)


def fig09(book) -> dict:
    names = [("MSP", "loao_msp_per_action.json"), ("Energy", "loao_energy_per_action.json"), ("Prototype", "loao_prototype_per_action.json")]
    folds = {}
    for name, fn in names:
        p = ROOT / "results" / "phase5_open_set" / fn
        data = json.loads(p.read_text(encoding="utf-8"))
        folds[name] = [float(r["AUROC"]) for r in data]
        book.record_file("Fig. 9", f"{name} AUROC", f"results/phase5_open_set/{fn}", "AUROC", "13 folds")
    for key in ("MSP_AUROC", "ENERGY_AUROC", "PROTO_AUROC", "MSP_FPR", "ENERGY_FPR", "PROTO_FPR"):
        book.value(key, "Fig. 9", key)

    fig, axes = plt.subplots(1, 2, figsize=(MM_DOUBLE, 3.2))
    ax = axes[0]
    data = [folds[n] for n, _ in names]
    cols = [OPENSET[n] for n, _ in names]
    bp = ax.boxplot(
        data,
        positions=[1, 2, 3],
        widths=0.46,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color=DARK_GRAY, lw=1.1),
        whiskerprops=dict(color=DARK_GRAY, lw=0.8),
        capprops=dict(color=DARK_GRAY, lw=0.8),
        boxprops=dict(lw=1.05),
    )
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
        patch.set_alpha(0.15)
        patch.set_edgecolor(c)
    rng = np.random.default_rng(0)
    for i, ((n, _), c) in enumerate(zip(names, cols), start=1):
        y = np.array(folds[n])
        x = i + rng.uniform(-0.08, 0.08, size=len(y))
        ax.scatter(x, y, s=12, color=c, alpha=0.7, zorder=3, linewidths=0)
    ax.axhline(0.5, color=GRAY, lw=0.8, linestyle=(0, (3, 2)))
    ax.text(3.28, 0.505, "0.500", fontsize=6.5, color=GRAY)
    ax.set_xticks([1, 2, 3], [n for n, _ in names])
    ax.set_ylim(0.0, 1.05)
    ax.set_ylabel("AUROC")
    despine(ax)
    panel_label(ax, "a")

    ax = axes[1]
    fpr = [
        book.value("MSP_FPR", "Fig. 9", "MSP FPR95"),
        book.value("ENERGY_FPR", "Fig. 9", "Energy FPR95"),
        book.value("PROTO_FPR", "Fig. 9", "Prototype FPR95"),
    ]
    xs = [1, 2, 3]
    for x, v, c in zip(xs, fpr, cols):
        ax.plot([x, x], [0, v], color=c, lw=1.15)
        ax.scatter([x], [v], s=40, color=c, zorder=3)
        ax.text(x, v + 0.03, fmt3(v), ha="center", fontsize=7, color=DARK_GRAY)
    ax.set_xticks(xs, [n for n, _ in names])
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("FPR95")
    despine(ax)
    panel_label(ax, "b")
    fig.tight_layout(w_pad=2.0)
    return save_figure(fig, "fig09_open_set", MAIN)


def fig10(book) -> dict:
    path = ROOT / "results" / "radarmind_v01" / "llm_online_test.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    rec0 = data["records"][0]
    mi = rec0["input"]["motion_intelligence"]
    tl = rec0["input"]["motion_timeline"]["events"][0]
    book.record_file("Fig. 10", "MI JSON", "results/radarmind_v01/llm_online_test.json", "records.0", "")
    for k in ("BOX_CONF", "BOX_UNC", "BOX_NWIN", "BOX_PEAK_V", "BOX_ENERGY", "LLM_N", "LLM_N_OK"):
        book.value(k, "Fig. 10", k)

    def fv(name):
        d = mi[name]
        return d.get("value"), d.get("source", "")

    act, _ = fv("activity")
    conf = float(fv("confidence")[0])
    unc = float(fv("uncertainty")[0])
    peak = float(fv("peak_velocity")[0])
    energy = float(fv("motion_intensity")[0])

    fig, axes = plt.subplots(1, 3, figsize=(MM_DOUBLE, 3.55))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis("off")

    ax = axes[0]
    ax.add_patch(Rectangle((0.25, 0.3), 9.5, 9.4, facecolor=pale(BLUE, 0.88), edgecolor="none"))
    ax.text(5, 9.25, "Model inference", ha="center", fontsize=8.5, fontweight="bold", color=DARK_GRAY)
    ax.text(5, 7.55, f"Predicted activity: {act}", ha="center", fontsize=7.5, color=DARK_GRAY)
    ax.text(5, 6.55, f"Confidence: {fmt3(conf)}", ha="center", fontsize=7.5, color=DARK_GRAY)
    ax.text(5, 5.55, f"Uncertainty: {fmt3(unc)}", ha="center", fontsize=7.5, color=DARK_GRAY)
    ax.add_patch(Rectangle((1.1, 1.3), 7.8, 3.2, facecolor=WHITE, edgecolor=LIGHT_GRAY, lw=0.6))
    ax.text(5, 3.7, f"{tl['activity']}", ha="center", fontsize=8, fontweight="bold", color=NAVY)
    ax.text(5, 2.7, f"{float(tl['start_time']):.0f}–{float(tl['end_time']):.0f} s", ha="center", fontsize=7, color=DARK_GRAY)
    ax.text(5, 1.85, f"{int(tl['n_windows'])} windows", ha="center", fontsize=7, color=DARK_GRAY)

    ax = axes[1]
    ax.add_patch(Rectangle((0.15, 0.3), 9.7, 9.4, facecolor=PALE_GRAY, edgecolor="none"))
    ax.text(5, 9.25, "Motion Intelligence", ha="center", fontsize=8.5, fontweight="bold", color=DARK_GRAY)
    rows = [
        ("activity", str(act), "MODEL", BLUE),
        ("confidence", fmt3(conf), "MODEL", BLUE),
        ("uncertainty", fmt3(unc), "MODEL", BLUE),
        ("peak |v|", f"{fmt3(peak)} m s−1", "PHYSICS", GREEN),
        ("motion energy", fmt3(energy), "PHYSICS", GREEN),
        ("direction", "UNKNOWN", "UNKNOWN", GRAY),
        ("periodicity", "UNKNOWN", "UNKNOWN", GRAY),
        ("subject", "UNKNOWN", "UNKNOWN", GRAY),
        ("environment", "UNKNOWN", "UNKNOWN", GRAY),
    ]
    y = 8.35
    for k, v, src, col in rows:
        ax.text(0.45, y, k, fontsize=6.4, color=DARK_GRAY, va="center")
        ax.text(4.55, y, v, fontsize=6.4, color=DARK_GRAY, va="center")
        ax.text(8.15, y, src, fontsize=5.8, color=col, va="center")
        y -= 0.82

    ax = axes[2]
    ax.add_patch(Rectangle((0.25, 0.3), 9.5, 9.4, facecolor=pale(GREEN, 0.90), edgecolor="none"))
    ax.text(5, 9.25, "Grounded language response", ha="center", fontsize=8.5, fontweight="bold", color=DARK_GRAY)
    ax.text(0.6, 7.9, "Question", fontsize=7, fontweight="bold", color=DARK_GRAY)
    ax.text(0.6, 6.85, "Does the timeline contain\nbend-and-pick?", fontsize=7, color=DARK_GRAY)
    ax.text(0.6, 5.15, "Response", fontsize=7, fontweight="bold", color=DARK_GRAY)
    ax.text(0.6, 3.85, "No bend-and-pick event is\npresent in the current timeline.", fontsize=7, color=DARK_GRAY)
    ax.text(0.6, 1.7, "available velocity descriptors\nare reported separately.", fontsize=6.2, color=GRAY)

    fig.text(0.18, 0.055, "Model inference", ha="center", fontsize=7, color=DARK_GRAY)
    fig.text(0.50, 0.055, "→  Motion Intelligence  →", ha="center", fontsize=7, color=DARK_GRAY)
    fig.text(0.82, 0.055, "Language output", ha="center", fontsize=7, color=DARK_GRAY)
    fig.add_artist(Line2D([0.16, 0.46], [0.018, 0.018], transform=fig.transFigure, color=GRAY, ls=(0, (4, 2)), lw=0.8, clip_on=False))
    fig.add_artist(Line2D([0.54, 0.84], [0.018, 0.018], transform=fig.transFigure, color=GRAY, ls=(0, (4, 2)), lw=0.8, clip_on=False))
    fig.text(0.50, 0.018, "×", ha="center", va="center", fontsize=10, color=GRAY, transform=fig.transFigure)
    fig.text(0.16, -0.012, "Raw radar", ha="left", fontsize=6.5, color=GRAY, transform=fig.transFigure)
    fig.text(0.84, -0.012, "Language layer", ha="right", fontsize=6.5, color=GRAY, transform=fig.transFigure)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(fig, "fig10_motion_intelligence_llm", MAIN)


def fig11(book) -> dict:
    nodes = [
        ("Closed-set HAR", "Supported", f"mRI {book.display('MRI_SI_ACC','Fig. 11','mRI')}\nRadHAR {book.display('RADHAR_XYZV_ACC','Fig. 11','RadHAR')}"),
        ("Subject\ngeneralization", "Partial", f"mRI SI {book.display('MRI_SI_ACC','Fig. 11','SI')}"),
        ("Cross-radar\ntransfer", "Unresolved", f"24→77 = {book.display('R0_24_TO_77','Fig. 11','24to77')}"),
        ("Continuous\ninference", "Partial", f"segment F1 = {book.display('CONT_SEG_F1','Fig. 11','seg')}"),
        ("Open-set\ndetection", "Partial", f"Prototype AUROC\n= {book.display('PROTO_AUROC','Fig. 11','proto')}"),
        ("Semantic\ninteraction", "Supported", f"stored tasks =\n{book.display('LLM_N_OK','Fig. 11','ok')}/{book.display('LLM_N','Fig. 11','n')}"),
    ]
    fig, ax = plt.subplots(figsize=(MM_DOUBLE, 2.85))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5.0)
    ax.axis("off")
    xs = np.linspace(1.05, 10.95, 6)
    ax.plot([xs[0], xs[-1]], [2.55, 2.55], color=LIGHT_GRAY, lw=1.15, zorder=0)
    for i, ((title, st, ev), x) in enumerate(zip(nodes, xs)):
        col = STATUS[st]
        ax.scatter([x], [2.55], s=78, color=col, zorder=3, edgecolors=WHITE, linewidths=1.1)
        ax.text(x, 3.15, title, ha="center", va="bottom", fontsize=6.7, fontweight="bold", color=DARK_GRAY)
        ax.text(x, 2.05, ev, ha="center", va="top", fontsize=6.1, color=DARK_GRAY)
        if i < 5:
            ax.annotate("", xy=(xs[i + 1] - 0.22, 2.55), xytext=(x + 0.22, 2.55), arrowprops=dict(arrowstyle="-|>", color=GRAY, lw=0.75))
    ax.text(xs[2], 0.72, "unseen-radar transfer remains unresolved", ha="center", fontsize=6.2, color=GRAY)
    ax.text(6.0, 0.18, "Radar → recognition → Motion Intelligence → timeline → language", ha="center", fontsize=7.2, color=DARK_GRAY)
    return save_figure(fig, "fig11_capability_summary", MAIN)
