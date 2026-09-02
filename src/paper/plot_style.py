"""Shared matplotlib style for RadarMind main-text figures.

Print-oriented: white background, one sans font, fixed semantic colors.
No gradients, shadows, 3D, or UI chrome.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


FONT = "DejaVu Sans"

# Okabe–Ito. One concept keeps one colour in every figure.
C_10 = "#0072B2"
C_24 = "#E69F00"
C_77 = "#009E73"
C_MRI = "#56B4E9"
C_RADHAR = "#CC79A7"
C_MSP = "#999999"
C_ENERGY = "#CC79A7"
C_PROTO = "#56B4E9"
C_M0 = "#999999"
C_M1 = "#0072B2"
C_M2 = "#D55E00"
C_ACTIVITY = "#0072B2"
C_RETRIEVAL = "#56B4E9"
C_RID = "#D55E00"
C_ACT = C_ACTIVITY
C_T1 = "#BBBBBB"
C_INK = "#222222"
C_GRID = "#D0D0D0"
C_MUTED = "#666666"
C_HATCH = "#B0B0B0"
C_SUPPORT = C_10
C_PARTIAL = C_24
C_UNRESOLVED = "#999999"

GHZ_COLOR = {"10": C_10, "24": C_24, "77": C_77}
DATASET_COLOR = {"mRI": C_MRI, "RadHAR": C_RADHAR}
OPENSET_COLOR = {"MSP": C_MSP, "Energy": C_ENERGY, "Prototype": C_PROTO}
MODEL_COLOR = {"M0": C_M0, "M1": C_M1, "M2": C_M2}
ACTIVITY_PALETTE = [
    C_10,
    C_24,
    C_77,
    C_M2,
    C_RADHAR,
    C_MRI,
    "#000000",
    "#F0E442",
]

ACCURACY_CMAP = LinearSegmentedColormap.from_list(
    "rb_acc",
    ["#F7F4ED", "#E8D5B5", "#C4A35A", "#6B8F71", "#2F4A6A"],
)


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Arial", "Helvetica"],
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "figure.titlesize": 9,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "lines.linewidth": 1.4,
            "lines.markersize": 5.0,
            "patch.linewidth": 0.8,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.unicode_minus": False,
        }
    )


def despine(ax, heatmap: bool = False) -> None:
    if heatmap:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.8)
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.9)
    ax.spines["bottom"].set_linewidth(0.9)
    ax.tick_params(width=0.8, length=3.0)


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.08) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=C_INK,
        clip_on=False,
    )


def fmt3(value: float) -> str:
    return f"{float(value):.3f}"


def save_figure(fig, stem: str, out_dir: Path) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for ext, dpi in ((".pdf", None), (".svg", None), (".png", 600)):
        dest = out_dir / f"{stem}{ext}"
        kw = {"bbox_inches": "tight", "facecolor": "white", "edgecolor": "none"}
        if dpi is not None:
            kw["dpi"] = dpi
        fig.savefig(dest, **kw)
        paths[ext.lstrip(".")] = str(dest).replace("\\", "/")
    plt.close(fig)
    return paths
