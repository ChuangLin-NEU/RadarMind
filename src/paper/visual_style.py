"""RadarMind SCI visual system. Fig. 2–11 must import colors only from here."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

NAVY = "#3E5F7D"
BLUE = "#5B8DB8"
TEAL = "#5F9EA0"
GREEN = "#6E9F72"
AMBER = "#D5A34A"
PURPLE = "#8C7AA9"
GRAY = "#9A9A9A"
DARK_GRAY = "#555555"
LIGHT_GRAY = "#E7E7E7"
PALE_GRAY = "#F4F4F4"

PALETTE = {
    "NAVY": NAVY,
    "BLUE": BLUE,
    "TEAL": TEAL,
    "GREEN": GREEN,
    "AMBER": AMBER,
    "PURPLE": PURPLE,
    "GRAY": GRAY,
    "DARK_GRAY": DARK_GRAY,
    "LIGHT_GRAY": LIGHT_GRAY,
    "PALE_GRAY": PALE_GRAY,
}

GHZ = {"10": BLUE, "24": AMBER, "77": GREEN}
DATASET = {"mRI": TEAL, "RadHAR": PURPLE, "CI4R": NAVY}
OPENSET = {"MSP": GRAY, "Energy": BLUE, "Prototype": GREEN}
ACTIVITY = NAVY
RETRIEVAL = TEAL
RADAR_ID = DARK_GRAY
STATUS = {"Supported": NAVY, "Partial": AMBER, "Unresolved": GRAY}

INK = DARK_GRAY
WHITE = "#FFFFFF"

MM_DOUBLE = 180.0 / 25.4
MM_SINGLE = 88.0 / 25.4


def _pick_font() -> str:
    available = {f.name for f in font_manager.fontManager.ttflist}
    for name in ("Arial", "Helvetica", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


FONT = _pick_font()


def apply_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [FONT, "Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8.5,
            "xtick.labelsize": 7.5,
            "ytick.labelsize": 7.5,
            "legend.fontsize": 7.5,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.3,
            "patch.linewidth": 0.8,
            "xtick.major.width": 0.7,
            "ytick.major.width": 0.7,
            "xtick.major.size": 2.8,
            "ytick.major.size": 2.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
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
            "axes.prop_cycle": mpl.cycler(color=[NAVY, BLUE, TEAL, GREEN, AMBER, PURPLE, GRAY, DARK_GRAY]),
        }
    )


def despine(ax, heatmap: bool = False) -> None:
    if heatmap:
        for side in ("top", "right", "bottom", "left"):
            ax.spines[side].set_visible(True)
            ax.spines[side].set_linewidth(0.8)
            ax.spines[side].set_color(DARK_GRAY)
        return
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.spines["left"].set_color(DARK_GRAY)
    ax.spines["bottom"].set_color(DARK_GRAY)
    ax.tick_params(colors=DARK_GRAY, width=0.7, length=2.8)
    ax.xaxis.label.set_color(DARK_GRAY)
    ax.yaxis.label.set_color(DARK_GRAY)


def panel_label(ax, letter: str, x: float = -0.08, y: float = 1.06) -> None:
    ax.text(
        x,
        y,
        f"({letter})",
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=DARK_GRAY,
        clip_on=False,
    )


def fmt3(value: float) -> str:
    return f"{float(value):.3f}"


def contrast_text(rgba) -> str:
    r, g, b = rgba[0], rgba[1], rgba[2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return WHITE if y < 0.50 else DARK_GRAY


def pale(hex_color: str, mix: float = 0.55) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    r = int(r + (255 - r) * mix)
    g = int(g + (255 - g) * mix)
    b = int(b + (255 - b) * mix)
    return f"#{r:02X}{g:02X}{b:02X}"


def navy_pale() -> str:
    return pale(NAVY, 0.82)


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
