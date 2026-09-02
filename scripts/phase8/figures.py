#!/usr/bin/env python3
"""Phase 8 figures. Fig4 R2 omitted if no cache."""
from __future__ import annotations

import csv
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"
P7 = ROOT / "results" / "phase7_radar_invariance"
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
RADARS = ["10GHz", "24GHz", "77GHz"]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.build_r1_cache import load_r1_gray224  # noqa: E402


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(encoding="utf-8")))


def pick_ts() -> str | None:
    split = read_csv(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv")
    man = read_csv(ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv")
    by = defaultdict(dict)
    for r in man:
        by[r["timestamp"]][r["radar_frequency"]] = r
    for r in split:
        if r.get("split") != "test" or r.get("action") not in CLASS_NAMES:
            continue
        ts = r["timestamp"]
        if set(by.get(ts, {})) >= set(RADARS):
            return ts
    return None


def fig1(figdir: Path) -> None:
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(10.5, 3.4))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3.2)
    boxes = [
        (0.2, 2.1, "10 GHz .dat\nXeThru 182 float", "#dceaf7"),
        (0.2, 1.1, "24 GHz .dat\nfc Tsweep NTS Bw", "#dceaf7"),
        (0.2, 0.1, "77 GHz Raw_0.bin\n8-lane IQ", "#dceaf7"),
        (3.4, 2.1, "zip on 01 / GCS\nNOT on this PC", "#f7e1c8"),
        (3.4, 1.1, "mapping: 327/327\nraw_mapping.csv", "#e8f0d8"),
        (3.4, 0.1, "R2 S(t,f_D)\nNOT BUILT", "#f3d4d4"),
        (6.8, 1.5, "official PNG\nR0 / R1 only", "#eeeeee"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.8, 0.9, facecolor=c, edgecolor="#333"))
        ax.text(x + 1.4, y + 0.45, t, ha="center", va="center", fontsize=8)
    ax.annotate("", xy=(3.3, 1.55), xytext=(3.05, 1.55), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Fig1  raw → representation  (R2 blocked on this host)")
    fig.tight_layout()
    fig.savefig(figdir / "fig1_pipeline.png", dpi=140)
    plt.close(fig)


def fig23(ts: str, figdir: Path) -> None:
    plt = _mpl()
    man = read_csv(ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv")
    hits = {r["radar_frequency"]: r for r in man if r["timestamp"] == ts}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, freq in zip(axes, RADARS):
        ax.imshow(np.asarray(Image.open(ROOT / hits[freq]["source_file"]).convert("L")), cmap="gray", aspect="auto")
        ax.set_title(freq)
        ax.axis("off")
    fig.suptitle(f"Fig2  R0 official PNG  ts={ts}  {hits['10GHz'].get('action')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "fig2_r0_triplet.png", dpi=140)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, freq in zip(axes, RADARS):
        x, _ = load_r1_gray224(ROOT / hits[freq]["source_file"], freq)
        ax.imshow(x, cmap="gray", aspect="auto")
        ax.set_title(freq)
        ax.axis("off")
    fig.suptitle(f"Fig3  R1 fair PNG  ts={ts}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "fig3_r1_triplet.png", dpi=140)
    plt.close(fig)
    (figdir / "fig4_r2_note.txt").write_text(
        "Fig4 R2 not drawn. No physical numerical cache on this host.\n", encoding="utf-8"
    )


def fig5(figdir: Path) -> None:
    plt = _mpl()
    rows = read_csv(OUT / "representation_fingerprint.csv")
    pix = [r for r in rows if r["kind"] == "pixels_16x16_logreg"]
    labs, ys = [], []
    for r in pix:
        labs.append(r["representation"])
        try:
            ys.append(float(r["radar_id_accuracy"]))
        except (TypeError, ValueError):
            ys.append(0.0)
    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar(labs, ys, color=["#4c72b0", "#55a868", "#cccccc"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID (pixels)")
    ax.set_title("Fig5  R0 / R1 / R2  (R2 empty)")
    fig.tight_layout()
    fig.savefig(figdir / "fig5_radar_id.png", dpi=140)
    plt.close(fig)


def fig6(figdir: Path) -> None:
    plt = _mpl()
    # R0/R1 only
    sys.path.insert(0, str(ROOT / "scripts"))
    from phase7.common import R0_CROSS  # noqa: E402

    fair = {r["protocol"]: r for r in read_csv(P7 / "representation_fairness_comparison.csv")}
    order = ["10", "24", "77"]
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    for ax, kind, src in zip(axes, ("R0", "R1"), (R0_CROSS, None)):
        M = np.full((3, 3), np.nan)
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                key = f"{a}→{b}"
                if kind == "R0":
                    M[i, j] = R0_CROSS.get(key, np.nan)
                else:
                    try:
                        M[i, j] = float(fair[key]["R1_accuracy"])
                    except (KeyError, TypeError, ValueError):
                        pass
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), order)
        ax.set_yticks(range(3), order)
        ax.set_title(kind)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig6  R0 vs R1 only; R2 not trained", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "fig6_cross_r0_r1.png", dpi=140)
    plt.close(fig)


def main() -> None:
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    fig1(figdir)
    ts = pick_ts()
    if ts:
        fig23(ts, figdir)
        print("triplet", ts, flush=True)
    fig5(figdir)
    fig6(figdir)
    print("figures", figdir, flush=True)


if __name__ == "__main__":
    main()
