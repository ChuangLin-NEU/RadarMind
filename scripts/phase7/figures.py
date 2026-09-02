#!/usr/bin/env python3
"""Phase 7 figures A–H. Missing panels are skipped with a note file."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import CLASS_NAMES, CROP, OUT, R0_CROSS, RADARS, read_csv  # noqa: E402
from phase7.build_r1_cache import load_r1_gray224  # noqa: E402


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def pick_timestamp() -> str | None:
    split = read_csv(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv")
    man = read_csv(ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv")
    by = defaultdict(dict)
    for r in man:
        by[r["timestamp"]][r["radar_frequency"]] = r
    for r in split:
        if r.get("split") != "test":
            continue
        ts = r["timestamp"]
        if set(by.get(ts, {})) >= set(RADARS) and by[ts]["10GHz"].get("action") in CLASS_NAMES:
            return ts
    return None


def fig_ab(ts: str, figdir: Path) -> None:
    plt = _mpl()
    man = read_csv(ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv")
    hits = {r["radar_frequency"]: r for r in man if r["timestamp"] == ts}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, freq in zip(axes, RADARS):
        png = ROOT / hits[freq]["source_file"]
        ax.imshow(np.asarray(Image.open(png).convert("L")), cmap="gray", aspect="auto")
        ax.set_title(f"{freq} official")
        ax.axis("off")
    fig.suptitle(f"Fig A  timestamp {ts}  {hits['10GHz'].get('action')}", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "figA_official_triplet.png", dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    for ax, freq in zip(axes, RADARS):
        x, _ = load_r1_gray224(ROOT / hits[freq]["source_file"], freq)
        ax.imshow(x, cmap="gray", aspect="auto")
        ax.set_title(f"{freq} R1 224")
        ax.axis("off")
    fig.suptitle(f"Fig B  timestamp {ts}  R1 fair PNG", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "figB_r1_triplet.png", dpi=140)
    plt.close(fig)


def fig_cd(figdir: Path) -> None:
    plt = _mpl()
    rows = read_csv(OUT / "representation_domain_statistics.csv")
    if not rows:
        return
    keys = ["entropy", "time_axis_variation", "doppler_axis_variation", "dynamic_range"]
    for tag, kind in (("figC_r0_domain_stats.png", "R0"), ("figD_r1_domain_stats.png", "R1")):
        sub = [r for r in rows if r["representation"] == kind]
        fig, axes = plt.subplots(1, 4, figsize=(12, 3.2))
        x = np.arange(len(RADARS))
        for ax, key in zip(axes, keys):
            ys = []
            for freq in RADARS:
                hit = next((r for r in sub if r["radar"] == freq), None)
                ys.append(float(hit[key]) if hit else 0.0)
            ax.bar(x, ys, color=["#4c72b0", "#55a868", "#c44e52"])
            ax.set_xticks(x, ["10", "24", "77"])
            ax.set_title(key, fontsize=8)
        fig.suptitle(f"{kind} domain statistics", fontsize=10)
        fig.tight_layout()
        fig.savefig(figdir / tag, dpi=140)
        plt.close(fig)


def _acc_map(kind: str) -> dict[str, float]:
    if kind == "R0":
        return dict(R0_CROSS)
    rows = read_csv(OUT / "representation_fairness_comparison.csv")
    out = {}
    for r in rows:
        proto = r.get("protocol", "")
        try:
            out[proto] = float(r["R1_accuracy"])
        except (TypeError, ValueError, KeyError):
            continue
    return out


def fig_e(figdir: Path) -> None:
    plt = _mpl()
    order = ["10GHz", "24GHz", "77GHz"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0))
    for ax, kind in zip(axes, ("R0", "R1")):
        acc = _acc_map(kind)
        M = np.full((3, 3), np.nan)
        for i, a in enumerate(order):
            for j, b in enumerate(order):
                key = f"{a.replace('GHz', '')}→{b.replace('GHz', '')}"
                if key in acc:
                    M[i, j] = acc[key]
        im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis")
        ax.set_xticks(range(3), ["10", "24", "77"])
        ax.set_yticks(range(3), ["10", "24", "77"])
        ax.set_xlabel("test")
        ax.set_ylabel("train")
        ax.set_title(kind)
        for i in range(3):
            for j in range(3):
                if not np.isnan(M[i, j]):
                    ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Fig E  R0 vs R1 cross-radar accuracy", fontsize=10)
    fig.tight_layout()
    fig.savefig(figdir / "figE_r0_vs_r1_matrix.png", dpi=140)
    plt.close(fig)


def fig_f(figdir: Path) -> None:
    plt = _mpl()
    rows = read_csv(OUT / "probe_results.csv")
    shared = [r for r in rows if r.get("protocol") == "multi_radar_shared"]
    if not shared:
        (figdir / "figF_note.txt").write_text("M3/probes not trained.\n", encoding="utf-8")
        return
    models = [r["model"] for r in shared]
    act = [float(r["activity_probe_accuracy"]) if r.get("activity_probe_accuracy") else 0 for r in shared]
    rad = [float(r["radar_probe_accuracy"]) if r.get("radar_probe_accuracy") else 0 for r in shared]
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    ax.bar(x - 0.18, act, 0.36, label="Activity Probe")
    ax.bar(x + 0.18, rad, 0.36, label="Radar-ID Probe")
    ax.set_xticks(x, models)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Fig F  Activity Probe / Radar-ID Probe")
    fig.tight_layout()
    fig.savefig(figdir / "figF_probes.png", dpi=140)
    plt.close(fig)


def fig_g(figdir: Path) -> None:
    plt = _mpl()
    rows = read_csv(OUT / "loro_results.csv")
    if not rows:
        (figdir / "figG_note.txt").write_text("LORO not trained.\n", encoding="utf-8")
        return
    folds = ["77GHz", "24GHz", "10GHz"]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    x = np.arange(len(folds))
    for off, model in ((-0.18, "M0"), (0.18, "M3")):
        ys = []
        for tgt in folds:
            hit = next((r for r in rows if r["model"] == model and r["test_radar"] == tgt), None)
            ys.append(float(hit["activity_accuracy"]) if hit and hit.get("activity_accuracy") else 0.0)
        ax.bar(x + off, ys, 0.36, label=model)
    ax.set_xticks(x, ["10+24→77", "10+77→24", "24+77→10"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title("Fig G  Leave-One-Radar-Out")
    fig.tight_layout()
    fig.savefig(figdir / "figG_loro.png", dpi=140)
    plt.close(fig)


def main() -> None:
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    ts = pick_timestamp()
    if ts:
        fig_ab(ts, figdir)
        print("Fig A/B", ts, flush=True)
    fig_cd(figdir)
    fig_e(figdir)
    fig_f(figdir)
    fig_g(figdir)
    print("figures in", figdir, flush=True)


if __name__ == "__main__":
    main()
