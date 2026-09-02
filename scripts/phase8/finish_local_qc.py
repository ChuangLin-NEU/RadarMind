#!/usr/bin/env python3
"""Local Phase 8B finish: extract cache_r2, R1 pair similarity, Fig 2. No training."""
from __future__ import annotations

import csv
import tarfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"
CROP = {"10GHz": (1, 1, 1, 1), "24GHz": (8, 8, 0, 0), "77GHz": (55, 58, 0, 0)}
RADARS = ["10GHz", "24GHz", "77GHz"]
SMOKE = ["1573836822", "1573837114", "1573837435"]


def pooled_png(path: Path, radar: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    top, bot, left, right = CROP[radar]
    crop = im.crop((left, top, w - right, h - bot))
    arr = np.asarray(crop.resize((16, 16), Image.BILINEAR), dtype=np.float32)
    return arr.reshape(-1)


def cosine(a, b) -> float:
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na <= 0 or nb <= 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def extract_tar() -> None:
    tar = OUT / "cache_r2.tar"
    dest = OUT / "cache_r2"
    if dest.is_dir() and any(dest.rglob("*.npz")):
        print("cache_r2 already extracted", flush=True)
        return
    if not tar.is_file():
        print("no cache_r2.tar", flush=True)
        return
    with tarfile.open(tar, "r") as t:
        t.extractall(OUT)
    print("extracted", tar, flush=True)


def r1_similarity() -> dict:
    man = list(csv.DictReader((OUT / "raw_mapping.csv").open(encoding="utf-8")))
    by = defaultdict(dict)
    for r in man:
        p = ROOT / r["local_png"]
        if p.is_file():
            by[r["timestamp"]][r["radar"]] = pooled_png(p, r["radar"])
    paired = [ts for ts, d in by.items() if len(d) == 3]
    same = []
    for ts in paired:
        for a, b in (("10GHz", "24GHz"), ("10GHz", "77GHz"), ("24GHz", "77GHz")):
            same.append(cosine(by[ts][a], by[ts][b]))
    rng = np.random.RandomState(42)
    ts_list = list(paired)
    diff = []
    for _ in range(min(2000, len(same) * 3)):
        t1, t2 = rng.choice(ts_list, 2, replace=False)
        a, b = rng.choice(RADARS, 2, replace=False)
        diff.append(cosine(by[t1][a], by[t2][b]))
    return {
        "same_pair_mean": float(np.mean(same)) if same else "",
        "diff_pair_mean": float(np.mean(diff)) if diff else "",
        "n_same": len(same),
        "n_diff": len(diff),
        "n_paired": len(paired),
    }


def fig2() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    man = list(csv.DictReader((OUT / "raw_mapping.csv").open(encoding="utf-8")))
    by = defaultdict(dict)
    for r in man:
        by[r["timestamp"]][r["radar"]] = ROOT / r["local_png"]
    fig, axes = plt.subplots(3, 3, figsize=(11, 8))
    for i, ts in enumerate(SMOKE):
        for j, radar in enumerate(RADARS):
            ax = axes[i, j]
            p = by[ts].get(radar)
            if p and p.is_file():
                ax.imshow(np.asarray(Image.open(p).convert("L")), cmap="gray")
            ax.set_title(f"{radar} {ts}", fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("Fig 2  official PNG (same timestamps as smoke)")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "fig2_official_png.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("wrote fig2", flush=True)


def main() -> None:
    extract_tar()
    (OUT / "figures").mkdir(parents=True, exist_ok=True)
    fig2()
    r1 = r1_similarity()
    print("R1 similarity", r1, flush=True)
    r2 = list(csv.DictReader((OUT / "r2_similarity.csv").open(encoding="utf-8")))
    r2d = next(x for x in r2 if x["representation"] == "R2-D")
    fields = ["representation", "same_pair_mean", "diff_pair_mean", "n_same", "n_diff"]
    rows = [
        {
            "representation": "R1",
            "same_pair_mean": r1["same_pair_mean"],
            "diff_pair_mean": r1["diff_pair_mean"],
            "n_same": r1["n_same"],
            "n_diff": r1["n_diff"],
        },
        r2d,
        {"representation": "R2-V", "same_pair_mean": "", "diff_pair_mean": "", "n_same": "", "n_diff": "not built"},
    ]
    with (OUT / "r2_similarity.csv").open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    labels = ["R1 same", "R1 diff", "R2-D same", "R2-D diff"]
    vals = [
        float(r1["same_pair_mean"]),
        float(r1["diff_pair_mean"]),
        float(r2d["same_pair_mean"]),
        float(r2d["diff_pair_mean"]),
    ]
    ax.bar(labels, vals)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("cosine (16×16)")
    fig.savefig(OUT / "figures" / "fig6_pair_similarity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("done", flush=True)


if __name__ == "__main__":
    main()
