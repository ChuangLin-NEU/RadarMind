#!/usr/bin/env python3
"""R0 vs R1 domain statistics + representation-only Radar-ID (pixels, not motion encoder)."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import CROP, OUT, P2_CACHE, RADARS, load_band, write_csv  # noqa: E402

STAT_FIELDS = [
    "representation",
    "radar",
    "n",
    "native_height_mean",
    "native_width_mean",
    "pixel_mean",
    "pixel_std",
    "entropy",
    "dynamic_range",
    "time_axis_variation",
    "doppler_axis_variation",
    "spectrum_low_energy_frac",
    "spectrum_centroid",
]


def _entropy(x: np.ndarray) -> float:
    lo, hi = float(x.min()), float(x.max())
    if hi - lo < 1e-8:
        return 0.0
    hist, _ = np.histogram(((x - lo) / (hi - lo)).ravel(), bins=64, range=(0, 1), density=False)
    p = hist.astype(np.float64)
    p = p / (p.sum() + 1e-12)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def _stats_stack(X: np.ndarray) -> dict:
    # X: N,H,W already z-scored per image; still report empirical moments
    means = X.mean(axis=(1, 2))
    stds = X.std(axis=(1, 2))
    ents, drs, tvars, dvars, lowf, cents = [], [], [], [], [], []
    for i in range(len(X)):
        x = X[i]
        ents.append(_entropy(x))
        drs.append(float(x.max() - x.min()))
        tvars.append(float(x.mean(axis=0).std()))  # variation across time bins
        dvars.append(float(x.mean(axis=1).std()))  # variation across doppler bins
        spec = np.abs(np.fft.rfft(x.mean(axis=1)))
        spec = spec / (spec.sum() + 1e-12)
        freqs = np.arange(len(spec), dtype=np.float64)
        lowf.append(float(spec[: max(1, len(spec) // 8)].sum()))
        cents.append(float((freqs * spec).sum()))
    return {
        "pixel_mean": float(means.mean()),
        "pixel_std": float(stds.mean()),
        "entropy": float(np.mean(ents)),
        "dynamic_range": float(np.mean(drs)),
        "time_axis_variation": float(np.mean(tvars)),
        "doppler_axis_variation": float(np.mean(dvars)),
        "spectrum_low_energy_frac": float(np.mean(lowf)),
        "spectrum_centroid": float(np.mean(cents)),
    }


def native_hw(freq: str, kind: str) -> tuple[float, float]:
    z = np.load(
        P2_CACHE / f"ci4r_{freq.replace('GHz', '')}ghz_png224.npz"
        if kind == "R0"
        else OUT / "cache" / f"ci4r_{freq.replace('GHz', '')}ghz_r1_png224.npz",
        allow_pickle=True,
    )
    if kind == "R1" and "native_height" in z.files:
        return float(z["native_height"].mean()), float(z["native_width"].mean())
    hs, ws = [], []
    for p in z["source_file"]:
        im = Image.open(ROOT / str(p))
        w, h = im.size
        hs.append(h)
        ws.append(w)
    return float(np.mean(hs)), float(np.mean(ws))


def pooled_pixels(X: np.ndarray) -> np.ndarray:
    # 16x16 appearance vector; representation-only, not a motion encoder
    out = np.zeros((len(X), 16, 16), np.float32)
    for i, x in enumerate(X):
        im = Image.fromarray((x - x.min()) / (x.max() - x.min() + 1e-6) * 255.0).convert("F")
        out[i] = np.asarray(im.resize((16, 16), Image.BILINEAR), dtype=np.float32)
    return out.reshape(len(X), -1)


def radar_id_pixels(kind: str) -> dict:
    Xs, ys, splits = [], [], []
    for ri, freq in enumerate(RADARS):
        d = load_band(freq, kind)
        Xs.append(d["X"])
        ys.append(np.full(len(d["y"]), ri, np.int64))
        splits.append(d["split"])
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys)
    split = np.concatenate(splits)
    feat = pooled_pixels(X)
    tr, te = split == "train", split == "test"
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(feat[tr])
    Xte = scaler.transform(feat[te])
    clf = LogisticRegression(max_iter=500, random_state=42)
    clf.fit(Xtr, y[tr])
    pred = clf.predict(Xte)
    return {
        "representation": kind,
        "classifier": "logreg_16x16_pixels",
        "n_train": int(tr.sum()),
        "n_test": int(te.sum()),
        "accuracy": float(accuracy_score(y[te], pred)),
        "macro_f1": float(f1_score(y[te], pred, average="macro", zero_division=0)),
        "note": "fit train pixels, eval test; not a motion-encoder probe",
    }


def main() -> None:
    rows = []
    for kind in ("R0", "R1"):
        for freq in RADARS:
            d = load_band(freq, kind)
            st = _stats_stack(d["X"])
            nh, nw = native_hw(freq, kind)
            rows.append(
                {
                    "representation": kind,
                    "radar": freq,
                    "n": int(len(d["y"])),
                    "native_height_mean": nh,
                    "native_width_mean": nw,
                    **st,
                }
            )
            print(kind, freq, {k: round(st[k], 4) for k in ("entropy", "time_axis_variation", "doppler_axis_variation")}, flush=True)
    write_csv(OUT / "representation_domain_statistics.csv", STAT_FIELDS, rows)
    probes = [radar_id_pixels("R0"), radar_id_pixels("R1")]
    write_csv(
        OUT / "representation_radar_id.csv",
        ["representation", "classifier", "n_train", "n_test", "accuracy", "macro_f1", "note"],
        probes,
    )
    (OUT / "representation_radar_id.json").write_text(json.dumps(probes, indent=2), encoding="utf-8")
    print("Radar-ID pixels", probes, flush=True)


if __name__ == "__main__":
    main()
