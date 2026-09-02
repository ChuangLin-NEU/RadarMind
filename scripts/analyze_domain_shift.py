"""Compare spectrogram energy / Doppler width across people and radar frequencies."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from ci4r_meta import CLASS_NAMES, FREQ_TO_DIR, list_samples  # noqa: E402


def features(path: Path) -> dict:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise RuntimeError(f"cannot read {path}")
    arr = image.astype(np.float32)
    col = arr.mean(axis=1)
    energy = float(arr.mean())
    width = float((col > (col.mean() + 0.5 * col.std())).sum())
    return {"energy": energy, "doppler_width": width}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--spec-root",
        default=str(ROOT / "data" / "CI4R_MULTI3" / "spectrograms" / "11_class_activity_data"),
    )
    args = parser.parse_args()
    spec_root = Path(args.spec_root)
    out_dir = ROOT / "results" / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for freq in ("77GHz", "24GHz", "10GHz"):
        folder = spec_root / FREQ_TO_DIR[freq]
        if not folder.exists():
            continue
        for sample in list_samples(spec_root, freq):
            feat = features(sample.path)
            rows.append({**feat, "freq": freq, "class": sample.class_name, "subject": sample.subject})

    if not rows:
        raise SystemExit("no spectrograms")

    summary = {}
    for freq in sorted({r["freq"] for r in rows}):
        items = [r for r in rows if r["freq"] == freq]
        summary[freq] = {
            "n": len(items),
            "energy_mean": float(np.mean([r["energy"] for r in items])),
            "energy_std": float(np.std([r["energy"] for r in items])),
            "doppler_width_mean": float(np.mean([r["doppler_width"] for r in items])),
            "doppler_width_std": float(np.std([r["doppler_width"] for r in items])),
        }
    (out_dir / "domain_shift_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    freqs = list(summary)
    axes[0].boxplot([[r["energy"] for r in rows if r["freq"] == f] for f in freqs], tick_labels=freqs)
    axes[0].set_title("spectrogram mean intensity")
    axes[1].boxplot([[r["doppler_width"] for r in rows if r["freq"] == f] for f in freqs], tick_labels=freqs)
    axes[1].set_title("occupied Doppler bins")
    fig.tight_layout()
    fig.savefig(out_dir / "domain_shift_freq.png", dpi=150)
    plt.close(fig)

    subjects = sorted({r["subject"] for r in rows if r["freq"] == "77GHz"})
    if subjects:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.boxplot(
            [[r["energy"] for r in rows if r["freq"] == "77GHz" and r["subject"] == s] for s in subjects],
            tick_labels=subjects,
        )
        ax.set_title("77 GHz intensity by subject")
        fig.tight_layout()
        fig.savefig(out_dir / "domain_shift_subject_77.png", dpi=150)
        plt.close(fig)

    rec = {
        "observation": "先看跨频强度和多普勒占用宽度是否 systematically 不同，再决定要不要做 f_D=2v/λ 归一化。",
        "n_rows": len(rows),
        "classes": CLASS_NAMES,
        "summary": summary,
    }
    (out_dir / "m8_notes.json").write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
