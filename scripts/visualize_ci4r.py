"""Save IQ / range / micro-Doppler figures used in Level 1 notes."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
AIRHAR = ROOT / "third_party" / "AIRHAR"
sys.path.insert(0, str(AIRHAR / "datasets"))
sys.path.insert(0, str(ROOT / "scripts"))
from ci4r_meta import CLASS_NAMES, list_samples  # noqa: E402


def save_png_gallery(spec_root: Path, out_dir: Path, freq: str = "77GHz") -> None:
    samples = list_samples(spec_root, freq)
    by_class: dict[str, list] = {}
    for sample in samples:
        by_class.setdefault(sample.class_name, []).append(sample)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    axes = axes.ravel()
    for i, class_name in enumerate(CLASS_NAMES):
        ax = axes[i]
        items = by_class.get(class_name, [])
        if not items:
            ax.set_axis_off()
            continue
        image = cv2.imread(str(items[0].path), cv2.IMREAD_GRAYSCALE)
        ax.imshow(image, aspect="auto", origin="lower", cmap="jet")
        ax.set_title(class_name)
        ax.set_xlabel("time")
        ax.set_ylabel("Doppler")
    axes[-1].set_axis_off()
    fig.tight_layout()
    fig.savefig(out_dir / f"micro_doppler_gallery_{freq}.png", dpi=150)
    plt.close(fig)


def visualize_bin(bin_path: Path, out_dir: Path) -> None:
    from data_prepare_CI4R import RGB_mD, bin2np

    data = bin2np(str(bin_path))
    nts, n_chirp = data.shape
    out_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax.plot(np.real(data[0, :2000]), label="I (range bin 0, first 2000 chirps)")
    ax.plot(np.imag(data[0, :2000]), label="Q", alpha=0.7)
    ax.set_xlabel("slow-time index")
    ax.set_ylabel("ADC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"{bin_path.stem}_iq.png", dpi=150)
    plt.close(fig)

    rp = np.fft.fftshift(np.fft.fft(data, axis=0), axes=0)
    rp_pos = np.abs(rp[nts // 2 :, :])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.imshow(20 * np.log10(rp_pos / rp_pos.max() + 1e-12), aspect="auto", origin="lower", cmap="jet")
    ax.set_xlabel("chirp")
    ax.set_ylabel("range bin")
    ax.set_title("range profile")
    fig.tight_layout()
    fig.savefig(out_dir / f"{bin_path.stem}_range.png", dpi=150)
    plt.close(fig)

    spec = RGB_mD(data)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(spec, aspect="auto", origin="lower", cmap="jet")
    ax.set_xlabel("time (resized)")
    ax.set_ylabel("Doppler (resized)")
    ax.set_title("STFT micro-Doppler 224x224")
    fig.tight_layout()
    fig.savefig(out_dir / f"{bin_path.stem}_stft.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--freq", default="77GHz")
    parser.add_argument(
        "--spec-root",
        default=str(ROOT / "data" / "CI4R_MULTI3" / "spectrograms" / "11_class_activity_data"),
    )
    parser.add_argument("--bin", default="")
    parser.add_argument("--out", default=str(ROOT / "results" / "figures"))
    args = parser.parse_args()
    out_dir = Path(args.out)
    save_png_gallery(Path(args.spec_root), out_dir, args.freq)
    print(f"wrote gallery to {out_dir}")
    if args.bin:
        visualize_bin(Path(args.bin), out_dir)
        print(f"wrote IQ/range/STFT from {args.bin}")


if __name__ == "__main__":
    main()
