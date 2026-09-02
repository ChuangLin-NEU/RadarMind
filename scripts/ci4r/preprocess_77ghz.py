"""Official AIRHAR 77 GHz preprocessing only (RGB_mD). No extra MTI/CFAR."""
from __future__ import annotations

from typing import Dict

import numpy as np

# third_party/AIRHAR/datasets/data_prepare_CI4R.py:RGB_mD
NTS = 256
WINDOW = 256
NFFT = 4096
OVERLAP = 200
SHIFT = WINDOW - OVERLAP  # 56
RANGE_BINS = (23, 31)  # sum rp[23:31, :]
CLIP_DB = (-45.0, 0.0)
OUT_HW = 224


def airhar_micro_doppler(iq: np.ndarray, window: int = WINDOW, nfft: int = NFFT, overlap: int = OVERLAP) -> np.ndarray:
    """Return 224x224 float32 spectrogram. Matches AIRHAR RGB_mD math.

    Steps actually executed in AIRHAR (not merely commented):
    1. Range FFT + fftshift along ADC axis (256)
    2. Keep positive-range half [128:]
    3. Sum range bins 23:31
    4. STFT, Hann 256, hop 56, nfft 4096
    5. |fftshift| flipud
    6. 20 log10 vs peak, clip [-45, 0]
    7. cubic resize to 224x224

    Commented in AIRHAR but NOT run: MTI filter.
    MATLAB datToImage_77.m is a different chain (MTI, range bin 8, overlap 197, getframe PNG).
    """
    if iq.ndim != 2 or iq.shape[0] != NTS:
        raise ValueError(f"expected IQ (256, n_chirps), got {iq.shape}")

    fft_raw = np.fft.fftshift(np.fft.fft(iq, axis=0), axes=0)
    rp = fft_raw[NTS // 2 :, :]
    rangedata = np.sum(rp[RANGE_BINS[0] : RANGE_BINS[1], :], axis=0)

    shift = window - overlap
    n_win = (len(rangedata) - window - 1) // shift
    if n_win <= 0:
        raise ValueError("not enough chirps for STFT")

    # vectorized STFT; same windows as the official for-loop
    idx = np.arange(n_win)[:, None] * shift + np.arange(window)[None, :]
    frames = rangedata[idx] * np.hanning(window)[None, :]
    spec = np.fft.fft(frames, n=nfft, axis=1).T  # (nfft, n_win)

    sx2 = np.abs(np.flipud(np.fft.fftshift(spec, axes=0)))
    peak = np.max(sx2)
    if peak <= 0:
        dopp = np.full(sx2.shape, CLIP_DB[0], dtype=np.float32)
    else:
        dopp = 20.0 * np.log10(np.abs(sx2 / peak))
        dopp = np.clip(dopp, CLIP_DB[0], CLIP_DB[1])
    return _resize_224(dopp.astype(np.float32))


def _resize_224(img: np.ndarray) -> np.ndarray:
    try:
        import cv2  # noqa: WPS433

        return cv2.resize(img, (OUT_HW, OUT_HW), interpolation=cv2.INTER_CUBIC).astype(np.float32)
    except Exception:
        # fallback: numpy bilinear-ish via zoom
        from numpy import linspace

        h, w = img.shape
        ys = linspace(0, h - 1, OUT_HW)
        xs = linspace(0, w - 1, OUT_HW)
        y0 = np.floor(ys).astype(int)
        x0 = np.floor(xs).astype(int)
        y1 = np.clip(y0 + 1, 0, h - 1)
        x1 = np.clip(x0 + 1, 0, w - 1)
        wy = ys - y0
        wx = xs - x0
        out = (
            img[y0][:, x0] * (1 - wy)[:, None] * (1 - wx)[None, :]
            + img[y1][:, x0] * wy[:, None] * (1 - wx)[None, :]
            + img[y0][:, x1] * (1 - wy)[:, None] * wx[None, :]
            + img[y1][:, x1] * wy[:, None] * wx[None, :]
        )
        return out.astype(np.float32)


def zscore(img: np.ndarray) -> np.ndarray:
    """AIRHAR prepare_alabma_dataset per-spectrogram z-score."""
    s = float(img.std())
    if s == 0:
        return img - img.mean()
    return ((img - img.mean()) / s).astype(np.float32)


def preprocess_record(iq: np.ndarray) -> Dict[str, np.ndarray]:
    spec = airhar_micro_doppler(iq)
    return {
        "spectrogram_db_224": spec,
        "spectrogram_zscore": zscore(spec),
    }
