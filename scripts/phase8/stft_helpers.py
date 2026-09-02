#!/usr/bin/env python3
"""STFT helpers.

myspecgramnew is NOT in the repo, the zip namelist, or MATLAB sources here.
Call site: sx = myspecgramnew(x, window, nfft, shift)
  window = scalar length
  shift  = window - noverlap

Compatible replacement (REIMPLEMENTED_MISSING_HELPER, not OFFICIAL):
  Hamming window (MATLAB specgram default when window is a length)
  n_frames = floor((N - window) / shift)
  column i = fft(hamming * x[i*shift : i*shift+window], n=nfft)

This n_frames rule matches the official 77 GHz sx1 width (~538) when
N ≈ 32000 and window=256, shift=59: floor((32000-256)/59) = 538.
"""
from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter

C_MPS = 299792458.0
HELPER_STATUS = "REIMPLEMENTED_MISSING_HELPER"
PREPROCESSING_VERSION = "phase8b_r2d_v1"


def myspecgramnew(x: np.ndarray, window: int, nfft: int, shift: int) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim != 1:
        x = np.ravel(x)
    n = int(x.size)
    n_frames = (n - int(window)) // int(shift)
    if n_frames <= 0:
        raise ValueError(f"signal too short for STFT: N={n} window={window} shift={shift}")
    win = np.hamming(int(window)).astype(np.float64)
    idx = np.arange(n_frames)[:, None] * int(shift) + np.arange(int(window))[None, :]
    frames = x[idx] * win[None, :]
    return np.fft.fft(frames, n=int(nfft), axis=1).T  # (nfft, n_frames)


def fftshift_stft(sx: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(sx, axes=0)


def doppler_axis_hz(nfft: int, prf_hz: float) -> np.ndarray:
    return np.fft.fftshift(np.fft.fftfreq(int(nfft), d=1.0 / float(prf_hz))).astype(np.float64)


def stft_time_axis_sec(n_frames: int, shift: int, window: int, prf_hz: float) -> np.ndarray:
    centers = (np.arange(n_frames, dtype=np.float64) * shift + 0.5 * window) / float(prf_hz)
    return centers


def mag_db(sx: np.ndarray) -> np.ndarray:
    mag = np.abs(sx)
    peak = float(np.max(mag)) if mag.size else 0.0
    if peak <= 0:
        return np.full(mag.shape, -80.0, dtype=np.float32)
    db = 20.0 * np.log10(np.maximum(mag / peak, 1e-12))
    return db.astype(np.float32)


def butter_highpass_mti(iq_range_time: np.ndarray, order: int = 4, wn: float = 0.01) -> np.ndarray:
    """Causal MTI used by XeThru and Ancortek scripts. iq: (range, time)."""
    b, a = butter(order, wn, btype="high")
    out = np.empty_like(iq_range_time, dtype=np.complex128)
    for k in range(iq_range_time.shape[0]):
        out[k, :] = lfilter(b, a, iq_range_time[k, :])
    return out


def velocity_axis_mps(doppler_hz: np.ndarray, fc_hz: float) -> np.ndarray:
    return (C_MPS * np.asarray(doppler_hz, dtype=np.float64)) / (2.0 * float(fc_hz))


def oddnumber(n: int) -> int:
    """Nearest odd >= n. Used only to reproduce ns = oddnumber(n)-1."""
    n = int(n)
    return n if (n % 2) else n + 1
