"""R2-D motion-factor extraction. No velocity. Bands pre-registered from common grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
GRID = json.loads((ROOT / "results" / "phase8_physical_representation" / "common_grid.json").read_text(encoding="utf-8"))
NORM = json.loads((ROOT / "results" / "phase8_physical_representation" / "r2_normalization.json").read_text(encoding="utf-8"))

# Pre-registered from TRAIN common Doppler support ±256 Hz. Not tuned on test.
FD_MIN = float(GRID["common_fd_min"])
FD_MAX = float(GRID["common_fd_max"])
LOW_ABS_HZ = 32.0
MID_ABS_HZ = 128.0
ONSET_MASS = 0.10
OCCUP_LO = 0.05
OCCUP_HI = 0.95
PEAK_REL = 0.25
ACF_MIN_LAG = 4
EPS = 1e-12

FACTOR_NAMES = [
    "abs_doppler_centroid_hz",
    "doppler_centroid_hz",
    "doppler_std_hz",
    "doppler_bandwidth_hz",
    "fd_q10_hz",
    "fd_q25_hz",
    "fd_q50_hz",
    "fd_q75_hz",
    "fd_q90_hz",
    "pos_doppler_energy_frac",
    "neg_doppler_energy_frac",
    "low_doppler_energy_frac",
    "mid_doppler_energy_frac",
    "high_doppler_energy_frac",
    "temporal_energy_mean",
    "temporal_energy_std",
    "temporal_energy_range",
    "peak_to_mean_ratio",
    "n_temporal_peaks",
    "dominant_temporal_freq",
    "periodicity_score",
    "autocorr_peak",
    "temporal_entropy",
    "spectral_entropy",
    "sparsity",
    "occupied_doppler_frac",
    "peak_doppler_magnitude",
    "n_doppler_peaks",
    "doppler_envelope_roughness",
    "energy_onset_tau",
    "energy_offset_tau",
    "active_duration_frac",
    "early_energy_frac",
    "middle_energy_frac",
    "late_energy_frac",
    "time_centroid",
    "time_spread",
    "time_doppler_covariance",
]


def fd_axis(n: int = 224) -> np.ndarray:
    return np.linspace(FD_MIN, FD_MAX, n, dtype=np.float64)


def tau_axis(n: int = 224) -> np.ndarray:
    return np.linspace(0.0, 1.0, n, dtype=np.float64)


def to_power(s: np.ndarray) -> np.ndarray:
    """Invert TRAIN z-score of clipped dB, then 10^(dB/10). Not velocity."""
    db = np.asarray(s, dtype=np.float64) * float(NORM["std"]) + float(NORM["mean"])
    return np.power(10.0, db / 10.0)


def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 0, None)
    z = p.sum()
    if z <= EPS:
        return 0.0
    p = p / z
    p = np.clip(p, EPS, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def _weighted_quantile(fd: np.ndarray, mass: np.ndarray, q: float) -> float:
    m = np.clip(mass, 0, None)
    z = m.sum()
    if z <= EPS:
        return float(np.median(fd))
    order = np.argsort(fd)
    cdf = np.cumsum(m[order]) / z
    return float(np.interp(q, cdf, fd[order]))


def _n_peaks(y: np.ndarray, rel: float = PEAK_REL) -> int:
    if y.size < 3:
        return 0
    thr = rel * float(np.max(y) + EPS)
    n = 0
    for i in range(1, len(y) - 1):
        if y[i] >= y[i - 1] and y[i] >= y[i + 1] and y[i] >= thr:
            n += 1
    return n


def extract_one(s: np.ndarray, fd: np.ndarray | None = None, tau: np.ndarray | None = None) -> dict:
    x = np.asarray(s, dtype=np.float64)
    if x.ndim == 3:
        x = x[0]
    nf, nt = x.shape
    fd = fd_axis(nf) if fd is None else np.asarray(fd, dtype=np.float64)
    tau = tau_axis(nt) if tau is None else np.asarray(tau, dtype=np.float64)
    w = to_power(x)
    etot = float(w.sum())
    if etot <= EPS:
        etot = EPS
    e_f = w.sum(axis=1)
    e_t = w.sum(axis=0)
    p_f = e_f / etot
    p_t = e_t / etot
    mu_f = float((p_f * fd).sum())
    mu_t = float((p_t * tau).sum())
    var_f = float((p_f * (fd - mu_f) ** 2).sum())
    var_t = float((p_t * (tau - mu_t) ** 2).sum())
    cov = float((w * (fd[:, None] - mu_f) * (tau[None, :] - mu_t)).sum() / etot)
    q10 = _weighted_quantile(fd, e_f, 0.10)
    q25 = _weighted_quantile(fd, e_f, 0.25)
    q50 = _weighted_quantile(fd, e_f, 0.50)
    q75 = _weighted_quantile(fd, e_f, 0.75)
    q90 = _weighted_quantile(fd, e_f, 0.90)
    q05 = _weighted_quantile(fd, e_f, OCCUP_LO)
    q95 = _weighted_quantile(fd, e_f, OCCUP_HI)
    low = np.abs(fd) < LOW_ABS_HZ
    mid = (np.abs(fd) >= LOW_ABS_HZ) & (np.abs(fd) < MID_ABS_HZ)
    high = np.abs(fd) >= MID_ABS_HZ
    et = e_t.copy()
    et_mean = float(et.mean())
    spec = np.abs(np.fft.rfft(et - et_mean))
    spec[0] = 0.0
    k = int(np.argmax(spec)) if spec.size else 0
    spec_sum = float(spec.sum())
    e0 = et - et_mean
    sd = float(e0.std()) + EPS
    zn = e0 / sd
    acf = np.correlate(zn, zn, mode="full")[nt - 1 :]
    acf = acf / (acf[0] + EPS)
    acf_peak = float(np.max(acf[ACF_MIN_LAG:])) if acf.size > ACF_MIN_LAG else 0.0
    csum = np.cumsum(et) / (et.sum() + EPS)
    onset = float(tau[int(np.searchsorted(csum, ONSET_MASS))])
    offset = float(tau[int(np.searchsorted(csum, 1.0 - ONSET_MASS))])
    if offset < onset:
        onset, offset = 0.0, 1.0
    n1, n2 = nt // 3, 2 * nt // 3
    l1 = float(w[:, :n1].sum() / etot)
    l2 = float(w[:, n1:n2].sum() / etot)
    l3 = float(w[:, n2:].sum() / etot)
    nrm = np.linalg.norm(w.ravel())
    l1n = float(np.abs(w).sum())
    hoyer = 0.0
    if nrm > EPS:
        n = w.size
        hoyer = float((np.sqrt(n) - l1n / nrm) / (np.sqrt(n) - 1.0 + EPS))
    return {
        "abs_doppler_centroid_hz": abs(mu_f),
        "doppler_centroid_hz": mu_f,
        "doppler_std_hz": float(np.sqrt(max(var_f, 0.0))),
        "doppler_bandwidth_hz": q95 - q05,
        "fd_q10_hz": q10,
        "fd_q25_hz": q25,
        "fd_q50_hz": q50,
        "fd_q75_hz": q75,
        "fd_q90_hz": q90,
        "pos_doppler_energy_frac": float(e_f[fd > 0].sum() / etot),
        "neg_doppler_energy_frac": float(e_f[fd < 0].sum() / etot),
        "low_doppler_energy_frac": float(e_f[low].sum() / etot),
        "mid_doppler_energy_frac": float(e_f[mid].sum() / etot),
        "high_doppler_energy_frac": float(e_f[high].sum() / etot),
        "temporal_energy_mean": et_mean,
        "temporal_energy_std": float(et.std()),
        "temporal_energy_range": float(et.max() - et.min()),
        "peak_to_mean_ratio": float(et.max() / (et_mean + EPS)),
        "n_temporal_peaks": float(_n_peaks(et)),
        "dominant_temporal_freq": float(k / nt),
        "periodicity_score": float(spec[k] / (spec_sum + EPS)) if spec_sum > 0 else 0.0,
        "autocorr_peak": acf_peak,
        "temporal_entropy": _entropy(e_t),
        "spectral_entropy": _entropy(e_f),
        "sparsity": hoyer,
        "occupied_doppler_frac": float((q95 - q05) / (FD_MAX - FD_MIN + EPS)),
        "peak_doppler_magnitude": float(e_f.max()),
        "n_doppler_peaks": float(_n_peaks(e_f)),
        "doppler_envelope_roughness": float(np.mean(np.abs(np.diff(e_f))) / (e_f.mean() + EPS)),
        "energy_onset_tau": onset,
        "energy_offset_tau": offset,
        "active_duration_frac": offset - onset,
        "early_energy_frac": l1,
        "middle_energy_frac": l2,
        "late_energy_frac": l3,
        "time_centroid": mu_t,
        "time_spread": float(np.sqrt(max(var_t, 0.0))),
        "time_doppler_covariance": cov,
    }
