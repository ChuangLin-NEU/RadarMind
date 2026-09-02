#!/usr/bin/env python3
"""Raw → numerical STFT. Plotting (figure/imagesc/isodata) is not applied."""
from __future__ import annotations

import struct
from typing import Any, Dict

import numpy as np

from .stft_helpers import (
    C_MPS,
    HELPER_STATUS,
    PREPROCESSING_VERSION,
    butter_highpass_mti,
    doppler_axis_hz,
    fftshift_stft,
    mag_db,
    myspecgramnew,
    oddnumber,
    stft_time_axis_sec,
    velocity_axis_mps,
)

# XeThru_mDopp_bulk.m
XETHRU_FLOATS_PER_FRAME = 182
XETHRU_RANGE_BINS = 91
XETHRU_PRF = 512.0
XETHRU_WINDOW = 150
XETHRU_NFFT = 4096
XETHRU_NOVERLAP = 128
XETHRU_RANGE_BIN_MATLAB = 27  # 1-based; comment: 18 front, 27 corner

# datToImage_Anchortech.m
ANCORTEK_WINDOW = 128
ANCORTEK_NFFT = 4096
ANCORTEK_NOVERLAP = 100
ANCORTEK_RANGE_BIN_MATLAB = 19

# AIRHAR RGB_mD (stop before resize/z-score/PNG)
AIRHAR_NTS = 256
AIRHAR_WINDOW = 256
AIRHAR_NFFT = 4096
AIRHAR_OVERLAP = 200
AIRHAR_RANGE = (23, 31)
AIRHAR_I_LANE = 3
AIRHAR_Q_LANE = 7
AIRHAR_N_LANES = 8
# datToImage_77.m
FC_77 = (77.1799e9 + 77.9474e9) / 2.0
PRF_77 = 1.0 / (40e-3 / 128.0)  # 3200 Hz


def parse_xethru_dat(raw: bytes) -> np.ndarray:
    """Restore I(range,time)+jQ. 182 float/frame = 91 I + 91 Q."""
    rec = 12 + XETHRU_FLOATS_PER_FRAME * 4
    n_full = len(raw) // rec
    if n_full <= 0:
        raise ValueError("XeThru file shorter than one frame")
    frames = []
    off = 0
    for _ in range(n_full):
        if off + rec > len(raw):
            break
        _cid, _info, _length = struct.unpack_from("<III", raw, off)
        off += 12
        data = np.frombuffer(raw, dtype="<f4", count=XETHRU_FLOATS_PER_FRAME, offset=off).copy()
        off += XETHRU_FLOATS_PER_FRAME * 4
        frames.append(data[:XETHRU_RANGE_BINS] + 1j * data[XETHRU_RANGE_BINS:])
    if not frames:
        raise ValueError("no XeThru frames parsed")
    return np.stack(frames, axis=1).astype(np.complex128)


def reconstruct_10ghz(raw: bytes) -> Dict[str, Any]:
    iq = parse_xethru_dat(raw)  # (91, n_slow)
    mti = butter_highpass_mti(iq)
    gate = mti[XETHRU_RANGE_BIN_MATLAB - 1, :]
    shift = XETHRU_WINDOW - XETHRU_NOVERLAP
    sx = myspecgramnew(gate, XETHRU_WINDOW, XETHRU_NFFT, shift)
    sx1 = fftshift_stft(sx)
    db = mag_db(sx1)
    prf = XETHRU_PRF
    t = stft_time_axis_sec(db.shape[1], shift, XETHRU_WINDOW, prf)
    fd = doppler_axis_hz(XETHRU_NFFT, prf)
    return {
        "spectrogram_db": db,
        "spectrogram_complex": sx1.astype(np.complex64),
        "time_axis_sec": t.astype(np.float32),
        "doppler_axis_hz": fd.astype(np.float32),
        "velocity_axis_mps": None,
        "fc_hz": None,
        "prf_hz": prf,
        "n_slow": int(iq.shape[1]),
        "raw_duration_sec": float(iq.shape[1] / prf),
        "range_bins": int(iq.shape[0]),
        "range_gate": XETHRU_RANGE_BIN_MATLAB,
        "window": XETHRU_WINDOW,
        "noverlap": XETHRU_NOVERLAP,
        "nfft": XETHRU_NFFT,
        "clutter": "butterworth_highpass_order4_Wn0.01_causal",
        "stft_helper": HELPER_STATUS,
        "has_physical_time": True,
        "has_doppler_hz": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "chain": "XeThru_mDopp_bulk.m numerical (no isodata, no imagesc)",
    }


def _parse_matlab_complex_lines(lines) -> np.ndarray:
    """Parse MATLAB textscan('%f') complex tokens like 1942+2152i."""
    out = np.empty(len(lines), dtype=np.complex128)
    for i, raw_line in enumerate(lines):
        s = raw_line.strip()
        if not s:
            out[i] = 0.0
            continue
        if s[-1] in "iIjJ":
            s = s[:-1]
        k = max(s.rfind("+"), s.rfind("-"))
        if k <= 0:
            out[i] = float(s)
        else:
            out[i] = float(s[:k]) + 1j * float(s[k:])
    return out


def parse_ancortek_dat(raw: bytes) -> Dict[str, Any]:
    """Read the real file header. Values are not Phase 7 inferences.

    Observed layout (CRLF text):
      fc
      Tsweep_ms
      NTS
      Bw
      I+Qi   (MATLAB complex per line; textscan('%f') reads these as complex)
    """
    text = raw.decode("ascii", errors="ignore")
    lines = text.splitlines()
    if len(lines) < 8:
        raise ValueError(f"Ancortek file too short: n_lines={len(lines)}")
    fc = float(lines[0])
    tsweep_ms = float(lines[1])
    nts = int(round(float(lines[2])))
    bw = float(lines[3])
    data = _parse_matlab_complex_lines(lines[4:])
    if nts <= 0 or data.size < nts:
        raise ValueError(f"bad NTS={nts} data={data.size}")
    nc = int(data.size // nts)
    data = data[: nc * nts]
    data_time = data.reshape((nts, nc), order="F")
    tsweep = tsweep_ms / 1000.0
    return {
        "fc": fc,
        "tsweep_ms": tsweep_ms,
        "tsweep_sec": tsweep,
        "nts": nts,
        "bw": bw,
        "n_sweeps": nc,
        "data_time": data_time,
        "raw_duration_sec": float(nc * tsweep),
        "n_floats_total": int(data.size + 4),
        "iq_format": "matlab_complex_text",
    }


def reconstruct_24ghz(raw: bytes) -> Dict[str, Any]:
    hdr = parse_ancortek_dat(raw)
    data_time = hdr["data_time"]
    nts = hdr["nts"]
    tmp = np.fft.fftshift(np.fft.fft(data_time, axis=0), axes=0)
    data_range = tmp[nts // 2 :, :]
    n_slow = data_range.shape[1]
    ns = oddnumber(n_slow) - 1
    if ns != n_slow:
        data_range = data_range[:, :ns]
        n_slow = ns
    mti = butter_highpass_mti(data_range)
    gate = mti[ANCORTEK_RANGE_BIN_MATLAB - 1, :]
    shift = ANCORTEK_WINDOW - ANCORTEK_NOVERLAP
    sx = myspecgramnew(gate, ANCORTEK_WINDOW, ANCORTEK_NFFT, shift)
    # official: sx1 = flipud(fftshift(sx,1)); we store increasing Doppler (no flip)
    sx1 = fftshift_stft(sx)
    db = mag_db(sx1)
    prf = 1.0 / hdr["tsweep_sec"]
    t = stft_time_axis_sec(db.shape[1], shift, ANCORTEK_WINDOW, prf)
    fd = doppler_axis_hz(ANCORTEK_NFFT, prf)
    v = velocity_axis_mps(fd, hdr["fc"]) if hdr["fc"] > 0 else None
    return {
        "spectrogram_db": db,
        "spectrogram_complex": sx1.astype(np.complex64),
        "time_axis_sec": t.astype(np.float32),
        "doppler_axis_hz": fd.astype(np.float32),
        "velocity_axis_mps": None if v is None else v.astype(np.float32),
        "fc_hz": hdr["fc"],
        "prf_hz": prf,
        "n_slow": int(n_slow),
        "raw_duration_sec": hdr["raw_duration_sec"],
        "nts": hdr["nts"],
        "bw_hz": hdr["bw"],
        "tsweep_sec": hdr["tsweep_sec"],
        "tsweep_ms": hdr["tsweep_ms"],
        "range_gate": ANCORTEK_RANGE_BIN_MATLAB,
        "window": ANCORTEK_WINDOW,
        "noverlap": ANCORTEK_NOVERLAP,
        "nfft": ANCORTEK_NFFT,
        "clutter": "range_fft + butterworth_highpass_order4_Wn0.01_causal",
        "stft_helper": HELPER_STATUS,
        "has_physical_time": True,
        "has_doppler_hz": True,
        "ylim_5p8_ghz": "plot_only_datToImage_Anchortech.m:103",
        "preprocessing_version": PREPROCESSING_VERSION,
        "chain": "datToImage_Anchortech.m numerical (no isodata, no imagesc)",
        "header": {k: hdr[k] for k in ("fc", "tsweep_ms", "tsweep_sec", "nts", "bw", "n_sweeps", "raw_duration_sec")},
    }


def parse_77ghz_bin(raw: bytes) -> np.ndarray:
    """AIRHAR bin2np: int16, 8 lanes, I=3 Q=7, Fortran reshape to (256, n_chirps)."""
    x = np.frombuffer(raw, dtype="<i2")
    if x.size == 0 or x.size % AIRHAR_N_LANES != 0:
        raise ValueError("77 GHz int16 length not divisible by 8")
    lanes = x.reshape((AIRHAR_N_LANES, -1), order="F")
    iq_1d = lanes[AIRHAR_I_LANE, :].astype(np.float64) + 1j * lanes[AIRHAR_Q_LANE, :].astype(np.float64)
    if iq_1d.size % AIRHAR_NTS != 0:
        raise ValueError("selected-lane length not divisible by 256")
    return iq_1d.reshape((AIRHAR_NTS, -1), order="F")


def reconstruct_77ghz(raw: bytes) -> Dict[str, Any]:
    iq = parse_77ghz_bin(raw)
    fft_raw = np.fft.fftshift(np.fft.fft(iq, axis=0), axes=0)
    rp = fft_raw[AIRHAR_NTS // 2 :, :]
    rangedata = np.sum(rp[AIRHAR_RANGE[0] : AIRHAR_RANGE[1], :], axis=0)
    window, nfft, overlap = AIRHAR_WINDOW, AIRHAR_NFFT, AIRHAR_OVERLAP
    shift = window - overlap
    n_win = (len(rangedata) - window - 1) // shift
    if n_win <= 0:
        raise ValueError("not enough chirps for AIRHAR STFT")
    idx = np.arange(n_win)[:, None] * shift + np.arange(window)[None, :]
    frames = rangedata[idx] * np.hanning(window)[None, :]
    spec = np.fft.fft(frames, n=nfft, axis=1).T
    sx1 = fftshift_stft(spec)  # increasing Doppler; do not flipud (that is plotting)
    db = mag_db(sx1)
    prf = PRF_77
    t = stft_time_axis_sec(db.shape[1], shift, window, prf)
    fd = doppler_axis_hz(nfft, prf)
    v = velocity_axis_mps(fd, FC_77)
    return {
        "spectrogram_db": db,
        "spectrogram_complex": sx1.astype(np.complex64),
        "time_axis_sec": t.astype(np.float32),
        "doppler_axis_hz": fd.astype(np.float32),
        "velocity_axis_mps": v.astype(np.float32),
        "fc_hz": FC_77,
        "prf_hz": prf,
        "n_slow": int(iq.shape[1]),
        "raw_duration_sec": float(iq.shape[1] / prf),
        "range_gate": f"sum_{AIRHAR_RANGE[0]}:{AIRHAR_RANGE[1]}",
        "window": window,
        "noverlap": overlap,
        "nfft": nfft,
        "clutter": "none_AIRHAR_MTI_commented_out",
        "stft_helper": "AIRHAR_numpy_hann",
        "has_physical_time": True,
        "has_doppler_hz": True,
        "preprocessing_version": PREPROCESSING_VERSION,
        "chain": "AIRHAR RGB_mD numerical (no resize 224, no z-score, no PNG)",
    }


def reconstruct_radar(radar: str, raw: bytes) -> Dict[str, Any]:
    if radar == "10GHz":
        return reconstruct_10ghz(raw)
    if radar == "24GHz":
        return reconstruct_24ghz(raw)
    if radar == "77GHz":
        return reconstruct_77ghz(raw)
    raise ValueError(radar)
