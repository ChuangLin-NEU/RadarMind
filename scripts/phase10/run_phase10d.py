#!/usr/bin/env python3
"""Phase 10D: unified measurement pipeline + signal-processing ablation.

24/77 only. No ResNet / Adapter / domain loss. Do not overwrite Phase 1–10C.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE.parent))

from phase8.archive_reader import ArchiveReader  # noqa: E402
from phase8.reconstruct import (  # noqa: E402
    AIRHAR_NTS,
    AIRHAR_RANGE,
    ANCORTEK_RANGE_BIN_MATLAB,
    parse_77ghz_bin,
    parse_ancortek_dat,
)
from phase8.stft_helpers import (  # noqa: E402
    C_MPS,
    butter_highpass_mti,
    doppler_axis_hz,
    fftshift_stft,
    mag_db,
    myspecgramnew,
    oddnumber,
    stft_time_axis_sec,
)

FC_24 = 24.75e9
FC_77 = (77.1799e9 + 77.9474e9) / 2.0
PRF_24 = 1000.0
PRF_77 = 3200.0
RADARS = ["24GHz", "77GHz"]
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
SEED = 42
NT = NV = 224
EPS = 1e-8
# Phase 10C common velocity. Do not retune.
V_LO = -3.0282065868377686
V_HI = 3.0267279148101807
# Pre-registered STFT. Not chosen by classification.
WIN_SEC = 0.100
OVERLAP_FRAC = 0.75
NFFT = 4096
WINDOW_FAMILY = "hamming"
# 24 official Butterworth: order 4, Wn=0.01 of Nyquist at PRF=1000 → 5 Hz.
MTI_ORDER = 4
MTI_FCUT_HZ = 0.01 * (PRF_24 / 2.0)
RID_DROP_CLEAR = 0.10


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def atomic_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".part")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, **arrays)
    os.replace(tmp, path)


def vel_from_fd(fd, fc):
    return (C_MPS * np.asarray(fd, np.float64)) / (2.0 * float(fc))


def interp_axis(x_src, x_dst, arr, axis: int):
    idx = np.interp(x_dst, x_src, np.arange(x_src.size, dtype=np.float64))
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x_src.size - 1)
    w = (idx - i0).astype(np.float64)
    if axis == 1:
        return arr[:, i0] * (1.0 - w) + arr[:, i1] * w
    return arr[i0, :] * (1.0 - w)[:, None] + arr[i1, :] * w[:, None]


def resample_grid(db, t, y, tau_grid, y_grid):
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    db = np.asarray(db, np.float64)
    if y[0] > y[-1]:
        y = y[::-1]
        db = db[::-1, :]
    tau = (t - t[0]) / (t[-1] - t[0])
    tmp = interp_axis(tau, tau_grid, db, axis=1)
    return interp_axis(y, y_grid, tmp, axis=0).astype(np.float32)


def pool16(X):
    X = np.asarray(X, np.float32)
    if X.ndim == 2:
        X = X[None]
    try:
        import torch

        t = torch.from_numpy(X)
        if t.ndim == 3:
            t = t.unsqueeze(1)
        return torch.nn.functional.adaptive_avg_pool2d(t, (16, 16)).flatten(1).numpy()
    except Exception:
        n, h, w = X.shape
        hs, ws = h // 16, w // 16
        out = X[:, : hs * 16, : ws * 16].reshape(n, 16, hs, 16, ws).mean(axis=(2, 4))
        return out.reshape(n, -1)


def probe(Xtr, ytr, Xte, yte):
    if len(set(map(str, ytr))) < 2:
        return {"accuracy": 0.0, "macro_f1": 0.0}
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def cosine_sim(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def corrcoef(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def motion_geometry(X, axis):
    W = np.maximum(np.asarray(X, np.float64) - np.min(X), 0.0) + EPS
    axis = np.asarray(axis, np.float64)
    mass_t = W.sum(axis=0)
    cen = (axis[:, None] * W).sum(axis=0) / mass_t
    spread = np.sqrt(((axis[:, None] - cen[None, :]) ** 2 * W).sum(axis=0) / mass_t)

    def env(mask):
        if not np.any(mask):
            return np.zeros(W.shape[1])
        ww, aa = W[mask, :], axis[mask]
        out = np.zeros(ww.shape[1])
        for t in range(ww.shape[1]):
            col = ww[:, t]
            s = col.sum()
            if s <= EPS:
                continue
            order = np.argsort(np.abs(aa))
            c = np.cumsum(col[order])
            k = min(int(np.searchsorted(c, 0.9 * s)), order.size - 1)
            out[t] = aa[order[k]]
        return out

    return {
        "pos_env": env(axis >= 0),
        "neg_env": env(axis < 0),
        "centroid": cen,
        "spread": spread,
        "energy": mass_t,
    }


def geo_compare(g24, g77):
    keys = ["pos_env", "neg_env", "centroid", "spread", "energy"]
    out = {}
    for k in keys:
        a, b = g24[k], g77[k]
        out[f"{k}_corr"] = corrcoef(a, b)
    out["mean_corr"] = float(np.mean([out[f"{k}_corr"] for k in keys]))
    out["envelope_corr"] = float(0.5 * (out["pos_env_corr"] + out["neg_env_corr"]))
    return out


def butter_mti_1d(x, prf, fcut_hz=MTI_FCUT_HZ, order=MTI_ORDER):
    nyq = float(prf) / 2.0
    wn = float(fcut_hz) / nyq
    if not (0.0 < wn < 1.0):
        raise ValueError(f"normalized cutoff {wn} out of (0,1) at PRF={prf}")
    b, a = butter(order, wn, btype="high")
    return lfilter(b, a, np.asarray(x, np.complex128))


def unified_window_samples(prf):
    n = int(round(WIN_SEC * float(prf)))
    hop = max(1, int(round(n * (1.0 - OVERLAP_FRAC))))
    return n, hop, n / float(prf), hop / float(prf)


def unified_stft(x, prf, fc):
    n_win, hop, dur, hop_s = unified_window_samples(prf)
    sx = myspecgramnew(x, n_win, NFFT, hop)
    sx1 = fftshift_stft(sx)
    db = mag_db(sx1)
    t = stft_time_axis_sec(db.shape[1], hop, n_win, prf)
    fd = doppler_axis_hz(NFFT, prf)
    v = vel_from_fd(fd, fc)
    return {
        "db": db,
        "t": t,
        "v": v,
        "n_win": n_win,
        "hop": hop,
        "duration_s": dur,
        "hop_s": hop_s,
        "noverlap": n_win - hop,
    }


def extract_24(raw: bytes) -> dict:
    hdr = parse_ancortek_dat(raw)
    data_time = hdr["data_time"]
    nts = hdr["nts"]
    tmp = np.fft.fftshift(np.fft.fft(data_time, axis=0), axes=0)
    data_range = tmp[nts // 2 :, :]
    n_slow = data_range.shape[1]
    ns = oddnumber(n_slow) - 1
    if ns != n_slow:
        data_range = data_range[:, :ns]
    x_pre = data_range[ANCORTEK_RANGE_BIN_MATLAB - 1, :]
    mti = butter_highpass_mti(data_range)
    x_off = mti[ANCORTEK_RANGE_BIN_MATLAB - 1, :]
    x_uni = butter_mti_1d(x_pre, PRF_24)
    return {
        "x_pre_mti": np.asarray(x_pre, np.complex64),
        "x_official": np.asarray(x_off, np.complex64),
        "x_unified_mti": np.asarray(x_uni, np.complex64),
        "fc": float(hdr["fc"]),
        "prf": float(1.0 / hdr["tsweep_sec"]),
        "bw": float(hdr["bw"]),
        "nts": int(nts),
        "n_range": int(data_range.shape[0]),
        "n_slow": int(data_range.shape[1]),
        "signal_kind": "complex",
        "range_gate": f"matlab_bin_{ANCORTEK_RANGE_BIN_MATLAB}",
    }


def extract_77(raw: bytes) -> dict:
    iq = parse_77ghz_bin(raw)
    fft_raw = np.fft.fftshift(np.fft.fft(iq, axis=0), axes=0)
    rp = fft_raw[AIRHAR_NTS // 2 :, :]
    x_pre = np.sum(rp[AIRHAR_RANGE[0] : AIRHAR_RANGE[1], :], axis=0)
    x_uni = butter_mti_1d(x_pre, PRF_77)
    return {
        "x_pre_mti": np.asarray(x_pre, np.complex64),
        "x_official": np.asarray(x_pre, np.complex64),
        "x_unified_mti": np.asarray(x_uni, np.complex64),
        "fc": FC_77,
        "prf": PRF_77,
        "bw": 77.9474e9 - 77.1799e9,
        "nts": AIRHAR_NTS,
        "n_range": int(rp.shape[0]),
        "n_slow": int(rp.shape[1]),
        "signal_kind": "complex",
        "range_gate": f"airhar_sum_{AIRHAR_RANGE[0]}:{AIRHAR_RANGE[1]}_coherent",
    }


def write_static_docs(out: Path) -> None:
    n24, hop24, d24, h24 = unified_window_samples(PRF_24)
    n77, hop77, d77, h77 = unified_window_samples(PRF_77)
    wn24 = MTI_FCUT_HZ / (PRF_24 / 2.0)
    wn77 = MTI_FCUT_HZ / (PRF_77 / 2.0)
    rows = [
        ["carrier_frequency", "24.75e9 Hz (raw header)", f"{FC_77:.0f} Hz = (77.1799e9+77.9474e9)/2", "different", "yes_velocity_scale", "no", "header; datToImage_77.m:8-10"],
        ["bandwidth", "1.5e9 Hz (header)", "0.7675e9 Hz (fstop-fstart). README 4e9 is chip, not this chirp", "different", "range_resolution", "no", "header_24ghz.json; datToImage_77.m:8-9,17"],
        ["PRF_sweep_rate", "1000 Hz = 1/Tsweep", "3200 Hz = 1/(40e-3/128)", "different", "nyquist_and_stft_hop", "no", "header; datToImage_77.m:13-20"],
        ["range_FFT", "fftshift(fft) along NTS, keep nts/2 positive incl. DC", "same pattern, NTS=256, keep 128", "same_formula", "range_profile", "already_same", "Anchortech.m:24-25; AIRHAR RGB_mD"],
        ["range_resolution", "c/(2*1.5e9) ≈ 0.0999 m", "c/(2*0.7675e9) ≈ 0.1954 m", "different", "spatial_cell", "no", "FMCW formula + Bw"],
        ["range_axis", "script has no rng_axis", "datToImage_77.m:72 rng_axis=([1:NTS]-1)*c/2/Bw on full 256; AIRHAR does not use it", "not_both_script_mapped", "gate_meters", "no", "24: no axis; 77 axis not applied to AIRHAR rp rows"],
        ["range_selection", "single MATLAB bin 19", "AIRHAR sum bins 23:31 (0-based, half profile); official MATLAB uses bin 8", "different", "which_range_cells", "only_if_physical_axis", "Anchortech.m:52; AIRHAR RGB_mD; datToImage_77.m:75"],
        ["clutter_removal", "Butterworth highpass after range FFT", "Phase 8B / AIRHAR: none. Official MATLAB conv [1 -2 3 -2 1]", "different", "stationary_reject", "yes_if_same_filter", "Anchortech.m:27-33; AIRHAR MTI commented"],
        ["MTI", "butter(4, Wn=0.01, high), causal filter(), per range bin", "Phase 8B: none", "different", "slow_time_highpass", "yes_physical_cutoff", "Anchortech.m:27-33"],
        ["slow_time_signal", "complex range-bin 19 after MTI", "complex coherent sum of bins 23:31, no MTI", "different", "STFT_input", "partial", "reconstruct.py"],
        ["STFT_window_type", "Hamming (myspecgramnew reimplementation)", "Hann (AIRHAR); official MATLAB myspecgramnew", "different", "spectral_leakage", "yes", "stft_helpers.py; AIRHAR RGB_mD"],
        ["STFT_window_samples", "128", "256", "different", "time_smear_samples", "unify_via_seconds", "Anchortech.m:51; AIRHAR"],
        ["STFT_window_duration_s", "0.128", "0.080", "different", "time_smear_seconds", "yes", "window/PRF"],
        ["overlap_samples", "100", "200 (AIRHAR); 197 (MATLAB 77)", "different", "column_rate", "unify_via_fraction", "scripts"],
        ["overlap_fraction", "100/128 = 0.781", "200/256 = 0.781 (AIRHAR)", "same_fraction_native", "hop_policy", "yes", "computed"],
        ["hop_duration_s", "0.028", "0.0175", "different", "temporal_sampling", "yes_via_fraction", "hop/PRF"],
        ["nfft", "4096", "4096", "same", "zero_pad", "yes", "both chains"],
        ["Doppler_resolution_hz", "1000/4096 ≈ 0.244 Hz", "3200/4096 ≈ 0.781 Hz", "different", "fd_bin", "compare_on_velocity", "PRF/nfft"],
        ["velocity_resolution_mps", f"{C_MPS*(PRF_24/NFFT)/(2*FC_24):.6f}", f"{C_MPS*(PRF_77/NFFT)/(2*FC_77):.6f}", "close", "v_bin", "already_close", "v=c*(PRF/nfft)/(2 fc)"],
        ["dynamic_range", "20 log10(|s|/max) per recording", "same in Phase 8B reconstruct", "same_formula", "display_gain", "already_same", "mag_db"],
        ["log_conversion", "20 log10", "20 log10", "same", "dB", "already_same", "mag_db"],
        ["normalization", "Phase 10C: TRAIN clip 1–99 + global z-score", "same procedure in 10C", "same_procedure", "amplitude", "keep_global_TRAIN", "r2v_normalization.json"],
    ]
    write_csv(
        out / "processing_operator_audit.csv",
        ["property", "24GHz", "77GHz", "same_or_different", "physical_relevance", "can_unify", "evidence"],
        [dict(zip(["property", "24GHz", "77GHz", "same_or_different", "physical_relevance", "can_unify", "evidence"], r)) for r in rows],
    )
    (out / "unified_stft_definition.md").write_text(
        "\n".join(
            [
                "# Unified STFT definition (U1)",
                "",
                "参数按两路共同支持的物理时间预注册，不按分类结果选。",
                "",
                f"- window family: **{WINDOW_FAMILY}**（24 官方 MATLAB / Phase 8B 已是 Hamming；77 官方 MATLAB 也是 `myspecgramnew`。AIRHAR 的 Hann 不采用，避免给 24 引入新窗。）",
                f"- target window duration: **{WIN_SEC:.3f} s**",
                f"- 24: {n24} samples, actual duration **{d24:.6f} s**",
                f"- 77: {n77} samples, actual duration **{d77:.6f} s**",
                f"- overlap fraction: **{OVERLAP_FRAC:.2f}**（不统一 sample count）",
                f"- 24 hop: {hop24} samples = **{h24:.6f} s**",
                f"- 77 hop: {hop77} samples = **{h77:.6f} s**",
                f"- nfft: **{NFFT}**",
                "- 比较轴: velocity，不是 FFT bin index",
                f"- common velocity: [{V_LO:.6f}, {V_HI:.6f}] m/s（沿用 Phase 10C，不改）",
                "- 各雷达仍用自己的官方 range selection 和官方 MTI 状态",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "unified_mti_definition.md").write_text(
        "\n".join(
            [
                "# Unified MTI definition (U2)",
                "",
                "只复现 24 GHz 官方 Butterworth，再按物理截止频率移植到 77。",
                "不跑 mean-subtraction / SVD / 其它高通。",
                "",
                "- type: Butterworth highpass, causal `lfilter`（对应 MATLAB `filter`，不是 `filtfilt`）",
                f"- order: **{MTI_ORDER}**",
                "- 24 official: `butter(4, 0.01, 'high')` at PRF=1000",
                f"- physical cutoff: Wn * (PRF/2) = 0.01 * 500 = **{MTI_FCUT_HZ:.4f} Hz**",
                f"- 24 normalized cutoff: **{wn24:.6f}** (PRF=1000, Nyquist=500)",
                f"- 77 normalized cutoff: **{wn77:.6f}** (PRF=3200, Nyquist=1600) so cutoff stays {MTI_FCUT_HZ:.4f} Hz",
                "- apply on official-gate slow-time (complex). LTI: 77 的 bin 求和后再滤，等于先滤再求和。",
                "- 24 U2 从 **pre-MTI** bin 19 用同一 5 Hz 滤波器重算，避免和官方链叠滤两次。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "unified_range_definition.md").write_text(
        "\n".join(
            [
                "# Unified range definition (U3)",
                "",
                "**U3 = NOT AVAILABLE**",
                "",
                "bin index 不能跨雷达当同一物理距离。要统一必须两路都有可核对的 `range_axis_m`。",
                "",
                "- 24 GHz：`datToImage_Anchortech.m` **没有** `rng_axis`。只有 bin 19。",
                "- 77 GHz 官方 MATLAB：有 `rng_axis=([1:NTS]-1)*c/2/Bw`，但是用在全 256 点，且官方门是半剖面 **第 8 行**，不是 AIRHAR 的 23:31。",
                "- Phase 8B / AIRHAR：对半剖面 `rp[23:31]` 复数求和，**没有** 把这些行标成米。",
                "",
                "距离分辨可用 `c/(2 Bw)` 计算，但官方目标距离区间不能在不额外假设 FFT 零距对准的情况下写成可靠米制门。",
                "按规则：不造统一 physical range gate。本阶段停在 U2。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "range_gate_audit.md").write_text(
        "\n".join(
            [
                "# Range gate audit (24 / 77)",
                "",
                "问的是：两套官方/Phase 8B pipeline 各自看哪一段距离。不能把 bin 19 改成 bin 19 跨雷达。",
                "",
                "## 24 GHz",
                "",
                "| 量 | 值 | 证据 |",
                "|---|---|---|",
                "| Bw | 1.5e9 Hz | raw header / `header_24ghz.json` |",
                "| range resolution | c/(2 Bw) ≈ 0.0999 m | FMCW 公式；脚本未写出 `dr` 字符串 |",
                "| 距离轴 | **UNKNOWN（脚本未写）** | `datToImage_Anchortech.m` 无 `rng_axis` |",
                "| 选择 | 半剖面 MATLAB bin **19**（0-based 18） | `myspecgramnew(Data_range_MTI(19,:), ...)` |",
                "| bin 19 的米 | **UNKNOWN** | 没有官方把第 19 行标成米 |",
                "",
                "若额外假设 fftshift 后半剖面第 1 行是 DC=0 m（脚本没写），则 18×0.0999 ≈ 1.80 m。这是推导，不是源码字段。**不算可靠 physical axis。**",
                "",
                "## 77 GHz",
                "",
                "| 量 | 值 | 证据 |",
                "|---|---|---|",
                "| Bw | 0.7675e9 Hz | `datToImage_77.m:8-9,17` |",
                "| range resolution | c/(2 Bw) ≈ 0.1954 m | 同上 |",
                "| 官方 MATLAB 轴 | `([1:256]-1)*c/2/Bw` | `datToImage_77.m:72`，针对 **全 NTS** |",
                "| 官方 MATLAB 门 | 半剖面第 **8** 行 | `myspecgramnew(rngpro(8,:), ...)` |",
                "| Phase 8B / AIRHAR 门 | `rp[23:31]` 复数求和 | `data_prepare_CI4R.py` RGB_mD |",
                "| AIRHAR 23:31 的米 | **UNKNOWN** | AIRHAR 未引用 `rng_axis` |",
                "",
                "官方 MATLAB 和第 8 行、AIRHAR 23:31 不是同一距离策略。本阶段 77 必须跟 Phase 8B/10C 一致，用 AIRHAR 求和门。",
                "",
                "## 是否在看同一人体距离区",
                "",
                "**不能可靠回答。** 24 没有米轴；77 的米轴没有接到 AIRHAR 行号上。",
                "两路官方门的物理重叠：**UNKNOWN**。",
                "",
                "## U3",
                "",
                "**NOT AVAILABLE**。不编 ±Δr。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def load_u0(cache: Path, paired, split):
    Xs, ys, splits, tss, acts, rads = [], [], [], [], [], []
    for ts, rec in paired.items():
        for radar in RADARS:
            z = np.load(cache / radar / f"{ts}.npz")
            x = np.asarray(z["r2v"], np.float32)
            if x.ndim == 3:
                x = x[0]
            Xs.append(x)
            ys.append(CLASS_NAMES.index(split[ts]["action"]))
            splits.append(split[ts]["split"])
            tss.append(ts)
            acts.append(split[ts]["action"])
            rads.append(radar)
    return {
        "X": np.stack(Xs),
        "y": np.asarray(ys, np.int64),
        "split": np.array(splits),
        "timestamp": np.array(tss),
        "action": np.array(acts),
        "radar": np.array(rads),
    }


def build_from_slow(slow_root: Path, paired, split, which: str, out_cache: Path, tau, vgrid):
    train_stack = []
    keys = []
    for ts, rec in paired.items():
        for radar in RADARS:
            z = np.load(slow_root / radar / f"{ts}.npz")
            x = z["x_official"] if which == "U1" else z["x_unified_mti"]
            st = unified_stft(x, float(z["prf"]), float(z["fc"]))
            img = resample_grid(st["db"], st["t"], st["v"], tau, vgrid)
            keys.append((ts, radar, split[ts]["split"], split[ts]["action"]))
            if split[ts]["split"] == "train":
                train_stack.append(img)
    stack = np.stack(train_stack)
    clip_lo = float(np.percentile(stack, 1))
    clip_hi = float(np.percentile(stack, 99))
    clipped = np.clip(stack, clip_lo, clip_hi)
    mu, sd = float(clipped.mean()), float(clipped.std())
    if sd <= 1e-8:
        sd = 1.0
    Xs, ys, splits, tss, acts, rads = [], [], [], [], [], []
    man = []
    i_tr = 0
    for ts, radar, sp, act in keys:
        if sp == "train":
            img = train_stack[i_tr]
            i_tr += 1
        else:
            z = np.load(slow_root / radar / f"{ts}.npz")
            x = z["x_official"] if which == "U1" else z["x_unified_mti"]
            st = unified_stft(x, float(z["prf"]), float(z["fc"]))
            img = resample_grid(st["db"], st["t"], st["v"], tau, vgrid)
        x = ((np.clip(img, clip_lo, clip_hi) - mu) / sd).astype(np.float32)
        dest = out_cache / radar / f"{ts}.npz"
        atomic_npz(dest, r2v=x[None], tau=tau.astype(np.float32), velocity_axis_mps=vgrid.astype(np.float32))
        Xs.append(x)
        ys.append(CLASS_NAMES.index(act))
        splits.append(sp)
        tss.append(ts)
        acts.append(act)
        rads.append(radar)
        man.append({"unit": which, "timestamp": ts, "radar": radar, "split": sp, "action": act, "path": str(dest)})
    return {
        "pack": {
            "X": np.stack(Xs),
            "y": np.asarray(ys, np.int64),
            "split": np.array(splits),
            "timestamp": np.array(tss),
            "action": np.array(acts),
            "radar": np.array(rads),
        },
        "norm": {"clip_lo": clip_lo, "clip_hi": clip_hi, "mean": mu, "std": sd, "source": "TRAIN_24_77_global"},
        "man": man,
    }


def pair_sep(pack):
    P = pool16(pack["X"])
    feat = defaultdict(dict)
    for i in range(len(pack["y"])):
        feat[str(pack["timestamp"][i])][pack["radar"][i]] = P[i]
    ts_te = [t for t in feat if pack["split"][np.where(pack["timestamp"] == t)[0][0]] == "test"]
    same, diff = [], []
    rng = np.random.RandomState(SEED)
    for t in ts_te:
        same.append(cosine_sim(feat[t]["24GHz"], feat[t]["77GHz"]))
    for _ in range(len(ts_te)):
        a, b = rng.choice(ts_te, 2, replace=False)
        diff.append(cosine_sim(feat[a]["24GHz"], feat[b]["77GHz"]))
    same_c = [corrcoef(feat[t]["24GHz"], feat[t]["77GHz"]) for t in ts_te]
    return {
        "same_cosine": float(np.mean(same)),
        "diff_cosine": float(np.mean(diff)),
        "sep_cosine": float(np.mean(same) - np.mean(diff)),
        "same_corr": float(np.mean(same_c)),
        "n_same": len(same),
        "n_diff": len(diff),
    }


def geo_mean(pack, vgrid):
    cors = []
    env = []
    spr = []
    te = pack["split"] == "test"
    ts_te = sorted(set(pack["timestamp"][te]))
    for ts in ts_te:
        i24 = np.where((pack["timestamp"] == ts) & (pack["radar"] == "24GHz"))[0][0]
        i77 = np.where((pack["timestamp"] == ts) & (pack["radar"] == "77GHz"))[0][0]
        g = geo_compare(motion_geometry(pack["X"][i24], vgrid), motion_geometry(pack["X"][i77], vgrid))
        cors.append(g["mean_corr"])
        env.append(g["envelope_corr"])
        spr.append(g["spread_corr"])
    return {"mean_corr": float(np.mean(cors)), "envelope_corr": float(np.mean(env)), "spread_corr": float(np.mean(spr))}


def fingerprint(pack):
    X, y_r, y_a, tr, te = pack["X"], pack["radar"], pack["y"], pack["split"] == "train", pack["split"] == "test"
    gstat, vel, tmp = [], [], []
    for x in X:
        gstat.append([x.mean(), x.std(), np.median(x), np.percentile(x, 10), np.percentile(x, 90), x.max() - x.min()])
        vel.append(x.mean(axis=1))
        tmp.append(x.mean(axis=0))
    gstat, vel, tmp = np.asarray(gstat), np.asarray(vel), np.asarray(tmp)
    P = pool16(X)
    rows = []
    for name, feat in (
        ("global_stats", gstat),
        ("velocity_marginal", vel),
        ("temporal_marginal", tmp),
        ("full_16x16", P),
    ):
        rid = probe(feat[tr], y_r[tr], feat[te], y_r[te])
        act = probe(feat[tr], y_a[tr], feat[te], y_a[te])
        rows.append({"features": name, "radar_id": rid["accuracy"], "activity": act["accuracy"], "n_dim": feat.shape[1]})
    return rows, P


def asymmetry(pack):
    tr = pack["split"] == "train"
    rows = []
    for r in RADARS:
        x = pack["X"][(pack["radar"] == r) & tr]
        vel = x.mean(axis=2)
        tmp = x.mean(axis=1)
        rows.append(
            {
                "radar": r,
                "mean": float(x.mean()),
                "std": float(x.std()),
                "velocity_profile_std": float(vel.std()),
                "temporal_profile_std": float(tmp.std()),
                "temporal_diff_std": float(np.diff(tmp, axis=1).std()),
                "sparsity_abs_gt_0p25": float((np.abs(x) > 0.25).mean()),
            }
        )
    return rows


def diagnose(name, pack, vgrid):
    P = pool16(pack["X"])
    tr, te = pack["split"] == "train", pack["split"] == "test"
    rid = probe(P[tr], pack["radar"][tr], P[te], pack["radar"][te])
    act = probe(P[tr], pack["y"][tr], P[te], pack["y"][te])
    sim = pair_sep(pack)
    geo = geo_mean(pack, vgrid)
    fp, _ = fingerprint(pack)
    print(name, "RID", rid["accuracy"], "ACT", act["accuracy"], "sep", sim["sep_cosine"], "geo", geo["mean_corr"], flush=True)
    return {
        "unit": name,
        "radar_id": rid["accuracy"],
        "radar_id_f1": rid["macro_f1"],
        "activity": act["accuracy"],
        "activity_f1": act["macro_f1"],
        **sim,
        **{f"geo_{k}": v for k, v in geo.items()},
        "fingerprint": fp,
        "asym": asymmetry(pack),
    }


def extract_all(zip_path, mapping, paired, split, slow_root, sanity_dir):
    reader = ArchiveReader(str(zip_path))
    reader.open()
    n_ok = n_skip = 0
    t0 = time.time()
    items = [(ts, radar, mapping[ts][radar]["archive_raw_member"]) for ts in paired for radar in RADARS]
    sanity_ts = []
    for act in CLASS_NAMES:
        for ts in paired:
            if split[ts]["action"] == act:
                sanity_ts.append(ts)
                break
    for i, (ts, radar, member) in enumerate(items, 1):
        dest = slow_root / radar / f"{ts}.npz"
        if dest.is_file():
            n_skip += 1
        else:
            raw = reader.read_member_bytes(member)
            rec = extract_24(raw) if radar == "24GHz" else extract_77(raw)
            atomic_npz(
                dest,
                x_pre_mti=rec["x_pre_mti"],
                x_official=rec["x_official"],
                x_unified_mti=rec["x_unified_mti"],
                fc=np.array(rec["fc"]),
                prf=np.array(rec["prf"]),
            )
            n_ok += 1
            if ts in sanity_ts:
                meta = {k: rec[k] for k in rec if not str(k).startswith("x_")}
                atomic_npz(
                    sanity_dir / f"{radar}_{ts}.npz",
                    x_pre_mti=rec["x_pre_mti"],
                    x_official=rec["x_official"],
                    x_unified_mti=rec["x_unified_mti"],
                    fc=np.array(rec["fc"]),
                    prf=np.array(rec["prf"]),
                    meta_json=np.array(json.dumps(meta, default=str)),
                )
        if i % 10 == 0 or i == len(items):
            print(f"extract {i}/{len(items)} ok={n_ok} skip={n_skip} {time.time()-t0:.0f}s", flush=True)
    reader.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", default=os.environ.get("CI4R_RAW_ZIP", os.environ.get("RADARMIND_CI4R_ZIP", "")))
    ap.add_argument("--p8-dir", default=str(ROOT / "results" / "phase8_physical_representation"))
    ap.add_argument("--u0-cache", default="")
    ap.add_argument("--split", default=str(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase10_unified_measurement"))
    ap.add_argument("--skip-extract", action="store_true")
    args = ap.parse_args()

    p8 = Path(args.p8_dir)
    out = Path(args.out)
    fig = out / "figures"
    slow = out / "cache_slowtime"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(exist_ok=True)
    (out / "sanity_slowtime").mkdir(exist_ok=True)
    write_static_docs(out)

    split = {str(r["timestamp"]): r for r in read_csv(Path(args.split))}
    mapping_rows = read_csv(p8 / "raw_mapping.csv")
    mapping = defaultdict(dict)
    for r in mapping_rows:
        if r["radar"] in RADARS:
            mapping[str(r["timestamp"])][r["radar"]] = r
    paired = {ts: rec for ts, rec in mapping.items() if all(x in rec for x in RADARS) and ts in split}
    if len(paired) != 327:
        print("paired", len(paired), flush=True)

    u0_cache = Path(args.u0_cache) if args.u0_cache else ROOT / "results" / "phase10_velocity_calibration" / "cache_r2v"
    alt_env = os.environ.get("RADARMIND_R2V_CACHE", "").strip()
    if not (u0_cache / "24GHz").is_dir() and alt_env:
        alt = Path(alt_env)
        if alt.is_dir():
            u0_cache = alt
    print("U0 cache", u0_cache, flush=True)

    if not args.skip_extract:
        if not Path(args.zip).is_file():
            raise FileNotFoundError(args.zip)
        extract_all(args.zip, mapping, paired, split, slow, out / "sanity_slowtime")
    missing = [slow / r / f"{ts}.npz" for ts in paired for r in RADARS if not (slow / r / f"{ts}.npz").is_file()]
    if missing:
        raise FileNotFoundError(f"slow-time missing {len(missing)} e.g. {missing[0]}")

    tau = np.linspace(0.0, 1.0, NT)
    vgrid = np.linspace(V_LO, V_HI, NV)
    packs = {"U0": load_u0(u0_cache, paired, split)}
    print("building U1", flush=True)
    u1 = build_from_slow(slow, paired, split, "U1", out / "cache_u1", tau, vgrid)
    packs["U1"] = u1["pack"]
    (out / "u1_normalization.json").write_text(json.dumps(u1["norm"], indent=2), encoding="utf-8")
    print("building U2", flush=True)
    u2 = build_from_slow(slow, paired, split, "U2", out / "cache_u2", tau, vgrid)
    packs["U2"] = u2["pack"]
    (out / "u2_normalization.json").write_text(json.dumps(u2["norm"], indent=2), encoding="utf-8")

    man = [{"unit": "U0", "timestamp": ts, "radar": r, "split": split[ts]["split"], "action": split[ts]["action"], "path": str(u0_cache / r / f"{ts}.npz")} for ts in paired for r in RADARS]
    man += u1["man"] + u2["man"]
    write_csv(out / "representation_manifest.csv", ["unit", "timestamp", "radar", "split", "action", "path"], man)

    diags = {k: diagnose(k, packs[k], vgrid) for k in ("U0", "U1", "U2")}
    rid_rows, act_rows, sim_rows, geo_rows, fp_rows, asy_rows = [], [], [], [], [], []
    for k in ("U0", "U1", "U2"):
        d = diags[k]
        rid_rows.append({"unit": k, "available": "YES", "radar_id": d["radar_id"], "macro_f1": d["radar_id_f1"]})
        act_rows.append({"unit": k, "available": "YES", "activity": d["activity"], "macro_f1": d["activity_f1"]})
        sim_rows.append({"unit": k, "same_cosine": d["same_cosine"], "diff_cosine": d["diff_cosine"], "sep_cosine": d["sep_cosine"], "same_corr": d["same_corr"]})
        geo_rows.append({"unit": k, "mean_corr": d["geo_mean_corr"], "envelope_corr": d["geo_envelope_corr"], "spread_corr": d["geo_spread_corr"]})
        for r in d["fingerprint"]:
            fp_rows.append({"unit": k, **r})
        for r in d["asym"]:
            asy_rows.append({"unit": k, **r})
    rid_rows.append({"unit": "U3", "available": "NO", "radar_id": "", "macro_f1": "", "note": "physical range axis not both script-mapped"})
    act_rows.append({"unit": "U3", "available": "NO", "activity": "", "macro_f1": ""})
    write_csv(out / "operator_ablation_radar_id.csv", ["unit", "available", "radar_id", "macro_f1", "note"], rid_rows)
    write_csv(out / "operator_ablation_activity.csv", ["unit", "available", "activity", "macro_f1"], act_rows)
    write_csv(out / "operator_ablation_similarity.csv", ["unit", "same_cosine", "diff_cosine", "sep_cosine", "same_corr"], sim_rows)
    write_csv(out / "operator_ablation_geometry.csv", ["unit", "mean_corr", "envelope_corr", "spread_corr"], geo_rows)
    write_csv(out / "fingerprint_decomposition.csv", ["unit", "features", "radar_id", "activity", "n_dim"], fp_rows)
    write_csv(out / "radar_asymmetry.csv", ["unit", "radar", "mean", "std", "velocity_profile_std", "temporal_profile_std", "temporal_diff_std", "sparsity_abs_gt_0p25"], asy_rows)

    d0, d1, d2 = diags["U0"], diags["U1"], diags["U2"]
    drop01 = d0["radar_id"] - d1["radar_id"]
    drop12 = d1["radar_id"] - d2["radar_id"]
    drop02 = d0["radar_id"] - d2["radar_id"]
    if drop01 >= RID_DROP_CLEAR:
        case, src = "MEAS-A", "STFT"
    elif drop12 >= RID_DROP_CLEAR or (drop02 >= RID_DROP_CLEAR and drop12 > drop01):
        case, src = "MEAS-B", "MTI"
    elif drop02 >= RID_DROP_CLEAR:
        case, src = "MEAS-B", "MTI"
    else:
        case, src = "MEAS-D", "HARDWARE/MEASUREMENT OPERATOR"
    nxt = "HARDWARE-AWARE MEASUREMENT MODEL" if case == "MEAS-D" else "TRAIN UNIFIED REPRESENTATION"
    if case == "MEAS-D" and max(d0["radar_id"], d1["radar_id"], d2["radar_id"]) >= 0.90:
        nxt = "HARDWARE-AWARE MEASUREMENT MODEL"

    # figures
    plt = _mpl()
    demo = next((ts for ts in paired if split[ts]["action"] == "Towards" and split[ts]["split"] == "train"), next(iter(paired)))

    fig1, axes = plt.subplots(1, 2, figsize=(8.8, 3.6))
    axes[0].text(0.05, 0.5, "24 GHz\nrange FFT → bin 19\nButterworth MTI\nSTFT 0.128 s Hamming", va="center", fontsize=10)
    axes[1].text(0.05, 0.5, "77 GHz\nrange FFT → sum 23:31\nno MTI\nSTFT 0.080 s Hann", va="center", fontsize=10)
    for ax, t in zip(axes, ("24 GHz native", "77 GHz native")):
        ax.set_title(t)
        ax.axis("off")
    fig1.savefig(fig / "fig1_pipelines.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    def show_pair(pack, path, title):
        figp, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
        for ax, radar in zip(axes, RADARS):
            i = np.where((pack["timestamp"] == demo) & (pack["radar"] == radar))[0][0]
            ax.imshow(pack["X"][i], origin="lower", aspect="auto", cmap="gray", extent=[0, 1, V_LO, V_HI])
            ax.set_title(f"{title} {radar}")
            ax.set_xlabel("tau")
            ax.set_ylabel("v (m/s)")
        figp.savefig(path, dpi=130, bbox_inches="tight")
        plt.close(figp)

    show_pair(packs["U0"], fig / "fig2_u0_pair.png", "U0")
    show_pair(packs["U1"], fig / "fig3_u1_pair.png", "U1")
    show_pair(packs["U2"], fig / "fig4_u2_pair.png", "U2")
    fig5, ax = plt.subplots(figsize=(5.2, 2.4))
    ax.text(0.5, 0.5, "U3 NOT AVAILABLE\nphysical range axis not both mapped", ha="center", va="center")
    ax.axis("off")
    fig5.savefig(fig / "fig5_u3_unavailable.png", dpi=130, bbox_inches="tight")
    plt.close(fig5)

    xs = ["U0", "U1", "U2"]
    fig6, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(xs, [diags[k]["radar_id"] for k in xs])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID")
    fig6.savefig(fig / "fig6_radar_id.png", dpi=130, bbox_inches="tight")
    plt.close(fig6)

    fig7, ax = plt.subplots(figsize=(5.2, 4.4))
    for k in xs:
        ax.scatter(diags[k]["radar_id"], diags[k]["activity"], s=40)
        ax.annotate(k, (diags[k]["radar_id"], diags[k]["activity"]))
    ax.set_xlabel("Radar-ID")
    ax.set_ylabel("Activity")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig7.savefig(fig / "fig7_activity_vs_rid.png", dpi=130, bbox_inches="tight")
    plt.close(fig7)

    fig8, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(xs, [diags[k]["sep_cosine"] for k in xs])
    ax.set_ylabel("same − different cosine")
    fig8.savefig(fig / "fig8_pair_sep.png", dpi=130, bbox_inches="tight")
    plt.close(fig8)

    fig9, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(xs, [diags[k]["geo_envelope_corr"] for k in xs])
    ax.set_ylabel("envelope correlation")
    fig9.savefig(fig / "fig9_geometry.png", dpi=130, bbox_inches="tight")
    plt.close(fig9)

    fig10, ax = plt.subplots(figsize=(7.2, 3.6))
    feats = ["global_stats", "velocity_marginal", "temporal_marginal", "full_16x16"]
    w = 0.25
    for i, k in enumerate(xs):
        vals = [next(r["radar_id"] for r in diags[k]["fingerprint"] if r["features"] == f) for f in feats]
        ax.bar(np.arange(4) + i * w, vals, w, label=k)
    ax.set_xticks(np.arange(4) + w)
    ax.set_xticklabels(feats, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_ylabel("Radar-ID")
    fig10.savefig(fig / "fig10_fingerprint.png", dpi=130, bbox_inches="tight")
    plt.close(fig10)

    answers = {
        "U0": {k: d0[k] for k in ("radar_id", "activity", "sep_cosine", "geo_mean_corr", "geo_envelope_corr")},
        "U1": {k: d1[k] for k in ("radar_id", "activity", "sep_cosine", "geo_mean_corr", "geo_envelope_corr")},
        "U2": {k: d2[k] for k in ("radar_id", "activity", "sep_cosine", "geo_mean_corr", "geo_envelope_corr")},
        "U3": "NOT AVAILABLE",
        "delta_U1_vs_U0_rid": drop01,
        "delta_U2_vs_U1_rid": drop12,
        "case": case,
        "main_source": src,
        "next_step": nxt,
        "har_trained": False,
    }
    (out / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    write_report(out, answers, d0, d1, d2)
    print("phase10d done", json.dumps(answers, indent=2), flush=True)


def write_report(out, ans, d0, d1, d2):
    lines = [
        "# Phase 10D — Unified measurement pipeline",
        "",
        "未覆盖 Phase 1–10C。未训 ResNet。10 GHz 不参与。",
        "normalization：每个 U 自己的 TRAIN 24+77 全局 clip+z-score。不是 per-radar。",
        "",
        "## OPERATOR AUDIT",
        "",
        "- STFT: 24 Hamming 0.128 s hop 0.028 s；77 Hann 0.080 s hop 0.0175 s",
        "- MTI: 24 Butterworth order4 Wn=0.01 (5 Hz)；77 Phase 8B 无",
        "- Range: 24 bin 19；77 AIRHAR sum 23:31。物理米轴不能两边都从脚本对上。",
        "",
        "## U0 native R2-V",
        "",
        f"- Radar-ID **{d0['radar_id']:.3f}**  Activity **{d0['activity']:.3f}**",
        f"- Pair sep **{d0['sep_cosine']:.3f}**  Geometry **{d0['geo_mean_corr']:.3f}**",
        "",
        "## U1 UNIFIED STFT",
        "",
        "available: **YES**",
        f"- Radar-ID **{d1['radar_id']:.3f}**  Activity **{d1['activity']:.3f}**",
        f"- Pair sep **{d1['sep_cosine']:.3f}**  Geometry **{d1['geo_mean_corr']:.3f}**",
        f"- Δ Radar-ID vs U0: **{ans['delta_U1_vs_U0_rid']:.3f}**",
        "",
        "## U2 + UNIFIED MTI",
        "",
        "available: **YES**（24 官方 Butterworth，77 用同一 5 Hz 物理截止）",
        f"- Radar-ID **{d2['radar_id']:.3f}**  Activity **{d2['activity']:.3f}**",
        f"- Pair sep **{d2['sep_cosine']:.3f}**  Geometry **{d2['geo_mean_corr']:.3f}**",
        f"- Δ Radar-ID vs U1: **{ans['delta_U2_vs_U1_rid']:.3f}**",
        "",
        "## U3 + UNIFIED RANGE",
        "",
        "available: **NO**  physical range axis: **NO**",
        "",
        f"## CASE **{ans['case']}**",
        f"MAIN DOMAIN-SHIFT SOURCE: **{ans['main_source']}**",
        f"NEXT STEP: **{ans['next_step']}**",
        "",
        "未训 ResNet。未自动进入 Phase 11。",
        "",
    ]
    (out / "phase10_unified_measurement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
