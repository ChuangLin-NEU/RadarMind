#!/usr/bin/env python3
"""Phase 10C: 24/77 velocity-domain calibration + measurement-operator audit.

No new Adapter / deep encoder / CORAL / DANN / contrastive / GAN.
ResNet18 CE only if a pre-registered representation gate passes.
10 GHz is excluded from R2-V (XeThru fc UNRESOLVED).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

C_MPS = 299792458.0
FC_24 = 24.75e9
FC_77 = (77.1799e9 + 77.9474e9) / 2.0  # datToImage_77.m:8-10
RADARS = ["24GHz", "77GHz"]
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
SEED = 42
NT = 224
NV = 224
EPS = 1e-8
# Pre-registered gate. Not tuned on TEST.
GATE_RID_DROP = 0.10
GATE_SEP_GAIN = 0.05
GATE_GEO_GAIN = 0.10
HAR_LR = 1e-3
HAR_BATCH = 16
HAR_EPOCHS = 20
HAR_PATIENCE = 6

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


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


def vel_from_fd(fd, fc: float) -> np.ndarray:
    return (C_MPS * np.asarray(fd, np.float64)) / (2.0 * float(fc))


def interp_axis(x_src, x_dst, arr, axis: int) -> np.ndarray:
    idx = np.interp(x_dst, x_src, np.arange(x_src.size, dtype=np.float64))
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.clip(i0 + 1, 0, x_src.size - 1)
    w = (idx - i0).astype(np.float64)
    if axis == 1:
        return arr[:, i0] * (1.0 - w) + arr[:, i1] * w
    return arr[i0, :] * (1.0 - w)[:, None] + arr[i1, :] * w[:, None]


def resample_grid(db, t, y, tau_grid, y_grid) -> np.ndarray:
    t = np.asarray(t, np.float64)
    y = np.asarray(y, np.float64)
    db = np.asarray(db, np.float64)
    if t[-1] <= t[0]:
        raise ValueError("non-increasing time")
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


def pca_probe(Xtr, ytr, Xte, yte, n=32):
    sc = StandardScaler()
    Ztr = sc.fit_transform(Xtr)
    Zte = sc.transform(Xte)
    k = min(n, Ztr.shape[0] - 1, Ztr.shape[1])
    pca = PCA(n_components=k, random_state=SEED)
    return probe(pca.fit_transform(Ztr), ytr, pca.transform(Zte), yte)


def cosine_sim(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def corrcoef(a, b):
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def load_native(path: Path) -> dict:
    z = np.load(path, allow_pickle=True)
    meta = {}
    if "meta_json" in z.files:
        raw = z["meta_json"]
        meta = json.loads(str(raw.item() if getattr(raw, "shape", ()) == () else raw))
    fd = np.asarray(z["doppler_axis_hz"], np.float64)
    fc = float(meta.get("fc_hz") or (FC_24 if "24" in str(path) else FC_77))
    if "velocity_axis_mps" in z.files:
        v = np.asarray(z["velocity_axis_mps"], np.float64)
    else:
        v = vel_from_fd(fd, fc)
    return {
        "db": np.asarray(z["spectrogram_db"], np.float32),
        "t": np.asarray(z["time_axis_sec"], np.float64),
        "fd": fd,
        "v": v,
        "fc": fc,
        "prf": float(meta.get("prf_hz") or (1000.0 if "24" in str(path) else 3200.0)),
        "meta": meta,
        "native_shape": list(np.asarray(z["spectrogram_db"]).shape),
    }


def invert_energy(x, norm: dict) -> np.ndarray:
    db = np.asarray(x, np.float64) * float(norm["std"]) + float(norm["mean"])
    return np.power(10.0, db / 10.0)


def motion_geometry(X, axis) -> dict:
    """X: (n_axis, n_time) normalized array. axis: physical coordinate."""
    W = invert_energy(X, {"mean": 0.0, "std": 1.0})
    # invert_energy with dummy norm treats X as already-dB-like; use shifted positive mass instead
    W = np.maximum(np.asarray(X, np.float64) - np.min(X), 0.0) + EPS
    axis = np.asarray(axis, np.float64)
    pos = axis >= 0
    neg = axis < 0
    mass_t = W.sum(axis=0)
    cen = (axis[:, None] * W).sum(axis=0) / mass_t
    spread = np.sqrt(((axis[:, None] - cen[None, :]) ** 2 * W).sum(axis=0) / mass_t)
    def env(mask):
        if not np.any(mask):
            return np.zeros(W.shape[1], np.float64)
        ww = W[mask, :]
        aa = axis[mask]
        # energy-weighted extreme: 90th percentile by mass along |axis|
        out = np.zeros(ww.shape[1], np.float64)
        for t in range(ww.shape[1]):
            col = ww[:, t]
            s = col.sum()
            if s <= EPS:
                continue
            order = np.argsort(np.abs(aa))
            c = np.cumsum(col[order])
            k = int(np.searchsorted(c, 0.9 * s))
            k = min(k, order.size - 1)
            out[t] = aa[order[k]]
        return out
    return {
        "pos_env": env(pos),
        "neg_env": env(neg),
        "centroid": cen,
        "spread": spread,
        "energy": mass_t,
    }


def geo_compare(g24, g77) -> dict:
    keys = ["pos_env", "neg_env", "centroid", "spread", "energy"]
    out = {}
    for k in keys:
        a, b = g24[k], g77[k]
        an = (a - a.mean()) / (a.std() + EPS)
        bn = (b - b.mean()) / (b.std() + EPS)
        out[f"{k}_corr"] = corrcoef(a, b)
        out[f"{k}_mae_norm"] = float(np.mean(np.abs(an - bn)))
    out["mean_corr"] = float(np.mean([out[f"{k}_corr"] for k in keys]))
    return out


def write_fc_audit(out: Path) -> dict:
    lines = [
        "# 77 GHz carrier-frequency audit",
        "",
        "本阶段只问：77 GHz 的 `fc` 能否从官方处理链可靠读出。",
        "不把数据集名字「77 GHz」写成 `fc = 77e9`。",
        "AIRHAR 不提供独立载频。raw `.bin` 无文本头。",
        "",
        "## Verdict",
        "",
        "**CONFIRMED_FROM_OFFICIAL_MATLAB**",
        "",
        f"- `fstart = 77.1799e9`",
        f"- `fstop = 77.9474e9`",
        f"- `fc = (fstart + fstop) / 2 = {FC_77:.0f} Hz = {FC_77/1e9:.5f} GHz`",
        "",
        "## Evidence",
        "",
        "| source | location | what it says | usable as fc |",
        "|---|---|---|---|",
        "| Official MATLAB | `third_party/CI4R-MULTI3/datToImage_77.m` lines 8–10 | `fstart=77.1799e9`; `fstop=77.9474e9`; `fc=(fstart+fstop)/2` | **YES** |",
        "| Official MATLAB | same file line 17 | `Bw = fstop - fstart` = 0.7675e9 | bandwidth, not fc |",
        "| Official MATLAB | same file lines 13–20 | `SweepTime=40e-3`, `NPpF=128` → PRF 3200 Hz | timing |",
        "| AIRHAR | `third_party/AIRHAR/datasets/data_prepare_CI4R.py` | `RGB_mD` / `bin2np` 无 `fc`、无 `v=c f_D/(2 fc)` | NO independent fc |",
        "| AIRHAR | `third_party/AIRHAR/datasets/CI4R/spec.json` | 只有网络训练超参 | NO |",
        "| Dataset README | `third_party/CI4R-MULTI3/README.md` line 9 | `TI IWR1443 BOOST (77 GHz FMCW radar with 4 GHz bandwidth)` | label + chip BW, **not** chirp fc |",
        "| Raw `*_Raw_0.bin` | DCA1000 int16 | 无 `fc` 文本头 | NO |",
        "| mmWave Studio `.cfg` | not in this repository | — | UNKNOWN |",
        "",
        "## What is not used",
        "",
        "- `fc = 77e9`（仅因数据集名叫 77 GHz）",
        "- README 的 4 GHz bandwidth（芯片能力；与脚本 chirp `Bw=0.7675 GHz` 不一致，不以 README 覆盖脚本）",
        "- AIRHAR 数值链：它做 Range FFT + Hann STFT，不换算速度",
        "",
        "## 24 GHz (already confirmed, restated)",
        "",
        f"- `fc = {FC_24:.0f} Hz = 24.75 GHz`",
        "- source: raw `.dat` header line 1, `datToImage_Anchortech.m` line 8 `fc = radarData(1)`",
        "- Phase 8B `header_24ghz.json` and native manifest `fc_hz=24750000000`",
        "",
        "## Decision",
        "",
        "R2-V 24↔77：**PROCEED**。",
        "10 GHz 仍不进入本阶段速度域。",
        "",
    ]
    (out / "fc_audit_77ghz.md").write_text("\n".join(lines), encoding="utf-8")
    return {
        "status": "CONFIRMED_FROM_OFFICIAL_MATLAB",
        "fc_24_hz": FC_24,
        "fc_77_hz": FC_77,
        "r2v": "AVAILABLE",
        "evidence": "datToImage_77.m:8-10",
    }


def write_operator_audit(out: Path) -> list:
    rows = [
        {
            "property": "carrier_frequency_hz",
            "24GHz": f"{FC_24:.0f}",
            "77GHz": f"{FC_77:.0f}",
            "comparable": "YES_AFTER_VELOCITY",
            "class": "A_coordinate_scaling",
            "evidence": "24: raw header / Anchortech.m:8; 77: datToImage_77.m:8-10",
            "likely_effect": "same radial v maps to different f_D; this is the H1 test",
        },
        {
            "property": "bandwidth_hz",
            "24GHz": "1.5e9 (header Bw)",
            "77GHz": "0.7675e9 (fstop-fstart); README says 4e9 (chip, not this chirp)",
            "comparable": "NO",
            "class": "C_spatial_measurement",
            "evidence": "header_24ghz.json; datToImage_77.m:8-9,17; README.md:9",
            "likely_effect": "different range resolution and range-bin mixing",
        },
        {
            "property": "PRF_or_sweep_rate_hz",
            "24GHz": "1000 (1/Tsweep, Tsweep=1.0 ms)",
            "77GHz": "3200 (1/(40e-3/128))",
            "comparable": "NO",
            "class": "B_sampling_resolution",
            "evidence": "header_24ghz.json; datToImage_77.m:13-20",
            "likely_effect": "different Nyquist and STFT hop in seconds",
        },
        {
            "property": "native_Doppler_support_hz",
            "24GHz": "[-500, +499.756]",
            "77GHz": "[-1600, +1599.219]",
            "comparable": "NO_IN_HZ_YES_IN_VELOCITY_INTERSECT",
            "class": "A_coordinate_scaling",
            "evidence": "Phase 8B native_representation_manifest / common_grid.json",
            "likely_effect": "Hz crop ±256 discards 77 GHz high-f_D that maps into 24 GHz v support",
        },
        {
            "property": "range_preprocessing",
            "24GHz": "range FFT, keep positive half, single bin 19",
            "77GHz": "MATLAB: range FFT per RX, MTI conv, bin 8; Phase8B AIRHAR: sum bins 23:31",
            "comparable": "NO",
            "class": "C_spatial_measurement",
            "evidence": "Anchortech.m:21-32,52; datToImage_77.m:44-75",
            "likely_effect": "different range gate / multipath mix",
        },
        {
            "property": "RX_channel_strategy",
            "24GHz": "single Ancortek channel in .dat",
            "77GHz": "MATLAB: 4 RX then RX1 for MTI; AIRHAR/Phase8B: I=3 Q=7 only",
            "comparable": "NO",
            "class": "C_spatial_measurement",
            "evidence": "Anchortech.m; datToImage_77.m:7,44-57; AIRHAR bin2np",
            "likely_effect": "beam / SNR / clutter difference",
        },
        {
            "property": "MTI_clutter",
            "24GHz": "Butterworth highpass order4 Wn=0.01",
            "77GHz": "MATLAB conv [1 -2 3 -2 1]; AIRHAR MTI commented out (Phase 8B uses AIRHAR: none)",
            "comparable": "NO",
            "class": "D_signal_conditioning",
            "evidence": "Anchortech.m:27-33; datToImage_77.m:50-57; AIRHAR RGB_mD MTI commented",
            "likely_effect": "stationary / slow-motion residual differs; Phase 8B 77 has no MTI",
        },
        {
            "property": "STFT_window_samples",
            "24GHz": "128 Hamming (myspecgramnew reimplementation)",
            "77GHz": "256 Hann (AIRHAR); MATLAB official 256 Hamming-like",
            "comparable": "NO",
            "class": "B_sampling_resolution",
            "evidence": "Anchortech.m:51; AIRHAR RGB_mD; datToImage_77.m:73",
            "likely_effect": "different temporal smearing",
        },
        {
            "property": "STFT_window_seconds",
            "24GHz": "128/1000 = 0.128 s",
            "77GHz": "256/3200 = 0.080 s",
            "comparable": "NO",
            "class": "B_sampling_resolution",
            "evidence": "window/PRF",
            "likely_effect": "24 GHz integrates longer motion; not removed by v-axis remap",
        },
        {
            "property": "STFT_overlap_and_hop",
            "24GHz": "noverlap=100, hop=28 samples = 0.028 s",
            "77GHz": "AIRHAR overlap=200, hop=56 samples = 0.0175 s; MATLAB noverlap=197 hop=59",
            "comparable": "NO",
            "class": "B_sampling_resolution",
            "evidence": "Anchortech.m:51; AIRHAR RGB_mD; datToImage_77.m:73",
            "likely_effect": "different frame rate of spectrogram columns",
        },
        {
            "property": "STFT_nfft",
            "24GHz": "4096",
            "77GHz": "4096",
            "comparable": "YES_COUNT_NO_HZ",
            "class": "B_sampling_resolution",
            "evidence": "both scripts",
            "likely_effect": "bin width 0.244 Hz vs 0.781 Hz",
        },
        {
            "property": "native_duration_sec",
            "24GHz": "10.0",
            "77GHz": "10.0 (32000 chirps; MATLAB NoF=500 would be 20 s, file is 10 s)",
            "comparable": "YES",
            "class": "B_sampling_resolution",
            "evidence": "header_24ghz.json; Phase 1 bin length",
            "likely_effect": "tau normalization is valid",
        },
        {
            "property": "native_range_resolution_m",
            "24GHz": "c/(2*1.5e9) ≈ 0.100 m",
            "77GHz": "c/(2*0.7675e9) ≈ 0.195 m",
            "comparable": "NO",
            "class": "C_spatial_measurement",
            "evidence": "Bw values above",
            "likely_effect": "range-bin 19 vs summed 23:31 cover different physical depth",
        },
        {
            "property": "native_velocity_resolution_mps",
            "24GHz": f"{C_MPS * (1000.0/4096.0) / (2*FC_24):.6f}",
            "77GHz": f"{C_MPS * (3200.0/4096.0) / (2*FC_77):.6f}",
            "comparable": "CLOSE",
            "class": "B_sampling_resolution",
            "evidence": "v=c*(PRF/nfft)/(2 fc)",
            "likely_effect": "bin width similar in m/s; not the main residual",
        },
        {
            "property": "dB_peak_normalization",
            "24GHz": "20 log10(|s|/max) per recording (Phase 8B mag_db)",
            "77GHz": "same in Phase 8B reconstruct",
            "comparable": "YES_FORMULA",
            "class": "D_signal_conditioning",
            "evidence": "scripts/phase8/stft_helpers.py mag_db",
            "likely_effect": "per-file gain removed; clutter floor still radar-specific",
        },
    ]
    write_csv(
        out / "measurement_operator_24_77.csv",
        ["property", "24GHz", "77GHz", "comparable", "class", "evidence", "likely_effect"],
        rows,
    )
    return rows


def write_stft_audit(out: Path) -> None:
    lines = [
        "# STFT equivalence audit (24 vs 77, Phase 8B numerical chain)",
        "",
        "只审计。不重新生成 R3。",
        "",
        "## Phase 8B chains used for native STFT",
        "",
        "| | 24 GHz | 77 GHz |",
        "|---|---|---|",
        "| source script | `datToImage_Anchortech.m` numerical | AIRHAR `RGB_mD` numerical |",
        "| helper | `myspecgramnew` reimplementation, Hamming | `np.hanning`, AIRHAR loop |",
        "| window (samples) | 128 | 256 |",
        "| window (seconds) | **0.128** | **0.080** |",
        "| overlap (samples) | 100 | 200 |",
        "| hop (samples) | 28 | 56 |",
        "| hop (seconds) | **0.028** | **0.0175** |",
        "| nfft | 4096 | 4096 |",
        "| PRF | 1000 Hz | 3200 Hz |",
        "| Doppler bin (Hz) | 0.244 | 0.781 |",
        "| Doppler bin (m/s) | see operator table | see operator table |",
        "",
        "## Official 77 GHz MATLAB (not the Phase 8B 77 chain)",
        "",
        "`datToImage_77.m:73`: window=256, nfft=4096, noverlap=197, hop=59.",
        "与 AIRHAR hop=56 不同；且 MATLAB 有卷积 MTI，AIRHAR/Phase8B 77 **无 MTI**。",
        "",
        "## Can native time-domain be re-unified?",
        "",
        "**YES, in principle.** Both radars still have slow-time range-gate series before STFT.",
        "A later R3 could fix:",
        "",
        "1. window duration in seconds (same physical time)",
        "2. overlap fraction",
        "3. frequency / velocity bin width",
        "4. window family (Hamming vs Hann)",
        "",
        "本阶段不自动重算。",
        "",
        "## Verdict",
        "",
        "**NOT EQUIVALENT.** Velocity remapping does not equalize window seconds or hop seconds.",
        "If H1 is weak, STFT mismatch is a SUPPORTED remaining operator difference.",
        "",
    ]
    (out / "stft_equivalence_audit.md").write_text("\n".join(lines), encoding="utf-8")


def pair_scores(feat24, feat77, ts_list, rng):
    same_cos, same_corr, diff_cos, diff_corr = [], [], [], []
    for ts in ts_list:
        a, b = feat24[ts], feat77[ts]
        same_cos.append(cosine_sim(a, b))
        same_corr.append(corrcoef(a, b))
    keys = list(ts_list)
    n_diff = max(len(keys), 1)
    for _ in range(n_diff):
        t1, t2 = rng.choice(keys, 2, replace=False)
        a, b = feat24[t1], feat77[t2]
        diff_cos.append(cosine_sim(a, b))
        diff_corr.append(corrcoef(a, b))
    return {
        "same_cosine": float(np.mean(same_cos)),
        "diff_cosine": float(np.mean(diff_cos)),
        "same_corr": float(np.mean(same_corr)),
        "diff_corr": float(np.mean(diff_corr)),
        "sep_cosine": float(np.mean(same_cos) - np.mean(diff_cos)),
        "n_same": len(same_cos),
        "n_diff": len(diff_cos),
    }


def maybe_train_har(out: Path, pack: dict, device) -> list:
    import random

    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, Dataset

    sys.path.insert(0, str(ROOT / "scripts"))
    from models.resnet18_spec import ResNet18Spec  # noqa: E402

    class DS(Dataset):
        def __init__(self, X, y):
            self.X = torch.from_numpy(np.asarray(X, np.float32))
            self.y = torch.from_numpy(np.asarray(y, np.int64))

        def __len__(self):
            return len(self.y)

        def __getitem__(self, i):
            return self.X[i], self.y[i]

    class EmbNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = ResNet18Spec(7, 1)

        def forward(self, x, return_emb=False):
            bb = self.backbone.backbone
            if x.ndim == 3:
                x = x.unsqueeze(1)
            h = bb.conv1(x)
            h = bb.bn1(h)
            h = bb.relu(h)
            h = bb.maxpool(h)
            h = bb.layer1(h)
            h = bb.layer2(h)
            h = bb.layer3(h)
            h = bb.layer4(h)
            h = bb.avgpool(h)
            emb = torch.flatten(h, 1)
            logits = bb.fc(emb)
            if return_emb:
                return logits, emb
            return logits

    def set_seed():
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)

    @torch.no_grad()
    def ev(net, X, y):
        net.eval()
        ps = []
        for i in range(0, len(y), 32):
            xb = torch.from_numpy(X[i : i + 32]).float().to(device)
            ps.append(net(xb).argmax(1).cpu().numpy())
        pred = np.concatenate(ps)
        return {
            "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "pred": pred,
        }

    jobs = [
        ("r2v_24_to_24", "24GHz", "24GHz", "within_radar"),
        ("r2v_77_to_77", "77GHz", "77GHz", "within_radar"),
        ("r2v_24_to_77", "24GHz", "77GHz", "single_source_cross_radar"),
        ("r2v_77_to_24", "77GHz", "24GHz", "single_source_cross_radar"),
    ]
    rows = []
    ckpt = out / "checkpoints_r2v"
    ckpt.mkdir(parents=True, exist_ok=True)
    for run_id, src, tgt, protocol in jobs:
        set_seed()
        tr = (pack["radar"] == src) & (pack["split"] == "train")
        va = (pack["radar"] == src) & (pack["split"] == "val")
        te = (pack["radar"] == tgt) & (pack["split"] == "test")
        dest = ckpt / run_id
        dest.mkdir(exist_ok=True)
        net = EmbNet().to(device)
        opt = torch.optim.AdamW(net.parameters(), lr=HAR_LR)
        crit = nn.CrossEntropyLoss()
        amp = device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda", enabled=amp)
        best, best_state, bad = -1.0, None, 0
        hist = []
        loader = DataLoader(DS(pack["X"][tr], pack["y"][tr]), batch_size=HAR_BATCH, shuffle=True)
        for epoch in range(HAR_EPOCHS):
            net.train()
            losses = []
            for xb, yb in loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=amp):
                    loss = crit(net(xb), yb)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                losses.append(float(loss.detach().cpu()))
            vm = ev(net, pack["X"][va], pack["y"][va])
            hist.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "val_acc": vm["accuracy"]})
            print(run_id, hist[-1], flush=True)
            if vm["accuracy"] > best:
                best = vm["accuracy"]
                best_state = {k: v.detach().cpu().clone() for k, v in net.state_dict().items()}
                torch.save(best_state, dest / "best.pt")
                bad = 0
            else:
                bad += 1
                if bad >= HAR_PATIENCE:
                    break
        net.load_state_dict(best_state)
        net.to(device)
        tm = ev(net, pack["X"][te], pack["y"][te])
        row = {
            "run_id": run_id,
            "protocol": protocol,
            "train_radar": src,
            "test_radar": tgt,
            "method": "source_only_ce",
            "representation": "R2-V",
            "n_classes": 7,
            "seed": SEED,
            "accuracy": tm["accuracy"],
            "macro_f1": tm["macro_f1"],
            "n_train": int(tr.sum()),
            "n_val": int(va.sum()),
            "n_test": int(te.sum()),
            "best_val_acc": float(best),
            "note": "R2-V CE; same HP as Phase 8C; no domain loss",
        }
        (dest / "metrics.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
        with (dest / "history.csv").open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_acc"])
            w.writeheader()
            w.writerows(hist)
        # embedding probes on this run's test+val of both radars
        net.eval()
        with torch.no_grad():
            embs, labs_r, labs_a, spl = [], [], [], []
            for i in range(0, len(pack["y"]), 32):
                xb = torch.from_numpy(pack["X"][i : i + 32]).float().to(device)
                _, e = net(xb, return_emb=True)
                embs.append(e.cpu().numpy())
                labs_r.append(pack["radar"][i : i + 32])
                labs_a.append(pack["y"][i : i + 32])
                spl.append(pack["split"][i : i + 32])
        E = np.concatenate(embs)
        R = np.concatenate(labs_r)
        A = np.concatenate(labs_a)
        S = np.concatenate(spl)
        trm, tem = S == "train", S == "test"
        actp = probe(E[trm], A[trm], E[tem], A[tem])
        ridp = probe(E[trm], R[trm], E[tem], R[tem])
        row["emb_activity"] = actp["accuracy"]
        row["emb_radar_id"] = ridp["accuracy"]
        rows.append(row)
        print("[har]", run_id, row["accuracy"], "rid", ridp["accuracy"], flush=True)
    write_csv(
        out / "training_runs.csv",
        [
            "run_id",
            "protocol",
            "train_radar",
            "test_radar",
            "method",
            "representation",
            "n_classes",
            "seed",
            "accuracy",
            "macro_f1",
            "n_train",
            "n_val",
            "n_test",
            "best_val_acc",
            "emb_activity",
            "emb_radar_id",
            "note",
        ],
        rows,
    )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p8-dir", default=str(ROOT / "results" / "phase8_physical_representation"))
    ap.add_argument("--native-root", default="")
    ap.add_argument("--r2d-root", default="")
    ap.add_argument("--split", default=str(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase10_velocity_calibration"))
    ap.add_argument("--skip-build", action="store_true")
    args = ap.parse_args()

    p8 = Path(args.p8_dir)
    native_root = Path(args.native_root) if args.native_root else p8 / "cache_native"
    r2d_root = Path(args.r2d_root) if args.r2d_root else p8 / "cache_r2"
    out = Path(args.out)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(parents=True, exist_ok=True)

    fc_info = write_fc_audit(out)
    op_rows = write_operator_audit(out)
    write_stft_audit(out)

    split = {str(r["timestamp"]): r for r in read_csv(Path(args.split))}
    man_path = p8 / "native_representation_manifest.csv"
    if not man_path.is_file():
        raise FileNotFoundError(man_path)
    man = [r for r in read_csv(man_path) if r["radar"] in RADARS]
    by = defaultdict(dict)
    for r in man:
        by[str(r["timestamp"])][r["radar"]] = r
    paired = {ts: rec for ts, rec in by.items() if all(x in rec for x in RADARS) and ts in split}
    if len(paired) != 327:
        print("paired", len(paired), "expected 327", flush=True)

    # --- locate native ---
    missing = []
    for ts, rec in paired.items():
        for radar in RADARS:
            p = native_root / radar / f"{ts}.npz"
            if not p.is_file():
                missing.append(str(p))
    if missing:
        (out / "answers.json").write_text(
            json.dumps(
                {
                    "R2_V": "BLOCKED_BY_NATIVE_CACHE",
                    "n_missing": len(missing),
                    "example": missing[:5],
                    "fc": fc_info,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        raise FileNotFoundError(f"native cache missing {len(missing)} files, e.g. {missing[0]}")

    # --- common velocity from physics (TRAIN axes; all files share radar axis) ---
    sample24 = load_native(native_root / "24GHz" / f"{next(iter(paired))}.npz")
    sample77 = load_native(native_root / "77GHz" / f"{next(iter(paired))}.npz")
    if abs(sample24["fc"] - FC_24) > 1e6:
        raise RuntimeError(f"24 GHz fc mismatch {sample24['fc']}")
    if abs(sample77["fc"] - FC_77) > 1e6:
        raise RuntimeError(f"77 GHz fc mismatch {sample77['fc']}")
    v24, v77 = sample24["v"], sample77["v"]
    common_min = float(max(v24.min(), v77.min()))
    common_max = float(min(v24.max(), v77.max()))
    tau_grid = np.linspace(0.0, 1.0, NT, dtype=np.float64)
    v_grid = np.linspace(common_min, common_max, NV, dtype=np.float64)
    grid = {
        "status": "BUILT_R2V_24_77",
        "type": "S(normalized_time, radial_velocity_mps)",
        "Nt": NT,
        "Nv": NV,
        "tau_min": 0.0,
        "tau_max": 1.0,
        "fc_24_hz": FC_24,
        "fc_77_hz": FC_77,
        "c_mps": C_MPS,
        "formula": "v = c * f_D / (2 * fc)",
        "24_doppler_min_hz": float(sample24["fd"].min()),
        "24_doppler_max_hz": float(sample24["fd"].max()),
        "77_doppler_min_hz": float(sample77["fd"].min()),
        "77_doppler_max_hz": float(sample77["fd"].max()),
        "24_velocity_min_mps": float(v24.min()),
        "24_velocity_max_mps": float(v24.max()),
        "77_velocity_min_mps": float(v77.min()),
        "77_velocity_max_mps": float(v77.max()),
        "common_velocity_min_mps": common_min,
        "common_velocity_max_mps": common_max,
        "note": "intersection of native velocity supports. Not tuned on HAR.",
        "excluded": "10GHz XeThru fc UNRESOLVED",
    }
    (out / "common_velocity_grid.json").write_text(json.dumps(grid, indent=2), encoding="utf-8")
    print("common v", common_min, common_max, flush=True)

    cache_v = out / "cache_r2v"
    man_rows = []
    train_stack = []
    train_keys = []
    for ts, rec in paired.items():
        if split[ts]["split"] != "train":
            continue
        for radar in RADARS:
            nav = load_native(native_root / radar / f"{ts}.npz")
            x = resample_grid(nav["db"], nav["t"], nav["v"], tau_grid, v_grid)
            train_stack.append(x)
            train_keys.append((ts, radar))
    stack = np.stack(train_stack, 0)
    clip_lo = float(np.percentile(stack, 1))
    clip_hi = float(np.percentile(stack, 99))
    clipped = np.clip(stack, clip_lo, clip_hi)
    mu = float(clipped.mean())
    sd = float(clipped.std())
    if sd <= 1e-8:
        sd = 1.0
    norm = {"clip_lo": clip_lo, "clip_hi": clip_hi, "mean": mu, "std": sd, "source": "TRAIN_24_77_only"}
    (out / "r2v_normalization.json").write_text(json.dumps(norm, indent=2), encoding="utf-8")

    def to_r2v(nav):
        x = resample_grid(nav["db"], nav["t"], nav["v"], tau_grid, v_grid)
        x = np.clip(x, clip_lo, clip_hi)
        return ((x - mu) / sd).astype(np.float32)

    for ts, rec in paired.items():
        for radar in RADARS:
            dest = cache_v / radar / f"{ts}.npz"
            nav = load_native(native_root / radar / f"{ts}.npz")
            if not (args.skip_build and dest.is_file()):
                x = to_r2v(nav)
                atomic_npz(
                    dest,
                    r2v=x[None, ...],
                    tau=tau_grid.astype(np.float32),
                    velocity_axis_mps=v_grid.astype(np.float32),
                )
            else:
                x = np.asarray(np.load(dest)["r2v"], np.float32)
                if x.ndim == 3:
                    x = x[0]
            src = rec[radar]
            man_rows.append(
                {
                    "timestamp": ts,
                    "action": split[ts]["action"],
                    "split": split[ts]["split"],
                    "radar": radar,
                    "fc_hz": nav["fc"],
                    "prf_hz": nav["prf"],
                    "native_doppler_min_hz": float(nav["fd"].min()),
                    "native_doppler_max_hz": float(nav["fd"].max()),
                    "native_velocity_min_mps": float(nav["v"].min()),
                    "native_velocity_max_mps": float(nav["v"].max()),
                    "common_velocity_min_mps": common_min,
                    "common_velocity_max_mps": common_max,
                    "native_shape": json.dumps(nav["native_shape"]),
                    "resampling": "linear_tau_and_velocity_to_224x224",
                    "r2v_path": str(dest),
                    "source_member": src.get("source_member", ""),
                }
            )
            if len(man_rows) % 40 == 0:
                print(f"r2v {len(man_rows)}/654 {radar} {ts}", flush=True)
    write_csv(
        out / "r2v_manifest.csv",
        [
            "timestamp",
            "action",
            "split",
            "radar",
            "fc_hz",
            "prf_hz",
            "native_doppler_min_hz",
            "native_doppler_max_hz",
            "native_velocity_min_mps",
            "native_velocity_max_mps",
            "common_velocity_min_mps",
            "common_velocity_max_mps",
            "native_shape",
            "resampling",
            "r2v_path",
            "source_member",
        ],
        man_rows,
    )

    # --- load packs ---
    def load_pack(kind: str):
        Xs, ys, splits, tss, acts, rads = [], [], [], [], [], []
        for ts, rec in paired.items():
            for radar in RADARS:
                if kind == "R2-D":
                    z = np.load(r2d_root / radar / f"{ts}.npz")
                    x = np.asarray(z["r2"], np.float32)
                else:
                    z = np.load(cache_v / radar / f"{ts}.npz")
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

    packs = {"R2-D": load_pack("R2-D"), "R2-V": load_pack("R2-V")}
    fd_grid = np.linspace(-256.0, 255.875, 224)
    probe_rows = []
    pair_rows = []
    geo_rows = []
    rng = np.random.RandomState(SEED)

    for name, pack in packs.items():
        P = pool16(pack["X"])
        tr, te = pack["split"] == "train", pack["split"] == "test"
        rid = probe(P[tr], pack["radar"][tr], P[te], pack["radar"][te])
        act = probe(P[tr], pack["y"][tr], P[te], pack["y"][te])
        act_pca = pca_probe(P[tr], pack["y"][tr], P[te], pack["y"][te], 32)
        probe_rows.append(
            {
                "representation": name,
                "n_radars": 2,
                "radar_id": rid["accuracy"],
                "radar_id_macro_f1": rid["macro_f1"],
                "activity": act["accuracy"],
                "activity_macro_f1": act["macro_f1"],
                "activity_pca32": act_pca["accuracy"],
                "n_train": int(tr.sum()),
                "n_test": int(te.sum()),
                "note": "16x16 logistic; 2-class radar; 7-class activity",
            }
        )
        feat24, feat77 = {}, {}
        for i in range(len(pack["y"])):
            ts = str(pack["timestamp"][i])
            if pack["radar"][i] == "24GHz":
                feat24[ts] = P[i]
            else:
                feat77[ts] = P[i]
        for split_name in ("train", "val", "test"):
            ts_list = [str(t) for t in paired if split[str(t)]["split"] == split_name]
            sc = pair_scores(feat24, feat77, ts_list, rng)
            pair_rows.append({"representation": name, "split": split_name, **sc})
        # geometry on TEST pairs
        axis = v_grid if name == "R2-V" else fd_grid
        cors = []
        for ts in [t for t in paired if split[t]["split"] == "test"]:
            i24 = np.where((pack["timestamp"] == ts) & (pack["radar"] == "24GHz"))[0][0]
            i77 = np.where((pack["timestamp"] == ts) & (pack["radar"] == "77GHz"))[0][0]
            g = geo_compare(motion_geometry(pack["X"][i24], axis), motion_geometry(pack["X"][i77], axis))
            g.update({"representation": name, "timestamp": ts, "action": split[ts]["action"]})
            geo_rows.append(g)
            cors.append(g["mean_corr"])
        print(name, "RID", rid["accuracy"], "ACT", act["accuracy"], flush=True)

    write_csv(
        out / "representation_probe.csv",
        [
            "representation",
            "n_radars",
            "radar_id",
            "radar_id_macro_f1",
            "activity",
            "activity_macro_f1",
            "activity_pca32",
            "n_train",
            "n_test",
            "note",
        ],
        probe_rows,
    )
    write_csv(
        out / "pair_alignment_24_77.csv",
        [
            "representation",
            "split",
            "same_cosine",
            "diff_cosine",
            "same_corr",
            "diff_corr",
            "sep_cosine",
            "n_same",
            "n_diff",
        ],
        pair_rows,
    )
    write_csv(
        out / "motion_geometry_alignment.csv",
        [
            "representation",
            "timestamp",
            "action",
            "pos_env_corr",
            "pos_env_mae_norm",
            "neg_env_corr",
            "neg_env_mae_norm",
            "centroid_corr",
            "centroid_mae_norm",
            "spread_corr",
            "spread_mae_norm",
            "energy_corr",
            "energy_mae_norm",
            "mean_corr",
        ],
        geo_rows,
    )

    def _get(rows, rep, key, split_name="test"):
        return next(r for r in rows if r["representation"] == rep and r.get("split", split_name) == split_name)

    p_d = next(r for r in probe_rows if r["representation"] == "R2-D")
    p_v = next(r for r in probe_rows if r["representation"] == "R2-V")
    s_d = _get(pair_rows, "R2-D", "sep", "test")
    s_v = _get(pair_rows, "R2-V", "sep", "test")
    geo_d = float(np.mean([r["mean_corr"] for r in geo_rows if r["representation"] == "R2-D"]))
    geo_v = float(np.mean([r["mean_corr"] for r in geo_rows if r["representation"] == "R2-V"]))
    env_d = float(np.mean([0.5 * (r["pos_env_corr"] + r["neg_env_corr"]) for r in geo_rows if r["representation"] == "R2-D"]))
    env_v = float(np.mean([0.5 * (r["pos_env_corr"] + r["neg_env_corr"]) for r in geo_rows if r["representation"] == "R2-V"]))
    spr_d = float(np.mean([r["spread_corr"] for r in geo_rows if r["representation"] == "R2-D"]))
    spr_v = float(np.mean([r["spread_corr"] for r in geo_rows if r["representation"] == "R2-V"]))

    rid_drop = float(p_d["radar_id"]) - float(p_v["radar_id"])
    sep_gain = float(s_v["sep_cosine"]) - float(s_d["sep_cosine"])
    geo_gain = geo_v - geo_d
    gate_a = rid_drop >= GATE_RID_DROP
    gate_b = sep_gain >= GATE_SEP_GAIN
    gate_c = geo_gain >= GATE_GEO_GAIN
    gate = "PASS" if (gate_a or gate_b or gate_c) else "FAIL"
    print("GATE", gate, "rid_drop", rid_drop, "sep_gain", sep_gain, "geo_gain", geo_gain, flush=True)

    har_rows = []
    if gate == "PASS":
        try:
            import torch

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print("HAR device", device, flush=True)
            har_rows = maybe_train_har(out, packs["R2-V"], device)
        except Exception as exc:
            print("HAR skipped", type(exc).__name__, exc, flush=True)
            (out / "training_runs.csv").write_text(
                "run_id,note\nSKIP,GATE PASS but torch/train failed: " + str(exc) + "\n",
                encoding="utf-8",
            )
    else:
        write_csv(
            out / "training_runs.csv",
            ["run_id", "note"],
            [{"run_id": "NOT_TRAINED", "note": "representation gate FAIL; no ResNet"}],
        )

    # sanity figures + diagnostics
    plt = _mpl()
    # pick 3 train timestamps per action
    picks = []
    for act in CLASS_NAMES:
        got = [ts for ts, rec in paired.items() if split[ts]["action"] == act and split[ts]["split"] == "train"]
        picks.extend(got[:3])
    sanity = []
    for ts in picks:
        z24 = np.load(cache_v / "24GHz" / f"{ts}.npz")["r2v"]
        z77 = np.load(cache_v / "77GHz" / f"{ts}.npz")["r2v"]
        a = z24[0] if z24.ndim == 3 else z24
        b = z77[0] if z77.ndim == 3 else z77
        sanity.append(
            {
                "timestamp": ts,
                "action": split[ts]["action"],
                "pos_mass_24": float(a[v_grid >= 0].mean()),
                "pos_mass_77": float(b[v_grid >= 0].mean()),
                "neg_mass_24": float(a[v_grid < 0].mean()),
                "neg_mass_77": float(b[v_grid < 0].mean()),
                "sign_agree": int(np.sign(a[v_grid >= 0].mean() - a[v_grid < 0].mean()) == np.sign(b[v_grid >= 0].mean() - b[v_grid < 0].mean())),
            }
        )
    write_csv(
        out / "sanity_pairs.csv",
        ["timestamp", "action", "pos_mass_24", "pos_mass_77", "neg_mass_24", "neg_mass_77", "sign_agree"],
        sanity,
    )

    # Fig 1 schematic
    fig1, ax = plt.subplots(figsize=(7.2, 3.6))
    fd = np.linspace(-500, 1600, 400)
    ax.plot(fd, vel_from_fd(fd, FC_24), label="24.75 GHz")
    ax.plot(fd, vel_from_fd(fd, FC_77), label="77.56365 GHz")
    ax.axhline(common_min, color="k", ls="--", lw=0.8)
    ax.axhline(common_max, color="k", ls="--", lw=0.8)
    ax.set_xlabel("Doppler f_D (Hz)")
    ax.set_ylabel("radial v (m/s)")
    ax.set_title("v = c f_D / (2 fc)")
    ax.legend()
    fig1.savefig(fig / "fig1_doppler_to_velocity.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    demo_ts = picks[0] if picks else next(iter(paired))
    # Fig 2 R2-D
    fig2, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, radar in zip(axes, RADARS):
        x = np.load(r2d_root / radar / f"{demo_ts}.npz")["r2"]
        if x.ndim == 3:
            x = x[0]
        ax.imshow(x, origin="lower", aspect="auto", cmap="gray")
        ax.set_title(f"R2-D {radar}")
        ax.set_xlabel("tau")
        ax.set_ylabel("Doppler Hz bins")
    fig2.savefig(fig / "fig2_r2d_pair.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    fig3, axes = plt.subplots(1, 2, figsize=(8.4, 3.4), sharey=True)
    for ax, radar in zip(axes, RADARS):
        x = np.load(cache_v / radar / f"{demo_ts}.npz")["r2v"]
        if x.ndim == 3:
            x = x[0]
        ax.imshow(x, origin="lower", aspect="auto", cmap="gray", extent=[0, 1, common_min, common_max])
        ax.set_title(f"R2-V {radar}")
        ax.set_xlabel("tau")
        ax.set_ylabel("v (m/s)")
    fig3.savefig(fig / "fig3_r2v_pair.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.bar([0, 1], [p_d["radar_id"], p_v["radar_id"]], color=["#4c72b0", "#dd8452"])
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["R2-D 24/77", "R2-V 24/77"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID (2-class)")
    fig4.savefig(fig / "fig4_radar_id.png", dpi=130, bbox_inches="tight")
    plt.close(fig4)

    fig5, ax = plt.subplots(figsize=(6.4, 3.6))
    labs = ["R2-D same", "R2-D diff", "R2-V same", "R2-V diff"]
    ax.bar(labs, [s_d["same_cosine"], s_d["diff_cosine"], s_v["same_cosine"], s_v["diff_cosine"]])
    ax.set_ylabel("cosine similarity")
    ax.tick_params(axis="x", rotation=15)
    fig5.savefig(fig / "fig5_pair_similarity.png", dpi=130, bbox_inches="tight")
    plt.close(fig5)

    fig6, axes = plt.subplots(2, 1, figsize=(7.2, 5.0), sharex=True)
    packv = packs["R2-V"]
    i24 = np.where((packv["timestamp"] == demo_ts) & (packv["radar"] == "24GHz"))[0][0]
    i77 = np.where((packv["timestamp"] == demo_ts) & (packv["radar"] == "77GHz"))[0][0]
    g24 = motion_geometry(packv["X"][i24], v_grid)
    g77 = motion_geometry(packv["X"][i77], v_grid)
    tau = np.linspace(0, 1, NT)
    axes[0].plot(tau, g24["pos_env"], label="24 +v")
    axes[0].plot(tau, g77["pos_env"], label="77 +v")
    axes[0].plot(tau, g24["neg_env"], label="24 -v", ls="--")
    axes[0].plot(tau, g77["neg_env"], label="77 -v", ls="--")
    axes[0].set_ylabel("envelope v")
    axes[0].legend(fontsize=8)
    axes[1].plot(tau, g24["centroid"], label="24 centroid")
    axes[1].plot(tau, g77["centroid"], label="77 centroid")
    axes[1].set_xlabel("tau")
    axes[1].set_ylabel("centroid v")
    axes[1].legend(fontsize=8)
    fig6.savefig(fig / "fig6_velocity_envelope.png", dpi=130, bbox_inches="tight")
    plt.close(fig6)

    if har_rows:
        fig7, ax = plt.subplots(figsize=(6.4, 3.6))
        ax.bar([r["run_id"] for r in har_rows], [float(r["accuracy"]) for r in har_rows])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("TEST acc")
        ax.tick_params(axis="x", rotation=20)
        fig7.savefig(fig / "fig7_har_transfer.png", dpi=130, bbox_inches="tight")
        plt.close(fig7)

    if rid_drop >= 0.20 or sep_gain >= 0.15:
        doppler_contrib = "STRONG"
    elif gate == "PASS":
        doppler_contrib = "MODERATE"
    else:
        doppler_contrib = "WEAK"
    meas_shift = {"STRONG": "WEAK", "MODERATE": "MODERATE", "WEAK": "STRONG"}[doppler_contrib]
    if gate == "FAIL":
        nxt = "UNIFIED STFT / MEASUREMENT MODEL"
    elif float(p_v["radar_id"]) >= 0.90:
        nxt = "UNIFIED STFT / MEASUREMENT MODEL"
    else:
        nxt = "RETURN TO ENCODER"

    # top 3 remaining operator diffs (skip the one H1 just tested if STRONG)
    remaining = [
        r
        for r in op_rows
        if r["comparable"] != "YES" and r["property"] not in ("dB_peak_normalization", "native_duration_sec")
    ]
    if doppler_contrib == "STRONG":
        remaining = [r for r in remaining if r["class"] != "A_coordinate_scaling"]
    # prefer B/C/D with explicit mismatch
    prefer = ["STFT_window_seconds", "MTI_clutter", "range_preprocessing", "RX_channel_strategy", "bandwidth_hz", "PRF_or_sweep_rate_hz"]
    top = []
    for key in prefer:
        hit = next((r for r in remaining if r["property"] == key), None)
        if hit and hit not in top:
            top.append(hit)
        if len(top) == 3:
            break
    while len(top) < 3 and remaining:
        r = remaining.pop(0)
        if r not in top:
            top.append(r)

    har_map = {r["run_id"]: r for r in har_rows}
    answers = {
        "PHYSICS": {
            "fc_24": "24.75 GHz",
            "fc_77": f"{FC_77/1e9:.5f} GHz",
            "evidence": "datToImage_77.m:8-10",
            "R2_V": "AVAILABLE",
            "common_velocity": [common_min, common_max],
        },
        "REPRESENTATION": {
            "R2D_radar_id": p_d["radar_id"],
            "R2V_radar_id": p_v["radar_id"],
            "R2D_activity": p_d["activity"],
            "R2V_activity": p_v["activity"],
        },
        "PAIR_ALIGNMENT": {
            "R2D": {"same": s_d["same_cosine"], "different": s_d["diff_cosine"]},
            "R2V": {"same": s_v["same_cosine"], "different": s_v["diff_cosine"]},
        },
        "MOTION_GEOMETRY": {
            "envelope_corr_r2d": env_d,
            "envelope_corr_r2v": env_v,
            "spread_corr_r2d": spr_d,
            "spread_corr_r2v": spr_v,
        },
        "GATE": gate,
        "gate_bits": {"A_rid_drop": gate_a, "B_sep_gain": gate_b, "C_geo_gain": gate_c},
        "rid_drop": rid_drop,
        "sep_gain": sep_gain,
        "geo_gain": geo_gain,
        "HAR": "TRAINED" if har_rows else "NOT TRAINED",
        "har": {k: har_map[k]["accuracy"] for k in har_map},
        "doppler_scaling": doppler_contrib,
        "measurement_model_shift": meas_shift,
        "next_step": nxt,
        "top_operator": [r["property"] for r in top],
    }
    (out / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    write_report(out, answers, p_d, p_v, s_d, s_v, env_d, env_v, spr_d, spr_v, top, har_rows, grid)
    print("phase10c done", json.dumps(answers, indent=2), flush=True)


def write_report(out, ans, p_d, p_v, s_d, s_v, env_d, env_v, spr_d, spr_v, top, har_rows, grid) -> None:
    lines = [
        "# Phase 10C — Two-radar velocity-domain calibration",
        "",
        "未覆盖 Phase 1–10B。未开新 Adapter / 对抗 / pair loss。10 GHz 不进 R2-V。",
        "",
        "## PHYSICS",
        "",
        "- 24 fc: **24.75 GHz**（raw header / `datToImage_Anchortech.m:8`）",
        f"- 77 fc: **{FC_77/1e9:.5f} GHz** = (77.1799e9+77.9474e9)/2",
        "- evidence: `third_party/CI4R-MULTI3/datToImage_77.m` lines 8–10",
        "- R2-V: **AVAILABLE**",
        f"- 24 velocity: [{grid['24_velocity_min_mps']:.4f}, {grid['24_velocity_max_mps']:.4f}] m/s",
        f"- 77 velocity: [{grid['77_velocity_min_mps']:.4f}, {grid['77_velocity_max_mps']:.4f}] m/s",
        f"- common velocity: [{grid['common_velocity_min_mps']:.4f}, {grid['common_velocity_max_mps']:.4f}] m/s",
        "",
        "共同速度区间只由两路 native support 求交，未按 HAR 调。",
        "",
        "## REPRESENTATION（24/77 两分类 Radar-ID，不要和三分类 0.993 横比）",
        "",
        f"- R2-D 24/77 Radar-ID: **{p_d['radar_id']:.3f}**",
        f"- R2-V Radar-ID: **{p_v['radar_id']:.3f}**",
        f"- R2-D Activity: **{p_d['activity']:.3f}**",
        f"- R2-V Activity: **{p_v['activity']:.3f}**",
        f"- R2-D Activity PCA32: {p_d['activity_pca32']:.3f}",
        f"- R2-V Activity PCA32: {p_v['activity_pca32']:.3f}",
        "",
        "## PAIR ALIGNMENT（TEST，24↔77）",
        "",
        f"- R2-D same cosine {s_d['same_cosine']:.3f}  different {s_d['diff_cosine']:.3f}  sep {s_d['sep_cosine']:.3f}",
        f"- R2-V same cosine {s_v['same_cosine']:.3f}  different {s_v['diff_cosine']:.3f}  sep {s_v['sep_cosine']:.3f}",
        f"- R2-D same corr {s_d['same_corr']:.3f}  R2-V same corr {s_v['same_corr']:.3f}",
        "",
        "## MOTION GEOMETRY（TEST 对，24 vs 77）",
        "",
        f"- velocity/Doppler envelope correlation: R2-D {env_d:.3f}  R2-V {env_v:.3f}",
        f"- spread correlation: R2-D {spr_d:.3f}  R2-V {spr_v:.3f}",
        "",
        f"## GATE: **{ans['GATE']}**",
        "",
        f"预注册：A ΔRadar-ID≥{GATE_RID_DROP} 或 B Δ(same−diff)≥{GATE_SEP_GAIN} 或 C Δgeometry≥{GATE_GEO_GAIN}。",
        f"- A rid_drop={ans['rid_drop']:.3f} → {ans['gate_bits']['A_rid_drop']}",
        f"- B sep_gain={ans['sep_gain']:.3f} → {ans['gate_bits']['B_sep_gain']}",
        f"- C geo_gain={ans['geo_gain']:.3f} → {ans['gate_bits']['C_geo_gain']}",
        "",
        f"HAR: **{ans['HAR']}**",
        "",
    ]
    if har_rows:
        lines.append("## HAR（R2-V ResNet18 CE，无 domain loss）")
        lines.append("")
        for r in har_rows:
            lines.append(
                f"- {r['train_radar']}→{r['test_radar']}: acc {float(r['accuracy']):.3f}  "
                f"emb Activity {float(r['emb_activity']):.3f}  emb Radar-ID {float(r['emb_radar_id']):.3f}"
            )
        lines.append("")
        lines.append("R0/R1/R2-D 对照见 Phase 8 `r0_r1_r2_cross_radar.csv`（同一 split）。")
        lines.append("")
    lines += [
        "## MEASUREMENT OPERATOR",
        "",
        "Top remaining differences:",
    ]
    for i, r in enumerate(top, 1):
        lines.append(f"{i}. **{r['property']}** [{r['class']}] {r['24GHz']} vs {r['77GHz']} — {r['likely_effect']} (SUPPORTED)")
    lines += [
        "",
        "## SCIENTIFIC CONCLUSION",
        "",
        f"- Doppler scaling contribution: **{ans['doppler_scaling']}**",
        f"- Measurement-model shift: **{ans['measurement_model_shift']}**",
        f"- NEXT STEP: **{ans['next_step']}**",
        "",
        "H1（跨雷达差主要来自载频 Doppler 缩放）：",
        "若 Radar-ID 明显下降且 pair / geometry 变好则部分成立；否则频率缩放不是主因。",
        "",
        "未自动进入下一阶段。未开新深度模型（除非本文件在 GATE PASS 下训了最小 CE）。",
        "",
    ]
    (out / "phase10_velocity_calibration_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
