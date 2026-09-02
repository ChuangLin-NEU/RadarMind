#!/usr/bin/env python3
"""Phase 10F: motion-conditioned measurement response + geometry feasibility.

U1 only. No action labels in calibration. No ResNet. No Phase 11.
Not geometry-aware unless metadata exists (it does not).
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.cross_decomposition import PLSRegression
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, r2_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RADARS = ["24GHz", "77GHz"]
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
SEED = 42
EPS = 1e-8
V_LO = -3.0282065868377686
V_HI = 3.0267279148101807
ENERGY_Q = 20.0  # same as Phase 10E
LOW_V = 1.0
MID_V = 2.0
ACF_MIN_LAG = 4
# Gate: B1 vs B0. Not chosen by Radar-ID.
GATE_MAE_RATIO = 0.90
GATE_CORR_GAIN = 0.10
ALPHAS = [0.1, 1.0, 10.0, 100.0, 1000.0]
PLS_K = [2, 4, 6, 8]

Q_NAMES = [
    "abs_velocity_centroid",
    "velocity_spread",
    "low_v_energy_frac",
    "mid_v_energy_frac",
    "high_v_energy_frac",
    "pos_velocity_frac",
    "neg_velocity_frac",
    "spectral_entropy",
    "sparsity",
    "temporal_energy_mean",
    "temporal_energy_std",
    "temporal_energy_range",
    "dominant_temporal_freq",
    "periodicity_score",
    "autocorr_peak",
    "active_duration_frac",
    "velocity_envelope_roughness",
]


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
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na <= EPS or nb <= EPS:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def corrcoef(a, b):
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    if a.size < 3 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
    return float(np.corrcoef(a, b)[0, 1])


def _entropy(p):
    p = np.clip(p, 0, None)
    z = p.sum()
    if z <= EPS:
        return 0.0
    p = np.clip(p / z, EPS, 1)
    return float(-(p * np.log(p)).sum() / np.log(len(p)))


def extract_q(s, v, norm):
    """Phase 10A formulas on velocity axis. U1 invert."""
    x = np.asarray(s, np.float64)
    if x.ndim == 3:
        x = x[0]
    db = x * float(norm["std"]) + float(norm["mean"])
    w = np.power(10.0, db / 10.0)
    nf, nt = w.shape
    v = np.asarray(v, np.float64)
    tau = np.linspace(0.0, 1.0, nt)
    etot = float(w.sum()) + EPS
    e_f, e_t = w.sum(1), w.sum(0)
    p_f = e_f / etot
    mu = float((p_f * v).sum())
    var = float((p_f * (v - mu) ** 2).sum())
    low, mid, high = np.abs(v) < LOW_V, (np.abs(v) >= LOW_V) & (np.abs(v) < MID_V), np.abs(v) >= MID_V
    et_mean = float(e_t.mean())
    spec = np.abs(np.fft.rfft(e_t - et_mean))
    spec[0] = 0.0
    k = int(np.argmax(spec)) if spec.size else 0
    e0 = e_t - et_mean
    zn = e0 / (float(e0.std()) + EPS)
    acf = np.correlate(zn, zn, mode="full")[nt - 1 :]
    acf = acf / (acf[0] + EPS)
    acf_peak = float(np.max(acf[ACF_MIN_LAG:])) if acf.size > ACF_MIN_LAG else 0.0
    csum = np.cumsum(e_t) / (e_t.sum() + EPS)
    onset = float(tau[int(np.searchsorted(csum, 0.10))])
    offset = float(tau[int(np.searchsorted(csum, 0.90))])
    if offset < onset:
        onset, offset = 0.0, 1.0
    nrm = np.linalg.norm(w.ravel())
    hoyer = 0.0
    if nrm > EPS:
        n = w.size
        hoyer = float((np.sqrt(n) - np.abs(w).sum() / nrm) / (np.sqrt(n) - 1.0 + EPS))
    return {
        "abs_velocity_centroid": abs(mu),
        "velocity_spread": float(np.sqrt(max(var, 0.0))),
        "low_v_energy_frac": float(e_f[low].sum() / etot),
        "mid_v_energy_frac": float(e_f[mid].sum() / etot),
        "high_v_energy_frac": float(e_f[high].sum() / etot),
        "pos_velocity_frac": float(e_f[v > 0].sum() / etot),
        "neg_velocity_frac": float(e_f[v < 0].sum() / etot),
        "spectral_entropy": _entropy(e_f),
        "sparsity": hoyer,
        "temporal_energy_mean": et_mean,
        "temporal_energy_std": float(e_t.std()),
        "temporal_energy_range": float(e_t.max() - e_t.min()),
        "dominant_temporal_freq": float(k / nt),
        "periodicity_score": float(spec[k] / (spec.sum() + EPS)),
        "autocorr_peak": acf_peak,
        "active_duration_frac": offset - onset,
        "velocity_envelope_roughness": float(np.mean(np.abs(np.diff(e_f))) / (e_f.mean() + EPS)),
    }


def pair_index(pack):
    by = defaultdict(dict)
    for i in range(len(pack["y"])):
        by[str(pack["timestamp"][i])][pack["radar"][i]] = i
    return {ts: rec for ts, rec in by.items() if all(r in rec for r in RADARS)}


def load_u1(cache, split):
    Xs, ys, splits, tss, acts, rads = [], [], [], [], [], []
    for ts, row in split.items():
        for radar in RADARS:
            z = np.load(cache / radar / f"{ts}.npz")
            x = np.asarray(z["r2v"], np.float32)
            if x.ndim == 3:
                x = x[0]
            Xs.append(x)
            ys.append(CLASS_NAMES.index(row["action"]))
            splits.append(row["split"])
            tss.append(ts)
            acts.append(row["action"])
            rads.append(radar)
    return {
        "X": np.stack(Xs),
        "y": np.asarray(ys, np.int64),
        "split": np.array(splits),
        "timestamp": np.array(tss),
        "action": np.array(acts),
        "radar": np.array(rads),
    }


def metrics(y, yhat):
    y, yhat = np.asarray(y, np.float64).ravel(), np.asarray(yhat, np.float64).ravel()
    mae = float(np.mean(np.abs(y - yhat)))
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - ((y - yhat) ** 2).sum() / sst) if sst > EPS else 0.0
    return {"r2": r2, "mae": mae, "corr": corrcoef(y, yhat)}


def region_metrics(y, yhat, v_valid):
    out = {}
    for name, m in (
        ("low", np.abs(v_valid) < LOW_V),
        ("mid", (np.abs(v_valid) >= LOW_V) & (np.abs(v_valid) < MID_V)),
        ("high", np.abs(v_valid) >= MID_V),
    ):
        if m.sum() == 0:
            out[name] = {"r2": "", "mae": "", "corr": ""}
            continue
        out[name] = metrics(y[:, m], yhat[:, m])
    return out


def fit_predict(Qtr, Ytr, Qva, Qte, Yva):
    sc = StandardScaler()
    Ztr, Zva, Zte = sc.fit_transform(Qtr), sc.transform(Qva), sc.transform(Qte)
    best, best_sc, best_name = None, -1e18, ""
    for a in ALPHAS:
        mdl = Ridge(alpha=a)
        mdl.fit(Ztr, Ytr)
        pred = mdl.predict(Zva)
        scv = metrics(Yva, pred)
        score = -scv["mae"] + scv["corr"]
        if score > best_sc:
            best, best_sc, best_name = mdl, score, f"ridge_{a}"
    for k in PLS_K:
        kk = min(k, Ztr.shape[0] - 1, Ztr.shape[1], Ytr.shape[1])
        if kk < 1:
            continue
        mdl = PLSRegression(n_components=kk)
        mdl.fit(Ztr, Ytr)
        scv = metrics(Yva, mdl.predict(Zva))
        score = -scv["mae"] + scv["corr"]
        if score > best_sc:
            best, best_sc, best_name = mdl, score, f"pls_{kk}"
    return best, best_name, sc, best.predict(Zte), best.predict(Zva)


def write_geometry(out: Path) -> None:
    (out / "geometry_feasibility_audit.md").write_text(
        "\n".join(
            [
                "# Geometry feasibility audit",
                "",
                "逐项只标 SUPPORTED / UNRESOLVED / UNAVAILABLE。没有字段就不猜。",
                "",
                "| 量 | 状态 | 证据 |",
                "|---|---|---|",
                "| subject distance | **UNAVAILABLE** | `raw_mapping.csv` / split / 文件名无距离字段 |",
                "| radar-target distance | **UNAVAILABLE** | 24/77 header 只有 fc、Bw、NTS，无逐条距离 |",
                "| azimuth | **UNAVAILABLE** | MATLAB / README / mapping 无该字段 |",
                "| elevation | **UNAVAILABLE** | 同上 |",
                "| body orientation | **UNAVAILABLE** | 无姿态或朝向记录 |",
                "| approach/recede direction | **UNRESOLVED** | Towards/Away 是动作标签，不是测得几何；禁止作 q |",
                "| range trajectory | **UNAVAILABLE** | 未保存逐 chirp 距离轨迹 |",
                "| angle-of-arrival | **UNAVAILABLE** | 无多通道测角处理 |",
                "| antenna geometry | **UNRESOLVED** | README 写 co-located，无基线/阵面坐标 |",
                "| target range bin in meters | **UNAVAILABLE** | 24 脚本无 `rng_axis`；77 AIRHAR `rp[23:31]` 未接到米 |",
                "",
                "查过：`third_party/CI4R-MULTI3/README.md`、`datToImage_Anchortech.m`、`datToImage_77.m`、",
                "`XeThru_mDopp_bulk.m`、`results/phase8_physical_representation/raw_mapping.csv`（timestamp/action/radar/路径）、",
                "`data_splits/ci4r_shared_motion_acquisition_split.csv`（timestamp/action/split）、",
                "Phase 10D `range_gate_audit.md`。文件名形如 `06050010_1573836822.dat`，前缀是被试/动作编码，不是距离或角度。",
                "",
                "README：`All activities were conducted along the radar line-of-sight`。这是采集说明，不是每条 acquisition 的几何量。",
                "文中六名被试有年龄/身高/体重描述，**没有**逐条 subject id 对上 327 timestamp。",
                "",
                "## True geometry-aware model",
                "",
                "**BLOCKED**",
                "",
                "本阶段只称：**motion-conditioned measurement response**。",
                "不能叫 geometry-aware。",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (out / "conditioning_factor_definitions.md").write_text(
        "\n".join(
            [
                "# Conditioning factors q (source-only)",
                "",
                "公式复用 Phase 10A（能量 = 反 z-score 后 10^(dB/10)），轴改为径向速度。",
                f"低/中/高速带预注册：|v|<{LOW_V}、[{LOW_V},{MID_V})、≥{MID_V} m/s。与 10E 低速门 |v|<1 对齐。不按 TEST 调。",
                "",
                "- 只从 **当前 source radar** 的这一条 U1 谱提取",
                "- 不用 action 标签，不用对端雷达",
                "",
                "\n".join(f"- `{n}`" for n in Q_NAMES),
                "",
            ]
        ),
        encoding="utf-8",
    )


def motion_geometry(X, axis):
    W = np.maximum(np.asarray(X, np.float64) - np.min(X), 0.0) + EPS
    axis = np.asarray(axis, np.float64)
    mass = W.sum(axis=0)
    cen = (axis[:, None] * W).sum(axis=0) / mass
    spread = np.sqrt(((axis[:, None] - cen[None, :]) ** 2 * W).sum(axis=0) / mass)

    def env(mask):
        if not np.any(mask):
            return np.zeros(W.shape[1])
        ww, aa = W[mask], axis[mask]
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

    return {"pos_env": env(axis >= 0), "neg_env": env(axis < 0), "spread": spread}


def geo_stats(pack, X, vgrid):
    pairs = pair_index(pack)
    env, spr = [], []
    for ts, rec in pairs.items():
        if pack["split"][rec["24GHz"]] != "test":
            continue
        g24 = motion_geometry(X[rec["24GHz"]], vgrid)
        g77 = motion_geometry(X[rec["77GHz"]], vgrid)
        env.append(0.5 * (corrcoef(g24["pos_env"], g77["pos_env"]) + corrcoef(g24["neg_env"], g77["neg_env"])))
        spr.append(corrcoef(g24["spread"], g77["spread"]))
    return {"envelope_corr": float(np.mean(env)), "spread_corr": float(np.mean(spr))}


def pair_sep(pack, X):
    P = pool16(X)
    feat = defaultdict(dict)
    for i in range(len(pack["y"])):
        feat[str(pack["timestamp"][i])][pack["radar"][i]] = P[i]
    ts_te = [t for t in feat if pack["split"][np.where(pack["timestamp"] == t)[0][0]] == "test"]
    rng = np.random.RandomState(SEED)
    same = [cosine_sim(feat[t]["24GHz"], feat[t]["77GHz"]) for t in ts_te]
    diff = []
    for _ in range(len(ts_te)):
        a, b = rng.choice(ts_te, 2, replace=False)
        diff.append(cosine_sim(feat[a]["24GHz"], feat[b]["77GHz"]))
    return float(np.mean(same) - np.mean(diff))


def fingerprint(X, pack):
    tr, te = pack["split"] == "train", pack["split"] == "test"
    vel = np.stack([x.mean(1) for x in X])
    P = pool16(X)
    return {
        "radar_id": probe(P[tr], pack["radar"][tr], P[te], pack["radar"][te])["accuracy"],
        "vel_rid": probe(vel[tr], pack["radar"][tr], vel[te], pack["radar"][te])["accuracy"],
        "activity": probe(P[tr], pack["y"][tr], P[te], pack["y"][te])["accuracy"],
    }


def ridge_coef(mdl, n_valid, n_q):
    coef = np.asarray(mdl.coef_, np.float64)
    if coef.ndim == 1:
        coef = coef.reshape(1, -1)
    if coef.shape == (n_q, n_valid):
        coef = coef.T
    if coef.shape != (n_valid, n_q):
        return None
    return coef


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u1-cache", default=str(ROOT / "results" / "phase10_unified_measurement" / "cache_u1"))
    ap.add_argument("--u1-norm", default=str(ROOT / "results" / "phase10_unified_measurement" / "u1_normalization.json"))
    ap.add_argument("--split", default=str(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase10_motion_conditioned"))
    args = ap.parse_args()

    out = Path(args.out)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(exist_ok=True)
    write_geometry(out)

    split = {str(r["timestamp"]): r for r in read_csv(Path(args.split))}
    pack = load_u1(Path(args.u1_cache), split)
    norm = json.loads(Path(args.u1_norm).read_text(encoding="utf-8"))
    Xz = pack["X"]
    vgrid = np.linspace(V_LO, V_HI, Xz.shape[1])
    db = Xz * float(norm["std"]) + float(norm["mean"])
    E = np.power(10.0, db / 10.0)
    P = E.mean(axis=2)
    radar, action, tsarr, spl = pack["radar"], pack["action"], pack["timestamp"], pack["split"]
    pairs = pair_index(pack)
    tr = spl == "train"

    Q_rows, Q = [], np.zeros((len(pack["y"]), len(Q_NAMES)))
    for i in range(len(pack["y"])):
        rec = extract_q(Xz[i], vgrid, norm)
        Q[i] = [rec[n] for n in Q_NAMES]
        Q_rows.append(
            {"timestamp": str(tsarr[i]), "radar": radar[i], "split": spl[i], "action": action[i], **{n: rec[n] for n in Q_NAMES}}
        )
    write_csv(out / "conditioning_factors.csv", ["timestamp", "radar", "split", "action"] + Q_NAMES, Q_rows)

    profiles = {r: P[(radar == r) & tr].mean(0) for r in RADARS}
    thr = float(np.percentile(0.5 * (profiles["24GHz"] + profiles["77GHz"]), ENERGY_Q))
    mask = (profiles["24GHz"] > thr) & (profiles["77GHz"] > thr)
    v_valid = vgrid[mask]
    print("mask bins", int(mask.sum()), "thr", thr, flush=True)

    def collect(direction, split_name):
        src, tgt = ("24GHz", "77GHz") if direction == "24to77" else ("77GHz", "24GHz")
        qs, ys, acts, tss = [], [], [], []
        for t, rec in pairs.items():
            if pack["split"][rec["24GHz"]] != split_name:
                continue
            d = np.log(P[rec["77GHz"]] + EPS) - np.log(P[rec["24GHz"]] + EPS)
            if direction == "77to24":
                d = -d
            qs.append(Q[rec[src]])
            ys.append(d[mask])
            acts.append(action[rec["24GHz"]])
            tss.append(t)
        return np.stack(qs), np.stack(ys), np.array(acts), tss

    pred_rows, region_rows, coef_rows = [], [], []
    hats = {}
    gate_ok = True
    for direction in ("24to77", "77to24"):
        Qtr, Ytr, Atr, _ = collect(direction, "train")
        Qva, Yva, Ava, _ = collect(direction, "val")
        Qte, Yte, Ate, Tte = collect(direction, "test")
        D0 = Ytr.mean(0)
        y0_te, y0_va = np.broadcast_to(D0, Yte.shape), np.broadcast_to(D0, Yva.shape)
        b0_te, b0_va = metrics(Yte, y0_te), metrics(Yva, y0_va)
        mdl, mname, scq, y1_te, y1_va = fit_predict(Qtr, Ytr, Qva, Qte, Yva)
        b1_te, b1_va = metrics(Yte, y1_te), metrics(Yva, y1_va)
        hats[direction] = {"Yte": Yte, "Y0": y0_te, "Y1": y1_te, "Ate": Ate, "D0": D0, "model": mdl, "scaler": scq, "name": mname}
        print(direction, mname, "B0", b0_te, "B1", b1_te, flush=True)
        pred_rows.append({"direction": direction, "model": "B0", "split": "test", **b0_te, "val_mae": b0_va["mae"], "val_corr": b0_va["corr"]})
        pred_rows.append({"direction": direction, "model": "B1", "split": "test", **b1_te, "val_mae": b1_va["mae"], "val_corr": b1_va["corr"], "estimator": mname})
        r0, r1 = region_metrics(Yte, y0_te, v_valid), region_metrics(Yte, y1_te, v_valid)
        n_reg = {
            "low": int((np.abs(v_valid) < LOW_V).sum()),
            "mid": int(((np.abs(v_valid) >= LOW_V) & (np.abs(v_valid) < MID_V)).sum()),
            "high": int((np.abs(v_valid) >= MID_V).sum()),
        }
        for reg in ("low", "mid", "high"):
            region_rows.append(
                {
                    "direction": direction,
                    "region": reg,
                    "scope": "energy_mask_10E",
                    "n_bins": n_reg[reg],
                    "B0_r2": r0[reg]["r2"],
                    "B0_mae": r0[reg]["mae"],
                    "B0_corr": r0[reg]["corr"],
                    "B1_r2": r1[reg]["r2"],
                    "B1_mae": r1[reg]["mae"],
                    "B1_corr": r1[reg]["corr"],
                    "note": "" if n_reg[reg] else "no valid bins in 10E energy mask",
                }
            )
        # per-bin R2
        for j, v in enumerate(v_valid):
            pred_rows.append(
                {
                    "direction": direction,
                    "model": "B1_perbin",
                    "velocity_mps": float(v),
                    "r2": float(r2_score(Yte[:, j], y1_te[:, j])) if np.std(Yte[:, j]) > 1e-12 else 0.0,
                    "mae": float(np.mean(np.abs(Yte[:, j] - y1_te[:, j]))),
                    "corr": corrcoef(Yte[:, j], y1_te[:, j]),
                }
            )
        # coefficients if Ridge
        coef = ridge_coef(mdl, Ytr.shape[1], len(Q_NAMES))
        if coef is not None:
            imp = np.mean(np.abs(coef), 0)
            for i, n in enumerate(Q_NAMES):
                coef_rows.append({"direction": direction, "factor": n, "mean_abs_coef": float(imp[i])})
                for reg, m in (("low", np.abs(v_valid) < LOW_V), ("mid", (np.abs(v_valid) >= LOW_V) & (np.abs(v_valid) < MID_V)), ("high", np.abs(v_valid) >= MID_V)):
                    if m.sum():
                        coef_rows.append({"direction": direction, "factor": n, "region": reg, "mean_abs_coef": float(np.mean(np.abs(coef[m, i])))})
        va_ok = (b1_va["mae"] <= GATE_MAE_RATIO * b0_va["mae"]) and (b1_va["corr"] >= b0_va["corr"] + GATE_CORR_GAIN)
        te_ok = (b1_te["mae"] <= GATE_MAE_RATIO * b0_te["mae"]) and (b1_te["corr"] >= b0_te["corr"] + GATE_CORR_GAIN)
        hats[direction]["gate_val"] = va_ok
        hats[direction]["gate_test"] = te_ok
        hats[direction]["b0_te"] = b0_te
        hats[direction]["b1_te"] = b1_te
        if not (va_ok and te_ok):
            gate_ok = False

    write_csv(out / "transfer_predictability.csv", ["direction", "model", "split", "r2", "mae", "corr", "val_mae", "val_corr", "estimator", "velocity_mps"], pred_rows)

    # Diagnostic only: D(v|q) on ALL bins, not used for gate / calibration.
    for direction in ("24to77", "77to24"):
        src = "24GHz" if direction == "24to77" else "77GHz"

        def collect_all(split_name):
            qs, ys = [], []
            for t, rec in pairs.items():
                if pack["split"][rec["24GHz"]] != split_name:
                    continue
                d = np.log(P[rec["77GHz"]] + EPS) - np.log(P[rec["24GHz"]] + EPS)
                if direction == "77to24":
                    d = -d
                qs.append(Q[rec[src]])
                ys.append(d)
            return np.stack(qs), np.stack(ys)

        Qtr, Ytr = collect_all("train")
        Qva, Yva = collect_all("val")
        Qte, Yte = collect_all("test")
        D0 = Ytr.mean(0)
        mdl, mname, scq, y1_te, y1_va = fit_predict(Qtr, Ytr, Qva, Qte, Yva)
        y0_te = np.broadcast_to(D0, Yte.shape)
        r0, r1 = region_metrics(Yte, y0_te, vgrid), region_metrics(Yte, y1_te, vgrid)
        for reg, m in (
            ("low", np.abs(vgrid) < LOW_V),
            ("mid", (np.abs(vgrid) >= LOW_V) & (np.abs(vgrid) < MID_V)),
            ("high", np.abs(vgrid) >= MID_V),
        ):
            region_rows.append(
                {
                    "direction": direction,
                    "region": reg,
                    "scope": "all_bins_diagnostic",
                    "n_bins": int(m.sum()),
                    "B0_r2": r0[reg]["r2"],
                    "B0_mae": r0[reg]["mae"],
                    "B0_corr": r0[reg]["corr"],
                    "B1_r2": r1[reg]["r2"],
                    "B1_mae": r1[reg]["mae"],
                    "B1_corr": r1[reg]["corr"],
                    "note": f"diagnostic {mname}; not used for gate",
                }
            )

    write_csv(
        out / "transfer_region_metrics.csv",
        ["direction", "region", "scope", "n_bins", "B0_r2", "B0_mae", "B0_corr", "B1_r2", "B1_mae", "B1_corr", "note"],
        region_rows,
    )
    if coef_rows:
        write_csv(out / "conditioning_coefficients.csv", ["direction", "factor", "region", "mean_abs_coef"], coef_rows)

    act_rows = []
    for direction, h in hats.items():
        for act in CLASS_NAMES:
            m = h["Ate"] == act
            if m.sum() == 0:
                continue
            e0 = float(np.mean(np.abs(h["Yte"][m] - h["Y0"][m])))
            e1 = float(np.mean(np.abs(h["Yte"][m] - h["Y1"][m])))
            act_rows.append({"direction": direction, "action": act, "n": int(m.sum()), "B0_mae": e0, "B1_mae": e1})
    write_csv(out / "action_conditioned_residual.csv", ["direction", "action", "n", "B0_mae", "B1_mae"], act_rows)

    near = np.abs(vgrid) < 0.1
    bg = np.abs(vgrid) >= MID_V
    lowf_rows = []
    for r in RADARS:
        m = (radar == r) & tr
        er = E[m]
        e_near_t = er[:, near, :].mean(1)
        persist = []
        for row in e_near_t:
            if row.size < 3 or float(row.std()) <= 1e-12:
                continue
            persist.append(float(np.corrcoef(row[:-1], row[1:])[0, 1]))
        pos_f = float(er[:, vgrid > 0, :].sum() / (er.sum() + EPS))
        neg_f = float(er[:, vgrid < 0, :].sum() / (er.sum() + EPS))
        pmean = er.mean(axis=(0, 2))
        lowf_rows.append(
            {
                "radar": r,
                "mean_energy": float(er.mean()),
                "var_energy": float(er.var()),
                "near0_mean_energy": float(er[:, near, :].mean()),
                "near0_var_energy": float(er[:, near, :].var()),
                "near0_energy_frac": float(er[:, near, :].sum() / (er.sum() + EPS)),
                "background_mean_|v|>=2": float(er[:, bg, :].mean()) if bg.any() else "",
                "ridge_over_background": float(er[:, near, :].mean() / (er[:, bg, :].mean() + EPS)) if bg.any() else "",
                "pos_frac": pos_f,
                "neg_frac": neg_f,
                "pos_neg_asymmetry": pos_f - neg_f,
                "temporal_persist_near0": float(np.mean(persist)) if persist else 0.0,
                "spectral_smooth_mean_P": float(np.mean(np.abs(np.diff(pmean)))),
                "near0_smooth_mean_P": float(np.mean(np.abs(np.diff(pmean[near])))) if near.sum() > 1 else "",
            }
        )
    write_csv(out / "low_velocity_forensics.csv", list(lowf_rows[0].keys()), lowf_rows)

    mask_rows = []
    P16 = pool16(Xz)
    trm, tem = spl == "train", spl == "test"
    base_rid = probe(P16[trm], radar[trm], P16[tem], radar[tem])["accuracy"]
    base_act = probe(P16[trm], pack["y"][trm], P16[tem], pack["y"][tem])["accuracy"]
    mask_rows.append({"mask": "none", "mode": "baseline", "radar_id": base_rid, "activity": base_act})
    for thr_v in (0.10, 0.25, 0.50):
        Xrm = Xz.copy()
        Xrm[:, np.abs(vgrid) < thr_v, :] = 0.0
        Prm = pool16(Xrm)
        Xkp = Xz.copy()
        Xkp[:, np.abs(vgrid) >= thr_v, :] = 0.0
        Pkp = pool16(Xkp)
        mask_rows.append(
            {
                "mask": f"remove |v|<{thr_v}",
                "mode": "zero_out_near_zero",
                "radar_id": probe(Prm[trm], radar[trm], Prm[tem], radar[tem])["accuracy"],
                "activity": probe(Prm[trm], pack["y"][trm], Prm[tem], pack["y"][tem])["accuracy"],
            }
        )
        mask_rows.append(
            {
                "mask": f"keep only |v|<{thr_v}",
                "mode": "keep_near_zero",
                "radar_id": probe(Pkp[trm], radar[trm], Pkp[tem], radar[tem])["accuracy"],
                "activity": probe(Pkp[trm], pack["y"][trm], Pkp[tem], pack["y"][tem])["accuracy"],
            }
        )
    write_csv(out / "velocity_mask_diagnostic.csv", ["mask", "mode", "radar_id", "activity"], mask_rows)
    print("mask diagnostic", mask_rows, flush=True)

    cal = None
    if gate_ok:
        E2 = E.copy()
        for t, rec in pairs.items():
            d24 = np.zeros(len(vgrid))
            d77 = np.zeros(len(vgrid))
            d24[mask] = hats["24to77"]["model"].predict(sc_or(hats["24to77"], Q[rec["24GHz"]]))[0]
            d77[mask] = hats["77to24"]["model"].predict(sc_or(hats["77to24"], Q[rec["77GHz"]]))[0]
            E2[rec["24GHz"]] = E[rec["24GHz"]] * np.exp(0.5 * d24)[:, None]
            E2[rec["77GHz"]] = E[rec["77GHz"]] * np.exp(0.5 * d77)[:, None]
        db2 = 10.0 * np.log10(np.maximum(E2, EPS))
        clip_lo, clip_hi = np.percentile(db2[tr], [1, 99])
        c = np.clip(db2, clip_lo, clip_hi)
        mu, sd = float(c[tr].mean()), float(c[tr].std()) + EPS
        Xc = ((c - mu) / sd).astype(np.float32)
        before = fingerprint(Xz, pack)
        after = fingerprint(Xc, pack)
        sep0, sep1 = pair_sep(pack, Xz), pair_sep(pack, Xc)
        geo0, geo1 = geo_stats(pack, Xz, vgrid), geo_stats(pack, Xc, vgrid)
        cal = {"before": before, "after": after, "sep_before": sep0, "sep_after": sep1, "geo_before": geo0, "geo_after": geo1}
        write_csv(
            out / "conditional_calibration_results.csv",
            ["item", "before", "after"],
            [
                {"item": "radar_id", "before": before["radar_id"], "after": after["radar_id"]},
                {"item": "vel_rid", "before": before["vel_rid"], "after": after["vel_rid"]},
                {"item": "activity", "before": before["activity"], "after": after["activity"]},
                {"item": "pair_sep", "before": sep0, "after": sep1},
                {"item": "motion_geometry_env", "before": geo0["envelope_corr"], "after": geo1["envelope_corr"]},
                {"item": "note", "before": "paired known-radar calibration", "after": "not unseen-radar DG"},
            ],
        )
        print("calibration", cal, flush=True)
    else:
        write_csv(
            out / "conditional_calibration_results.csv",
            ["item", "value"],
            [{"item": "status", "value": "NOT_RUN"}, {"item": "reason", "value": "predictability gate FAIL"}],
        )

    # importance ranking: mean |coef| over valid bins, average the two directions
    imp = defaultdict(list)
    imp_dir = defaultdict(lambda: defaultdict(float))
    for r in coef_rows:
        if r.get("region"):
            continue
        imp[r["factor"]].append(r["mean_abs_coef"])
        imp_dir[r["direction"]][r["factor"]] = r["mean_abs_coef"]
    top = sorted(((k, float(np.mean(v))) for k, v in imp.items()), key=lambda kv: -kv[1])[:5]
    top_24 = sorted(imp_dir["24to77"].items(), key=lambda kv: -kv[1])[:3]
    top_77 = sorted(imp_dir["77to24"].items(), key=lambda kv: -kv[1])[:3]

    # figures
    plt = _mpl()
    fig1, ax = plt.subplots(figsize=(7.0, 3.4))
    ax.plot(v_valid, hats["24to77"]["D0"], color="k", lw=2.0, label="B0 fixed mean D(v)")
    y1 = hats["24to77"]["Y1"]
    if len(y1) >= 4:
        idx = np.linspace(0, len(y1) - 1, 4).astype(int)
        for k, i in enumerate(idx):
            ax.plot(v_valid, y1[i], alpha=0.75, label=f"B1 D(v|q) TEST {k+1}")
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("D")
    ax.legend(fontsize=8)
    fig1.savefig(fig / "fig1_fixed_vs_conditional.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(7.0, 3.6))
    ax.plot(v_valid, hats["24to77"]["Yte"].mean(0), label="actual mean D TEST")
    ax.plot(v_valid, hats["24to77"]["Y1"].mean(0), label="B1 pred")
    ax.plot(v_valid, hats["24to77"]["Y0"].mean(0), ls="--", label="B0")
    ax.legend()
    fig2.savefig(fig / "fig2_actual_vs_pred.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(5.6, 3.4))
    labs, mae0, mae1 = [], [], []
    for d in ("24to77", "77to24"):
        labs.append(d)
        mae0.append(hats[d]["b0_te"]["mae"])
        mae1.append(hats[d]["b1_te"]["mae"])
    x = np.arange(2)
    ax.bar(x - 0.18, mae0, 0.36, label="B0")
    ax.bar(x + 0.18, mae1, 0.36, label="B1")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.legend()
    fig3.savefig(fig / "fig3_b0_b1_mae.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(7.2, 3.8))
    if top:
        ax.barh([t[0] for t in top[::-1]], [t[1] for t in top[::-1]])
    fig4.savefig(fig / "fig4_coefficients.png", dpi=130, bbox_inches="tight")
    plt.close(fig4)

    fig5, ax = plt.subplots(figsize=(7.0, 3.4))
    pb = [r for r in pred_rows if r.get("model") == "B1_perbin" and r["direction"] == "24to77"]
    if pb:
        ax.plot([float(r["velocity_mps"]) for r in pb], [float(r["r2"]) for r in pb])
        ax.set_ylabel("per-bin R²")
    fig5.savefig(fig / "fig5_r2_by_v.png", dpi=130, bbox_inches="tight")
    plt.close(fig5)

    fig6, ax = plt.subplots(figsize=(6.8, 3.4))
    regs = [r for r in region_rows if r["direction"] == "24to77" and r.get("scope") == "all_bins_diagnostic"]
    if not regs:
        regs = [r for r in region_rows if r["direction"] == "24to77" and r.get("scope") != "all_bins_diagnostic"]
    ax.bar([r["region"] for r in regs], [float(r["B1_mae"] or 0) for r in regs])
    ax.set_ylabel("B1 MAE")
    fig6.savefig(fig / "fig6_region_err.png", dpi=130, bbox_inches="tight")
    plt.close(fig6)

    fig7, ax = plt.subplots(figsize=(7.0, 3.6))
    ar = [r for r in act_rows if r["direction"] == "24to77"]
    ax.bar([r["action"] for r in ar], [r["B1_mae"] for r in ar])
    ax.tick_params(axis="x", rotation=20)
    fig7.savefig(fig / "fig7_action_resid.png", dpi=130, bbox_inches="tight")
    plt.close(fig7)

    fig8, ax = plt.subplots(figsize=(7.2, 3.6))
    rm = [r for r in mask_rows if r.get("mode") in ("baseline", "zero_out_near_zero")]
    ax.plot([r["mask"] for r in rm], [r["radar_id"] for r in rm], marker="o", label="Radar-ID (remove near-0)")
    ax.plot([r["mask"] for r in rm], [r["activity"] for r in rm], marker="o", label="Activity (remove near-0)")
    ax.tick_params(axis="x", rotation=15)
    ax.legend()
    ax.set_ylim(0, 1.05)
    fig8.savefig(fig / "fig8_velocity_mask.png", dpi=130, bbox_inches="tight")
    plt.close(fig8)

    fig9, ax = plt.subplots(figsize=(7.0, 3.6))
    for r in RADARS:
        ax.plot(vgrid, profiles[r], label=r)
    ax.axvline(-0.1, color="k", ls="--", lw=0.7)
    ax.axvline(0.1, color="k", ls="--", lw=0.7)
    ax.legend()
    fig9.savefig(fig / "fig9_near_zero.png", dpi=130, bbox_inches="tight")
    plt.close(fig9)

    # case
    b1s = [hats[d]["b1_te"]["r2"] for d in hats]
    if gate_ok:
        case, nxt = "MC-A", "CONDITIONAL MEASUREMENT MODEL"
    elif max(b1s) >= 0.10 or any(hats[d]["b1_te"]["mae"] < 0.95 * hats[d]["b0_te"]["mae"] for d in hats):
        case, nxt = "MC-B", "SPATIAL / GEOMETRY DATA NEEDED"
    else:
        case, nxt = "MC-C", "SPATIAL / GEOMETRY DATA NEEDED"

    best_act = min((r for r in act_rows if r["direction"] == "24to77"), key=lambda r: r["B1_mae"], default={"action": ""})
    worst_act = max((r for r in act_rows if r["direction"] == "24to77"), key=lambda r: r["B1_mae"], default={"action": ""})

    answers = {
        "geometry_aware": "BLOCKED",
        "gate": "PASS" if gate_ok else "FAIL",
        "case": case,
        "next_step": nxt,
        "24to77": {"B0": hats["24to77"]["b0_te"], "B1": hats["24to77"]["b1_te"], "estimator": hats["24to77"]["name"]},
        "77to24": {"B0": hats["77to24"]["b0_te"], "B1": hats["77to24"]["b1_te"], "estimator": hats["77to24"]["name"]},
        "regions": {f"{r['direction']}_{r['region']}_{r.get('scope','mask')}": r for r in region_rows},
        "gate_rule": {"mae_ratio": GATE_MAE_RATIO, "corr_gain": GATE_CORR_GAIN},
        "gate_by_direction": {d: {"val": hats[d]["gate_val"], "test": hats[d]["gate_test"]} for d in hats},
        "top_conditioners": [t[0] for t in top[:3]],
        "top_24to77": [t[0] for t in top_24],
        "top_77to24": [t[0] for t in top_77],
        "v_valid_min": float(v_valid.min()) if v_valid.size else None,
        "v_valid_max": float(v_valid.max()) if v_valid.size else None,
        "n_valid_bins": int(mask.sum()),
        "best_action": best_act.get("action"),
        "worst_action": worst_act.get("action"),
        "mask": mask_rows,
        "calibration": cal,
        "har_trained": False,
        "kind": "paired known-radar; not unseen-radar DG; not geometry-aware",
    }
    (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    write_report(out, answers, hats, region_rows, top, mask_rows, best_act, worst_act, case, nxt, gate_ok, cal, top_24, top_77)
    print("phase10f done", json.dumps(answers, indent=2, default=str), flush=True)


def sc_or(h, q):
    return h["scaler"].transform(np.asarray(q, np.float64)[None, :])


def write_report(out, ans, hats, regions, top, masks, best_act, worst_act, case, nxt, gate_ok, cal, top_24=None, top_77=None):
    lines = [
        "# Phase 10F — Motion-conditioned measurement response",
        "",
        "不是 geometry-aware。不是 unseen-radar DG。配对 TRAIN 上学跨雷达响应。未训 ResNet。",
        "",
        "## Geometry",
        "",
        "range / distance / aspect / orientation：**UNAVAILABLE** 或 UNRESOLVED。",
        "True geometry-aware model: **BLOCKED**",
        "",
        "## Predictability",
        "",
    ]
    for d in ("24to77", "77to24"):
        b0, b1 = hats[d]["b0_te"], hats[d]["b1_te"]
        lines.append(f"### {d} ({hats[d]['name']})")
        lines.append(f"- B0: R² {b0['r2']:.3f}  MAE {b0['mae']:.3f}  corr {b0['corr']:.3f}")
        lines.append(f"- B1: R² {b1['r2']:.3f}  MAE {b1['mae']:.3f}  corr {b1['corr']:.3f}")
        lines.append("")
    lines += ["## Velocity regions", ""]
    for r in regions:
        if not r.get("B1_mae") and r.get("B1_mae") != 0:
            lines.append(f"- {r['direction']} {r['region']} ({r.get('scope')}): n={r.get('n_bins')} UNAVAILABLE")
            continue
        lines.append(
            f"- {r['direction']} {r['region']} ({r.get('scope')}): n={r.get('n_bins')}  "
            f"B0 R² {float(r['B0_r2']):.3f} MAE {float(r['B0_mae']):.3f}  "
            f"B1 R² {float(r['B1_r2']):.3f} MAE {float(r['B1_mae']):.3f}"
        )
    lines.append("")
    lines += [
        f"10E energy mask: {ans.get('n_valid_bins')} bins, v in [{ans.get('v_valid_min')}, {ans.get('v_valid_max')}] m/s。",
        "官方 valid bin **全部**落在 |v|<1。中/高速 D(v) 在该 mask 下没有定义。",
        "",
        "## Top conditioners",
        "",
        "两方向平均 |coef|：",
    ]
    for i, (n, s) in enumerate(top[:3], 1):
        lines.append(f"{i}. {n} (mean |coef| {s:.4f})")
    if top_24:
        lines.append("24→77: " + ", ".join(f"{n}" for n, _ in top_24))
    if top_77:
        lines.append("77→24: " + ", ".join(f"{n}" for n, _ in top_77))
    lines += ["", f"## Gate: **{'PASS' if gate_ok else 'FAIL'}**", ""]
    lines.append("Gate 只看 D(v) 的 MAE/corr，不看 Radar-ID。")
    lines.append("这是 paired known-radar calibration，不是 unseen-radar DG。")
    lines.append("")
    if cal:
        lines += [
            "## Conditional calibration (paired known-radar)",
            f"- Radar-ID {cal['before']['radar_id']:.3f} → {cal['after']['radar_id']:.3f}",
            f"- vel-marginal {cal['before']['vel_rid']:.3f} → {cal['after']['vel_rid']:.3f}",
            f"- Activity {cal['before']['activity']:.3f} → {cal['after']['activity']:.3f}",
            f"- sep {cal['sep_before']:.3f} → {cal['sep_after']:.3f}",
            "",
        ]
    lines += ["## Low-|v| mask (diagnostic only)", ""]
    for r in masks:
        lines.append(f"- {r['mask']}: Radar-ID {r['radar_id']:.3f}  Activity {r['activity']:.3f}")
    lines += [
        "",
        f"Action residual 24→77: best **{best_act.get('action')}**  worst **{worst_act.get('action')}**",
        "",
        f"## CASE **{case}**",
        f"NEXT STEP: **{nxt}**",
        "",
        "未自动进入 Phase 11。",
        "",
    ]
    (out / "phase10_motion_conditioned_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
