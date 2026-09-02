#!/usr/bin/env python3
"""Phase 10E: empirical hardware-aware measurement-response audit.

U1 only (unified STFT R2-V). No U2. No ResNet. No Phase 11.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
RADARS = ["24GHz", "77GHz"]
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]
SEED = 42
EPS = 1e-8
V_LO = -3.0282065868377686
V_HI = 3.0267279148101807
# TRAIN-only energy mask: both-radar mean energy must exceed this TRAIN percentile.
ENERGY_Q = 20.0
# Research gate (pre-registered, not tuned on TEST).
GATE_VEL_RID = 0.90
GATE_ACT_DROP = 0.05
GATE_SEP_DROP = 0.02


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


def eta2(values, groups):
    y = np.asarray(values, np.float64)
    g = np.asarray(groups)
    if y.size < 3:
        return 0.0
    mu = y.mean()
    sst = float(((y - mu) ** 2).sum())
    if sst <= EPS:
        return 0.0
    ssb = 0.0
    for u in np.unique(g):
        yy = y[g == u]
        ssb += len(yy) * float((yy.mean() - mu) ** 2)
    return float(np.clip(ssb / sst, 0, 1))


def invert_db(Xz, norm):
    return np.asarray(Xz, np.float64) * float(norm["std"]) + float(norm["mean"])


def db_to_e(db):
    return np.power(10.0, np.asarray(db, np.float64) / 10.0)


def e_to_db(e):
    return 10.0 * np.log10(np.maximum(np.asarray(e, np.float64), EPS))


def global_z(db, tr):
    clip_lo = float(np.percentile(db[tr], 1))
    clip_hi = float(np.percentile(db[tr], 99))
    c = np.clip(db, clip_lo, clip_hi)
    mu = float(c[tr].mean())
    sd = float(c[tr].std())
    if sd <= 1e-8:
        sd = 1.0
    return ((c - mu) / sd).astype(np.float32), {"clip_lo": clip_lo, "clip_hi": clip_hi, "mean": mu, "std": sd}


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

    return {"pos_env": env(axis >= 0), "neg_env": env(axis < 0), "spread": spread, "energy": mass}


def pair_index(pack):
    by = defaultdict(dict)
    for i in range(len(pack["y"])):
        by[str(pack["timestamp"][i])][pack["radar"][i]] = i
    return {ts: rec for ts, rec in by.items() if all(r in rec for r in RADARS)}


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
    return {
        "same_cosine": float(np.mean(same)),
        "diff_cosine": float(np.mean(diff)),
        "sep_cosine": float(np.mean(same) - np.mean(diff)),
    }


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


def fingerprint(X, pack, split_eval="test"):
    tr = pack["split"] == "train"
    te = pack["split"] == split_eval
    gstat, vel, tmp = [], [], []
    for x in X:
        gstat.append([x.mean(), x.std(), np.median(x), np.percentile(x, 10), np.percentile(x, 90), x.max() - x.min()])
        vel.append(x.mean(axis=1))
        tmp.append(x.mean(axis=0))
    gstat, vel, tmp = np.asarray(gstat), np.asarray(vel), np.asarray(tmp)
    P = pool16(X)
    rows = {}
    for name, feat in (
        ("global_stats", gstat),
        ("velocity_marginal", vel),
        ("temporal_marginal", tmp),
        ("full_16x16", P),
    ):
        rid = probe(feat[tr], pack["radar"][tr], feat[te], pack["radar"][te])
        act = probe(feat[tr], pack["y"][tr], feat[te], pack["y"][te])
        rows[name] = {"radar_id": rid["accuracy"], "activity": act["accuracy"], "activity_f1": act["macro_f1"]}
    return rows


def load_u1(cache: Path, split: dict):
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


def write_definition(out: Path) -> None:
    (out / "measurement_model_definition.md").write_text(
        "\n".join(
            [
                "# Empirical measurement-response model",
                "",
                "这是工作模型，**不是** exact radar forward model。",
                "",
                "```",
                "S_r(tau, v) ≈ G_r(v) * M(tau, v) + B_r(v) + noise",
                "```",
                "",
                "- `M`：未观测的人体运动谱",
                "- `G_r(v)`：雷达相关的速度响应（经验）",
                "- `B_r(v)`：雷达相关的背景/偏移（经验）",
                "",
                "输入：Phase 10D **U1**（统一 STFT 后的 R2-V）。不用 U2。",
                "先把 U1 z-score 反回 clip 后的 dB，再 `E=10^(dB/10)`。",
                "",
                "## Energy mask（TRAIN only）",
                "",
                f"对 TRAIN 的 `P_r(v)=mean_tau E_r(tau,v)` 取两路平均，阈值 = 该 224 点曲线的第 **{ENERGY_Q:.0f}** 分位。",
                "两路均值都高于阈值的 bin 才解释 transfer / 才写入 D(v)。",
                "",
                "## Transfer",
                "",
                "`D_i(v) = log(P77_i(v)+eps) - log(P24_i(v)+eps)`，只在 mask 上。",
                "",
                "- `transfer_stability_score = 1 / (1 + mean std_D(v) on valid bins)`",
                "- 越接近 1 越稳。不按 TEST 调。",
                "",
                "## Models（拟合不用 action 标签）",
                "",
                "- M0：不校正",
                "- M1：每雷达一个 scale+offset，把 TRAIN dB 的全局均值/方差对齐到两雷达平均（对称 canonical）",
                "- M2：每个速度 bin 一个 gain/offset，对齐该 bin 的 TRAIN 均值/方差到两雷达平均",
                "- M3：TRAIN 成对平均 D(v)；`log E24' = log E24 + 0.5 D`，`log E77' = log E77 - 0.5 D`",
                "",
                "VAL 选模型。TEST 只评一次。",
                "",
                "## Gate",
                "",
                f"同时：TEST 速度边际 Radar-ID ≤ {GATE_VEL_RID:.2f}；Activity 相对 M0 下降 ≤ {GATE_ACT_DROP:.2f}；pair sep 下降 ≤ {GATE_SEP_DROP:.2f}。",
                "",
            ]
        ),
        encoding="utf-8",
    )


def diagnose(name, pack, X, vgrid, split_name="test"):
    fp = fingerprint(X, pack, split_name)
    sim = pair_sep(pack, X)
    geo = geo_stats(pack, X, vgrid)
    return {
        "model": name,
        "split": split_name,
        "radar_id": fp["full_16x16"]["radar_id"],
        "activity": fp["full_16x16"]["activity"],
        "activity_f1": fp["full_16x16"]["activity_f1"],
        "vel_rid": fp["velocity_marginal"]["radar_id"],
        "global_rid": fp["global_stats"]["radar_id"],
        "temporal_rid": fp["temporal_marginal"]["radar_id"],
        **sim,
        **geo,
        "fp": fp,
    }


def second_order_rid(X, pack):
    feats = []
    for x in X:
        x = np.asarray(x, np.float64)
        vel_var = x.var(axis=1)
        tmp_var = x.var(axis=0)
        ac_v = [corrcoef(x[:-k, :].ravel(), x[k:, :].ravel()) if x.shape[0] > k else 0.0 for k in (1, 2, 4)]
        ac_t = [corrcoef(x[:, :-k].ravel(), x[:, k:].ravel()) if x.shape[1] > k else 0.0 for k in (1, 2, 4)]
        feats.append(np.concatenate([vel_var, tmp_var, ac_v, ac_t]))
    F = np.stack(feats)
    tr, te = pack["split"] == "train", pack["split"] == "test"
    return probe(F[tr], pack["radar"][tr], F[te], pack["radar"][te])["accuracy"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u1-cache", default=str(ROOT / "results" / "phase10_unified_measurement" / "cache_u1"))
    ap.add_argument("--u1-norm", default=str(ROOT / "results" / "phase10_unified_measurement" / "u1_normalization.json"))
    ap.add_argument("--split", default=str(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase10_hardware_measurement"))
    args = ap.parse_args()

    out = Path(args.out)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(exist_ok=True)
    write_definition(out)

    split = {str(r["timestamp"]): r for r in read_csv(Path(args.split))}
    pack = load_u1(Path(args.u1_cache), split)
    norm = json.loads(Path(args.u1_norm).read_text(encoding="utf-8"))
    Xz = pack["X"]
    db0 = invert_db(Xz, norm)
    E0 = db_to_e(db0)
    vgrid = np.linspace(V_LO, V_HI, Xz.shape[1])
    tr = pack["split"] == "train"
    va = pack["split"] == "val"
    radar, action, ts = pack["radar"], pack["action"], pack["timestamp"]
    pairs = pair_index(pack)
    train_ts = [t for t, rec in pairs.items() if pack["split"][rec["24GHz"]] == "train"]
    print("loaded U1", Xz.shape, "train pairs", len(train_ts), flush=True)

    P = E0.mean(axis=2)  # (n, v)
    stat_rows = []
    profiles = {}
    for r in RADARS:
        m = (radar == r) & tr
        pr = P[m]
        profiles[r] = {
            "mean": pr.mean(0),
            "median": np.median(pr, 0),
            "std": pr.std(0),
            "q10": np.percentile(pr, 10, 0),
            "q90": np.percentile(pr, 90, 0),
        }
        for i, v in enumerate(vgrid):
            stat_rows.append(
                {
                    "radar": r,
                    "velocity_mps": float(v),
                    "mean": float(profiles[r]["mean"][i]),
                    "median": float(profiles[r]["median"][i]),
                    "std": float(profiles[r]["std"][i]),
                    "q10": float(profiles[r]["q10"][i]),
                    "q90": float(profiles[r]["q90"][i]),
                }
            )
    write_csv(out / "velocity_response_statistics.csv", ["radar", "velocity_mps", "mean", "median", "std", "q10", "q90"], stat_rows)

    both = 0.5 * (profiles["24GHz"]["mean"] + profiles["77GHz"]["mean"])
    thr = float(np.percentile(both, ENERGY_Q))
    mask = (profiles["24GHz"]["mean"] > thr) & (profiles["77GHz"]["mean"] > thr)
    print("energy mask", int(mask.sum()), "bins thr", thr, flush=True)

    D_list, D_act = [], []
    for t in train_ts:
        rec = pairs[t]
        d = np.log(P[rec["77GHz"]] + EPS) - np.log(P[rec["24GHz"]] + EPS)
        D_list.append(d)
        D_act.append(action[rec["24GHz"]])
    D = np.stack(D_list)
    D_act = np.array(D_act)
    D_mean = np.where(mask, D.mean(0), 0.0)
    D_med = np.where(mask, np.median(D, 0), 0.0)
    D_std = np.where(mask, D.std(0), np.nan)
    stab_rows = []
    for i, v in enumerate(vgrid):
        cv = float(D_std[i] / (abs(D_mean[i]) + EPS)) if mask[i] else ""
        stab_rows.append(
            {
                "velocity_mps": float(v),
                "valid": int(mask[i]),
                "transfer_mean": float(D_mean[i]) if mask[i] else "",
                "transfer_median": float(D_med[i]) if mask[i] else "",
                "transfer_std": float(D_std[i]) if mask[i] else "",
                "coefficient_of_variation": cv,
                "valid_fraction": float(mask.mean()),
            }
        )
    write_csv(
        out / "transfer_stability.csv",
        ["velocity_mps", "valid", "transfer_mean", "transfer_median", "transfer_std", "coefficient_of_variation", "valid_fraction"],
        stab_rows,
    )
    write_csv(
        out / "paired_transfer_profile.csv",
        ["velocity_mps", "mean_P24", "mean_P77", "ratio_77_over_24", "diff_77_minus_24", "D_mean", "D_std", "valid"],
        [
            {
                "velocity_mps": float(vgrid[i]),
                "mean_P24": float(profiles["24GHz"]["mean"][i]),
                "mean_P77": float(profiles["77GHz"]["mean"][i]),
                "ratio_77_over_24": float(profiles["77GHz"]["mean"][i] / (profiles["24GHz"]["mean"][i] + EPS)),
                "diff_77_minus_24": float(profiles["77GHz"]["mean"][i] - profiles["24GHz"]["mean"][i]),
                "D_mean": float(D_mean[i]) if mask[i] else "",
                "D_std": float(D_std[i]) if mask[i] else "",
                "valid": int(mask[i]),
            }
            for i in range(len(vgrid))
        ],
    )
    stab_score = float(1.0 / (1.0 + np.nanmean(D_std[mask])))
    mean_std_d = float(np.nanmean(D_std[mask]))

    act_rows = []
    D_by_act = {}
    for act in CLASS_NAMES:
        sel = D_act == act
        if sel.sum() == 0:
            continue
        da = D[sel].mean(0)
        D_by_act[act] = da
        for i, v in enumerate(vgrid):
            if not mask[i]:
                continue
            act_rows.append({"action": act, "velocity_mps": float(v), "D_mean": float(da[i]), "n_train": int(sel.sum())})
    write_csv(out / "action_conditioned_transfer.csv", ["action", "velocity_mps", "D_mean", "n_train"], act_rows)
    eta_bins = [eta2(D[:, i], D_act) for i in range(len(vgrid)) if mask[i]]
    mean_eta = float(np.mean(eta_bins)) if eta_bins else 0.0
    action_indep = "A_consistent" if mean_eta < 0.20 else "B_action_dependent"

    resid_norm = []
    resid_act = []
    for j, t in enumerate(train_ts):
        dv = D[j]
        resid_norm.append(float(np.linalg.norm((dv - D_mean)[mask])))
        resid_act.append(D_act[j])
    by_act_var = []
    for act in CLASS_NAMES:
        sel = np.array(resid_act) == act
        if sel.sum() == 0:
            continue
        vv = np.asarray(resid_norm)[sel]
        by_act_var.append({"action": act, "mean_residual_norm": float(vv.mean()), "std_residual_norm": float(vv.std()), "n": int(sel.sum())})
    by_act_var.sort(key=lambda r: r["mean_residual_norm"])
    write_csv(out / "pair_transfer_residuals.csv", ["action", "mean_residual_norm", "std_residual_norm", "n"], by_act_var)

    # --- models on dB / energy (no action labels) ---
    db = db0.copy()
    mu24, sd24 = float(db[(radar == "24GHz") & tr].mean()), float(db[(radar == "24GHz") & tr].std()) + EPS
    mu77, sd77 = float(db[(radar == "77GHz") & tr].mean()), float(db[(radar == "77GHz") & tr].std()) + EPS
    mu_c, sd_c = 0.5 * (mu24 + mu77), 0.5 * (sd24 + sd77)
    a24, b24 = sd_c / sd24, mu_c - (sd_c / sd24) * mu24
    a77, b77 = sd_c / sd77, mu_c - (sd_c / sd77) * mu77
    db_m1 = db.copy()
    db_m1[radar == "24GHz"] = a24 * db[radar == "24GHz"] + b24
    db_m1[radar == "77GHz"] = a77 * db[radar == "77GHz"] + b77

    db_m2 = db.copy()
    for r, ar in (("24GHz", "24"), ("77GHz", "77")):
        m = (radar == r) & tr
        mu_v = db[m].mean(axis=(0, 2))
        sd_v = db[m].std(axis=(0, 2)) + EPS
        if r == "24GHz":
            mu24v, sd24v = mu_v, sd_v
        else:
            mu77v, sd77v = mu_v, sd_v
    muc_v = 0.5 * (mu24v + mu77v)
    sdc_v = 0.5 * (sd24v + sd77v)
    g24, o24 = sdc_v / sd24v, muc_v - (sdc_v / sd24v) * mu24v
    g77, o77 = sdc_v / sd77v, muc_v - (sdc_v / sd77v) * mu77v
    db_m2[radar == "24GHz"] = db[radar == "24GHz"] * g24[None, :, None] + o24[None, :, None]
    db_m2[radar == "77GHz"] = db[radar == "77GHz"] * g77[None, :, None] + o77[None, :, None]

    E3 = E0.copy()
    gain = np.exp(0.5 * D_mean)  # 0 on invalid bins
    E3[radar == "24GHz"] = E0[radar == "24GHz"] * gain[None, :, None]
    E3[radar == "77GHz"] = E0[radar == "77GHz"] / np.maximum(gain[None, :, None], EPS)
    db_m3 = e_to_db(E3)

    X0 = Xz.astype(np.float32)
    X1, n1 = global_z(db_m1, tr)
    X2, n2 = global_z(db_m2, tr)
    X3, n3 = global_z(db_m3, tr)
    models = {"M0": X0, "M1": X1, "M2": X2, "M3": X3}
    (out / "calibration_params.json").write_text(
        json.dumps(
            {
                "M1": {"a24": a24, "b24": b24, "a77": a77, "b77": b77},
                "energy_threshold": thr,
                "n_valid_bins": int(mask.sum()),
                "transfer_stability_score": stab_score,
                "mean_D_std": mean_std_d,
                "mean_eta2_action": mean_eta,
                "norms": {"M1": n1, "M2": n2, "M3": n3},
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    cal_rows = []
    diags = {}
    for name, X in models.items():
        dte = diagnose(name, pack, X, vgrid, "test")
        dva = diagnose(name, pack, X, vgrid, "val")
        diags[name] = {"test": dte, "val": dva}
        cal_rows.append(
            {
                "model": name,
                "val_radar_id": dva["radar_id"],
                "val_vel_rid": dva["vel_rid"],
                "val_activity": dva["activity"],
                "val_sep": dva["sep_cosine"],
                "test_radar_id": dte["radar_id"],
                "test_vel_rid": dte["vel_rid"],
                "test_activity": dte["activity"],
                "test_activity_f1": dte["activity_f1"],
                "test_sep": dte["sep_cosine"],
                "test_same": dte["same_cosine"],
                "test_diff": dte["diff_cosine"],
                "test_env_corr": dte["envelope_corr"],
                "test_spread_corr": dte["spread_corr"],
                "test_global_rid": dte["global_rid"],
                "test_temporal_rid": dte["temporal_rid"],
            }
        )
        print(name, "VAL velRID", dva["vel_rid"], "ACT", dva["activity"], "TEST velRID", dte["vel_rid"], "ACT", dte["activity"], flush=True)
    write_csv(out / "measurement_calibration_results.csv", list(cal_rows[0].keys()), cal_rows)

    m0a = diags["M0"]["val"]["activity"]
    candidates = [k for k in ("M1", "M2", "M3") if diags[k]["val"]["activity"] >= m0a - GATE_ACT_DROP]
    if not candidates:
        candidates = ["M1", "M2", "M3"]
    best = min(candidates, key=lambda k: (diags[k]["val"]["vel_rid"], -diags[k]["val"]["activity"]))
    print("VAL-selected", best, flush=True)

    fp_rows = []
    for name, X in models.items():
        for feat, rec in diags[name]["test"]["fp"].items():
            fp_rows.append({"model": name, "features": feat, **rec})
    write_csv(out / "fingerprint_decomposition.csv", ["model", "features", "radar_id", "activity", "activity_f1"], fp_rows)

    Xb = models[best]
    ho = second_order_rid(Xb, pack)
    ho0 = second_order_rid(X0, pack)
    # residual after best
    mu24s = Xb[(radar == "24GHz") & tr].mean(axis=0)
    mu77s = Xb[(radar == "77GHz") & tr].mean(axis=0)
    res_v = (mu77s - mu24s).mean(axis=1)
    res_t = (mu77s - mu24s).mean(axis=0)
    write_csv(
        out / "residual_fingerprint.csv",
        ["kind", "value"],
        [
            {"kind": "best_model", "value": best},
            {"kind": "residual_spectrum_l2", "value": float(np.linalg.norm(mu77s - mu24s))},
            {"kind": "residual_velocity_l2", "value": float(np.linalg.norm(res_v))},
            {"kind": "residual_temporal_l2", "value": float(np.linalg.norm(res_t))},
            {"kind": "higher_order_radar_id_M0", "value": ho0},
            {"kind": "higher_order_radar_id_best", "value": ho},
            {"kind": "best_global_rid", "value": diags[best]["test"]["global_rid"]},
            {"kind": "best_vel_rid", "value": diags[best]["test"]["vel_rid"]},
            {"kind": "best_temporal_rid", "value": diags[best]["test"]["temporal_rid"]},
        ],
    )

    region_rows = []
    absv = np.abs(vgrid)
    regions = {
        "low_abs_v": absv < 1.0,
        "high_abs_v": absv >= 1.0,
        "positive_v": vgrid >= 0,
        "negative_v": vgrid < 0,
    }
    for rname, rm in regions.items():
        feat = X0.mean(axis=2)[:, rm]
        rid0 = probe(feat[tr], radar[tr], feat[pack["split"] == "test"], radar[pack["split"] == "test"])
        featb = Xb.mean(axis=2)[:, rm]
        ridb = probe(featb[tr], radar[tr], featb[pack["split"] == "test"], radar[pack["split"] == "test"])
        region_rows.append({"region": rname, "n_bins": int(rm.sum()), "M0_velregion_rid": rid0["accuracy"], "best_velregion_rid": ridb["accuracy"]})
    write_csv(out / "velocity_region_domain_shift.csv", ["region", "n_bins", "M0_velregion_rid", "best_velregion_rid"], region_rows)

    d0, dbest = diags["M0"]["test"], diags[best]["test"]
    gate = (
        dbest["vel_rid"] <= GATE_VEL_RID
        and dbest["activity"] >= d0["activity"] - GATE_ACT_DROP
        and dbest["sep_cosine"] >= d0["sep_cosine"] - GATE_SEP_DROP
    )
    vel_drop = d0["vel_rid"] - dbest["vel_rid"]
    act_drop = d0["activity"] - dbest["activity"]
    if gate:
        case, nxt = "HW-A", "CALIBRATED SHARED ENCODER"
    elif vel_drop >= 0.10 and act_drop > GATE_ACT_DROP:
        case, nxt = "HW-D", "DISENTANGLED SENSOR-MOTION MODEL"
    elif mean_eta >= 0.20 or mean_std_d >= 1.0:
        case, nxt = "HW-C", "PAIR-CONDITIONAL / GEOMETRY-AWARE MODEL"
    elif vel_drop >= 0.05 or stab_score >= 0.55:
        case, nxt = "HW-B", "PHYSICS-INFORMED MEASUREMENT MODEL"
    else:
        case, nxt = "HW-C", "PAIR-CONDITIONAL / GEOMETRY-AWARE MODEL"
    if dbest["vel_rid"] >= 0.97 and mean_eta >= 0.15:
        case, nxt = "HW-C", "PAIR-CONDITIONAL / GEOMETRY-AWARE MODEL"

    # figures
    plt = _mpl()
    fig1, ax = plt.subplots(figsize=(7.2, 3.2))
    ax.text(0.5, 0.55, r"$S_r(\tau,v)\approx G_r(v)\,M(\tau,v)+B_r(v)$", ha="center", fontsize=13)
    ax.text(0.5, 0.25, "empirical measurement-response  ·  not a forward model", ha="center")
    ax.axis("off")
    fig1.savefig(fig / "fig1_model_schematic.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(vgrid, profiles["24GHz"]["mean"], label="24 mean E")
    ax.plot(vgrid, profiles["77GHz"]["mean"], label="77 mean E")
    ax.axhline(thr, color="k", ls="--", lw=0.8, label="mask thr")
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("TRAIN mean energy")
    ax.legend()
    fig2.savefig(fig / "fig2_mean_marginals.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    fig3, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(vgrid[mask], D_mean[mask])
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("mean D(v)")
    fig3.savefig(fig / "fig3_paired_D.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)

    fig4, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(vgrid[mask], D_mean[mask])
    ax.fill_between(vgrid[mask], (D_mean - D_std)[mask], (D_mean + D_std)[mask], alpha=0.3)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("D(v)")
    fig4.savefig(fig / "fig4_D_mean_std.png", dpi=130, bbox_inches="tight")
    plt.close(fig4)

    fig5, ax = plt.subplots(figsize=(7.2, 3.8))
    for act, da in D_by_act.items():
        ax.plot(vgrid[mask], da[mask], label=act, lw=1)
    ax.set_xlabel("v (m/s)")
    ax.set_ylabel("D_action(v)")
    ax.legend(fontsize=7, ncol=2)
    fig5.savefig(fig / "fig5_action_transfer.png", dpi=130, bbox_inches="tight")
    plt.close(fig5)

    xs = ["M0", "M1", "M2", "M3"]
    fig6, ax = plt.subplots(figsize=(6.0, 3.4))
    ax.bar(xs, [diags[k]["test"]["radar_id"] for k in xs], 0.4, label="16x16")
    ax.bar(np.arange(4) + 0.4, [diags[k]["test"]["vel_rid"] for k in xs], 0.4, label="vel marginal")
    ax.set_xticks(np.arange(4) + 0.2)
    ax.set_xticklabels(xs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig6.savefig(fig / "fig6_radar_id.png", dpi=130, bbox_inches="tight")
    plt.close(fig6)

    fig7, ax = plt.subplots(figsize=(5.2, 4.4))
    for k in xs:
        ax.scatter(diags[k]["test"]["radar_id"], diags[k]["test"]["activity"])
        ax.annotate(k, (diags[k]["test"]["radar_id"], diags[k]["test"]["activity"]))
    ax.set_xlabel("Radar-ID")
    ax.set_ylabel("Activity")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    fig7.savefig(fig / "fig7_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig7)

    fig8, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar(xs, [diags[k]["test"]["sep_cosine"] for k in xs])
    ax.set_ylabel("pair sep")
    fig8.savefig(fig / "fig8_pair_sep.png", dpi=130, bbox_inches="tight")
    plt.close(fig8)

    fig9, ax = plt.subplots(figsize=(7.2, 3.4))
    ax.plot(vgrid, X0[(radar == "24GHz") & tr].mean(axis=(0, 2)), label="M0 24")
    ax.plot(vgrid, X0[(radar == "77GHz") & tr].mean(axis=(0, 2)), label="M0 77")
    ax.plot(vgrid, Xb[(radar == "24GHz") & tr].mean(axis=(0, 2)), ls="--", label=f"{best} 24")
    ax.plot(vgrid, Xb[(radar == "77GHz") & tr].mean(axis=(0, 2)), ls="--", label=f"{best} 77")
    ax.set_xlabel("v (m/s)")
    ax.legend(fontsize=8)
    fig9.savefig(fig / "fig9_residual.png", dpi=130, bbox_inches="tight")
    plt.close(fig9)

    fig10, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.bar([r["region"] for r in region_rows], [r["M0_velregion_rid"] for r in region_rows])
    ax.set_ylim(0, 1.05)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("M0 region Radar-ID")
    fig10.savefig(fig / "fig10_regions.png", dpi=130, bbox_inches="tight")
    plt.close(fig10)

    answers = {
        "transfer_stability_score": stab_score,
        "mean_D_std": mean_std_d,
        "mean_eta2_action": mean_eta,
        "action_independence": action_indep,
        "best_model": best,
        "gate": bool(gate),
        "case": case,
        "next_step": nxt,
        "n_valid_bins": int(mask.sum()),
        "M0": {k: diags["M0"]["test"][k] for k in ("radar_id", "vel_rid", "activity", "sep_cosine")},
        "M1": {k: diags["M1"]["test"][k] for k in ("radar_id", "vel_rid", "activity", "sep_cosine")},
        "M2": {k: diags["M2"]["test"][k] for k in ("radar_id", "vel_rid", "activity", "sep_cosine")},
        "M3": {k: diags["M3"]["test"][k] for k in ("radar_id", "vel_rid", "activity", "sep_cosine")},
        "higher_order_M0": ho0,
        "higher_order_best": ho,
        "most_stable_action": by_act_var[0]["action"] if by_act_var else "",
        "most_variable_action": by_act_var[-1]["action"] if by_act_var else "",
        "har_trained": False,
    }
    (out / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    write_report(out, answers, diags, by_act_var, region_rows, stab_score, mean_eta, mean_std_d, action_indep, best, case, nxt, ho0, ho)
    print("phase10e done", json.dumps(answers, indent=2), flush=True)


def write_report(out, ans, diags, by_act, regions, stab, eta, std_d, indep, best, case, nxt, ho0, ho):
    def line(m):
        d = diags[m]["test"]
        return f"- {m}: Radar-ID {d['radar_id']:.3f}  vel-marginal {d['vel_rid']:.3f}  Activity {d['activity']:.3f}  sep {d['sep_cosine']:.3f}"

    lines = [
        "# Phase 10E — Empirical measurement-response audit",
        "",
        "工作模型，不是 exact forward model。输入 U1。未用 U2。未训 ResNet。",
        "",
        "## Transfer",
        "",
        f"- valid bins: {ans['n_valid_bins']} / 224（TRAIN 能量第 {ENERGY_Q:.0f} 分位）",
        f"- transfer_stability_score: **{stab:.3f}**  (mean std D = {std_d:.3f})",
        f"- action η² (mean over valid v): **{eta:.3f}** → {indep}",
        "",
        "## Calibration (TEST)",
        "",
        line("M0"),
        line("M1"),
        line("M2"),
        line("M3"),
        "",
        f"VAL 选定：**{best}**。Gate: **{'PASS' if ans['gate'] else 'FAIL'}**",
        "",
        "## Residual",
        "",
        f"- higher-order Radar-ID M0 **{ho0:.3f}** / best **{ho:.3f}**",
        "",
        "## Action residual norm",
        "",
    ]
    for r in by_act:
        lines.append(f"- {r['action']}: {r['mean_residual_norm']:.3f}")
    lines += [
        "",
        "## Regions (M0 velocity-marginal Radar-ID)",
        "",
    ]
    for r in regions:
        lines.append(f"- {r['region']}: {r['M0_velregion_rid']:.3f}")
    lines += [
        "",
        f"## CASE **{case}**",
        f"NEXT STEP: **{nxt}**",
        "",
        "未自动进入下一阶段。",
        "",
    ]
    (out / "phase10_hardware_measurement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
