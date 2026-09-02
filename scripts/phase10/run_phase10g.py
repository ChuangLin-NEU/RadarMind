#!/usr/bin/env python3
"""Phase 10G: motion-conditioned 2D measurement model (low-rank + Ridge).

Not geometry-aware. Not unseen-radar DG. No ResNet / translator / Phase 11.
Reuses Phase 10F source-only q and Phase 10D U1. Does not modify 1–10F.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "phase10"))
import run_phase10f as p10f  # noqa: E402

RADARS = p10f.RADARS
CLASS_NAMES = p10f.CLASS_NAMES
SEED = p10f.SEED
EPS = p10f.EPS
V_LO, V_HI = p10f.V_LO, p10f.V_HI
LOW_V, MID_V = p10f.LOW_V, p10f.MID_V
ENERGY_Q = p10f.ENERGY_Q
Q_NAMES = p10f.Q_NAMES
ALPHAS = p10f.ALPHAS
KS = (2, 5, 10, 20)
NEAR_V = 0.10
# Gate: B2 vs B1 on 2D Delta / reconstruction / activity. Not chosen by Radar-ID.
GATE_MAE_RATIO = 0.90
GATE_SIM_GAIN = 0.05
GATE_ACT_DROP = 0.05
ORACLE_GOOD_R2 = 0.50
ORACLE_POOR_R2 = 0.30
COND_GAP_R2 = 0.15


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_csv(path: Path, fields, rows) -> None:
    p10f.write_csv(path, fields, rows)


def metrics_flat(y, yhat):
    y, yhat = np.asarray(y, np.float64).ravel(), np.asarray(yhat, np.float64).ravel()
    err = y - yhat
    sst = float(((y - y.mean()) ** 2).sum())
    r2 = float(1.0 - (err ** 2).sum() / sst) if sst > EPS else 0.0
    return {
        "r2": r2,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "corr": p10f.corrcoef(y, yhat),
    }


def ssim_pair(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    try:
        from skimage.metrics import structural_similarity

        data_range = float(max(a.max(), b.max()) - min(a.min(), b.min()))
        if data_range <= EPS:
            return 0.0
        return float(structural_similarity(a, b, data_range=data_range))
    except Exception:
        mu_a, mu_b = float(a.mean()), float(b.mean())
        sa, sb = float(a.var()), float(b.var())
        cov = float(((a - mu_a) * (b - mu_b)).mean())
        c1, c2 = 1e-4, 9e-4
        return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a**2 + mu_b**2 + c1) * (sa + sb + c2) + EPS))


def recon_row(pred, tgt, tag):
    sims, corrs, maes, ssims = [], [], [], []
    for i in range(len(tgt)):
        sims.append(p10f.cosine_sim(pred[i], tgt[i]))
        corrs.append(p10f.corrcoef(pred[i], tgt[i]))
        maes.append(float(np.mean(np.abs(pred[i] - tgt[i]))))
        ssims.append(ssim_pair(pred[i], tgt[i]))
    return {
        "item": tag,
        "cosine": float(np.mean(sims)),
        "corr": float(np.mean(corrs)),
        "mae": float(np.mean(maes)),
        "ssim": float(np.mean(ssims)),
    }


def region_masks(v, nt):
    near = np.abs(v) < NEAR_V
    low = np.abs(v) < LOW_V
    mid = (np.abs(v) >= LOW_V) & (np.abs(v) < MID_V)
    high = np.abs(v) >= MID_V
    t0, t1 = nt // 3, 2 * nt // 3
    return {
        "near-zero": (near, slice(None)),
        "low": (low, slice(None)),
        "mid": (mid, slice(None)),
        "high": (high, slice(None)),
        "early": (slice(None), slice(0, t0)),
        "middle": (slice(None), slice(t0, t1)),
        "late": (slice(None), slice(t1, None)),
    }


def take_region(arr, spec):
    vm, tm = spec
    if isinstance(vm, slice):
        return arr[:, :, tm] if arr.ndim == 3 else arr[:, tm]
    sub = arr[:, vm, :] if arr.ndim == 3 else arr[vm, :]
    if isinstance(tm, slice):
        return sub[:, :, tm] if sub.ndim == 3 else sub[:, tm]
    return sub


def fit_ridge_val(Qtr, Ytr, Qva, Qte, Yva):
    sc = StandardScaler()
    Ztr, Zva, Zte = sc.fit_transform(Qtr), sc.transform(Qva), sc.transform(Qte)
    best, best_sc, best_a = None, -1e18, None
    for a in ALPHAS:
        mdl = Ridge(alpha=a)
        mdl.fit(Ztr, Ytr)
        scv = metrics_flat(Yva, mdl.predict(Zva))
        score = -scv["mae"] + scv["corr"]
        if score > best_sc:
            best, best_sc, best_a = mdl, score, a
    return best, sc, float(best_a), best.predict(Zte), best.predict(Zva)


def classify_frac(clf, sc, X, label):
    pred = clf.predict(sc.transform(p10f.pool16(X)))
    return float(np.mean(pred == label))


def describe_pc(pc, v, name):
    """Describe visible structure only. No body-part words."""
    nv, nt = pc.shape
    t0, t1 = nt // 3, 2 * nt // 3
    e_v = np.mean(np.abs(pc), 1)
    e_t = np.mean(np.abs(pc), 0)
    notes = []
    if e_v[np.abs(v) < NEAR_V].mean() > 1.3 * (e_v.mean() + EPS):
        notes.append("near-zero velocity energy")
    if e_v[v > 0].mean() > 1.2 * (e_v[v < 0].mean() + EPS):
        notes.append("stronger on +v")
    elif e_v[v < 0].mean() > 1.2 * (e_v[v > 0].mean() + EPS):
        notes.append("stronger on -v")
    if e_v[np.abs(v) >= MID_V].mean() > 1.2 * (e_v[np.abs(v) < LOW_V].mean() + EPS):
        notes.append("high-|v| structure")
    if float(np.mean(np.abs(np.diff(e_v)))) > 1.2 * float(np.mean(e_v) / max(nv, 1)):
        notes.append("spectral widening / rough velocity envelope")
    if e_t[:t0].mean() > 1.2 * (e_t[t1:].mean() + EPS):
        notes.append("early-time")
    elif e_t[t1:].mean() > 1.2 * (e_t[:t0].mean() + EPS):
        notes.append("late-time")
    if e_t[t0:t1].mean() > 1.2 * (0.5 * (e_t[:t0].mean() + e_t[t1:].mean()) + EPS):
        notes.append("mid-time")
    return "; ".join(notes) if notes else "no dominant axis-aligned structure"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--u1-cache", default=str(ROOT / "results" / "phase10_unified_measurement" / "cache_u1"))
    ap.add_argument("--u1-norm", default=str(ROOT / "results" / "phase10_unified_measurement" / "u1_normalization.json"))
    ap.add_argument("--split", default=str(ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"))
    ap.add_argument("--q-csv", default=str(ROOT / "results" / "phase10_motion_conditioned" / "conditioning_factors.csv"))
    ap.add_argument("--out", default=str(ROOT / "results" / "phase10_conditional_2d_measurement"))
    args = ap.parse_args()

    out = Path(args.out)
    fig = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig.mkdir(exist_ok=True)

    split = {str(r["timestamp"]): r for r in p10f.read_csv(Path(args.split))}
    pack = p10f.load_u1(Path(args.u1_cache), split)
    norm = json.loads(Path(args.u1_norm).read_text(encoding="utf-8"))
    Sz = pack["X"]
    vgrid = np.linspace(V_LO, V_HI, Sz.shape[1])
    nv, nt = Sz.shape[1], Sz.shape[2]
    db = Sz * float(norm["std"]) + float(norm["mean"])
    E = np.power(10.0, db / 10.0)
    Xlog = np.log(E + EPS).astype(np.float64)
    radar, action, tsarr, spl = pack["radar"], pack["action"], pack["timestamp"], pack["split"]
    pairs = p10f.pair_index(pack)
    qrows = p10f.read_csv(Path(args.q_csv))
    Qmap = {(r["timestamp"], r["radar"]): np.array([float(r[n]) for n in Q_NAMES], np.float64) for r in qrows}

    def collect(split_name, sign=1):
        xs24, xs77, dlt, q24, q77, acts, tss = [], [], [], [], [], [], []
        for t, rec in pairs.items():
            if pack["split"][rec["24GHz"]] != split_name:
                continue
            a24, a77 = Xlog[rec["24GHz"]], Xlog[rec["77GHz"]]
            xs24.append(a24)
            xs77.append(a77)
            dlt.append(sign * (a77 - a24))
            q24.append(Qmap[(t, "24GHz")])
            q77.append(Qmap[(t, "77GHz")])
            acts.append(action[rec["24GHz"]])
            tss.append(t)
        return {
            "x24": np.stack(xs24),
            "x77": np.stack(xs77),
            "delta": np.stack(dlt),
            "q24": np.stack(q24),
            "q77": np.stack(q77),
            "action": np.array(acts),
            "ts": tss,
        }

    tr, va, te = collect("train", 1), collect("val", 1), collect("test", 1)
    Dtr = tr["delta"]
    n_tr = len(Dtr)
    Dmean = Dtr.mean(0)
    Dstd = Dtr.std(0)
    add_v, add_t = Dmean.mean(1), Dmean.mean(0)
    additive = add_v[:, None] + add_t[None, :] - float(Dmean.mean())
    inter = Dmean - additive
    tot = float(np.var(Dmean)) + EPS
    stat_rows = [
        {"item": "mean_abs_Delta", "value": float(np.mean(np.abs(Dmean)))},
        {"item": "std_mean_over_pairs", "value": float(Dstd.mean())},
        {"item": "velocity_marginal_var_frac", "value": float(np.var(add_v) / tot)},
        {"item": "temporal_marginal_var_frac", "value": float(np.var(add_t) / tot)},
        {"item": "additive_var_frac", "value": float(np.var(additive) / tot)},
        {"item": "interaction_var_frac", "value": float(np.var(inter) / tot)},
        {"item": "n_train_pairs", "value": n_tr},
        {"item": "shape_v", "value": nv},
        {"item": "shape_tau", "value": nt},
        {"item": "representation", "value": "log(E+eps) after U1 invert; not a forward model"},
    ]
    write_csv(out / "delta_2d_statistics.csv", ["item", "value"], stat_rows)
    print("Delta stats", {r["item"]: r["value"] for r in stat_rows if r["item"] != "representation"}, flush=True)

    # energy mask (same 10E/10F) for B1 tiling
    P = E.mean(axis=2)
    trm = spl == "train"
    profiles = {r: P[(radar == r) & trm].mean(0) for r in RADARS}
    thr = float(np.percentile(0.5 * (profiles["24GHz"] + profiles["77GHz"]), ENERGY_Q))
    vmask = (profiles["24GHz"] > thr) & (profiles["77GHz"] > thr)

    # PCA on TRAIN 24→77 Delta; 77→24 uses -Delta with its own PCA
    sel_rows, ev_rows, man_rows = [], [], []
    pcas = {}
    for direction, sign in (("24to77", 1), ("77to24", -1)):
        Ytr = (sign * Dtr).reshape(n_tr, -1)
        Yva = (sign * va["delta"]).reshape(len(va["delta"]), -1)
        pca20 = PCA(n_components=20, random_state=SEED)
        pca20.fit(Ytr)
        ev = np.cumsum(pca20.explained_variance_ratio_)
        for k in KS:
            ev_rows.append({"direction": direction, "k": k, "explained_var": float(ev[k - 1]), "split": "TRAIN"})
        best_k, best_mae = None, 1e18
        for k in KS:
            pca = PCA(n_components=k, random_state=SEED)
            pca.fit(Ytr)
            rec_va = pca.inverse_transform(pca.transform(Yva))
            mae = float(np.mean(np.abs(Yva - rec_va)))
            rec_tr = pca.inverse_transform(pca.transform(Ytr))
            sel_rows.append(
                {
                    "direction": direction,
                    "k": k,
                    "train_explained_var": float(np.cumsum(pca.explained_variance_ratio_)[-1]),
                    "val_recon_mae": mae,
                    "train_recon_mae": float(np.mean(np.abs(Ytr - rec_tr))),
                }
            )
            if mae < best_mae:
                best_mae, best_k = mae, k
        pcas[direction] = PCA(n_components=best_k, random_state=SEED).fit(Ytr)
        man_rows.append(
            {
                "direction": direction,
                "selected_k": best_k,
                "train_explained_var": float(np.cumsum(pcas[direction].explained_variance_ratio_)[-1]),
                "val_recon_mae": best_mae,
                "n_pixels": nv * nt,
                "note": "K by VAL recon MAE only; TEST unused",
            }
        )
        print(direction, "selected K", best_k, "VAL MAE", best_mae, flush=True)

    write_csv(out / "low_rank_selection.csv", ["direction", "k", "train_explained_var", "val_recon_mae", "train_recon_mae"], sel_rows)
    write_csv(out / "basis_explained_variance.csv", ["direction", "k", "explained_var", "split"], ev_rows)
    write_csv(out / "delta_basis_manifest.csv", ["direction", "selected_k", "train_explained_var", "val_recon_mae", "n_pixels", "note"], man_rows)

    # B1: 10F D(v|q) on official mask, tiled over tau
    b1_models = {}
    for direction, qkey, sign in (("24to77", "q24", 1), ("77to24", "q77", -1)):
        def dv(pack_s, s=sign):
            P24 = np.exp(pack_s["x24"]).mean(2)
            P77 = np.exp(pack_s["x77"]).mean(2)
            d = np.log(P77 + EPS) - np.log(P24 + EPS)
            return (s * d)[:, vmask]

        Ytr, Yva = dv(tr), dv(va)
        mdl, scq, alpha, yte, yva = fit_ridge_val(tr[qkey], Ytr, va[qkey], te[qkey], Yva)
        b1_models[direction] = {"model": mdl, "scaler": scq, "alpha": alpha}

        def tile(pack_s, mdl=mdl, scq=scq, qk=qkey, sg=sign):
            dhat = np.zeros((len(pack_s["delta"]), nv), np.float64)
            dhat[:, vmask] = mdl.predict(scq.transform(pack_s[qk]))
            return np.repeat(dhat[:, :, None], nt, axis=2)

        b1_models[direction]["tile"] = tile

    transfer_rows, region_rows, coef_rows = [], [], []
    recon_rows, radar_rows, act_rows = [], [], []
    resid_rows, action_rows, oracle_rows = [], [], []
    hats = {}
    regs = region_masks(vgrid, nt)

    # shared probes on log-energy TRAIN
    Xtr_all = np.concatenate([tr["x24"], tr["x77"]], 0)
    yrad_tr = np.array(["24GHz"] * len(tr["x24"]) + ["77GHz"] * len(tr["x77"]))
    yact_tr = np.concatenate([tr["action"], tr["action"]])
    yact_idx = np.array([CLASS_NAMES.index(a) for a in yact_tr])
    sc_r, sc_a = StandardScaler(), StandardScaler()
    clf_r = LogisticRegression(max_iter=2000, random_state=SEED)
    clf_a = LogisticRegression(max_iter=2000, random_state=SEED)
    clf_r.fit(sc_r.fit_transform(p10f.pool16(Xtr_all)), yrad_tr)
    clf_a.fit(sc_a.fit_transform(p10f.pool16(Xtr_all)), yact_idx)

    for direction, qkey, src_key, tgt_key, src_lab, tgt_lab, sign in (
        ("24to77", "q24", "x24", "x77", "24GHz", "77GHz", 1),
        ("77to24", "q77", "x77", "x24", "77GHz", "24GHz", -1),
    ):
        pca = pcas[direction]
        k = int(pca.n_components_)
        Ytr = (sign * Dtr).reshape(n_tr, -1)
        Yva = (sign * va["delta"]).reshape(len(va["delta"]), -1)
        Yte = (sign * te["delta"]).reshape(len(te["delta"]), -1)
        scores_tr = pca.transform(Ytr)
        scores_va = pca.transform(Yva)
        scores_te = pca.transform(Yte)
        oracle_te = pca.inverse_transform(scores_te).reshape(-1, nv, nt)
        b0_te = np.broadcast_to(pca.mean_.reshape(nv, nt), te["delta"].shape)
        b1_te = b1_models[direction]["tile"](te)
        mdl, scq, alpha, pred_sc, _ = fit_ridge_val(tr[qkey], scores_tr, va[qkey], te[qkey], scores_va)
        b2_te = pca.inverse_transform(pred_sc).reshape(-1, nv, nt)
        true_d = sign * te["delta"]

        hats[direction] = {
            "k": k,
            "alpha": alpha,
            "pca": pca,
            "true": true_d,
            "B0": b0_te,
            "B1": b1_te,
            "B2": b2_te,
            "oracle": oracle_te,
            "src": te[src_key],
            "tgt": te[tgt_key],
            "action": te["action"],
            "q": te[qkey],
            "model": mdl,
            "scaler": scq,
        }

        m0, m1, m2, mo = metrics_flat(true_d, b0_te), metrics_flat(true_d, b1_te), metrics_flat(true_d, b2_te), metrics_flat(true_d, oracle_te)
        print(direction, "K", k, "B0", m0, "B1", m1, "B2", m2, "ORACLE", mo, flush=True)
        for name, mm in (("B0", m0), ("B1", m1), ("B2", m2), ("Oracle", mo)):
            transfer_rows.append({"direction": direction, "model": name, **mm, "k": k if name in ("B2", "Oracle") else "", "ridge_alpha": alpha if name == "B2" else ""})
        hats[direction].update({"m0": m0, "m1": m1, "m2": m2, "mo": mo})

        for rname, spec in regs.items():
            yt, nbin = take_region(true_d, spec), take_region(true_d, spec).size // max(len(true_d), 1)
            if take_region(true_d[:1], spec).size == 0:
                region_rows.append({"direction": direction, "region": rname, "n": 0, "note": "UNAVAILABLE"})
                continue
            for name, pred in (("B0", b0_te), ("B1", b1_te), ("B2", b2_te), ("Oracle", oracle_te)):
                mm = metrics_flat(yt, take_region(pred, spec))
                region_rows.append({"direction": direction, "region": rname, "model": name, "n_pixels_per_pair": int(take_region(true_d[:1], spec).size), **mm})

        coef = np.asarray(mdl.coef_, np.float64)
        if coef.ndim == 1:
            coef = coef.reshape(1, -1)
        if coef.shape[1] != len(Q_NAMES) and coef.shape[0] == len(Q_NAMES):
            coef = coef.T
        for j in range(min(3, coef.shape[0])):
            order = np.argsort(-np.abs(coef[j]))
            for rank, qi in enumerate(order[:8], 1):
                coef_rows.append(
                    {
                        "direction": direction,
                        "component": f"PC{j+1}",
                        "factor": Q_NAMES[qi],
                        "std_coef": float(coef[j, qi]),
                        "abs_coef": float(abs(coef[j, qi])),
                        "rank": rank,
                    }
                )

        src, tgt = te[src_key], te[tgt_key]
        pred_b0, pred_b1, pred_b2, pred_or = src + b0_te, src + b1_te, src + b2_te, src + oracle_te
        hats[direction].update({"pred_b0": pred_b0, "pred_b1": pred_b1, "pred_b2": pred_b2, "pred_or": pred_or})
        for tag, arr in (
            ("source", src),
            ("B0_predicted", pred_b0),
            ("B1_predicted", pred_b1),
            ("B2_predicted", pred_b2),
            ("Oracle_predicted", pred_or),
        ):
            row = recon_row(arr, tgt, tag)
            row["direction"] = direction
            row["gain_vs_source"] = row["cosine"] - recon_row(src, tgt, "source")["cosine"] if tag != "source" else 0.0
            recon_rows.append(row)

        # radar-domain fidelity (not invariance)
        for tag, arr in (("source", src), ("B2_predicted", pred_b2), ("true_target", tgt), ("Oracle_predicted", pred_or)):
            radar_rows.append(
                {
                    "direction": direction,
                    "item": tag,
                    "frac_source_label": classify_frac(clf_r, sc_r, arr, src_lab),
                    "frac_target_label": classify_frac(clf_r, sc_r, arr, tgt_lab),
                }
            )

        yte_act = np.array([CLASS_NAMES.index(a) for a in te["action"]])
        for tag, arr in (("source", src), ("B2_predicted", pred_b2), ("true_target", tgt)):
            pred_a = clf_a.predict(sc_a.transform(p10f.pool16(arr)))
            act_rows.append({"direction": direction, "item": tag, "activity": float(np.mean(pred_a == yte_act))})

        resid = true_d - b2_te
        add_v_r, add_t_r = resid.mean(0).mean(1), resid.mean(0).mean(0)
        additive_r = add_v_r[:, None] + add_t_r[None, :] - float(resid.mean())
        resid_rows.append(
            {
                "direction": direction,
                "item": "overall_mae",
                "value": float(np.mean(np.abs(resid))),
            }
        )
        resid_rows.append({"direction": direction, "item": "overall_std", "value": float(resid.std())})
        resid_rows.append({"direction": direction, "item": "mean_bias", "value": float(resid.mean())})
        resid_rows.append({"direction": direction, "item": "interaction_std", "value": float((resid.mean(0) - additive_r).std())})
        for rname, spec in regs.items():
            chunk = take_region(resid, spec)
            if chunk.size == 0:
                resid_rows.append({"direction": direction, "item": f"{rname}_mae", "value": "UNAVAILABLE"})
            else:
                resid_rows.append({"direction": direction, "item": f"{rname}_mae", "value": float(np.mean(np.abs(chunk)))})
                resid_rows.append({"direction": direction, "item": f"{rname}_std", "value": float(chunk.std())})

        r_v = resid.mean(2)
        scores_b2_tr = mdl.predict(scq.transform(tr[qkey]))
        resid_tr = (sign * Dtr) - pca.inverse_transform(scores_b2_tr).reshape(n_tr, nv, nt)
        resid_va = (sign * va["delta"]) - pca.inverse_transform(mdl.predict(scq.transform(va[qkey]))).reshape(-1, nv, nt)
        _, _, _, rhat_te, _ = fit_ridge_val(tr[qkey], resid_tr.mean(2), va[qkey], te[qkey], resid_va.mean(2))
        rq = metrics_flat(r_v, rhat_te)
        resid_rows.append({"direction": direction, "item": "residual_vel_marginal_q_r2", "value": rq["r2"]})
        resid_rows.append({"direction": direction, "item": "residual_vel_marginal_q_corr", "value": rq["corr"]})
        hats[direction]["resid"] = resid
        hats[direction]["resid_q"] = rq

        for act in CLASS_NAMES:
            m = te["action"] == act
            if m.sum() == 0:
                continue
            action_rows.append(
                {
                    "direction": direction,
                    "action": act,
                    "n": int(m.sum()),
                    "B2_mae": float(np.mean(np.abs(true_d[m] - b2_te[m]))),
                    "Oracle_mae": float(np.mean(np.abs(true_d[m] - oracle_te[m]))),
                    "oracle_gap": float(np.mean(np.abs(true_d[m] - b2_te[m])) - np.mean(np.abs(true_d[m] - oracle_te[m]))),
                }
            )

        oracle_rows.append(
            {
                "direction": direction,
                "oracle_r2": mo["r2"],
                "oracle_mae": mo["mae"],
                "oracle_corr": mo["corr"],
                "B2_r2": m2["r2"],
                "B2_mae": m2["mae"],
                "B2_corr": m2["corr"],
                "gap_r2": mo["r2"] - m2["r2"],
                "gap_mae": m2["mae"] - mo["mae"],
            }
        )

    write_csv(out / "transfer_prediction_2d.csv", ["direction", "model", "r2", "mae", "rmse", "corr", "k", "ridge_alpha"], transfer_rows)
    write_csv(out / "region_prediction_metrics.csv", ["direction", "region", "model", "n_pixels_per_pair", "r2", "mae", "rmse", "corr", "n", "note"], region_rows)
    write_csv(out / "conditioning_coefficients.csv", ["direction", "component", "factor", "std_coef", "abs_coef", "rank"], coef_rows)
    write_csv(out / "target_reconstruction.csv", ["direction", "item", "cosine", "corr", "mae", "ssim", "gain_vs_source"], recon_rows)
    write_csv(out / "radar_domain_prediction.csv", ["direction", "item", "frac_source_label", "frac_target_label"], radar_rows)
    write_csv(out / "activity_preservation.csv", ["direction", "item", "activity"], act_rows)
    write_csv(out / "residual_analysis.csv", ["direction", "item", "value"], resid_rows)
    write_csv(out / "action_residual.csv", ["direction", "action", "n", "B2_mae", "Oracle_mae", "oracle_gap"], action_rows)
    write_csv(out / "oracle_vs_predicted.csv", ["direction", "oracle_r2", "oracle_mae", "oracle_corr", "B2_r2", "B2_mae", "B2_corr", "gap_r2", "gap_mae"], oracle_rows)

    # gate
    def get_recon(direction, item):
        for r in recon_rows:
            if r["direction"] == direction and r["item"] == item:
                return r
        return {"cosine": 0.0}

    def get_act(direction, item):
        for r in act_rows:
            if r["direction"] == direction and r["item"] == item:
                return r["activity"]
        return 0.0

    gate_ok = True
    for d in ("24to77", "77to24"):
        mae_ok = hats[d]["m2"]["mae"] <= GATE_MAE_RATIO * hats[d]["m1"]["mae"]
        sim_ok = get_recon(d, "B2_predicted")["cosine"] >= get_recon(d, "B1_predicted")["cosine"] + GATE_SIM_GAIN
        act_ok = get_act(d, "B2_predicted") >= get_act(d, "source") - GATE_ACT_DROP
        hats[d]["gate"] = {"mae": mae_ok, "sim": sim_ok, "act": act_ok}
        if not (mae_ok and sim_ok and act_ok):
            gate_ok = False

    # oracle interpretation
    o_r2 = [hats[d]["mo"]["r2"] for d in ("24to77", "77to24")]
    gaps = [hats[d]["mo"]["r2"] - hats[d]["m2"]["r2"] for d in ("24to77", "77to24")]
    if min(o_r2) >= ORACLE_GOOD_R2 and min(gaps) >= COND_GAP_R2:
        interp, case, nxt = "CONDITIONING-LIMITED", "C2D-B", "SPATIAL / LATENT VARIABLE DATA NEEDED"
    elif max(o_r2) < ORACLE_POOR_R2:
        interp, case, nxt = "BASIS-LIMITED", "C2D-C", "STOP EMPIRICAL SENSOR TRANSLATION"
    elif min(o_r2) >= ORACLE_GOOD_R2 and gate_ok:
        interp, case, nxt = "CONDITIONING-LIMITED" if max(gaps) >= COND_GAP_R2 else "low-rank+q sufficient", "C2D-A" if gate_ok else "C2D-B", "CONDITIONAL MEASUREMENT MODEL VALIDATED" if gate_ok else "SPATIAL / LATENT VARIABLE DATA NEEDED"
    elif min(o_r2) >= ORACLE_POOR_R2:
        interp, case, nxt = "BOTH", "C2D-B" if min(o_r2) >= ORACLE_GOOD_R2 else "C2D-C", "SPATIAL / LATENT VARIABLE DATA NEEDED" if min(o_r2) >= ORACLE_GOOD_R2 else "STOP EMPIRICAL SENSOR TRANSLATION"
    else:
        interp, case, nxt = "BOTH", "C2D-C", "STOP EMPIRICAL SENSOR TRANSLATION"
    if gate_ok:
        case, nxt = "C2D-A", "CONDITIONAL MEASUREMENT MODEL VALIDATED"
        if min(o_r2) < ORACLE_GOOD_R2:
            interp = "BOTH"
    else:
        if min(o_r2) >= ORACLE_GOOD_R2 and min(gaps) >= COND_GAP_R2:
            interp, case, nxt = "CONDITIONING-LIMITED", "C2D-B", "SPATIAL / LATENT VARIABLE DATA NEEDED"
        elif max(o_r2) < ORACLE_POOR_R2:
            interp, case, nxt = "BASIS-LIMITED", "C2D-C", "STOP EMPIRICAL SENSOR TRANSLATION"
        else:
            interp, case, nxt = "BOTH", "C2D-B" if mean_ok(o_r2) else "C2D-C", "SPATIAL / LATENT VARIABLE DATA NEEDED" if mean_ok(o_r2) else "STOP EMPIRICAL SENSOR TRANSLATION"

    best_act = min((r for r in action_rows if r["direction"] == "24to77"), key=lambda r: r["B2_mae"], default={"action": ""})
    worst_act = max((r for r in action_rows if r["direction"] == "24to77"), key=lambda r: r["B2_mae"], default={"action": ""})

    pc_notes = {}
    for d in ("24to77", "77to24"):
        comps = pcas[d].components_.reshape(-1, nv, nt)
        pc_notes[d] = [describe_pc(comps[j], vgrid, f"PC{j+1}") for j in range(min(3, comps.shape[0]))]

    make_figures(fig, Dmean, Dstd, pcas, hats, vgrid, recon_rows, action_rows)
    answers = {
        "kind": "motion-conditioned 2D measurement model; not geometry-aware; not unseen-radar DG",
        "selected_k": {d: hats[d]["k"] for d in hats},
        "train_explained_var": {r["direction"]: r["train_explained_var"] for r in man_rows},
        "oracle": {d: hats[d]["mo"] for d in hats},
        "transfer": {d: {"B0": hats[d]["m0"], "B1": hats[d]["m1"], "B2": hats[d]["m2"]} for d in hats},
        "gate": "PASS" if gate_ok else "FAIL",
        "gate_detail": {d: hats[d]["gate"] for d in hats},
        "interpretation": interp,
        "case": case,
        "next_step": nxt,
        "best_action": best_act.get("action"),
        "worst_action": worst_act.get("action"),
        "pc_notes": pc_notes,
        "residual_q": {d: hats[d]["resid_q"] for d in hats},
        "har_trained": False,
        "geometry_aware": "BLOCKED",
    }
    (out / "answers.json").write_text(json.dumps(answers, indent=2, default=str), encoding="utf-8")
    write_report(out, answers, hats, man_rows, recon_rows, radar_rows, act_rows, resid_rows, action_rows, coef_rows, pc_notes, interp, case, nxt, gate_ok)
    print("phase10g done", json.dumps(answers, indent=2, default=str)[:4000], flush=True)


def mean_ok(xs):
    return float(np.mean(xs)) >= ORACLE_GOOD_R2


def make_figures(fig, Dmean, Dstd, pcas, hats, vgrid, recon_rows, action_rows):
    plt = _mpl()
    nv, nt = Dmean.shape

    def im(ax, z, title, vmin=None, vmax=None):
        lim = float(np.percentile(np.abs(z), 98)) + EPS
        ax.imshow(z, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-(vmax or lim), vmax=(vmax or lim), extent=[0, 1, V_LO, V_HI])
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("tau")
        ax.set_ylabel("v")

    fig1, ax = plt.subplots(figsize=(7.2, 2.6))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    boxes = [(0.2, 1.1, "U1 24/77\nlog-energy"), (2.3, 1.1, "Delta\nX77-X24"), (4.4, 1.1, "TRAIN PCA\nK=2/5/10/20"), (6.5, 1.1, "q_source\nRidge a_k"), (8.4, 1.1, "Delta_hat\n2D field")]
    for x, y, t in boxes:
        ax.add_patch(plt.Rectangle((x, y), 1.7, 1.4, fill=False))
        ax.text(x + 0.85, y + 0.7, t, ha="center", va="center", fontsize=8)
    for x in (1.9, 4.0, 6.1, 8.2):
        ax.annotate("", xy=(x + 0.25, 1.8), xytext=(x, 1.8), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Motion-conditioned 2D measurement model (empirical, not geometry-aware)")
    fig1.savefig(fig / "fig1_schematic.png", dpi=130, bbox_inches="tight")
    plt.close(fig1)

    fig2, axs = plt.subplots(1, 2, figsize=(8.4, 3.4))
    im(axs[0], Dmean, "mean Delta (TRAIN)")
    im(axs[1], Dstd, "std Delta (TRAIN)")
    fig2.savefig(fig / "fig2_mean_delta.png", dpi=130, bbox_inches="tight")
    plt.close(fig2)

    comps = pcas["24to77"].components_.reshape(-1, nv, nt)
    fig3, axs = plt.subplots(1, 3, figsize=(10.2, 3.2))
    for j, ax in enumerate(axs):
        if j < comps.shape[0]:
            im(ax, comps[j], f"PC{j+1}")
        else:
            ax.axis("off")
    fig3.savefig(fig / "fig3_pcs.png", dpi=130, bbox_inches="tight")
    plt.close(fig3)

    h = hats["24to77"]
    i0 = 0
    fig4, axs = plt.subplots(1, 4, figsize=(12.0, 3.0))
    im(axs[0], h["true"][i0], "true Delta")
    im(axs[1], h["B0"][i0], "B0")
    im(axs[2], h["B1"][i0], "B1 D(v|q)")
    im(axs[3], h["B2"][i0], "B2 low-rank")
    fig4.savefig(fig / "fig4_true_vs_models.png", dpi=130, bbox_inches="tight")
    plt.close(fig4)

    fig5, ax = plt.subplots(figsize=(6.4, 3.4))
    labs = ["B0", "B1", "B2", "Oracle"]
    x = np.arange(len(labs))
    ax.bar(x - 0.18, [hats["24to77"][k]["mae"] for k in ("m0", "m1", "m2", "mo")], 0.36, label="24→77")
    ax.bar(x + 0.18, [hats["77to24"][k]["mae"] for k in ("m0", "m1", "m2", "mo")], 0.36, label="77→24")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("2D Delta MAE")
    ax.legend()
    fig5.savefig(fig / "fig5_errors.png", dpi=130, bbox_inches="tight")
    plt.close(fig5)

    fig6, axs = plt.subplots(1, 3, figsize=(10.2, 3.2))
    im(axs[0], h["src"][i0], "source 24")
    im(axs[1], h["pred_b2"][i0], "B2 predicted 77")
    im(axs[2], h["tgt"][i0], "true 77")
    fig6.savefig(fig / "fig6_source_pred_target.png", dpi=130, bbox_inches="tight")
    plt.close(fig6)

    fig7, ax = plt.subplots(figsize=(6.6, 3.4))
    items = ["source", "B1_predicted", "B2_predicted", "Oracle_predicted"]
    for d in ("24to77", "77to24"):
        ys = [next(r["cosine"] for r in recon_rows if r["direction"] == d and r["item"] == it) for it in items]
        ax.plot(items, ys, marker="o", label=d)
    ax.tick_params(axis="x", rotation=15)
    ax.set_ylabel("cosine vs true target")
    ax.legend()
    fig7.savefig(fig / "fig7_recon_sim.png", dpi=130, bbox_inches="tight")
    plt.close(fig7)

    fig8, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.bar([0, 1], [hats["24to77"]["mo"]["r2"], hats["77to24"]["mo"]["r2"]], 0.35, label="Oracle R2")
    ax.bar([0.35, 1.35], [hats["24to77"]["m2"]["r2"], hats["77to24"]["m2"]["r2"]], 0.35, label="B2 R2")
    ax.set_xticks([0.18, 1.18])
    ax.set_xticklabels(["24→77", "77→24"])
    ax.legend()
    fig8.savefig(fig / "fig8_oracle_vs_b2.png", dpi=130, bbox_inches="tight")
    plt.close(fig8)

    fig9, ax = plt.subplots(figsize=(5.0, 3.6))
    im(ax, h["resid"].mean(0), "mean residual B2 (TEST 24→77)")
    fig9.savefig(fig / "fig9_residual.png", dpi=130, bbox_inches="tight")
    plt.close(fig9)

    fig10, ax = plt.subplots(figsize=(7.2, 3.6))
    ar = [r for r in action_rows if r["direction"] == "24to77"]
    x = np.arange(len(ar))
    ax.bar(x - 0.18, [r["B2_mae"] for r in ar], 0.36, label="B2")
    ax.bar(x + 0.18, [r["Oracle_mae"] for r in ar], 0.36, label="Oracle")
    ax.set_xticks(x)
    ax.set_xticklabels([r["action"] for r in ar], rotation=20)
    ax.legend()
    fig10.savefig(fig / "fig10_action_gap.png", dpi=130, bbox_inches="tight")
    plt.close(fig10)


def write_report(out, ans, hats, man_rows, recon_rows, radar_rows, act_rows, resid_rows, action_rows, coef_rows, pc_notes, interp, case, nxt, gate_ok):
    lines = [
        "# Phase 10G — Motion-conditioned 2D measurement model",
        "",
        "经验 paired difference，不是正演/反演，不是 geometry-aware，不是 unseen-radar DG。",
        "未训 ResNet / translator。未改 Phase 1–10F。",
        "",
        f"Selected K: {ans['selected_k']}",
        f"TRAIN explained variance: {ans['train_explained_var']}",
        "",
        "## Oracle (TEST 投影到 TRAIN 基，不是部署模型)",
        "",
    ]
    for d in ("24to77", "77to24"):
        mo = hats[d]["mo"]
        lines.append(f"- {d}: R² {mo['r2']:.3f}  MAE {mo['mae']:.3f}  corr {mo['corr']:.3f}")
    lines += ["", "## Transfer (true Delta vs hat)", ""]
    for d in ("24to77", "77to24"):
        lines.append(f"### {d}")
        for name, key in (("B0", "m0"), ("B1", "m1"), ("B2", "m2")):
            m = hats[d][key]
            lines.append(f"- {name}: R² {m['r2']:.3f}  MAE {m['mae']:.3f}  corr {m['corr']:.3f}")
        lines.append("")
    lines += ["## PC 结构（只据图，不写肢体）", ""]
    for d, notes in pc_notes.items():
        for i, n in enumerate(notes, 1):
            lines.append(f"- {d} PC{i}: {n}")
    lines += ["", f"## Oracle gap 解释: **{interp}**", ""]
    if interp == "CONDITIONING-LIMITED":
        lines.append("未观测的 acquisition 变量可能还需要。候选（不是已证实原因）：range / distance / aspect / orientation / spatial scattering。")
        lines.append("不能声称 geometry 造成了残差：几何元数据 UNAVAILABLE。")
    lines += ["", f"## Gate: **{'PASS' if gate_ok else 'FAIL'}**", f"CASE **{case}**", f"NEXT: **{nxt}**", "", "未自动进入 Phase 11。", ""]
    (out / "phase10_conditional_2d_measurement_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
