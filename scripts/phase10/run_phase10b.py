#!/usr/bin/env python3
"""Phase 10B: known-radar sensor calibration audit. No GAN. ResNet only if VAL gate passes."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase8.run_r2_benchmark import CLASS_NAMES, RADARS, SEED, load_r2_pack  # noqa: E402
from phase10.extract_factors import FACTOR_NAMES, extract_one  # noqa: E402

OUT = ROOT / "results" / "phase10_sensor_calibration"
FIG = OUT / "figures"
P10A = ROOT / "results" / "phase10_motion_factors"
EPS = 1e-6
GATE_RID = 0.80
GATE_ACT = 0.75


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def pool16(X):
    import torch

    t = torch.from_numpy(np.asarray(X, np.float32))
    if t.ndim == 3:
        t = t.unsqueeze(1)
    return torch.nn.functional.adaptive_avg_pool2d(t, (16, 16)).flatten(1).numpy()


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


def paired(pack):
    by = defaultdict(dict)
    for i in range(len(pack["y"])):
        by[str(pack["timestamp"][i])][pack["radar"][i]] = i
    return {ts: rec for ts, rec in by.items() if all(r in rec for r in RADARS)}


def fit_c1(X):
    return float(X.mean()), float(X.std()) + EPS


def apply_c1(X, mu, sd):
    return ((X - mu) / sd).astype(np.float32)


def fit_c2(X):
    return X.mean(axis=(0, 2)).astype(np.float64), X.std(axis=(0, 2)).astype(np.float64) + EPS


def apply_c2(X, mu, sd):
    return ((X - mu[None, :, None]) / sd[None, :, None]).astype(np.float32)


def fit_c3(X):
    mu = float(X.mean())
    dop = X.mean(axis=(0, 2))
    tmp = X.mean(axis=(0, 1))
    resid = X - dop[None, :, None] - tmp[None, None, :] + mu
    return mu, dop, tmp, float(resid.std()) + EPS


def apply_c3(X, mu, dop, tmp, sd):
    return ((X - dop[None, :, None] - tmp[None, None, :] + mu) / sd).astype(np.float32)


def fit_c4(Xsrc, Xref):
    nf = Xsrc.shape[1]
    a = np.ones(nf, np.float64)
    b = np.zeros(nf, np.float64)
    for f in range(nf):
        x = Xsrc[:, f, :].ravel()
        y = Xref[:, f, :].ravel()
        A = np.stack([x, np.ones_like(x)], 1)
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        if np.isfinite(coef).all():
            a[f], b[f] = coef
    return a, b


def apply_c4(X, a, b):
    return (a[None, :, None] * X + b[None, :, None]).astype(np.float32)


def cosine_sim(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
    return np.sum(a * b, axis=1)


def pair_stats(pack, X, pairs, split_name="test"):
    ts_list = [t for t, rec in pairs.items() if pack["split"][rec["10GHz"]] == split_name]
    P = pool16(X)
    same, diff = [], []
    rng = np.random.RandomState(42)
    for t in ts_list:
        rec = pairs[t]
        z = {r: P[rec[r]][None] for r in RADARS}
        same.extend(
            [
                float(cosine_sim(z["10GHz"], z["24GHz"])[0]),
                float(cosine_sim(z["10GHz"], z["77GHz"])[0]),
                float(cosine_sim(z["24GHz"], z["77GHz"])[0]),
            ]
        )
    for _ in range(max(len(same), 1)):
        if len(ts_list) < 2:
            break
        a, b = rng.choice(ts_list, 2, replace=False)
        r1, r2 = rng.choice(RADARS, 2, replace=False)
        diff.append(float(cosine_sim(P[pairs[a][r1]][None], P[pairs[b][r2]][None])[0]))
    return {
        "same_sim": float(np.mean(same)) if same else "",
        "diff_sim": float(np.mean(diff)) if diff else "",
        "same_dist": float(np.mean(1.0 - np.asarray(same))) if same else "",
        "n_same": len(same),
    }


def apply_all(pack, params, kind):
    X = pack["X"]
    out = np.empty_like(X)
    for r in RADARS:
        m = pack["radar"] == r
        p = params[r]
        if kind == "C0":
            out[m] = X[m]
        elif kind == "C1":
            out[m] = apply_c1(X[m], *p)
        elif kind == "C2":
            out[m] = apply_c2(X[m], *p)
        elif kind == "C3":
            out[m] = apply_c3(X[m], *p)
        elif kind == "C4":
            out[m] = apply_c4(X[m], *p)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pack = load_r2_pack()
    X0 = pack["X"]
    split, radar, action = pack["split"], pack["radar"], pack["action"]
    tr, va, te = split == "train", split == "val", split == "test"
    pairs = paired(pack)
    print("loaded", X0.shape, flush=True)

    # --- bias decomposition (TRAIN) ---
    bias_rows, curve_rows = [], []
    fd = np.linspace(-256.0, 255.875, X0.shape[1])
    for r in RADARS:
        xt = X0[(radar == r) & tr]
        flat = xt.reshape(len(xt), -1)
        e_f = xt.mean(axis=(0, 2))
        e_t = xt.mean(axis=(0, 1))
        pos = e_f[fd > 0].sum()
        neg = e_f[fd < 0].sum()
        occ = float((np.abs(xt) > 0.25).mean())
        bias_rows.append(
            {
                "radar": r,
                "n_train": int(len(xt)),
                "mean": float(xt.mean()),
                "std": float(xt.std()),
                "median": float(np.median(xt)),
                "p10": float(np.percentile(xt, 10)),
                "p90": float(np.percentile(xt, 90)),
                "dynamic_range": float(xt.max() - xt.min()),
                "sparsity_hoyer": float(
                    (np.sqrt(xt.size) - np.abs(xt).sum() / (np.linalg.norm(xt) + EPS)) / (np.sqrt(xt.size) - 1)
                ),
                "occupancy_abs_gt_0p25": occ,
                "pos_doppler_mass": float(pos / (pos + neg + EPS)),
                "neg_doppler_mass": float(neg / (pos + neg + EPS)),
                "doppler_profile_std": float(e_f.std()),
                "temporal_profile_std": float(e_t.std()),
                "doppler_profile_range": float(e_f.max() - e_f.min()),
                "temporal_profile_range": float(e_t.max() - e_t.min()),
            }
        )
        mu_f, sd_f = fit_c2(xt)
        for i, f in enumerate(fd):
            curve_rows.append(
                {
                    "radar": r,
                    "doppler_hz": float(f),
                    "train_mean": float(mu_f[i]),
                    "train_std": float(sd_f[i]),
                    "occupancy": float((np.abs(xt[:, i, :]) > 0.25).mean()),
                }
            )
    write_csv(
        OUT / "sensor_bias_decomposition.csv",
        list(bias_rows[0].keys()),
        bias_rows,
    )

    # --- Radar-ID ablation ---
    gstat, dop, tmpp = [], [], []
    for i in range(len(X0)):
        x = X0[i]
        gstat.append([x.mean(), x.std(), np.median(x), np.percentile(x, 10), np.percentile(x, 90), x.max() - x.min()])
        dop.append(x.mean(axis=1))
        tmpp.append(x.mean(axis=0))
    gstat, dop, tmpp = np.asarray(gstat), np.asarray(dop), np.asarray(tmpp)
    fac_rows = read_csv(P10A / "motion_factors.csv")
    fmap = {(r["timestamp"], r["radar"]): [float(r[k]) for k in FACTOR_NAMES] for r in fac_rows}
    F = np.stack([fmap[(str(pack["timestamp"][i]), pack["radar"][i])] for i in range(len(X0))])
    P = pool16(X0)
    ablate = []
    for name, feat in (
        ("global_stats", gstat),
        ("doppler_marginal", dop),
        ("temporal_marginal", tmpp),
        ("motion_factors", F),
        ("r2_16x16", P),
    ):
        rid = probe(feat[tr], radar[tr], feat[te], radar[te])
        act = probe(feat[tr], action[tr], feat[te], action[te])
        ablate.append(
            {
                "features": name,
                "n_dim": feat.shape[1],
                "radar_id": rid["accuracy"],
                "activity": act["accuracy"],
                "note": "logistic; TRAIN fit / TEST eval; known-radar identity available as label only",
            }
        )
    write_csv(OUT / "radar_id_ablation.csv", ["features", "n_dim", "radar_id", "activity", "note"], ablate)

    # --- fit calibrations on TRAIN ---
    params = {"C0": {r: None for r in RADARS}, "C1": {}, "C2": {}, "C3": {}, "C4": {}}
    for r in RADARS:
        xt = X0[(radar == r) & tr]
        params["C1"][r] = fit_c1(xt)
        params["C2"][r] = fit_c2(xt)
        params["C3"][r] = fit_c3(xt)
    # C4: map each radar to 77 using TRAIN pairs
    ref = "77GHz"
    for r in RADARS:
        if r == ref:
            params["C4"][r] = (np.ones(X0.shape[1]), np.zeros(X0.shape[1]))
            continue
        src, dst = [], []
        for t, rec in pairs.items():
            if pack["split"][rec[r]] != "train":
                continue
            src.append(X0[rec[r]])
            dst.append(X0[rec[ref]])
        params["C4"][r] = fit_c4(np.stack(src), np.stack(dst))

    # save C2/C4 curves
    for r in RADARS:
        mu, sd = params["C2"][r]
        a, b = params["C4"][r]
        for i, f in enumerate(fd):
            curve_rows[i + RADARS.index(r) * len(fd)].update(
                {"c2_mu": float(mu[i]), "c2_sigma": float(sd[i]), "c4_a": float(a[i]), "c4_b": float(b[i])}
            )
    write_csv(
        OUT / "calibration_curves.csv",
        ["radar", "doppler_hz", "train_mean", "train_std", "occupancy", "c2_mu", "c2_sigma", "c4_a", "c4_b"],
        curve_rows,
    )

    Xc = {k: apply_all(pack, params[k], k) for k in ("C0", "C1", "C2", "C3", "C4")}

    probe_rows, sim_rows, trade = [], [], []
    val_scores = {}
    for k in ("C0", "C1", "C2", "C3", "C4"):
        pk = pool16(Xc[k])
        act_te = probe(pk[tr], action[tr], pk[te], action[te])
        rid_te = probe(pk[tr], radar[tr], pk[te], radar[te])
        act_va = probe(pk[tr], action[tr], pk[va], action[va])
        rid_va = probe(pk[tr], radar[tr], pk[va], radar[va])
        sim = pair_stats(pack, Xc[k], pairs, "test")
        probe_rows.append(
            {
                "calibration": k,
                "kind": "known_radar_train_stats",
                "activity_test": act_te["accuracy"],
                "activity_val": act_va["accuracy"],
                "activity_f1_test": act_te["macro_f1"],
                "radar_id_test": rid_te["accuracy"],
                "radar_id_val": rid_va["accuracy"],
                "gate_pass": int(rid_va["accuracy"] < GATE_RID and act_va["accuracy"] > GATE_ACT),
            }
        )
        sim_rows.append({"calibration": k, **sim})
        trade.append({"representation": k, "radar_id": rid_te["accuracy"], "activity_probe": act_te["accuracy"], "note": "16x16 logistic; known-radar calibration"})
        val_scores[k] = (act_va["accuracy"] - rid_va["accuracy"], act_va["accuracy"], -rid_va["accuracy"])
        print(k, "act_te", act_te["accuracy"], "rid_te", rid_te["accuracy"], "val", act_va["accuracy"], rid_va["accuracy"], flush=True)

    write_csv(
        OUT / "calibration_probe_results.csv",
        ["calibration", "kind", "activity_test", "activity_val", "activity_f1_test", "radar_id_test", "radar_id_val", "gate_pass"],
        probe_rows,
    )
    write_csv(OUT / "calibration_similarity.csv", ["calibration", "same_sim", "diff_sim", "same_dist", "n_same"], sim_rows)
    # keep 10A points on the map
    for r in read_csv(P10A / "information_tradeoff.csv"):
        trade.append({"representation": r["representation"], "radar_id": r["radar_id"], "activity_probe": r["activity_probe"], "note": r.get("note", "Phase 10A")})
    write_csv(OUT / "information_tradeoff.csv", ["representation", "radar_id", "activity_probe", "note"], trade)

    passed = [r["calibration"] for r in probe_rows if int(r["gate_pass"]) == 1]
    if passed:
        c_best = max(passed, key=lambda k: val_scores[k])
        gate = "PASS"
    else:
        c_best = max(("C0", "C1", "C2", "C3", "C4"), key=lambda k: val_scores[k])
        gate = "FAIL"
    print("gate", gate, "c_best", c_best, "passed", passed, flush=True)

    # factors on C_best (diagnostic even if gate fails)
    fac_cal = []
    mats = []
    for i in range(len(X0)):
        rec = extract_one(Xc[c_best][i])
        mats.append([rec[n] for n in FACTOR_NAMES])
        fac_cal.append({"timestamp": str(pack["timestamp"][i]), "radar": radar[i], "action": action[i], "split": split[i], **rec})
    Fc = np.asarray(mats)
    act_f = probe(Fc[tr], action[tr], Fc[te], action[te])
    rid_f = probe(Fc[tr], radar[tr], Fc[te], radar[te])
    write_csv(
        OUT / "calibrated_factor_probes.csv",
        ["calibration", "activity", "radar_id", "note"],
        [
            {
                "calibration": c_best,
                "activity": act_f["accuracy"],
                "radar_id": rid_f["accuracy"],
                "note": "same 38 factors as Phase 10A; extracted on calibrated R2-D",
            }
        ],
    )

    # calibration budget: unlabeled C2 on one target, activity probe from other two
    bud_rows = []
    rng = np.random.RandomState(42)
    budgets = [1, 5, 10, 20, "all"]
    for tgt in RADARS:
        srcs = [r for r in RADARS if r != tgt]
        idx_tgt_tr = np.where((radar == tgt) & tr)[0]
        ts_tgt = sorted({str(pack["timestamp"][i]) for i in idx_tgt_tr})
        rng.shuffle(ts_tgt)
        for b in budgets:
            k = len(ts_tgt) if b == "all" else min(int(b), len(ts_tgt))
            use_ts = set(ts_tgt[:k])
            use_i = [i for i in idx_tgt_tr if str(pack["timestamp"][i]) in use_ts]
            mu, sd = fit_c2(X0[use_i])
            # sources: own C2 from full train; target: budget C2
            Xb = np.empty_like(X0)
            for r in srcs:
                Xb[radar == r] = apply_c2(X0[radar == r], *params["C2"][r])
            Xb[radar == tgt] = apply_c2(X0[radar == tgt], mu, sd)
            pb = pool16(Xb)
            src_tr = np.isin(radar, srcs) & tr
            tgt_te = (radar == tgt) & te
            act = probe(pb[src_tr], action[src_tr], pb[tgt_te], action[tgt_te])
            rid = probe(pb[tr], radar[tr], pb[te], radar[te])
            bud_rows.append(
                {
                    "target": tgt,
                    "budget": b,
                    "n_unlabeled": k,
                    "activity_target_test": act["accuracy"],
                    "radar_id_all_test": rid["accuracy"],
                    "note": "C2 from unlabeled target TRAIN only; sources use full TRAIN C2; not DG",
                }
            )
            print("budget", tgt, b, "act", act["accuracy"], "rid", rid["accuracy"], flush=True)
    write_csv(
        OUT / "calibration_budget.csv",
        ["target", "budget", "n_unlabeled", "activity_target_test", "radar_id_all_test", "note"],
        bud_rows,
    )

    # figures
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.bar([r["features"] for r in ablate], [r["radar_id"] for r in ablate])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(FIG / "fig1_radar_id_ablation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    x = np.arange(5)
    labs = ["C0", "C1", "C2", "C3", "C4"]
    ax.bar(x - 0.18, [next(r for r in probe_rows if r["calibration"] == k)["activity_test"] for k in labs], 0.36, label="Activity")
    ax.bar(x + 0.18, [next(r for r in probe_rows if r["calibration"] == k)["radar_id_test"] for k in labs], 0.36, label="Radar-ID")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(FIG / "fig2_calibration_probes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for row in trade:
        try:
            ax.scatter(float(row["radar_id"]), float(row["activity_probe"]), s=36)
            ax.annotate(row["representation"], (float(row["radar_id"]), float(row["activity_probe"])), fontsize=6)
        except Exception:
            continue
    ax.set_xlabel("Radar-ID")
    ax.set_ylabel("Activity Probe")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Information Trade-off (ideal: upper left)")
    fig.savefig(FIG / "fig3_information_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    for r in RADARS:
        mu, _sd = params["C2"][r]
        ax.plot(fd, mu, label=r)
    ax.set_xlabel("Doppler Hz")
    ax.set_ylabel("TRAIN mean S")
    ax.legend()
    ax.set_title("C2 Doppler envelope (known-radar TRAIN)")
    fig.savefig(FIG / "fig4_c2_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.4))
    for r in ("10GHz", "24GHz"):
        a, b = params["C4"][r]
        axes[0].plot(fd, a, label=r)
        axes[1].plot(fd, b, label=r)
    axes[0].set_title("C4 a(f) → 77")
    axes[1].set_title("C4 b(f) → 77")
    for ax in axes:
        ax.legend()
        ax.set_xlabel("Doppler Hz")
    fig.savefig(FIG / "fig5_c4_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar(labs, [next(r for r in sim_rows if r["calibration"] == k)["same_sim"] for k in labs])
    ax.set_ylabel("same-acquisition cosine similarity")
    fig.savefig(FIG / "fig6_pair_similarity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    p0 = next(r for r in probe_rows if r["calibration"] == "C0")
    pb = next(r for r in probe_rows if r["calibration"] == c_best)
    s0 = next(r for r in sim_rows if r["calibration"] == "C0")
    sb = next(r for r in sim_rows if r["calibration"] == c_best)
    rid0, ridb = float(p0["radar_id_test"]), float(pb["radar_id_test"])
    act0, actb = float(p0["activity_test"]), float(pb["activity_test"])
    if ridb <= rid0 - 0.15 and actb >= act0 - 0.08:
        case = "CAL-A"
        nxt = "CALIBRATED SHARED ENCODER"
    elif ridb >= rid0 - 0.05:
        case = "CAL-C"
        nxt = "MEASUREMENT-MODEL CALIBRATION"
    else:
        case = "CAL-B"
        nxt = "MEASUREMENT-MODEL CALIBRATION"
    if gate == "FAIL" and case == "CAL-A":
        nxt = "CALIBRATED SHARED ENCODER"
    if gate == "FAIL" and case != "CAL-A":
        nxt = "MEASUREMENT-MODEL CALIBRATION" if case != "CAL-C" else "STOP CROSS-RADAR ALIGNMENT"
        if case == "CAL-C":
            nxt = "MEASUREMENT-MODEL CALIBRATION"

    strongest = max(ablate, key=lambda r: r["radar_id"] if r["features"] != "r2_16x16" else -1)
    answers = {
        "gate": gate,
        "c_best": c_best,
        "passed": passed,
        "har_trained": False,
        "case": case,
        "next_step": nxt,
        "c0_radar": rid0,
        "c0_activity": act0,
        "best_radar": ridb,
        "best_activity": actb,
        "best_same_sim": sb["same_sim"],
        "c0_same_sim": s0["same_sim"],
        "factor_activity": act_f["accuracy"],
        "factor_radar": rid_f["accuracy"],
        "ablation": {r["features"]: r["radar_id"] for r in ablate},
        "kind": "KNOWN-RADAR CALIBRATION",
    }
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    write_report(bias_rows, ablate, probe_rows, sim_rows, bud_rows, answers, c_best, gate, case, nxt)
    print("phase10b done", answers, flush=True)
    if gate == "PASS":
        print("GATE PASS but this script does not auto-train ResNet; report only.", flush=True)


def write_report(bias, ablate, probes, sims, budget, ans, c_best, gate, case, nxt) -> None:
    lines = [
        "# Phase 10B Sensor Calibration Audit",
        "",
        "未覆盖 Phase 1–10A。未使用 GAN / 像素翻译网络。",
        "C1–C4 都是 **KNOWN-RADAR CALIBRATION**：要用该雷达 TRAIN 统计。",
        "不能叫 Universal，也不能叫 unseen-radar DG。",
        "Phase 8C LORO 仍是 strict DG 对照，本阶段不改。",
        "",
        "## 雷达指纹来自哪里",
        "",
    ]
    for r in ablate:
        lines.append(f"- {r['features']}: Radar-ID {r['radar_id']:.3f}  Activity {r['activity']:.3f}")
    lines += ["", "TRAIN 全局统计：", ""]
    for r in bias:
        lines.append(
            f"- {r['radar']}: mean={r['mean']:.3f} std={r['std']:.3f} "
            f"doppler_profile_std={r['doppler_profile_std']:.3f} pos_mass={r['pos_doppler_mass']:.3f}"
        )
    lines += ["", "## Calibration probes（16×16 logistic，TRAIN 拟合）", ""]
    for r in probes:
        lines.append(
            f"- {r['calibration']}: TEST Activity {float(r['activity_test']):.3f}  "
            f"Radar-ID {float(r['radar_id_test']):.3f}  "
            f"VAL Activity {float(r['activity_val']):.3f} Radar-ID {float(r['radar_id_val']):.3f}  "
            f"gate={'PASS' if int(r['gate_pass']) else 'FAIL'}"
        )
    lines += ["", "## Pair similarity（TEST，16×16 cosine）", ""]
    for r in sims:
        lines.append(f"- {r['calibration']}: same {r['same_sim']}  diff {r['diff_sim']}")
    lines += [
        "",
        f"## Gate（工程：VAL Radar-ID < 0.80 且 Activity > 0.75）",
        "",
        f"**{gate}**  诊断用 C_best = {c_best}",
        "HAR trained: **NO**" if gate == "FAIL" else "HAR：门通过，本阶段仍不自动开训新网络。",
        "",
        f"校准后 38 factors（{c_best}）：Activity {ans['factor_activity']:.3f}  Radar-ID {ans['factor_radar']:.3f}",
        "",
        "## Unlabeled target calibration budget",
        "",
        "C2 用目标雷达 TRAIN 无标签样本估剖面。源雷达用自己的满 TRAIN C2。**不是 DG。**",
        "",
    ]
    for r in budget:
        lines.append(
            f"- target {r['target']} budget={r['budget']} n={r['n_unlabeled']}  "
            f"target HAR probe {float(r['activity_target_test']):.3f}  "
            f"Radar-ID {float(r['radar_id_all_test']):.3f}"
        )
    lines += [
        "",
        f"## CASE **{case}**",
        f"下一步：**{nxt}**",
        "",
        "CAL-A：简单校准就能把 Radar-ID 压下去且动作还在。",
        "CAL-B：Radar-ID 降一点，动作也掉。",
        "CAL-C：Radar-ID 基本不降。",
        "",
        "未自动开始 Calibrated Shared Motion Encoder。",
    ]
    (OUT / "phase10_sensor_calibration_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
