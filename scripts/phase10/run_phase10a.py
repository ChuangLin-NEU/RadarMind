#!/usr/bin/env python3
"""Phase 10A: motion-factor audit. Linear probes only. No new deep training."""
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
from phase10.extract_factors import FACTOR_NAMES, LOW_ABS_HZ, MID_ABS_HZ, extract_one  # noqa: E402

OUT = ROOT / "results" / "phase10_motion_factors"
FIG = OUT / "figures"
P9 = ROOT / "results" / "phase9_radarbridge_v2"
K_SUBSET = 8
EPS = 1e-8


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def write_csv(path: Path, fields, rows) -> None:
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


def eta2(values: np.ndarray, groups) -> float:
    y = np.asarray(values, dtype=np.float64)
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


def probe(Xtr, ytr, Xte, yte):
    if len(set(map(str, ytr))) < 2:
        return {"accuracy": None, "macro_f1": None}
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=2000, random_state=SEED)
    clf.fit(sc.fit_transform(Xtr), ytr)
    pred = clf.predict(sc.transform(Xte))
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def downsample16(X):
    t = np.asarray(X, np.float32)
    if t.ndim == 3:
        tt = __import__("torch")
        return tt.nn.functional.adaptive_avg_pool2d(tt.from_numpy(t).unsqueeze(1), (16, 16)).flatten(1).numpy()
    return t.reshape(len(t), -1)


def cosine_dist(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + EPS)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + EPS)
    return 1.0 - np.sum(a * b, axis=1)


def collect_factors(pack) -> tuple[np.ndarray, list]:
    rows = []
    mats = []
    for i in range(len(pack["y"])):
        rec = extract_one(pack["X"][i])
        mats.append([rec[k] for k in FACTOR_NAMES])
        rows.append(
            {
                "timestamp": str(pack["timestamp"][i]),
                "action": pack["action"][i],
                "split": pack["split"][i],
                "radar": pack["radar"][i],
                **{k: rec[k] for k in FACTOR_NAMES},
            }
        )
    return np.asarray(mats, dtype=np.float64), rows


def train_norm(F, split, radar, mode: str):
    tr = split == "train"
    out = np.zeros_like(F)
    if mode == "raw":
        return F.copy()
    if mode == "global":
        mu = F[tr].mean(axis=0)
        sd = F[tr].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        return (F - mu) / sd
    for r in RADARS:
        m = radar == r
        mu = F[m & tr].mean(axis=0)
        sd = F[m & tr].std(axis=0)
        sd = np.where(sd < 1e-8, 1.0, sd)
        out[m] = (F[m] - mu) / sd
    return out


def paired_index(pack):
    by = defaultdict(dict)
    for i in range(len(pack["y"])):
        by[str(pack["timestamp"][i])][pack["radar"][i]] = i
    return {ts: rec for ts, rec in by.items() if all(r in rec for r in RADARS)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    pack = load_r2_pack()
    F, rows = collect_factors(pack)
    write_csv(OUT / "motion_factors.csv", ["timestamp", "action", "split", "radar"] + FACTOR_NAMES, rows)
    split, radar, action, ts = pack["split"], pack["radar"], pack["action"], pack["timestamp"].astype(str)
    tr, te = split == "train", split == "test"
    Fg = train_norm(F, split, radar, "global")
    Fp = train_norm(F, split, radar, "per_radar")
    pairs = paired_index(pack)
    train_ts = [t for t, rec in pairs.items() if pack["split"][rec["10GHz"]] == "train"]

    stab_rows = []
    for j, name in enumerate(FACTOR_NAMES):
        cvs, d1024, d1077, d2477, corr_pairs = [], [], [], [], []
        v10 = []
        v24 = []
        v77 = []
        for t in train_ts:
            rec = pairs[t]
            a = F[rec["10GHz"], j]
            b = F[rec["24GHz"], j]
            c = F[rec["77GHz"], j]
            mu = np.mean([a, b, c])
            cvs.append(float(np.std([a, b, c]) / (abs(mu) + EPS)))
            d1024.append(abs(a - b))
            d1077.append(abs(a - c))
            d2477.append(abs(b - c))
            v10.append(a)
            v24.append(b)
            v77.append(c)
        def _corr(x, y):
            if np.std(x) < 1e-12 or np.std(y) < 1e-12:
                return 0.0
            return float(np.corrcoef(x, y)[0, 1])
        intra_cv = float(np.mean(cvs))
        e_act = eta2(F[tr, j], action[tr])
        e_rad = eta2(F[tr, j], radar[tr])
        score = e_act / (intra_cv + 1e-6)
        stab_rows.append(
            {
                "factor": name,
                "intra_acq_cv": intra_cv,
                "corr_10_24": _corr(v10, v24),
                "corr_10_77": _corr(v10, v77),
                "corr_24_77": _corr(v24, v77),
                "absdiff_10_24": float(np.mean(d1024)),
                "absdiff_10_77": float(np.mean(d1077)),
                "absdiff_24_77": float(np.mean(d2477)),
                "eta2_activity": e_act,
                "eta2_radar": e_rad,
                "motion_signal_strength": e_act,
                "radar_domain_variation": intra_cv,
                "stability_score": score,
            }
        )
    stab_rows.sort(key=lambda r: (r["intra_acq_cv"], -r["eta2_activity"]))
    for i, r in enumerate(stab_rows):
        r["rank_radar_stable"] = i + 1
    write_csv(
        OUT / "motion_factor_stability.csv",
        [
            "factor",
            "rank_radar_stable",
            "intra_acq_cv",
            "corr_10_24",
            "corr_10_77",
            "corr_24_77",
            "absdiff_10_24",
            "absdiff_10_77",
            "absdiff_24_77",
            "eta2_activity",
            "eta2_radar",
            "motion_signal_strength",
            "radar_domain_variation",
            "stability_score",
        ],
        stab_rows,
    )
    write_csv(
        OUT / "factor_effect_sizes.csv",
        ["factor", "eta2_activity", "eta2_radar", "intra_acq_cv", "stability_score"],
        stab_rows,
    )

    by_name = {r["factor"]: r for r in stab_rows}
    stable = [r["factor"] for r in sorted(stab_rows, key=lambda r: r["intra_acq_cv"])[:K_SUBSET]]
    act_top = [r["factor"] for r in sorted(stab_rows, key=lambda r: -r["eta2_activity"])[:K_SUBSET]]
    rad_top = [r["factor"] for r in sorted(stab_rows, key=lambda r: -r["eta2_radar"])[:K_SUBSET]]
    idx = {n: i for i, n in enumerate(FACTOR_NAMES)}

    def cols(names):
        return [idx[n] for n in names]

    probe_rows = []
    for label, names, mat, note in (
        ("all_raw", FACTOR_NAMES, F, "RAW factors; scaler inside probe"),
        ("all_global", FACTOR_NAMES, Fg, "TRAIN global z-score; not per-radar"),
        ("all_per_radar", FACTOR_NAMES, Fp, "per-radar TRAIN z-score; known-radar diagnostic only"),
        ("stable_global", stable, Fg, f"top {K_SUBSET} lowest intra-acq CV on TRAIN"),
        ("activity_top_global", act_top, Fg, f"top {K_SUBSET} eta2_activity on TRAIN"),
        ("radar_specific_global", rad_top, Fg, f"top {K_SUBSET} eta2_radar on TRAIN"),
    ):
        X = mat[:, cols(names)] if names is not FACTOR_NAMES else mat
        # when names is FACTOR_NAMES, mat already full
        if names is not FACTOR_NAMES:
            X = mat[:, cols(names)]
        else:
            X = mat
        act = probe(X[tr], action[tr], X[te], action[te])
        rid = probe(X[tr], radar[tr], X[te], radar[te])
        probe_rows.append(
            {
                "subset": label,
                "n_factors": len(names),
                "factors": ";".join(names) if len(names) <= 12 else f"{len(names)} factors",
                "activity_accuracy": act["accuracy"],
                "activity_macro_f1": act["macro_f1"],
                "radar_id_accuracy": rid["accuracy"],
                "radar_id_macro_f1": rid["macro_f1"],
                "note": note,
            }
        )
    r2_tr, r2_te = downsample16(pack["X"][tr]), downsample16(pack["X"][te])
    r2_act = probe(r2_tr, action[tr], r2_te, action[te])
    r2_rid = probe(r2_tr, radar[tr], r2_te, radar[te])
    p9 = read_csv(P9 / "probe_results.csv")
    a0 = next((r for r in p9 if r["model"] == "A0" and r["level"] == "shared_embedding"), {})
    a1 = next((r for r in p9 if r["model"] == "A1" and r["level"] == "shared_embedding"), {})
    a2 = next((r for r in p9 if r["model"] == "A2" and r["level"] == "shared_embedding"), {})
    inp = next((r for r in p9 if r["model"] == "input_R2D"), {})
    write_csv(
        OUT / "factor_probe_results.csv",
        ["subset", "n_factors", "factors", "activity_accuracy", "activity_macro_f1", "radar_id_accuracy", "radar_id_macro_f1", "note"],
        probe_rows,
    )

    # paired distances on global-normalized factors
    test_ts = [t for t, rec in pairs.items() if pack["split"][rec["10GHz"]] == "test"]
    same, diff = [], []
    rng = np.random.RandomState(42)
    for t in test_ts:
        rec = pairs[t]
        z = {r: Fg[rec[r]][None] for r in RADARS}
        same.extend(
            [
                float(cosine_dist(z["10GHz"], z["24GHz"])[0]),
                float(cosine_dist(z["10GHz"], z["77GHz"])[0]),
                float(cosine_dist(z["24GHz"], z["77GHz"])[0]),
            ]
        )
    for _ in range(len(same)):
        a, b = rng.choice(test_ts, 2, replace=False)
        r1, r2 = rng.choice(RADARS, 2, replace=False)
        diff.append(float(cosine_dist(Fg[pairs[a][r1]][None], Fg[pairs[b][r2]][None])[0]))
    r2_same = []
    pool = downsample16(pack["X"])
    for t in test_ts:
        rec = pairs[t]
        r2_same.extend(
            [
                float(cosine_dist(pool[rec["10GHz"]][None], pool[rec["24GHz"]][None])[0]),
                float(cosine_dist(pool[rec["10GHz"]][None], pool[rec["77GHz"]][None])[0]),
                float(cosine_dist(pool[rec["24GHz"]][None], pool[rec["77GHz"]][None])[0]),
            ]
        )
    p9d = {r["model"]: r for r in read_csv(P9 / "paired_embedding_distance.csv")}
    same_st, diff_st = [], []
    js = cols(stable)
    for t in test_ts:
        rec = pairs[t]
        z = {r: Fg[rec[r], js][None] for r in RADARS}
        same_st.extend(
            [
                float(cosine_dist(z["10GHz"], z["24GHz"])[0]),
                float(cosine_dist(z["10GHz"], z["77GHz"])[0]),
                float(cosine_dist(z["24GHz"], z["77GHz"])[0]),
            ]
        )
    for _ in range(len(same_st)):
        a, b = rng.choice(test_ts, 2, replace=False)
        r1, r2 = rng.choice(RADARS, 2, replace=False)
        diff_st.append(float(cosine_dist(Fg[pairs[a][r1], js][None], Fg[pairs[b][r2], js][None])[0]))
    pair_rows = [
        {"representation": "R2-D_16x16", "same_acq_mean": float(np.mean(r2_same)), "diff_acq_mean": "", "n": len(r2_same), "note": "cosine distance on 16x16 R2"},
        {"representation": "A0_embedding", "same_acq_mean": p9d.get("A0", {}).get("same_acquisition_mean", ""), "diff_acq_mean": "", "n": 147, "note": "Phase 9 frozen"},
        {"representation": "A2_embedding", "same_acq_mean": p9d.get("A2", {}).get("same_acquisition_mean", ""), "diff_acq_mean": "", "n": 147, "note": "Phase 9 frozen"},
        {"representation": "MotionFactors_global", "same_acq_mean": float(np.mean(same)), "diff_acq_mean": float(np.mean(diff)), "n": len(same), "note": "global TRAIN z-score factor cosine distance"},
        {"representation": "MotionFactors_stable", "same_acq_mean": float(np.mean(same_st)), "diff_acq_mean": float(np.mean(diff_st)), "n": len(same_st), "note": "stable subset cosine distance"},
    ]
    write_csv(OUT / "paired_factor_distance.csv", ["representation", "same_acq_mean", "diff_acq_mean", "n", "note"], pair_rows)

    act_stab = []
    for act in CLASS_NAMES:
        cvs = []
        for t in train_ts:
            rec = pairs[t]
            if pack["action"][rec["10GHz"]] != act:
                continue
            for j in range(len(FACTOR_NAMES)):
                vs = [F[rec[r], j] for r in RADARS]
                mu = np.mean(vs)
                cvs.append(float(np.std(vs) / (abs(mu) + EPS)))
        act_stab.append({"action": act, "mean_intra_cv": float(np.mean(cvs)) if cvs else "", "n": len(cvs) // max(len(FACTOR_NAMES), 1)})
    act_stab.sort(key=lambda r: r["mean_intra_cv"] if r["mean_intra_cv"] != "" else 9)
    write_csv(OUT / "action_factor_stability.csv", ["action", "mean_intra_cv", "n"], act_stab)

    pr_all = next(r for r in probe_rows if r["subset"] == "all_global")
    pr_st = next(r for r in probe_rows if r["subset"] == "stable_global")
    trade = [
        {"representation": "R2-D input 16x16", "radar_id": inp.get("radar_id_probe") or r2_rid["accuracy"], "activity_probe": inp.get("activity_probe") or r2_act["accuracy"], "note": "Phase 9 / this audit"},
        {"representation": "A0 embedding", "radar_id": a0.get("radar_id_probe", ""), "activity_probe": a0.get("activity_probe", ""), "note": "Phase 9"},
        {"representation": "A1 embedding", "radar_id": a1.get("radar_id_probe", ""), "activity_probe": a1.get("activity_probe", ""), "note": "Phase 9"},
        {"representation": "A2 embedding", "radar_id": a2.get("radar_id_probe", ""), "activity_probe": a2.get("activity_probe", ""), "note": "Phase 9"},
        {"representation": "Motion Factors All", "radar_id": pr_all["radar_id_accuracy"], "activity_probe": pr_all["activity_accuracy"], "note": "global TRAIN z-score"},
        {"representation": "Motion Factor Stable Subset", "radar_id": pr_st["radar_id_accuracy"], "activity_probe": pr_st["activity_accuracy"], "note": ";".join(stable)},
    ]
    write_csv(OUT / "information_tradeoff.csv", ["representation", "radar_id", "activity_probe", "note"], trade)

    # figures
    plt = _mpl()
    fig, ax = plt.subplots(figsize=(8.4, 3.2))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 3)
    boxes = [(0.3, 1.1, "R2-D\nS(τ, f_D)", "#dceaf7"), (3.2, 1.1, "energy\n10^(dB/10)", "#f7e6c8"), (6.2, 1.1, "Motion\nFactor Vector", "#d9ead3")]
    for x, y, t, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 2.2, 1.2, facecolor=c, edgecolor="#333"))
        ax.text(x + 1.1, y + 0.6, t, ha="center", va="center", fontsize=9)
        if x < 5:
            ax.annotate("", xy=(x + 3.15, 1.7), xytext=(x + 2.55, 1.7), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Fig 1  Motion-factor extraction")
    fig.savefig(FIG / "fig1_extraction.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 7.2))
    names = [r["factor"] for r in stab_rows][::-1]
    cvs = [r["intra_acq_cv"] for r in stab_rows][::-1]
    ax.barh(names, cvs)
    ax.set_xlabel("intra-acquisition cross-radar CV (TRAIN)")
    fig.savefig(FIG / "fig2_stability_ranking.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    xs = [r["eta2_radar"] for r in stab_rows]
    ys = [r["eta2_activity"] for r in stab_rows]
    ax.scatter(xs, ys, s=22)
    for r in stab_rows:
        if r["eta2_activity"] > 0.15 or r["eta2_radar"] > 0.4:
            ax.annotate(r["factor"], (r["eta2_radar"], r["eta2_activity"]), fontsize=6)
    ax.set_xlabel("eta2 radar")
    ax.set_ylabel("eta2 activity")
    ax.set_title("Fig 3  activity vs radar effect")
    fig.savefig(FIG / "fig3_effect_scatter.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    labs = ["R2 16x16", "A0", "A2", "Factors"]
    vals = [float(pair_rows[0]["same_acq_mean"]), float(pair_rows[1]["same_acq_mean"] or 0), float(pair_rows[2]["same_acq_mean"] or 0), float(pair_rows[3]["same_acq_mean"])]
    ax.bar(labs, vals)
    ax.set_ylabel("same-acquisition cosine distance")
    fig.savefig(FIG / "fig4_paired_distance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    labs = ["R2", "A0", "A2", "All F", "Stable"]
    actv = [float(trade[0]["activity_probe"]), float(trade[1]["activity_probe"]), float(trade[3]["activity_probe"]), float(trade[4]["activity_probe"]), float(trade[5]["activity_probe"])]
    ridv = [float(trade[0]["radar_id"]), float(trade[1]["radar_id"]), float(trade[3]["radar_id"]), float(trade[4]["radar_id"]), float(trade[5]["radar_id"])]
    x = np.arange(len(labs))
    ax.bar(x - 0.18, actv, 0.36, label="Activity")
    ax.bar(x + 0.18, ridv, 0.36, label="Radar-ID")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.savefig(FIG / "fig5_probes.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.bar([r["action"] for r in act_stab], [r["mean_intra_cv"] for r in act_stab])
    ax.set_ylabel("mean intra-acq CV")
    fig.savefig(FIG / "fig6_action_stability.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 7: a few key factor distributions
    keys = ["doppler_std_hz", "periodicity_score", "active_duration_frac", "spectral_entropy"]
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2))
    for ax, k in zip(axes.ravel(), keys):
        j = FACTOR_NAMES.index(k)
        for r in RADARS:
            ax.hist(F[radar == r, j], bins=18, histtype="step", label=r, density=True)
        ax.set_title(k)
        ax.legend(fontsize=7)
    fig.savefig(FIG / "fig7_factor_by_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for row in trade:
        try:
            ax.scatter(float(row["radar_id"]), float(row["activity_probe"]), s=40)
            ax.annotate(row["representation"], (float(row["radar_id"]), float(row["activity_probe"])), fontsize=7)
        except Exception:
            continue
    ax.set_xlabel("Radar-ID")
    ax.set_ylabel("Activity Probe")
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.set_title("Information Trade-off Map  (ideal: upper left)")
    fig.savefig(FIG / "fig8_information_tradeoff.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # case
    act_s, rid_s = float(pr_st["activity_accuracy"]), float(pr_st["radar_id_accuracy"])
    act_a, rid_a = float(pr_all["activity_accuracy"]), float(pr_all["radar_id_accuracy"])
    if (act_s >= 0.40 or act_a >= 0.40) and (rid_s <= 0.80 or rid_a <= 0.80) and min(rid_s, rid_a) < 0.90:
        case = "A"
        nxt = "PHYSICS-INFORMED ENCODER"
    elif min(rid_s, rid_a) <= 0.80 and max(act_s, act_a) < 0.30:
        case = "C"
        nxt = "STOP SHARED REPRESENTATION"
    else:
        case = "B"
        nxt = "SENSOR CALIBRATION"
    most_stable = [r["factor"] for r in stab_rows[:5]]
    most_radar = [r["factor"] for r in sorted(stab_rows, key=lambda r: -r["eta2_radar"])[:5]]
    most_act = [r["factor"] for r in sorted(stab_rows, key=lambda r: -r["eta2_activity"])[:5]]
    ideal = [r["factor"] for r in stab_rows if r["eta2_activity"] >= 0.08 and r["eta2_radar"] <= 0.15]
    d1024, d1077, d2477 = [], [], []
    for t in train_ts:
        rec = pairs[t]
        a, b, c = Fg[rec["10GHz"]], Fg[rec["24GHz"]], Fg[rec["77GHz"]]
        d1024.append(float(np.mean(np.abs(a - b))))
        d1077.append(float(np.mean(np.abs(a - c))))
        d2477.append(float(np.mean(np.abs(b - c))))
    pair_shift = {
        "10_24": float(np.mean(d1024)),
        "10_77": float(np.mean(d1077)),
        "24_77": float(np.mean(d2477)),
    }
    answers = {
        "n_factors": len(FACTOR_NAMES),
        "most_radar_stable": most_stable,
        "most_radar_specific": most_radar,
        "most_activity_informative": most_act,
        "ideal_high_act_low_radar": ideal,
        "stable_subset": stable,
        "activity_top_subset": act_top,
        "radar_specific_subset": rad_top,
        "probe_all_activity": act_a,
        "probe_all_radar": rid_a,
        "probe_stable_activity": act_s,
        "probe_stable_radar": rid_s,
        "pair_factors": float(np.mean(same)),
        "pair_r2": float(np.mean(r2_same)),
        "action_best": act_stab[0]["action"] if act_stab else "",
        "action_worst": act_stab[-1]["action"] if act_stab else "",
        "pair_shift": pair_shift,
        "case": case,
        "next_step": nxt,
        "low_hz": LOW_ABS_HZ,
        "mid_hz": MID_ABS_HZ,
        "k_subset": K_SUBSET,
    }
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    write_report(stab_rows, probe_rows, pair_rows, act_stab, trade, answers, pair_shift)
    print("phase10a done", answers, flush=True)


def write_report(stab, probes, pairs, acts, trade, ans, shift) -> None:
    lines = [
        "# Phase 10A Motion Factor Audit",
        "",
        "未训深度网络。未改 Phase 1–9。未使用径向速度 / m/s。",
        "XeThru effective fc 仍 UNRESOLVED。因子只基于时间、Doppler Hz、谱能量、时间动态。",
        "",
        "## 定义",
        "",
        "见 `motion_factor_definitions.md`。",
        f"预注册频带（共同支撑 ±256 Hz，不用 test 调）：low |f_D|<{ans['low_hz']} Hz；mid [{ans['low_hz']},{ans['mid_hz']})；high ≥{ans['mid_hz']} Hz。",
        "",
        "能量：把 R2-D 的 TRAIN z-score 还原为 clipped dB，再 `W=10^(dB/10)`。",
        "",
        "stability_score = eta²_activity / (intra-acquisition CV + 1e-6)。",
        "eta² 与 intra-CV 都只在 TRAIN acquisition 上算。",
        "per-radar 归一化只作 known-radar 诊断，不声称 unseen generalization。",
        "",
        f"## 因子数：{ans['n_factors']}",
        "",
        "最雷达稳定（intra-CV 最低）：",
        ", ".join(ans["most_radar_stable"]),
        "",
        "最依赖雷达（eta²_radar 最高）：",
        ", ".join(ans["most_radar_specific"]),
        "",
        "最能区分动作（eta²_activity 最高）：",
        ", ".join(ans["most_activity_informative"]),
        "",
        "同时偏高 activity / 偏低 radar（eta²_act≥0.08 且 eta²_rad≤0.15）：",
        (", ".join(ans["ideal_high_act_low_radar"]) if ans["ideal_high_act_low_radar"] else "无"),
        "",
        "## Probe",
        "",
    ]
    for r in probes:
        lines.append(
            f"- {r['subset']}: Activity {r['activity_accuracy']:.3f} (F1 {r['activity_macro_f1']:.3f})  "
            f"Radar-ID {r['radar_id_accuracy']:.3f}"
        )
    lines += [
        "",
        "对照（Phase 9，未重训）：",
        f"- R2-D 16×16: Activity {trade[0]['activity_probe']}  Radar-ID {trade[0]['radar_id']}",
        f"- A0 embedding: Activity {trade[1]['activity_probe']}  Radar-ID {trade[1]['radar_id']}",
        f"- A2 embedding: Activity {trade[3]['activity_probe']}  Radar-ID {trade[3]['radar_id']}",
        "",
        "## 同 acquisition 距离（cosine distance，越小越近）",
        "",
    ]
    for r in pairs:
        lines.append(f"- {r['representation']}: {r['same_acq_mean']}")
    lines += [
        "",
        "## 动作稳定性（TRAIN 平均 intra-CV）",
        "",
        f"- 最稳定: {ans['action_best']}",
        f"- 最不稳定: {ans['action_worst']}",
        "",
    ]
    for r in acts:
        lines.append(f"- {r['action']}: {r['mean_intra_cv']}")
    lines += [
        "",
        "## 频段对位移（TRAIN 平均 |Δfactor|，跨因子平均）",
        "",
        f"- 10 vs 24: {shift['10_24']:.4f}",
        f"- 10 vs 77: {shift['10_77']:.4f}",
        f"- 24 vs 77: {shift['24_77']:.4f}",
        "",
        "24 GHz 是否仍是最大位移，看上面三对哪个最大。",
        "",
        "## Motion Intelligence",
        "",
        "可对接、且不得写成 m/s 的量：",
        "- periodicity_score / dominant_temporal_freq：周期，单位是「每次采集的周期数」，不是速度。",
        "- temporal_energy_std / peak_to_mean_ratio：运动强弱的相对量。",
        "- doppler_std_hz / doppler_bandwidth_hz：Doppler 展宽，单位 Hz。",
        "- active_duration_frac：归一化时长上的有效占比。",
        "mean_velocity / peak_velocity 仍保持 UNKNOWN。",
        "",
        "## CASE",
        "",
        f"**CASE {ans['case']}**",
        f"下一步：**{ans['next_step']}**",
        "",
        "CASE A：Activity 可用且 Radar-ID 明显低于 0.993。",
        "CASE B：Activity 与 Radar-ID 都高，因子仍偏 sensor。",
        "CASE C：Radar-ID 降了但 Activity 接近随机。",
        "",
        "未训练新 encoder。未进入后续深度训练。",
    ]
    (OUT / "phase10_motion_factor_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
