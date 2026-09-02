#!/usr/bin/env python3
"""Phase 8C tables, probes, retrieval, figures. Reads frozen R0/R1; does not retrain them."""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase8.run_r2_benchmark import CLASS_NAMES, EmbNet, RADARS, SEED, load_r2_pack  # noqa: E402

OUT = ROOT / "results" / "phase8_physical_representation"
CKPT = OUT / "checkpoints_r2"
P2 = ROOT / "results" / "phase2_cross_radar" / "training_runs.csv"
P7 = ROOT / "results" / "phase7_radar_invariance"
P3 = ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv"


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def read_csv(path: Path) -> list:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, fields, rows) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def fnum(x, nd=6):
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return ""


def load_net(run_id: str, device):
    net = EmbNet(7)
    state = torch.load(CKPT / run_id / "best.pt", map_location="cpu")
    net.load_state_dict(state)
    return net.to(device).eval()


@torch.no_grad()
def embed(net, X, device, batch=32):
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        _, e = net(xb, return_emb=True)
        out.append(e.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0, 512), np.float32)


def probe(Etr, ytr, Ete, yte):
    if len(set(ytr)) < 2:
        return {"accuracy": None, "macro_f1": None}
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=500, random_state=SEED)
    clf.fit(sc.fit_transform(Etr), ytr)
    pred = clf.predict(sc.transform(Ete))
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def retrieval_pair(qa, ga, k=5):
    a = qa["emb"] / (np.linalg.norm(qa["emb"], axis=1, keepdims=True) + 1e-8)
    b = ga["emb"] / (np.linalg.norm(ga["emb"], axis=1, keepdims=True) + 1e-8)
    sim = a @ b.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    exact1, exact5, act1, act5 = [], [], [], []
    for i in range(len(qa["ts"])):
        hits_ts = ga["ts"][nn[i]]
        hits_act = ga["act"][nn[i]]
        exact1.append(hits_ts[0] == qa["ts"][i])
        exact5.append(qa["ts"][i] in hits_ts)
        act1.append(hits_act[0] == qa["act"][i])
        act5.append(qa["act"][i] in hits_act)
    return {
        "exact_top1": float(np.mean(exact1)),
        "exact_top5": float(np.mean(exact5)),
        "action_top1": float(np.mean(act1)),
        "action_top5": float(np.mean(act5)),
        "n": int(len(exact1)),
    }


def project_2d(E):
    try:
        import umap

        return umap.UMAP(n_components=2, random_state=42).fit_transform(E), "umap"
    except Exception:
        from sklearn.manifold import TSNE

        perp = min(30, max(5, len(E) // 4))
        return TSNE(n_components=2, random_state=42, init="pca", perplexity=perp).fit_transform(E), "tsne"


def r0_map():
    rows = read_csv(P2)
    out = {}
    for r in rows:
        if r.get("method") != "source_only":
            continue
        key = f"{r['train_radar'][:2]}→{r['test_radar'][:2]}"
        if r["train_radar"] == r["test_radar"]:
            key = f"{r['train_radar'][:2]}→{r['train_radar'][:2]}"
        out[key] = r
    return out


def r1_map():
    rows = read_csv(P7 / "training_runs.csv")
    out = {}
    for r in rows:
        if r.get("model") != "R1_source_only":
            continue
        if r["train_radar"] == r["test_radar"]:
            key = f"{r['train_radar'][:2]}→{r['train_radar'][:2]}"
        else:
            key = f"{r['train_radar'][:2]}→{r['test_radar'][:2]}"
        out[key] = r
    return out


def r2_map():
    rows = read_csv(OUT / "r2_training_runs.csv")
    out = {}
    for r in rows:
        proto = r.get("protocol", "")
        if proto == "within_radar":
            key = f"{r['train_radar'][:2]}→{r['train_radar'][:2]}"
        elif proto == "single_source_cross_radar":
            key = f"{r['train_radar'][:2]}→{r['test_radar'][:2]}"
        else:
            continue
        out[key] = r
    return out


def main() -> None:
    figdir = OUT / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_r2_pack()
    r0, r1, r2 = r0_map(), r1_map(), r2_map()
    keys = ["10→10", "24→24", "77→77", "10→24", "10→77", "24→10", "24→77", "77→10", "77→24"]
    cmp_rows = []
    for k in keys:
        a, b, c = r0.get(k, {}), r1.get(k, {}), r2.get(k, {})
        r0a = float(a["accuracy"]) if a.get("accuracy") not in (None, "") else None
        r1a = float(b["accuracy"]) if b.get("accuracy") not in (None, "") else None
        r2a = float(c["accuracy"]) if c.get("accuracy") not in (None, "") else None
        r0f = float(a["macro_f1"]) if a.get("macro_f1") not in (None, "") else None
        r1f = float(b["macro_f1"]) if b.get("macro_f1") not in (None, "") else None
        r2f = float(c["macro_f1"]) if c.get("macro_f1") not in (None, "") else None
        cmp_rows.append(
            {
                "protocol": k,
                "R0_acc": r0a,
                "R1_acc": r1a,
                "R2_acc": r2a,
                "R0_f1": r0f,
                "R1_f1": r1f,
                "R2_f1": r2f,
                "delta_R2_vs_R0": (None if r0a is None or r2a is None else r2a - r0a),
                "delta_R2_vs_R1": (None if r1a is None or r2a is None else r2a - r1a),
            }
        )
    write_csv(
        OUT / "r0_r1_r2_cross_radar.csv",
        ["protocol", "R0_acc", "R1_acc", "R2_acc", "R0_f1", "R1_f1", "R2_f1", "delta_R2_vs_R0", "delta_R2_vs_R1"],
        cmp_rows,
    )

    # LORO
    p7loro = read_csv(P7 / "loro_results.csv")
    r1_loro = {(r["train_radars"], r["test_radar"]): r for r in p7loro if r["model"] == "M0"}
    r2_runs = {r["run_id"]: r for r in read_csv(OUT / "r2_training_runs.csv")}
    loro_rows = []
    for srcs, tgt, rid in (
        ("10+24", "77GHz", "r2_loro_10_24_to_77"),
        ("10+77", "24GHz", "r2_loro_10_77_to_24"),
        ("24+77", "10GHz", "r2_loro_24_77_to_10"),
    ):
        a = r1_loro.get((srcs, tgt), {})
        b = r2_runs.get(rid, {})
        loro_rows.append(
            {
                "protocol": f"{srcs}→{tgt[:2]}",
                "R1_ce_acc": a.get("activity_accuracy", ""),
                "R1_ce_f1": a.get("macro_f1", ""),
                "R2_ce_acc": b.get("accuracy", ""),
                "R2_ce_f1": b.get("macro_f1", ""),
                "note": "CE baseline only; M3 excluded from this table",
            }
        )
    write_csv(
        OUT / "r2_loro_comparison.csv",
        ["protocol", "R1_ce_acc", "R1_ce_f1", "R2_ce_acc", "R2_ce_f1", "note"],
        loro_rows,
    )

    # probes + retrieval from shared CE
    net = load_net("r2_shared_ce", device)
    tr = pack["split"] == "train"
    te = pack["split"] == "test"
    Etr = embed(net, pack["X"][tr], device)
    Ete = embed(net, pack["X"][te], device)
    act = probe(Etr, pack["action"][tr], Ete, pack["action"][te])
    rad = probe(Etr, pack["radar"][tr], Ete, pack["radar"][te])
    p3 = {r["model"]: r for r in read_csv(P3) if r["protocol"] == "multi_radar_shared"}
    p7p = {r["model"]: r for r in read_csv(P7 / "probe_results.csv") if r["protocol"] == "multi_radar_shared"}
    fp = read_csv(OUT / "r2_fingerprint.csv")
    rep_id = {r["representation"]: r.get("accuracy") for r in fp}
    write_csv(
        OUT / "representation_probe_comparison.csv",
        [
            "level",
            "representation",
            "activity_probe",
            "radar_id_probe",
            "note",
        ],
        [
            {"level": "representation_16x16", "representation": "R0", "activity_probe": "", "radar_id_probe": "1.000", "note": "Phase 7 pixel logreg"},
            {"level": "representation_16x16", "representation": "R1", "activity_probe": "", "radar_id_probe": "1.000", "note": "Phase 7 pixel logreg"},
            {"level": "representation_16x16", "representation": "R2-D", "activity_probe": "", "radar_id_probe": rep_id.get("R2-D", "0.993197"), "note": "Phase 8B pixel logreg; not encoder"},
            {
                "level": "encoder_embedding",
                "representation": "R0",
                "activity_probe": p3.get("M0", {}).get("activity_probe_accuracy", ""),
                "radar_id_probe": p3.get("M0", {}).get("radar_probe_accuracy", "1.0"),
                "note": "Phase 3 M0 shared CE",
            },
            {
                "level": "encoder_embedding",
                "representation": "R1",
                "activity_probe": p7p.get("M0", {}).get("activity_probe_accuracy", ""),
                "radar_id_probe": p7p.get("M0", {}).get("radar_probe_accuracy", "1.0"),
                "note": "Phase 7 M0 shared CE",
            },
            {
                "level": "encoder_embedding",
                "representation": "R2-D",
                "activity_probe": act["accuracy"],
                "radar_id_probe": rad["accuracy"],
                "note": "Phase 8C shared CE; linear probe fit train eval test",
            },
        ],
    )

    te_idx = np.where(te)[0]
    by_rad = {}
    for radar in RADARS:
        m = pack["radar"][te_idx] == radar
        by_rad[radar] = {
            "emb": Ete[m],
            "ts": pack["timestamp"][te_idx][m],
            "act": pack["action"][te_idx][m],
        }
    ret_rows = []
    for q, g in (("10GHz", "24GHz"), ("10GHz", "77GHz"), ("24GHz", "10GHz"), ("24GHz", "77GHz"), ("77GHz", "10GHz"), ("77GHz", "24GHz")):
        rec = retrieval_pair(by_rad[q], by_rad[g])
        rec.update({"query": q, "gallery": g, "representation": "R2-D"})
        ret_rows.append(rec)
    write_csv(
        OUT / "r2_retrieval.csv",
        ["representation", "query", "gallery", "exact_top1", "exact_top5", "action_top1", "action_top5", "n"],
        ret_rows,
    )

    # clipping vs correctness (within models)
    ret = { (r["timestamp"], r["radar"]): r for r in read_csv(OUT / "doppler_band_retention.csv") }
    clip_rows = []
    for radar, rid in (("10GHz", "r2_within_10"), ("24GHz", "r2_within_24"), ("77GHz", "r2_within_77")):
        pred_p = CKPT / rid / "test_pred.npz"
        if not pred_p.is_file() or not ret:
            continue
        z = np.load(pred_p, allow_pickle=True)
        for i in range(len(z["timestamp"])):
            ts = str(z["timestamp"][i])
            rr = ret.get((ts, radar), {})
            clip_rows.append(
                {
                    "radar": radar,
                    "timestamp": ts,
                    "correct": int(z["correct"][i]),
                    "retained_energy_fraction": rr.get("retained_energy_fraction", ""),
                }
            )
    if clip_rows:
        write_csv(OUT / "retention_vs_correctness.csv", ["radar", "timestamp", "correct", "retained_energy_fraction"], clip_rows)

    # figures
    plt = _mpl()
    order_cross = ["10→24", "10→77", "24→10", "24→77", "77→10", "77→24"]
    fig, ax = plt.subplots(figsize=(8.2, 4.0))
    x = np.arange(len(order_cross))
    w = 0.25
    for i, (lab, key) in enumerate((("R0", "R0_acc"), ("R1", "R1_acc"), ("R2-D", "R2_acc"))):
        ys = []
        for k in order_cross:
            row = next(r for r in cmp_rows if r["protocol"] == k)
            ys.append(0.0 if row[key] in (None, "") else float(row[key]))
        ax.bar(x + (i - 1) * w, ys, w, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(order_cross)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(figdir / "fig7_r0_r1_r2_cross.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    x = np.arange(3)
    within = ["10→10", "24→24", "77→77"]
    for i, (lab, key) in enumerate((("R0", "R0_acc"), ("R1", "R1_acc"), ("R2-D", "R2_acc"))):
        ys = [float(next(r for r in cmp_rows if r["protocol"] == k)[key] or 0) for k in within]
        ax.bar(x + (i - 1) * 0.25, ys, 0.25, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels(["10", "24", "77"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.savefig(figdir / "fig8_r0_r1_r2_within.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.bar(
        ["R0 rep", "R1 rep", "R2-D rep", "R0 emb", "R1 emb", "R2-D emb"],
        [
            1.0,
            1.0,
            float(rep_id.get("R2-D") or 0.993197),
            1.0,
            1.0,
            float(rad["accuracy"] or 0),
        ],
    )
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID")
    fig.savefig(figdir / "fig9_radar_id_rep_vs_emb.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(
        ["R0 act", "R1 act", "R2-D act", "R0 rid", "R1 rid", "R2-D rid"],
        [
            float(p3.get("M0", {}).get("activity_probe_accuracy") or 0),
            float(p7p.get("M0", {}).get("activity_probe_accuracy") or 0),
            float(act["accuracy"] or 0),
            1.0,
            1.0,
            float(rad["accuracy"] or 0),
        ],
    )
    ax.set_ylim(0, 1.05)
    fig.savefig(figdir / "fig10_activity_vs_radar_probe.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    summ_p = OUT / "doppler_band_retention_summary.json"
    if summ_p.is_file():
        summ = json.loads(summ_p.read_text(encoding="utf-8"))
        fig, ax = plt.subplots(figsize=(5.2, 3.6))
        xs = RADARS
        ax.bar(xs, [summ[r]["mean"] for r in xs], yerr=[summ[r]["std"] for r in xs])
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("retained energy in ±256 Hz")
        fig.savefig(figdir / "fig11_doppler_retention.png", dpi=130, bbox_inches="tight")
        plt.close(fig)

    Z, kind = project_2d(Ete)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for u in CLASS_NAMES:
        m = pack["action"][te_idx] == u
        ax.scatter(Z[m, 0], Z[m, 1], s=12, label=u, alpha=0.75)
    ax.legend(fontsize=7)
    ax.set_title(f"Fig 12  R2-D embedding by activity ({kind})")
    fig.savefig(figdir / "fig12_r2_embed_activity.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for u in RADARS:
        m = pack["radar"][te_idx] == u
        ax.scatter(Z[m, 0], Z[m, 1], s=12, label=u, alpha=0.75)
    ax.legend(fontsize=7)
    ax.set_title(f"Fig 13  R2-D embedding by radar ({kind})")
    fig.savefig(figdir / f"fig13_r2_embed_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("analyze done", act, rad, flush=True)


if __name__ == "__main__":
    main()
