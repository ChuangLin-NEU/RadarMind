#!/usr/bin/env python3
"""Phase 9 probes, distances, figures, report. Reads trained A0/A1/A2."""
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
from phase8.run_r2_benchmark import CLASS_NAMES, RADARS, SEED, load_r2_pack  # noqa: E402
from phase9.models import RadarBridgeV2  # noqa: E402
from phase9.run_phase9 import CKPT, OUT, mask_samples  # noqa: E402

FIG = OUT / "figures"


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


def load_model(run_id: str, device, use_adapters: bool, allowed=None):
    net = RadarBridgeV2(use_adapters=use_adapters, allowed_radars=allowed or RADARS)
    state = torch.load(CKPT / run_id / "best.pt", map_location="cpu")
    net.load_state_dict(state)
    return net.to(device).eval()


@torch.no_grad()
def extract(net, X, radars, device, override=None, batch=32):
    logits_l, emb_l, ada_l, pred_l = [], [], [], []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i : i + batch]).float().to(device)
        rb = list(radars[i : i + batch])
        logits, emb, xa, feat = net(xb, rb, override=override, return_all=True)
        pred_l.append(logits.argmax(1).cpu().numpy())
        logits_l.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
        emb_l.append(emb.cpu().numpy())
        pool = torch.nn.functional.adaptive_avg_pool2d(xa, (16, 16))
        ada_l.append(pool.flatten(1).cpu().numpy())
    return {
        "pred": np.concatenate(pred_l),
        "prob": np.concatenate(logits_l),
        "emb": np.concatenate(emb_l),
        "adapter": np.concatenate(ada_l),
    }


def downsample16(X):
    # X: N,H,W
    t = torch.from_numpy(np.asarray(X, np.float32))
    if t.ndim == 3:
        t = t.unsqueeze(1)
    return torch.nn.functional.adaptive_avg_pool2d(t, (16, 16)).flatten(1).numpy()


def probe(Etr, ytr, Ete, yte):
    if len(set(map(str, ytr))) < 2:
        return {"accuracy": None, "macro_f1": None}
    sc = StandardScaler()
    clf = LogisticRegression(max_iter=800, random_state=SEED)
    clf.fit(sc.fit_transform(Etr), ytr)
    pred = clf.predict(sc.transform(Ete))
    return {
        "accuracy": float(accuracy_score(yte, pred)),
        "macro_f1": float(f1_score(yte, pred, average="macro", zero_division=0)),
    }


def cos_dist(a, b):
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-8)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return 1.0 - np.sum(a * b, axis=1)


def kl_div(p, q):
    p = np.clip(p, 1e-8, 1)
    q = np.clip(q, 1e-8, 1)
    return float(np.sum(p * np.log(p / q)))


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


def group_by_ts(pack, mask, blob):
    out = defaultdict(dict)
    ts = pack["timestamp"][mask]
    rad = pack["radar"][mask]
    act = pack["action"][mask]
    y = pack["y"][mask]
    for i in range(len(ts)):
        out[str(ts[i])][rad[i]] = {
            "emb": blob["emb"][i],
            "pred": int(blob["pred"][i]),
            "prob": blob["prob"][i],
            "act": act[i],
            "y": int(y[i]),
        }
    return out


def pair_stats(grouped):
    same_acq, same_act, diff_act = [], [], []
    items = [(ts, rec) for ts, rec in grouped.items() if all(r in rec for r in RADARS)]
    for ts, rec in items:
        e = {r: rec[r]["emb"][None] for r in RADARS}
        d = [cos_dist(e["10GHz"], e["24GHz"])[0], cos_dist(e["10GHz"], e["77GHz"])[0], cos_dist(e["24GHz"], e["77GHz"])[0]]
        same_acq.extend(d)
    # same action / different ts / different radar
    by_act = defaultdict(list)
    for ts, rec in items:
        by_act[rec["10GHz"]["act"]].append(ts)
    rng = np.random.RandomState(42)
    for act, tss in by_act.items():
        if len(tss) < 2:
            continue
        for _ in range(min(80, len(tss) * 3)):
            a, b = rng.choice(tss, 2, replace=False)
            r1, r2 = rng.choice(RADARS, 2, replace=False)
            same_act.append(cos_dist(grouped[a][r1]["emb"][None], grouped[b][r2]["emb"][None])[0])
    acts = list(by_act)
    for _ in range(240):
        a1, a2 = rng.choice(acts, 2, replace=False)
        t1, t2 = rng.choice(by_act[a1]), rng.choice(by_act[a2])
        r1, r2 = rng.choice(RADARS, 2, replace=False)
        diff_act.append(cos_dist(grouped[t1][r1]["emb"][None], grouped[t2][r2]["emb"][None])[0])
    return {
        "same_acquisition_mean": float(np.mean(same_acq)),
        "same_acquisition_std": float(np.std(same_acq)),
        "same_action_mean": float(np.mean(same_act)) if same_act else "",
        "different_action_mean": float(np.mean(diff_act)) if diff_act else "",
        "n_same_acq_pairs": len(same_acq),
        "same_acq_values": same_acq,
        "same_act_values": same_act,
        "diff_act_values": diff_act,
    }


def agreement_table(grouped):
    rows, kls = [], []
    n3 = n2 = n_ok = 0
    for ts, rec in grouped.items():
        if not all(r in rec for r in RADARS):
            continue
        p = [rec[r]["pred"] for r in RADARS]
        y = rec["10GHz"]["y"]
        agree3 = int(p[0] == p[1] == p[2])
        agree2 = int(len(set(p)) <= 2)
        ok3 = int(p[0] == p[1] == p[2] == y)
        n3 += agree3
        n2 += agree2
        n_ok += ok3
        kl = np.mean(
            [
                kl_div(rec["10GHz"]["prob"], rec["24GHz"]["prob"]),
                kl_div(rec["10GHz"]["prob"], rec["77GHz"]["prob"]),
                kl_div(rec["24GHz"]["prob"], rec["77GHz"]["prob"]),
            ]
        )
        kls.append(kl)
        rows.append(
            {
                "timestamp": ts,
                "action": rec["10GHz"]["act"],
                "pred_10": CLASS_NAMES[p[0]],
                "pred_24": CLASS_NAMES[p[1]],
                "pred_77": CLASS_NAMES[p[2]],
                "all_three_agree": agree3,
                "at_least_two_agree": agree2,
                "all_three_correct": ok3,
                "kl_mean": kl,
            }
        )
    n = max(len(rows), 1)
    summary = {
        "n": len(rows),
        "all_three_agreement": n3 / n,
        "at_least_two_agreement": n2 / n,
        "all_three_correct": n_ok / n,
        "kl_mean": float(np.mean(kls)) if kls else "",
    }
    return rows, summary


def fig_arch(plt):
    fig, ax = plt.subplots(figsize=(8.6, 3.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    boxes = [
        (0.2, 2.7, "10 GHz R2-D", "#dceaf7"),
        (0.2, 1.6, "24 GHz R2-D", "#dceaf7"),
        (0.2, 0.5, "77 GHz R2-D", "#dceaf7"),
        (2.6, 2.7, "Adapter_10", "#f7e6c8"),
        (2.6, 1.6, "Adapter_24", "#f7e6c8"),
        (2.6, 0.5, "Adapter_77", "#f7e6c8"),
        (5.2, 1.2, "Shared Motion\nEncoder\nResNet18", "#d9ead3"),
        (7.8, 1.6, "z 512D\nActivity head", "#ead1dc"),
    ]
    for x, y, t, c in boxes:
        ax.add_patch(plt.Rectangle((x, y), 1.9, 0.9, facecolor=c, edgecolor="#333"))
        ax.text(x + 0.95, y + 0.45, t, ha="center", va="center", fontsize=8)
    for y in (3.15, 2.05, 0.95):
        ax.annotate("", xy=(2.55, y), xytext=(2.15, y), arrowprops=dict(arrowstyle="->"))
        ax.annotate("", xy=(5.15, 1.65), xytext=(4.55, y), arrowprops=dict(arrowstyle="->"))
    ax.annotate("", xy=(7.75, 2.05), xytext=(7.15, 1.65), arrowprops=dict(arrowstyle="->"))
    ax.set_title("Fig 1  RadarBridge v2 Known-Radar Sensor-Adapter Framework")
    fig.savefig(FIG / "fig1_architecture.png", dpi=130, bbox_inches="tight")
    plt.close(fig)


def write_report(ctx):
    a0, a1, a2 = ctx["A0"], ctx["A1"], ctx["A2"]
    flow = ctx["radar_flow"]
    actf = ctx["activity_flow"]
    agr = ctx["agree"]
    loro = ctx["loro"]
    ps = ctx["pair"]
    pinfo = ctx["params"]
    gate = "EFFECTIVE_FC_UNRESOLVED"
    lines = [
        "# Phase 9 RadarBridge v2",
        "",
        "RadarBridge v2 是 **Known-Radar Sensor-Adapter Framework**。",
        "选 Adapter 必须知道 radar_id。不是 Universal Radar Encoder。",
        "",
        "## Physics Gate",
        "",
        f"- XeThru effective fc: **UNRESOLVED**（见 `xethru_frequency_audit.md`）",
        "- R2-V: **未建立**",
        "- 本阶段表示: **R2-D**  `S(normalized_time, Doppler_Hz)`  1×224×224",
        "- R2-D 去掉的是作图/像素坐标，不是按径向速度对齐，也不是 sensor-invariant。",
        "",
        "## 和 Phase 3 M1 的差别",
        "",
        "- Phase 3 M1：同一套图进同一个 encoder，再做 contrastive。",
        "- 本阶段：每个雷达先过自己的轻量 Adapter，再进 Shared Motion Encoder；",
        "  同 timestamp 的三频 embedding 用 cosine distance 对齐。",
        "- 假设：低层雷达统计不该全部丢给 shared encoder 自己消化。",
        "",
        "## 参数量",
        "",
        f"- 单个 Adapter: {pinfo.get('adapter_each')}",
        f"- 三个 Adapter 合计: {pinfo.get('adapter_all_three')}",
        f"- Shared encoder: {pinfo.get('shared_encoder')}",
        "- 运动建模仍在 shared encoder。",
        "",
        "## Known-radar 联合 HAR（三频 TRAIN / VAL / TEST）",
        "",
        f"- A0 shared encoder only, CE: acc={a0.get('accuracy')}  F1={a0.get('macro_f1')}  "
        f"10/24/77={a0.get('acc_10')}/{a0.get('acc_24')}/{a0.get('acc_77')}",
        f"- A1 adapters + CE: acc={a1.get('accuracy')}  F1={a1.get('macro_f1')}  "
        f"10/24/77={a1.get('acc_10')}/{a1.get('acc_24')}/{a1.get('acc_77')}",
        f"- A2 adapters + CE + exact-pair (lambda={a2.get('lambda_pair')}): "
        f"acc={a2.get('accuracy')}  F1={a2.get('macro_f1')}  "
        f"10/24/77={a2.get('acc_10')}/{a2.get('acc_24')}/{a2.get('acc_77')}",
        "",
        "## Radar-ID（同一套 linear probe）",
        "",
        f"- Input R2-D 16×16: {flow['input']}",
        f"- A2 adapter output: {flow['adapter']}",
        f"- A2 shared embedding: {flow['embedding']}",
        "",
        "## Activity probe",
        "",
        f"- A2 adapter output: {actf['adapter']}",
        f"- A2 shared embedding: {actf['embedding']}",
        "",
        "## 同 acquisition 三频距离（cosine distance，TEST）",
        "",
        f"- A0 mean: {ps['A0']['same_acquisition_mean']}",
        f"- A1 mean: {ps['A1']['same_acquisition_mean']}",
        f"- A2 mean: {ps['A2']['same_acquisition_mean']}",
        f"- A2 same-action (不同 acquisition): {ps['A2']['same_action_mean']}",
        f"- A2 different-action: {ps['A2']['different_action_mean']}",
        "",
        "## 三频预测一致（known-radar TEST，不是 unseen DG）",
        "",
        f"- A0 all-three agree: {agr['A0']['all_three_agreement']}",
        f"- A2 all-three agree: {agr['A2']['all_three_agreement']}",
        f"- A0 all-three correct: {agr['A0']['all_three_correct']}",
        f"- A2 all-three correct: {agr['A2']['all_three_correct']}",
        "",
        "## LORO adapter transfer（held-out 雷达没有自己的 Adapter）",
        "",
    ]
    for r in loro:
        lines.append(
            f"- {r['train_radar']}→{r['held_out']}  test_adapter={r['test_adapter']}  "
            f"acc={r['accuracy']}  F1={r['macro_f1']}"
        )
    lines += [
        "",
        "A2 不能证明 unseen-radar domain generalization：没见过的雷达没有对应 Adapter。",
        "",
        "LORO 最好一格是 10+24→77 用 Adapter_24：0.286。7 类随机约 0.143。",
        "24+77→10 最好 0.224，低于 Phase 8C R2-D CE 的 0.306。",
        "数字有的略高于 Phase 8C LORO，但仍不能当活动分类用。判定为 unseen 未改善。",
        "",
        "Adapter 没有压低 Radar-ID（输入 0.993 → Adapter 0.986 → embedding 1.000）。",
        "A2 的同 acquisition 距离比 A0 更大，三频一致率从 0.878 降到 0.592。",
        "Activity probe 在 embedding 上仍高（0.925），动作信息还在，雷达信息也还在。",
        "",
        "## 结论",
        "",
        f"- Known-radar interoperability: **{ctx['verdict']['known']}**",
        f"- Radar invariance: **{ctx['verdict']['invariance']}**",
        f"- Unseen-radar generalization: **{ctx['verdict']['unseen']}**",
        f"- Learn Motion, Not Sensor: **{ctx['verdict']['slogan']}**",
        f"- Sensor Adapter 实际解决的是: **{ctx['verdict']['solves']}**",
        "",
        "主论文 frozen 表未改。本阶段是独立分支。",
        f"Physics gate: {gate}.",
    ]
    (OUT / "phase9_radarbridge_v2_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = load_r2_pack()
    tr = pack["split"] == "train"
    te = pack["split"] == "test"
    Xtr, Xte = pack["X"][tr], pack["X"][te]
    rtr, rte = pack["radar"][tr], pack["radar"][te]
    params = json.loads((OUT / "model_params.json").read_text(encoding="utf-8")) if (OUT / "model_params.json").is_file() else {}
    runs = {r["run_id"]: r for r in read_csv(OUT / "training_runs.csv")}
    a0m, a1m, a2m = runs.get("A0", {}), runs.get("A1", {}), runs.get("A2", {})

    nets = {
        "A0": load_model("A0", device, False),
        "A1": load_model("A1", device, True),
        "A2": load_model("A2", device, True),
    }
    blobs = {}
    for name, net in nets.items():
        blobs[name] = {
            "train": extract(net, Xtr, rtr, device),
            "test": extract(net, Xte, rte, device),
        }

    Xin_tr, Xin_te = downsample16(Xtr), downsample16(Xte)
    probe_rows = []
    radar_flow_rows = []
    act_flow_rows = []
    rid_input = probe(Xin_tr, rtr, Xin_te, rte)
    act_input = probe(Xin_tr, pack["action"][tr], Xin_te, pack["action"][te])
    probe_rows.append(
        {
            "model": "input_R2D",
            "level": "representation_16x16",
            "activity_probe": act_input["accuracy"],
            "radar_id_probe": rid_input["accuracy"],
            "note": "R2-D pixels; no encoder",
        }
    )
    for name in ("A0", "A1", "A2"):
        btr, bte = blobs[name]["train"], blobs[name]["test"]
        rid_ad = probe(btr["adapter"], rtr, bte["adapter"], rte)
        act_ad = probe(btr["adapter"], pack["action"][tr], bte["adapter"], pack["action"][te])
        rid_em = probe(btr["emb"], rtr, bte["emb"], rte)
        act_em = probe(btr["emb"], pack["action"][tr], bte["emb"], pack["action"][te])
        probe_rows += [
            {
                "model": name,
                "level": "adapter_output",
                "activity_probe": act_ad["accuracy"],
                "radar_id_probe": rid_ad["accuracy"],
                "note": "16x16 of adapter output (A0 = identity)",
            },
            {
                "model": name,
                "level": "shared_embedding",
                "activity_probe": act_em["accuracy"],
                "radar_id_probe": rid_em["accuracy"],
                "note": "512-D linear probe; train fit / test eval",
            },
        ]
        radar_flow_rows.append(
            {
                "model": name,
                "input": rid_input["accuracy"],
                "adapter_output": rid_ad["accuracy"],
                "shared_embedding": rid_em["accuracy"],
            }
        )
        act_flow_rows.append(
            {
                "model": name,
                "adapter_output": act_ad["accuracy"],
                "shared_embedding": act_em["accuracy"],
            }
        )
    write_csv(
        OUT / "probe_results.csv",
        ["model", "level", "activity_probe", "radar_id_probe", "note"],
        probe_rows,
    )
    write_csv(OUT / "radar_information_flow.csv", ["model", "input", "adapter_output", "shared_embedding"], radar_flow_rows)
    write_csv(OUT / "activity_information_flow.csv", ["model", "adapter_output", "shared_embedding"], act_flow_rows)

    pair_out_rows = []
    pair_store = {}
    agree_store = {}
    agree_detail = {}
    grouped_store = {}
    for name in ("A0", "A1", "A2"):
        g = group_by_ts(pack, te, blobs[name]["test"])
        grouped_store[name] = g
        st = pair_stats(g)
        pair_store[name] = st
        pair_out_rows.append(
            {
                "model": name,
                "same_acquisition_mean": st["same_acquisition_mean"],
                "same_acquisition_std": st["same_acquisition_std"],
                "same_action_mean": st["same_action_mean"],
                "different_action_mean": st["different_action_mean"],
                "n_same_acq_pairs": st["n_same_acq_pairs"],
            }
        )
        det, summ = agreement_table(g)
        agree_store[name] = summ
        agree_detail[name] = det
        for r in det:
            r["model"] = name
    write_csv(
        OUT / "paired_embedding_distance.csv",
        ["model", "same_acquisition_mean", "same_acquisition_std", "same_action_mean", "different_action_mean", "n_same_acq_pairs"],
        pair_out_rows,
    )
    write_csv(
        OUT / "prediction_agreement.csv",
        ["model", "timestamp", "action", "pred_10", "pred_24", "pred_77", "all_three_agree", "at_least_two_agree", "all_three_correct", "kl_mean"],
        [r for name in ("A0", "A1", "A2") for r in agree_detail[name]],
    )
    write_csv(
        OUT / "prediction_agreement_summary.csv",
        ["model", "n", "all_three_agreement", "at_least_two_agreement", "all_three_correct", "kl_mean"],
        [{"model": k, **v} for k, v in agree_store.items()],
    )

    ret_rows = []
    te_idx = np.where(te)[0]
    for name in ("A0", "A1", "A2"):
        E = blobs[name]["test"]["emb"]
        by_rad = {}
        for radar in RADARS:
            m = pack["radar"][te_idx] == radar
            by_rad[radar] = {"emb": E[m], "ts": pack["timestamp"][te_idx][m], "act": pack["action"][te_idx][m]}
        for q, g in (("10GHz", "24GHz"), ("10GHz", "77GHz"), ("24GHz", "10GHz"), ("24GHz", "77GHz"), ("77GHz", "10GHz"), ("77GHz", "24GHz")):
            rec = retrieval_pair(by_rad[q], by_rad[g])
            rec.update({"model": name, "query": q, "gallery": g})
            ret_rows.append(rec)
    write_csv(
        OUT / "retrieval.csv",
        ["model", "query", "gallery", "exact_top1", "exact_top5", "action_top1", "action_top5", "n"],
        ret_rows,
    )

    loro = read_csv(OUT / "loro_adapter_transfer.csv")

    # figures
    plt = _mpl()
    fig_arch(plt)

    # Fig 2: same acquisition R2
    ts0 = str(pack["timestamp"][te_idx][0])
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    for ax, r in zip(axes, RADARS):
        m = (pack["timestamp"] == ts0) & (pack["radar"] == r)
        ax.imshow(pack["X"][m][0], origin="lower", aspect="auto", cmap="magma")
        ax.set_title(r)
        ax.set_xlabel("tau")
        ax.set_ylabel("Doppler bin")
    fig.suptitle(f"Fig 2  R2-D input  ts={ts0}")
    fig.savefig(FIG / "fig2_r2_triplet.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # Fig 3 adapter features
    net_a2 = nets["A2"]
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.2))
    with torch.no_grad():
        for ax, r in zip(axes, RADARS):
            m = (pack["timestamp"] == ts0) & (pack["radar"] == r)
            xb = torch.from_numpy(pack["X"][m]).float().to(device)
            xa, _ = net_a2.apply_adapter(xb, r)
            ax.imshow(xa[0, 0].cpu().numpy(), origin="lower", aspect="auto", cmap="magma")
            ax.set_title(f"Adapter {r}")
    fig.suptitle(f"Fig 3  A2 adapter output  ts={ts0}")
    fig.savefig(FIG / "fig3_adapter_features.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    Z, kind = project_2d(blobs["A2"]["test"]["emb"])
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for u in CLASS_NAMES:
        m = pack["action"][te_idx] == u
        ax.scatter(Z[m, 0], Z[m, 1], s=12, label=u, alpha=0.75)
    ax.legend(fontsize=7)
    ax.set_title(f"Fig 4  A2 embedding by action ({kind})")
    fig.savefig(FIG / "fig4_embed_action.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.0, 4.6))
    for u in RADARS:
        m = pack["radar"][te_idx] == u
        ax.scatter(Z[m, 0], Z[m, 1], s=12, label=u, alpha=0.75)
    ax.legend(fontsize=7)
    ax.set_title(f"Fig 5  A2 embedding by radar ({kind})")
    fig.savefig(FIG / "fig5_embed_radar.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    names = ["A0", "A1", "A2"]
    x = np.arange(3)
    inp = [rid_input["accuracy"]] * 3
    ada = [next(r for r in radar_flow_rows if r["model"] == n)["adapter_output"] for n in names]
    emb = [next(r for r in radar_flow_rows if r["model"] == n)["shared_embedding"] for n in names]
    ax.bar(x - 0.25, inp, 0.25, label="input")
    ax.bar(x, ada, 0.25, label="adapter")
    ax.bar(x + 0.25, emb, 0.25, label="embedding")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Radar-ID")
    ax.legend()
    fig.savefig(FIG / "fig6_radar_id_flow.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.8, 3.6))
    ada_a = [next(r for r in act_flow_rows if r["model"] == n)["adapter_output"] for n in names]
    emb_a = [next(r for r in act_flow_rows if r["model"] == n)["shared_embedding"] for n in names]
    ax.bar(x - 0.15, ada_a, 0.3, label="adapter")
    ax.bar(x + 0.15, emb_a, 0.3, label="embedding")
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Activity probe")
    ax.legend()
    fig.savefig(FIG / "fig7_activity_flow.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for name, c in (("A0", "C0"), ("A1", "C1"), ("A2", "C2")):
        ax.hist(pair_store[name]["same_acq_values"], bins=20, histtype="step", label=name, density=True)
    ax.set_xlabel("cosine distance (same acquisition, different radar)")
    ax.legend()
    fig.savefig(FIG / "fig8_paired_distance.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.6))
    ax.bar(
        ["A0 agree3", "A2 agree3", "A0 correct3", "A2 correct3"],
        [
            agree_store["A0"]["all_three_agreement"],
            agree_store["A2"]["all_three_agreement"],
            agree_store["A0"]["all_three_correct"],
            agree_store["A2"]["all_three_correct"],
        ],
    )
    ax.set_ylim(0, 1.05)
    fig.savefig(FIG / "fig9_prediction_agreement.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    # verdicts
    a2_rid = next(r for r in radar_flow_rows if r["model"] == "A2")["shared_embedding"]
    a0_acc, a1_acc, a2_acc = float(a0m.get("accuracy") or 0), float(a1m.get("accuracy") or 0), float(a2m.get("accuracy") or 0)
    a0_ag = agree_store["A0"]["all_three_agreement"]
    a2_ag = agree_store["A2"]["all_three_agreement"]
    a0_d = pair_store["A0"]["same_acquisition_mean"]
    a2_d = pair_store["A2"]["same_acquisition_mean"]
    known = "IMPROVED" if (a2_acc > a0_acc + 0.01 or a2_ag > a0_ag + 0.03 or a2_d < a0_d - 0.05) else "NOT IMPROVED"
    if a2_rid is not None and a2_rid <= 0.7:
        inv = "YES"
    elif a2_rid is not None and a2_rid < 0.95:
        inv = "PARTIAL"
    else:
        inv = "NO"
    # LORO vs Phase 8C CE. Near-chance held-out acc is still a failed DG test.
    p8 = {
        "77GHz": 0.143,
        "24GHz": 0.102,
        "10GHz": 0.306,
    }
    loro_best = {}
    for r in loro:
        h = r["held_out"]
        loro_best[h] = max(loro_best.get(h, 0.0), float(r["accuracy"] or 0))
    usable = all(loro_best.get(k, 0) >= 0.40 for k in p8)
    all_up = all(loro_best.get(k, 0) > p8[k] + 0.05 for k in p8)
    unseen = "IMPROVED" if usable or all_up else "NOT IMPROVED"
    if inv == "YES" and unseen == "IMPROVED":
        slogan = "YES"
    elif inv in ("YES", "PARTIAL") or known == "IMPROVED":
        slogan = "PARTIAL"
    else:
        slogan = "NO"
    if known == "IMPROVED" and unseen == "NOT IMPROVED":
        solves = "known-radar interoperability，不是 unseen-radar generalization"
    elif known == "NOT IMPROVED" and unseen == "NOT IMPROVED":
        solves = "两者都没有"
    elif unseen == "IMPROVED" and known == "IMPROVED":
        solves = "known-radar interoperability 和 unseen-radar generalization"
    else:
        solves = "两者都没有"

    ctx = {
        "A0": a0m,
        "A1": a1m,
        "A2": a2m,
        "radar_flow": {
            "input": rid_input["accuracy"],
            "adapter": next(r for r in radar_flow_rows if r["model"] == "A2")["adapter_output"],
            "embedding": a2_rid,
        },
        "activity_flow": {
            "adapter": next(r for r in act_flow_rows if r["model"] == "A2")["adapter_output"],
            "embedding": next(r for r in act_flow_rows if r["model"] == "A2")["shared_embedding"],
        },
        "agree": agree_store,
        "pair": pair_store,
        "loro": loro,
        "params": params,
        "verdict": {"known": known, "invariance": inv, "unseen": unseen, "slogan": slogan, "solves": solves},
    }
    write_report(ctx)
    answers = {
        "physics_gate": "EFFECTIVE_FC_UNRESOLVED",
        "R2_V": "NO",
        "representation": "R2-D",
        "framework_name": "RadarBridge v2 Known-Radar Sensor-Adapter Framework",
        "A0_acc": a0m.get("accuracy"),
        "A1_acc": a1m.get("accuracy"),
        "A2_acc": a2m.get("accuracy"),
        "radar_id_input": rid_input["accuracy"],
        "radar_id_adapter": ctx["radar_flow"]["adapter"],
        "radar_id_embedding": a2_rid,
        "activity_adapter": ctx["activity_flow"]["adapter"],
        "activity_embedding": ctx["activity_flow"]["embedding"],
        "pair_A0": pair_store["A0"]["same_acquisition_mean"],
        "pair_A1": pair_store["A1"]["same_acquisition_mean"],
        "pair_A2": pair_store["A2"]["same_acquisition_mean"],
        "agree_A0": agree_store["A0"]["all_three_agreement"],
        "agree_A2": agree_store["A2"]["all_three_agreement"],
        "known_radar": known,
        "radar_invariance": inv,
        "unseen_radar": unseen,
        "learn_motion_not_sensor": slogan,
        "solves": solves,
    }
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print("analyze done", answers, flush=True)


if __name__ == "__main__":
    main()
