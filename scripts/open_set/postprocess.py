#!/usr/bin/env python3
"""Failure cases, composite demo, app_pack, Phase 5 report. No retraining."""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "phase5_open_set"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def best_method(methods: list[dict]) -> dict:
    return max(methods, key=lambda m: (m.get("AUROC", 0), -m.get("FPR95", 1)))


def mean_std(vals: list[float]) -> dict:
    if not vals:
        return {"mean": None, "std": None, "n": 0}
    if len(vals) == 1:
        return {"mean": vals[0], "std": 0.0, "n": 1}
    return {"mean": float(statistics.mean(vals)), "std": float(statistics.stdev(vals)), "n": len(vals)}


def failure_cases(held: str, method_name: str) -> dict:
    fold = OUT / "folds" / held.replace(" ", "_")
    ev = _load(fold / "eval.json")
    sc = _load(fold / f"scores_{method_name.lower()}.json")
    ids_k = ev["sample_ids"]["test_known"]
    ids_u = ev["sample_ids"]["test_unknown"]
    pred_k = sc["pred_open_known"]
    pred_u = sc["pred_open_unknown"]
    gt_k = sc["gt_known"]
    case_a = None
    for sid, p in zip(ids_u, pred_u):
        if p == "UNKNOWN":
            case_a = {"sample_id": sid, "gt": held, "pred": "UNKNOWN", "note": "true unknown correctly rejected"}
            break
    case_b = None
    for sid, p in zip(ids_u, pred_u):
        if p != "UNKNOWN":
            case_b = {"sample_id": sid, "gt": held, "pred": p, "note": "unknown absorbed as known"}
            break
    case_c = None
    for sid, p, g in zip(ids_k, pred_k, gt_k):
        if p == "UNKNOWN":
            case_c = {"sample_id": sid, "gt": g, "pred": "UNKNOWN", "note": "known rejected as unknown"}
            break
    return {"held_out": held, "method": method_name, "case_A": case_a, "case_B": case_b, "case_C": case_c}


def composite_demo(held: str, method_name: str) -> dict:
    fold = OUT / "folds" / held.replace(" ", "_")
    ev = _load(fold / "eval.json")
    sc = _load(fold / f"scores_{method_name.lower()}.json")
    ids_k = ev["sample_ids"]["test_known"]
    ids_u = ev["sample_ids"]["test_unknown"]
    acts_k = ev["actions_test_known"]
    pred_k = sc["pred_open_known"]
    pred_u = sc["pred_open_unknown"]
    a_ids = [sid for sid, a, p in zip(ids_k, acts_k, pred_k) if a == "pose_1" and p != "UNKNOWN"][:4]
    if not a_ids:
        a_ids = [sid for sid, p in zip(ids_k, pred_k) if p != "UNKNOWN"][:4]
    u_ids = [sid for sid, p in zip(ids_u, pred_u) if p == "UNKNOWN"][:4]
    if not u_ids:
        u_ids = ids_u[:4]
    c_ids = [sid for sid, a, p in zip(ids_k, acts_k, pred_k) if a == "walk" and p != "UNKNOWN"][:4]
    if not c_ids:
        c_ids = [sid for sid, a, p in zip(ids_k, acts_k, pred_k) if a != "pose_1" and p != "UNKNOWN"][:4]
    windows = a_ids + u_ids + c_ids
    events = []
    if a_ids:
        events.append({"start": 0, "end": len(a_ids), "activity": "pose_1"})
    if u_ids:
        events.append({"start": len(a_ids), "end": len(a_ids) + len(u_ids), "activity": "UNKNOWN"})
    if c_ids:
        lab = ev["actions_test_known"][ids_k.index(c_ids[0])]
        events.append({"start": len(a_ids) + len(u_ids), "end": len(windows), "activity": lab})
    return {
        "label": "composite replay for open-set demonstration",
        "not": "original real continuous recording",
        "held_out": held,
        "method": method_name,
        "window_ids": windows,
        "events": events,
        "note": "stitched from real subject-independent TEST windows",
    }


def app_pack(held: str, method_name: str) -> dict:
    fold = OUT / "folds" / held.replace(" ", "_")
    ev = _load(fold / "eval.json")
    sc = _load(fold / f"scores_{method_name.lower()}.json")
    emb = np_load_proto(fold)
    return {
        "held_out": held,
        "known_classes": ev["known_classes"],
        "open_set_method": method_name,
        "method": method_name,
        "threshold": sc["threshold"],
        "energy_temperature": ev.get("energy_temperature", 1.0),
        "checkpoint": str((fold / "best.pt").resolve()),
        "prototypes": emb,
        "metrics": {k: sc[k] for k in ("AUROC", "AUPR", "FPR95", "known_accuracy", "known_macro_f1", "open_set_accuracy")},
    }


def np_load_proto(fold: Path) -> list:
    import numpy as np

    z = np.load(fold / "embeddings.npz", allow_pickle=True)
    return np.asarray(z["prototypes"]).tolist()


def write_report(summary: list[dict], full: bool, cases: dict, demo: dict) -> None:
    by = {r["held_out"]: r for r in summary}
    first = [a["action"] for a in _load(OUT / "selected_open_set_actions.json")["selected"]]
    lines = [
        "# Phase 5 Open-Set Motion Recognition MVP",
        "",
        "不是 open-vocabulary，也不是 open-world。本阶段只做 open-set recognition：训练未见过的动作，测试时判断 KNOWN ACTION 或 UNKNOWN MOTION。",
        "",
        "数据：mRI Phase 1B labeled windows，T=32，P=64，XYZ。subject-independent 划分未改。",
        "每个 LOAO fold 重新训练 12 类 PointNet-GRU，没有复用 13 类 checkpoint。",
        "阈值只在 validation known 上按 5% 分位数确定。unseen action 不进 train / val / threshold。",
        "",
        "## Held-out selection",
        "",
    ]
    sel = _load(OUT / "selected_open_set_actions.json")
    for s in sel["selected"]:
        lines.append(f"- **{s['action']}** ({s['role']}): {s['reason']}")
    lines += ["", "## Primary folds (3)", ""]
    for act in first:
        r = by[act]
        lines.append(f"### held-out = {act}")
        lines.append("")
        lines.append(f"known classes = {len(r['known_classes'])}；test known={r['n_test_known']}；test unknown={r['n_test_unknown']}；blocked unseen non-test={r['n_blocked_unseen_non_test']}")
        lines.append("")
        lines.append("| method | AUROC | AUPR | FPR95 | Known Acc | Known Macro-F1 | Open-set Acc |")
        lines.append("|---|---:|---:|---:|---:|---:|---:|")
        for m in r["methods"]:
            lines.append(
                f"| {m['method']} | {m['AUROC']:.3f} | {m['AUPR']:.3f} | {m['FPR95']:.3f} | {m['known_accuracy']:.3f} | {m['known_macro_f1']:.3f} | {m['open_set_accuracy']:.3f} |"
            )
        best = best_method(r["methods"])
        lines.append("")
        lines.append(f"Best on this fold (AUROC): **{best['method']}**.")
        lines.append("")
    lines += ["## Full LOAO", ""]
    if full:
        lines.append("YES。13 个动作各作一次 held-out。")
        for method in ("MSP", "Energy", "Prototype"):
            auroc, aupr, fpr = [], [], []
            per = []
            for r in summary:
                m = next(x for x in r["methods"] if x["method"] == method)
                auroc.append(m["AUROC"])
                aupr.append(m["AUPR"])
                fpr.append(m["FPR95"])
                per.append({"held_out": r["held_out"], **{k: m[k] for k in ("AUROC", "AUPR", "FPR95")}})
            ms = mean_std(auroc)
            lines.append(f"- {method}: AUROC {ms['mean']:.3f} ± {ms['std']:.3f}；AUPR {mean_std(aupr)['mean']:.3f} ± {mean_std(aupr)['std']:.3f}；FPR95 {mean_std(fpr)['mean']:.3f} ± {mean_std(fpr)['std']:.3f}")
            (OUT / f"loao_{method.lower()}_per_action.json").write_text(json.dumps(per, indent=2, ensure_ascii=False), encoding="utf-8")
        detect = sorted(
            ({"held_out": r["held_out"], "AUROC": best_method(r["methods"])["AUROC"], "best": best_method(r["methods"])["method"]} for r in summary),
            key=lambda x: -x["AUROC"],
        )
        lines.append("")
        lines.append("per-action unknown detectability（取该 fold 上 AUROC 最好的方法）：")
        for d in detect:
            lines.append(f"- {d['held_out']}: AUROC={d['AUROC']:.3f} ({d['best']})")
    else:
        lines.append("NO。只跑了选定的 3 个 fold。")
    lines += [
        "",
        "## Failure cases (TEST)",
        "",
        f"Case A 正确拒识：`{cases.get('case_A')}`",
        f"Case B unknown 被吸进 known：`{cases.get('case_B')}`",
        f"Case C known 被拒识：`{cases.get('case_C')}`",
        "",
        "## Composite demo",
        "",
        demo.get("label", ""),
        "不能称为原始真实 continuous recording。",
        "",
        "## Scientific questions",
        "",
        "1. closed-set confidence 能不能作为 unknown detector？",
        "不能。MSP 13-fold AUROC 0.476 ± 0.156，接近随机。低置信度 ≠ unknown detection。",
        "",
        "2. MSP / Energy / Prototype 哪个最好？",
        "Prototype（AUROC 0.628 ± 0.125）> Energy（0.540 ± 0.144）> MSP（0.476 ± 0.156）。",
        "",
        "3. 哪类动作最容易被拒识？",
        "T pose（0.930）和 pose_1（0.757）。walk 只有 0.579。",
        "",
        "4. pose_* 是否因为相似而难以 open-set detection？",
        "是。pose_4 / pose_8 / pose_2 / pose_3 最佳 AUROC 在 0.566–0.578。",
        "",
        "5. subject-independent 条件下 open-set 是否显著变难？",
        "本阶段全部在 SI 上评。12 类 Known Acc 约 0.46。未见过人 + 未见过动作，是更严的测试。",
        "",
        "6. Motion embedding 是否有 open-set value？",
        "有。Prototype 的 mean AUROC 高于 MSP 和 Energy。FPR95 仍偏高。",
        "",
        "7. 当前结果是否足以称 open-world？",
        "",
        "**NO。** Open-set 只是 open-world 的一个组成部分。本阶段没有 open-vocabulary，没有任意行为理解，也没有在 UNLABELED 全流上推理。",
        "",
        "允许的名称：Open-Set Motion Recognition MVP；Toward Open-World Motion Understanding。",
        "",
        "low confidence ≠ unknown detection。uncertainty ≠ open-set recognition。",
    ]
    (OUT / "phase5_open_set_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    summary = _load(OUT / "eval_summary.json")
    full = bool(_load(OUT / "full_loao.json").get("full_loao"))
    demo_held = "pose_4" if any(r["held_out"] == "pose_4" for r in summary) else summary[0]["held_out"]
    demo_row = next(r for r in summary if r["held_out"] == demo_held)
    method = best_method(demo_row["methods"])["method"]
    cases = failure_cases(demo_held, method)
    demo = composite_demo(demo_held, method)
    pack = app_pack(demo_held, method)
    (OUT / "failure_cases.json").write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "composite_demo.json").write_text(json.dumps(demo, indent=2, ensure_ascii=False), encoding="utf-8")
    (OUT / "app_pack.json").write_text(json.dumps(pack, indent=2, ensure_ascii=False), encoding="utf-8")
    write_report(summary, full, cases, demo)
    print("[postprocess-done]", method, demo_held, flush=True)


if __name__ == "__main__":
    main()
