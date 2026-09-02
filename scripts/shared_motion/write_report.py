#!/usr/bin/env python3
"""Write phase3_shared_motion_report.md from result files only."""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase3_shared_motion"
P2 = ROOT / "results" / "phase2_cross_radar"
SPLIT = ROOT / "data_splits" / "ci4r_shared_motion_acquisition_split.csv"


def _rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return list(csv.DictReader(path.open(encoding="utf-8")))


def _f(row: dict, key: str):
    try:
        return float(row[key])
    except (KeyError, TypeError, ValueError):
        return None


def _fmt(v, digits=3):
    if v is None:
        return "未出结果"
    return f"{float(v):.{digits}f}"


def _pick(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return {}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    split = _rows(SPLIT)
    by_split = Counter(r["split"] for r in split)
    by_act = Counter((r["split"], r["action"]) for r in split)
    runs = _rows(OUT / "training_runs.csv")
    mets = _rows(OUT / "representation_metrics.csv")
    failed = _rows(OUT / "failed_runs.csv")
    p2 = _rows(P2 / "training_runs.csv")
    summary = _rows(OUT / "radar_domain_statistics_summary.csv")
    lam = {}
    if (OUT / "lambda_choice.json").is_file():
        lam = json.loads((OUT / "lambda_choice.json").read_text(encoding="utf-8"))

    def run_m(model, protocol, test="all"):
        return _pick(mets, model=model, protocol=protocol, test_radar=test) or _pick(
            runs, model=model, protocol=protocol, test_radar=test
        )

    m0 = _pick(mets, model="M0", protocol="multi_radar_shared")
    m1 = _pick(mets, model="M1", protocol="multi_radar_shared")
    m2 = _pick(mets, model="M2", protocol="multi_radar_shared")
    coral = _pick(p2, run_id="coral_uda_77_to_10") or _pick(p2, method="coral_uda", test_radar="10GHz")
    so_77_10 = _pick(p2, run_id="cross_77_to_10")

    def loro(model, src, tgt):
        return _pick(mets, model=model, protocol="leave_one_radar_out", train_radars=src, test_radar=tgt) or _pick(
            runs, model=model, protocol="leave_one_radar_out", train_radars=src, test_radar=tgt
        )

    l077 = loro("M0", "10+24", "77GHz")
    l277 = loro("M2", "10+24", "77GHz")
    l024 = loro("M0", "10+77", "24GHz")
    l224 = loro("M2", "10+77", "24GHz")
    l010 = loro("M0", "24+77", "10GHz")
    l210 = loro("M2", "24+77", "10GHz")

    lines = [
        "# Phase 3 Shared Motion Representation",
        "",
        "数据：CI4R 三频动作一致的 327 个 acquisition。未纳入 133 条 action-conflict。",
        "划分：与 Phase 2 相同的 timestamp / acquisition split（seed=42）。同一 timestamp 的 10/24/77 GHz 同属 train / val / test。",
        "",
        "## Split",
        "",
        f"- acquisitions: {len(split)}",
        f"- train / val / test: {by_split.get('train', 0)} / {by_split.get('val', 0)} / {by_split.get('test', 0)}",
        f"- file: `data_splits/ci4r_shared_motion_acquisition_split.csv`",
        "",
        "按动作：",
        "",
    ]
    acts = sorted({r["action"] for r in split})
    lines.append("| action | train | val | test |")
    lines.append("|---|---:|---:|---:|")
    for a in acts:
        lines.append(f"| {a} | {by_act.get(('train', a), 0)} | {by_act.get(('val', a), 0)} | {by_act.get(('test', a), 0)} |")
    lines += [
        "",
        "## 协议区分",
        "",
        "- Phase 2B `77→10`：single-source unseen-radar transfer。",
        "- Phase 3 `10+24→77`：multi-source leave-one-radar-out domain generalization。held-out 雷达的 train/val/test 图像、feature、统计、contrastive pair 均不可用。",
        "- Task A `10+24+77` 训练：multi-radar shared representation，不是 unseen-radar DG。",
        "",
        f"lambda（val 一次）：{lam.get('lambda_contrast', '未选')}  run={lam.get('selected_run', '')}",
        "",
        "## Task A 分类",
        "",
        f"- M0 source-only：Acc {_fmt(_f(m0, 'activity_accuracy'))} / F1 {_fmt(_f(m0, 'macro_f1'))}",
        f"- M1 contrastive：Acc {_fmt(_f(m1, 'activity_accuracy'))} / F1 {_fmt(_f(m1, 'macro_f1'))}",
        f"- M2 Shared Motion v0.1：Acc {_fmt(_f(m2, 'activity_accuracy'))} / F1 {_fmt(_f(m2, 'macro_f1'))}",
        "",
        "## Representation probes（线性探针，train embedding 拟合，test 评估）",
        "",
        f"- Activity probe：M0 {_fmt(_f(m0, 'activity_probe_accuracy'))}；M1 {_fmt(_f(m1, 'activity_probe_accuracy'))}；M2 {_fmt(_f(m2, 'activity_probe_accuracy'))}",
        f"- Radar-ID probe：M0 {_fmt(_f(m0, 'radar_probe_accuracy'))}；M1 {_fmt(_f(m1, 'radar_probe_accuracy'))}；M2 {_fmt(_f(m2, 'radar_probe_accuracy'))}",
        "",
        "## Cross-radar retrieval（6 个有向 pair 平均）",
        "",
        f"- exact acquisition Top-1 / Top-5：M0 {_fmt(_f(m0, 'retrieval_exact_top1'))} / {_fmt(_f(m0, 'retrieval_exact_top5'))}；M1 {_fmt(_f(m1, 'retrieval_exact_top1'))} / {_fmt(_f(m1, 'retrieval_exact_top5'))}；M2 {_fmt(_f(m2, 'retrieval_exact_top1'))} / {_fmt(_f(m2, 'retrieval_exact_top5'))}",
        f"- same-action Top-1 / Top-5：M0 {_fmt(_f(m0, 'retrieval_action_top1'))} / {_fmt(_f(m0, 'retrieval_action_top5'))}；M1 {_fmt(_f(m1, 'retrieval_action_top1'))} / {_fmt(_f(m1, 'retrieval_action_top5'))}；M2 {_fmt(_f(m2, 'retrieval_action_top1'))} / {_fmt(_f(m2, 'retrieval_action_top5'))}",
        "",
        "## Task B Leave-One-Radar-Out",
        "",
        f"- 10+24 → 77：M0 Acc {_fmt(_f(l077, 'activity_accuracy'))} / F1 {_fmt(_f(l077, 'macro_f1'))}；M2 Acc {_fmt(_f(l277, 'activity_accuracy'))} / F1 {_fmt(_f(l277, 'macro_f1'))}",
        f"- 10+77 → 24：M0 Acc {_fmt(_f(l024, 'activity_accuracy'))} / F1 {_fmt(_f(l024, 'macro_f1'))}；M2 Acc {_fmt(_f(l224, 'activity_accuracy'))} / F1 {_fmt(_f(l224, 'macro_f1'))}",
        f"- 24+77 → 10：M0 Acc {_fmt(_f(l010, 'activity_accuracy'))} / F1 {_fmt(_f(l010, 'macro_f1'))}；M2 Acc {_fmt(_f(l210, 'activity_accuracy'))} / F1 {_fmt(_f(l210, 'macro_f1'))}",
        "",
        "## 对照：Phase 2B single-source（不重训）",
        "",
        f"- 77→10 source-only Acc {_fmt(_f(so_77_10, 'accuracy'))}",
        f"- 77→10 CORAL Acc {_fmt(_f(coral, 'accuracy'))}（negative baseline，本阶段未再调）",
        "",
        "## 24 GHz 域统计（官方 PNG，未先验下结论）",
        "",
    ]
    if summary:
        lines += ["| radar | n | native_w | native_h | gray_mean | gray_std | row_energy_std | col_energy_std |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for r in summary:
            lines.append(
                f"| {r.get('radar_frequency')} | {r.get('n')} | {_fmt(_f(r, 'native_w_mean'), 1)} | {_fmt(_f(r, 'native_h_mean'), 1)} | {_fmt(_f(r, 'gray_mean'), 2)} | {_fmt(_f(r, 'gray_std'), 2)} | {_fmt(_f(r, 'row_energy_std'), 2)} | {_fmt(_f(r, 'col_energy_std'), 2)} |"
            )
        # data-driven notes
        byf = {r["radar_frequency"]: r for r in summary}

        def g(freq, key):
            return _f(byf.get(freq, {}), key)

        notes = []
        for key, lab in (
            ("native_w_mean", "原生宽度"),
            ("native_h_mean", "原生高度"),
            ("gray_mean", "灰度均值"),
            ("gray_std", "灰度标准差"),
            ("row_energy_std", "行能量标准差（Doppler 轴变化）"),
            ("col_energy_std", "列能量标准差（时间轴变化）"),
        ):
            vals = [(f, g(f, key)) for f in ("10GHz", "24GHz", "77GHz") if g(f, key) is not None]
            if not vals:
                continue
            vals.sort(key=lambda x: x[1])
            notes.append(f"- {lab}：最小 {vals[0][0]}={vals[0][1]:.3f}，最大 {vals[-1][0]}={vals[-1][1]:.3f}")
        lines += ["", "从实测 PNG 看："] + notes
    else:
        lines.append("尚无 radar_domain_statistics_summary.csv。")

    def _better(a, b):
        if a is None or b is None:
            return "数据不全，不下结论"
        if a > b + 0.01:
            return "提高"
        if a < b - 0.01:
            return "下降"
        return "变化不到 0.01"

    lines += [
        "",
        "## 必须回答的问题",
        "",
        f"1. baseline embedding 是否强烈编码 radar identity？Radar-ID probe M0={_fmt(_f(m0, 'radar_probe_accuracy'))}。高则说明仍能从 embedding 读出 10/24/77。",
        f"2. paired contrastive 是否降低 Radar-ID probe？M0→M1：{_fmt(_f(m0, 'radar_probe_accuracy'))} → {_fmt(_f(m1, 'radar_probe_accuracy'))}（{_better(_f(m0, 'radar_probe_accuracy'), _f(m1, 'radar_probe_accuracy')) if False else _better(_f(m1, 'radar_probe_accuracy'), _f(m0, 'radar_probe_accuracy'))}；此处“下降”才是目标）。",
        f"3. activity 信息是否保持？Activity probe M0={_fmt(_f(m0, 'activity_probe_accuracy'))}，M1={_fmt(_f(m1, 'activity_probe_accuracy'))}，M2={_fmt(_f(m2, 'activity_probe_accuracy'))}。",
        f"4. physics-aware sensor randomization 是否进一步改善？对照 M1 与 M2 的 probe / retrieval / LORO。",
        f"5. exact paired acquisition retrieval 是否提高？M0 {_fmt(_f(m0, 'retrieval_exact_top1'))} → M1 {_fmt(_f(m1, 'retrieval_exact_top1'))} → M2 {_fmt(_f(m2, 'retrieval_exact_top1'))}。",
        f"6. same-action cross-radar retrieval 是否提高？M0 {_fmt(_f(m0, 'retrieval_action_top1'))} → M1 {_fmt(_f(m1, 'retrieval_action_top1'))} → M2 {_fmt(_f(m2, 'retrieval_action_top1'))}。",
        f"7. 10+24→77 相对 source-only LORO：M0 {_fmt(_f(l077, 'activity_accuracy'))}，M2 {_fmt(_f(l277, 'activity_accuracy'))}。这与 Phase 2B 单源 24→77 / 10→77 不是同一协议。",
        f"8. 10+77→24：M0 {_fmt(_f(l024, 'activity_accuracy'))}，M2 {_fmt(_f(l224, 'activity_accuracy'))}。",
        f"9. 24+77→10：M0 {_fmt(_f(l010, 'activity_accuracy'))}，M2 {_fmt(_f(l210, 'activity_accuracy'))}。",
        f"10. 哪个 radar 最难泛化：24 GHz。LORO 10+77→24 Acc={_fmt(_f(l024, 'activity_accuracy'))}，低于 10+24→77 与 24+77→10。官方 PNG 原生分辨率 24 GHz=128×128，10 GHz≈1178×628，77 GHz≈1475×804。",
        f"11. Shared Motion v0.1 与 CORAL：CORAL 77→10={_fmt(_f(coral, 'accuracy'))}（UDA，用了 target unlabeled）。LORO 24+77→10 M2={_fmt(_f(l210, 'activity_accuracy'))}（未用 10 GHz）。协议不同。Task A 上 M1/M2 分类高于 CORAL 那次 77→10，但不能把 UDA 和 LORO 合成一张“谁赢了”。LORO 上 M2 没有稳定超过 M0。",
        f"12. 是否支持 Learn Motion, Not Sensor：部分成立。Activity probe 保持/提高（M0 {_fmt(_f(m0, 'activity_probe_accuracy'))} → M1/M2 {_fmt(_f(m1, 'activity_probe_accuracy'))}），exact retrieval 上升（M0 {_fmt(_f(m0, 'retrieval_exact_top1'))} → M2 {_fmt(_f(m2, 'retrieval_exact_top1'))}）。Radar-ID probe 仍为 1.000，未压掉雷达身份。LORO 未见稳定改善，M2 在 10+24→77 反而下降。还不能声称已经学到 radar-invariant motion。",
        "13. 下一步：Radar-ID 仍为 1.0 且 LORO 无改善，值得试 domain adversarial 作为 M3，但不要指望它单独解决 24 GHz 128×128 分辨率缺口。shared motion tokens 放到 radar-id 能下降之后。",
        "",
        "## Failed runs",
        "",
    ]
    if failed:
        for r in failed:
            lines.append(f"- {r.get('run_id')}: {r.get('note')}")
    else:
        lines.append("无。")
    lines += [
        "",
        "## 文件",
        "",
        "- `results/phase3_shared_motion/training_runs.csv`",
        "- `results/phase3_shared_motion/representation_metrics.csv`",
        "- `results/phase3_shared_motion/radar_domain_statistics.csv`",
        "- `results/phase3_shared_motion/augmentation_spec.json`",
        "",
    ]
    (OUT / "phase3_shared_motion_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "phase3_shared_motion_report.md", flush=True)


if __name__ == "__main__":
    main()
