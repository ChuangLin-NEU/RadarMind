#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.continuous.common import OUT  # noqa: E402


def _rows(name: str) -> list[dict]:
    p = OUT / name
    if not p.is_file():
        return []
    return list(csv.DictReader(p.open(encoding="utf-8")))


def _f(row, key):
    try:
        return float(row[key])
    except (TypeError, ValueError, KeyError):
        return None


def _fmt(v, d=3):
    if v is None:
        return "未出结果"
    return f"{float(v):.{d}f}"


def _mean(rows, key, mode=None):
    xs = [_f(r, key) for r in rows if (mode is None or r.get("mode") == mode) and _f(r, key) is not None]
    return sum(xs) / len(xs) if xs else None


def main() -> None:
    man = _rows("mri_recording_manifest.csv")
    runs = _rows("continuous_runs.csv")
    segs = _rows("segment_metrics.csv")
    bnds = _rows("boundary_metrics.csv")
    demos = []
    if (OUT / "demo_registry.json").is_file():
        demos = json.loads((OUT / "demo_registry.json").read_text(encoding="utf-8")).get("demos", [])
    sm = {}
    if (OUT / "smoothing_choice.json").is_file():
        sm = json.loads((OUT / "smoothing_choice.json").read_text(encoding="utf-8"))
    n_rec = len(man)
    n_multi = sum(1 for r in man if int(r.get("labeled_segments") or 0) >= 2)
    test_sub = sorted({r["subject"] for r in man if r.get("split") == "test"})
    lines = [
        "# Phase 4 Continuous Motion Understanding MVP",
        "",
        "不是 open-world recognition。分类器仍是 mRI 13 类 subject-independent PointNet-GRU，未重训。",
        "协议名称：labeled-interval continuous replay（continuous multi-action inference over labeled intervals）。",
        "Phase 1B 已有 labeled-segment windows 按 Frame # 串接。UNLABELED 只有 GT，未做雷达推理。",
        "true full-stream inference including unlabeled intervals 需要原始 mRI radar CSV。数字未改。",
        "",
        "## 1. mRI 是否真正包含 continuous multi-action recordings？",
        "",
        f"是。20 条 recording（一人一 CSV）。其中 {n_multi}/{n_rec} 条有不少于 2 个 labeled 动作段。",
        "例如 subject10 一条 `subject10_0504.csv` 内含 T pose、pose_1…pose_10、free_form、walk。",
        "没有把训练窗手工拼成假连续流。",
        "",
        "## 2. 完整 recording 的层级",
        "",
        "subject → 一条 radar CSV → videolabels JSON 的多个 [start,end]（Frame #）→ 帧 → 点。",
        "段外、段间空隙为 UNLABELED。相邻段共用边界帧时，后一段从该帧起算，不把空隙贴给邻段。",
        "",
        "## 3. sliding-window 如何定义",
        "",
        "T=32，stride=16（Phase 1B labeled-segment cache）。绝对帧 = segment_start + 窗偏移。",
        "这是 labeled-interval continuous inference，不是 full-stream。本机没有 raw CSV，不能在 UNLABELED 上滑窗。",
        "评测只走原 SI **test subjects**：" + ", ".join(test_sub) + "。val 只用于选平滑阈值。",
        "",
        f"## 4. smoothing 是否改善",
        "",
        f"val 选定 confidence_aware conf_min={sm.get('conf_min')}（val macro-F1={_fmt(sm.get('val_macro_f1'))}）。",
        f"- raw window Acc {_fmt(_mean(runs, 'window_accuracy', 'raw'))} / F1 {_fmt(_mean(runs, 'window_macro_f1', 'raw'))}",
        f"- isolated_flip Acc {_fmt(_mean(runs, 'window_accuracy', 'isolated_flip'))} / F1 {_fmt(_mean(runs, 'window_macro_f1', 'isolated_flip'))}",
        f"- confidence_aware Acc {_fmt(_mean(runs, 'window_accuracy', 'confidence_aware'))} / F1 {_fmt(_mean(runs, 'window_macro_f1', 'confidence_aware'))}",
        "",
        "## 5–7. Mode A 指标（只评 labeled）",
        "",
        f"- frame Acc {_fmt(_mean(runs, 'frame_accuracy', 'confidence_aware'))} / F1 {_fmt(_mean(runs, 'frame_macro_f1', 'confidence_aware'))}",
        f"- window Acc {_fmt(_mean(runs, 'window_accuracy', 'confidence_aware'))} / F1 {_fmt(_mean(runs, 'window_macro_f1', 'confidence_aware'))}",
        f"- segment F1 {_fmt(_mean(segs, 'segment_f1', 'confidence_aware'))}（P {_fmt(_mean(segs, 'segment_precision', 'confidence_aware'))} / R {_fmt(_mean(segs, 'segment_recall', 'confidence_aware'))}）",
        f"- median boundary error {_fmt(_mean(bnds, 'median_frames', 'confidence_aware'), 1)} frames / {_fmt(_mean(bnds, 'median_seconds', 'confidence_aware'), 2)} s",
        "",
        "## 8. 连续混淆",
        "",
        "见 `figures/fig4_confusion.png`。pose_* 之间最容易连成一段。",
        "",
        "## 9. velocity / motion-energy",
        "",
        "来自窗内 Doppler 的 PHYSICS_DERIVED 统计，单位未经标定。可画曲线，不能单独当分割器。",
        "",
        "## 10–11. LLM",
        "",
        "只吃 Timeline JSON + series 摘要。pose_* 不得意译。低置信度必须写「模型低置信度地判断为 …」。",
        "uncertainty 指示器 ≠ open-set。",
        "",
        "## 12. Continuous Replay",
        "",
        "ReplayEngine mode=continuous：按绝对 Frame # seek。覆盖范围为 labeled windows，不是整条 CSV 的 full-stream。",
        "",
        "## 13. 能不能称为 open-world？",
        "",
        "**不能。** 没有 open-set / open-vocabulary 评测。当前是 Continuous Motion Understanding MVP。",
        "",
        "## 14. cross-environment 能不能做？",
        "",
        "**不能。** 见 `cross_environment_feasibility.md`。ProtocolRegistry.cross_environment.available=false。",
        "",
        "## 15. 离 PPT 完整目标还缺什么",
        "",
        "- sensor-invariant representation（Phase 3 Radar-ID probe 仍为 1.0）",
        "- 真正 open-set / open-vocabulary",
        "- 有环境标注后的 cross-environment",
        "- 新雷达集（MiliPoint / MM-Fi / FAU）与 Live Radar",
        "- 在 UNLABELED 上的 background detector（本阶段明确不做）",
        "",
        "## Demos",
        "",
    ]
    for d in demos:
        lines.append(f"- {d.get('role')} {d.get('recording')} ({d.get('subject')}): {d.get('why')}")
    lines += ["", "指标文件：continuous_runs.csv / segment_metrics.csv / boundary_metrics.csv", ""]
    (OUT / "phase4_continuous_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote report", flush=True)


if __name__ == "__main__":
    main()
