#!/usr/bin/env python3
"""RadarMind v0.2 final report from existing result files."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
V02 = ROOT / "results" / "radarmind_v02"


def _j(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None


def main() -> None:
    V02.mkdir(parents=True, exist_ok=True)
    p1a_r = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "random_record" / "resnet18" / "metrics.json") or {}
    p1a_p = _j(ROOT / "results" / "phase1_ci4r" / "baselines" / "prefix_tail4_disjoint" / "resnet18" / "metrics.json") or {}
    p1b = _j(ROOT / "results" / "phase1_pointcloud" / "baselines" / "mri_si_gru_xyz" / "metrics.json") or {}
    rad = _j(ROOT / "results" / "phase1_pointcloud" / "baselines" / "radhar_official_gru_xyzv" / "metrics.json") or {}
    p5 = _j(ROOT / "results" / "phase5_open_set" / "eval_summary.json")
    full = _j(ROOT / "results" / "phase5_open_set" / "full_loao.json") or {}
    smoke = ROOT / "results" / "radarmind_v02" / "smoke_test_report.md"
    smoke_txt = smoke.read_text(encoding="utf-8") if smoke.is_file() else "smoke not run"
    lines = [
        "# RadarMind v0.2",
        "",
        "## A. 当前实际完成了什么",
        "",
        "- CI4R 77 GHz micro-Doppler HAR（ResNet18）",
        "- mRI / RadHAR 点云 HAR（PointNet-GRU）",
        "- mRI subject-independent",
        "- CI4R 跨雷达（within / zero-shot / CORAL）",
        "- Shared Motion Representation v0.1",
        "- labeled-interval continuous replay + Timeline + DeepSeek",
        "- Replay Application",
        "- Open-Set Motion Recognition MVP（LOAO，MSP / Energy / Prototype）",
        "",
        "## B. 每一项用的真实 dataset",
        "",
        "- Phase 1A：CI4R 77 GHz IQ → spectrogram cache",
        "- Phase 1B：mRI videolabel 对齐后的 labeled windows；RadHAR official txt",
        "- Phase 2B / 3：CI4R official PNG，7 类，327 条 timestamp 对齐 triple",
        "- Phase 4：mRI Phase 1B labeled-segment windows，不是 raw CSV full-stream",
        "- Phase 5：同一份 mRI cache，T=32，P=64，XYZ",
        "",
        "## C. 已经得到的 scientific findings",
        "",
        f"- CI4R random-record Acc={p1a_r.get('accuracy')}；prefix-group-disjoint Acc={p1a_p.get('accuracy')}。后者不是 subject-independent。",
        f"- mRI SI PointNet-GRU XYZ Acc={p1b.get('accuracy')} Macro-F1={p1b.get('macro_f1')}。random-segment 明显高于 SI。",
        f"- RadHAR official PointNet-GRU XYZV Acc={rad.get('accuracy')} Macro-F1={rad.get('macro_f1')}。",
        "- 跨雷达 zero-shot 很难（24→77 最差）。CORAL 77→10 低于 source-only，是负结果。",
        "- Shared encoder 提高活动分类，但 Radar-ID probe 仍为 1.000，不是 sensor-invariant。",
        "- Phase 4 只能做 labeled-interval continuous inference。true full-stream 需要原始 mRI CSV。",
        "- Phase 5：13-fold LOAO。Prototype AUROC 0.628 ± 0.125，好于 Energy / MSP。MSP 接近随机。不足以称 open-world。",
        "",
        "## D. Negative results",
        "",
        "- CI4R prefix-group-disjoint 大幅低于 random-record。",
        "- mRI SI 远低于 random-segment。",
        "- CORAL 未超过 source-only。",
        "- Shared Motion 的雷达身份仍可被线性探针完全分开。",
        "- Phase 3 LORO M2 未稳定超过 M0。",
        "- closed-set 低置信度不能直接当成 unknown detection。",
        "",
        "## E. 仍只是 MVP 的部分",
        "",
        "- Shared Motion Representation v0.1",
        "- Continuous labeled-interval inference（不是 full-stream segmentation）",
        "- Open-Set Motion Recognition MVP",
        "- DeepSeek 只读 MotionIntelligence / Timeline",
        "",
        "## F. PPT 目标仍未完成",
        "",
        "- open-world understanding",
        "- open-vocabulary recognition",
        "- sensor-invariant motion representation",
        "- continuous full-stream segmentation（含 UNLABELED）",
        "- cross-environment",
        "- MM-Fi / MiliPoint / FAU",
        "- Live Radar",
        "",
        "## G. 下一步研究优先级",
        "",
        "1. 拿到原始 mRI CSV 后做 true full-stream（含 UNLABELED）",
        "2. 在 open-set 上改进 pose_* 混淆（相似姿态会被吸进 known）",
        "3. 跨环境，而不是再挖 Phase 3 adversarial",
        "4. 新数据（MM-Fi 等）只在现有协议走通之后再加",
        "",
        "Open-set 只是 open-world 的一部分。当前不能称 open-world。",
        "",
        "## Smoke",
        "",
        smoke_txt.splitlines()[2] if smoke.is_file() and len(smoke_txt.splitlines()) > 2 else smoke_txt,
    ]
    (V02 / "radarmind_v02_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("[v02-report]", V02 / "radarmind_v02_report.md", flush=True)


if __name__ == "__main__":
    main()
