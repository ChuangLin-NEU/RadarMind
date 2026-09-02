#!/usr/bin/env python3
"""Assemble RadarMind v0.1 artifacts. Does not overwrite Phase 1A/1B."""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from radarmind.data.base import DatasetRegistry
from radarmind.engine import RadarMindEngine
from radarmind.models.registry import MODEL_SPECS, ModelRegistry
from radarmind.paths import (
    CI4R_CACHE,
    CI4R_PREFIX_SPLIT,
    CI4R_RANDOM_SPLIT,
    CI4R_RESNET_PREFIX,
    CI4R_RESNET_RANDOM,
    MRI_CACHE,
    MRI_GRU_SI_XYZ,
    MRI_SI_SPLIT,
    PHASE1A,
    PHASE1B,
    RADHAR_CACHE,
    RADHAR_GRU_XYZV,
    RADHAR_SPLIT,
    RADHAR_VOXEL,
    ROOT as RROOT,
    SPLITS,
    V01,
)
from radarmind.protocols import ProtocolRegistry
from radarmind.results_io import ablation_block, generalization_table, temporal_rows
from radarmind.tests.smoke import main as run_smoke


def dump(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def artifact_registry() -> dict:
    models = []
    for name, spec in MODEL_SPECS.items():
        ckpt = Path(spec["checkpoint"])
        models.append(
            {
                "name": name,
                "dataset": spec["dataset"],
                "protocol": spec["protocol"],
                "features": spec["features"],
                "checkpoint": spec["checkpoint"],
                "checkpoint_exists": ckpt.is_file(),
                "bytes": ckpt.stat().st_size if ckpt.is_file() else 0,
            }
        )
    return {
        "version": "0.1.0",
        "frozen": {
            "phase1_ci4r": str(PHASE1A),
            "phase1_pointcloud": str(PHASE1B),
            "data_splits": str(SPLITS),
            "overwrite": False,
        },
        "datasets": [
            {
                "dataset": "ci4r",
                "representation": "micro_doppler",
                "radar_type": "77GHz_FMCW",
                "cache": str(CI4R_CACHE),
                "cache_exists": CI4R_CACHE.is_file(),
                "splits": {
                    "within_distribution": str(CI4R_RANDOM_SPLIT),
                    "prefix_group_disjoint": str(CI4R_PREFIX_SPLIT),
                },
                "note": "prefix_group_disjoint is not subject-independent",
            },
            {
                "dataset": "mri",
                "representation": "point_cloud",
                "radar_type": "mRI_mmwave",
                "cache": str(MRI_CACHE),
                "cache_exists": MRI_CACHE.is_file(),
                "splits": {"subject_independent": str(MRI_SI_SPLIT)},
            },
            {
                "dataset": "radhar",
                "representation": "point_cloud",
                "radar_type": "TI_IWR1443",
                "cache": str(RADHAR_CACHE),
                "cache_exists": RADHAR_CACHE.is_file(),
                "splits": {"official_split": str(RADHAR_SPLIT)},
            },
        ],
        "models": models,
        "metrics": {
            "ci4r_random_resnet18": str(PHASE1A / "baselines/random_record/resnet18/metrics.json"),
            "ci4r_prefix_resnet18": str(PHASE1A / "baselines/prefix_tail4_disjoint/resnet18/metrics.json"),
            "phase1b_training_runs": str(PHASE1B / "training_runs.csv"),
        },
        "source_result_files": [
            "results/phase1_ci4r/baselines/random_record/resnet18/metrics.json",
            "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json",
            "results/phase1_pointcloud/training_runs.csv",
        ],
    }


def _scan_until(dataset, protocol, model_name, want_correct: bool, max_n=80):
    eng = RadarMindEngine()
    ds = DatasetRegistry.load_dataset(dataset)
    ids = ds.sample_ids(split="test", protocol=protocol)
    hits = []
    for sid in ids[:max_n]:
        s = ds.get(sid)
        if s.action_gt == "UNKNOWN":
            continue
        p = eng.infer(s, model_name=model_name, protocol=protocol)
        ok = p.activity == s.action_gt
        if ok == want_correct:
            hits.append(
                {
                    "sample_id": sid,
                    "action_gt": s.action_gt,
                    "pred": p.activity,
                    "confidence": p.confidence,
                    "entropy": p.prediction_entropy,
                    "split": "test",
                    "protocol": protocol,
                    "model": model_name,
                    "correct": ok,
                    "subject_id": s.subject_id,
                }
            )
            if len(hits) >= 1:
                break
    return hits


def _radhar_sequence(max_windows=24):
    ds = DatasetRegistry.load_dataset("radhar")
    seqs = ds.sequence_ids(split="test", protocol="official_split")
    best = None
    for seq in seqs:
        wins = ds.windows_in_sequence(seq)
        if len(wins) >= 8:
            best = (seq, wins[:max_windows])
            break
    if best is None and seqs:
        wins = ds.windows_in_sequence(seqs[0])
        best = (seqs[0], wins[:max_windows])
    return best


def pick_demos() -> list[dict]:
    demos = []
    ok = _scan_until("ci4r", "within_distribution", "ci4r_resnet18", True)
    if ok:
        h = ok[0]
        demos.append(
            {
                "demo_id": "demo_1",
                "title": "CI4R 正确识别",
                "dataset": "ci4r",
                "protocol": "within_distribution",
                "model": "ci4r_resnet18",
                "sample_id": h["sample_id"],
                "split": "test",
                "expected": "correct",
                **h,
            }
        )
    bad = _scan_until("ci4r", "prefix_group_disjoint", "ci4r_resnet18_prefix", False, max_n=120)
    if bad:
        h = bad[0]
        demos.append(
            {
                "demo_id": "demo_2",
                "title": "CI4R prefix-group-disjoint 失败例",
                "dataset": "ci4r",
                "protocol": "prefix_group_disjoint",
                "model": "ci4r_resnet18_prefix",
                "sample_id": h["sample_id"],
                "split": "test",
                "expected": "incorrect",
                "note": "prefix-group-disjoint ≠ subject-independent",
                **h,
            }
        )
    ok = _scan_until("mri", "subject_independent", "mri_pointnet_gru", True, max_n=120)
    if ok:
        h = ok[0]
        ds = DatasetRegistry.load_dataset("mri")
        seg = ds._segment_of(h["sample_id"])
        wins = ds.windows_in_sequence(seg)
        demos.append(
            {
                "demo_id": "demo_3",
                "title": "mRI subject-independent 正确例",
                "dataset": "mri",
                "protocol": "subject_independent",
                "model": "mri_pointnet_gru",
                "sample_id": h["sample_id"],
                "sequence_id": seg,
                "window_ids": wins[:16],
                "split": "test",
                "expected": "correct",
                **h,
            }
        )
    bad = _scan_until("mri", "subject_independent", "mri_pointnet_gru", False, max_n=120)
    if bad:
        h = bad[0]
        ds = DatasetRegistry.load_dataset("mri")
        seg = ds._segment_of(h["sample_id"])
        wins = ds.windows_in_sequence(seg)
        demos.append(
            {
                "demo_id": "demo_4",
                "title": "mRI subject-independent 失败例",
                "dataset": "mri",
                "protocol": "subject_independent",
                "model": "mri_pointnet_gru",
                "sample_id": h["sample_id"],
                "sequence_id": seg,
                "window_ids": wins[:16],
                "split": "test",
                "expected": "incorrect",
                **h,
            }
        )
    seq = _radhar_sequence()
    if seq:
        seq_id, wins = seq
        ds = DatasetRegistry.load_dataset("radhar")
        s0 = ds.get(wins[0])
        demos.append(
            {
                "demo_id": "demo_5",
                "title": "RadHAR official test 时间序列",
                "dataset": "radhar",
                "protocol": "official_split",
                "model": "radhar_pointnet_gru",
                "sample_id": wins[0],
                "sequence_id": seq_id,
                "window_ids": wins,
                "split": "test",
                "n_windows": len(wins),
                "action_gt": s0.action_gt,
                "subject_id": "UNKNOWN",
            }
        )
    return demos


def write_demo_outputs(demos: list[dict]) -> tuple[list[dict], list[dict]]:
    eng = RadarMindEngine()
    tldir = V01 / "demo_timelines"
    tldir.mkdir(parents=True, exist_ok=True)
    timelines = []
    examples = []
    for d in demos:
        kwargs = {
            "dataset": d["dataset"],
            "protocol": d["protocol"],
            "model_name": d["model"],
        }
        if d.get("window_ids"):
            kwargs["sequence_id"] = d.get("sequence_id")
            kwargs["window_ids"] = d["window_ids"]
        else:
            kwargs["sample_id"] = d["sample_id"]
        st = eng.replay_load(**kwargs)
        tl = st.timeline
        dump(tldir / f"{d['demo_id']}.json", tl)
        timelines.append({"demo_id": d["demo_id"], "path": f"demo_timelines/{d['demo_id']}.json", "n_events": len(tl.get("events", []))})
        examples.append(
            {
                "demo_id": d["demo_id"],
                "sample_id": d.get("sample_id"),
                "motion_intelligence": st.intelligence,
                "prediction": st.prediction.to_dict() if st.prediction else None,
            }
        )
    return timelines, examples


def copy_figures() -> list[str]:
    dest = V01 / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    copied = []
    candidates = [
        PHASE1A / "baselines/figures/fig6_split_comparison.png",
        PHASE1A / "figures/fig6_split_comparison.png",
        PHASE1B / "figures/fig_split_comparison.png",
    ]
    for src in PHASE1A.rglob("*.png"):
        if src.name.startswith("fig") or src.name == "cm.png":
            candidates.append(src)
    for src in PHASE1B.rglob("*.png"):
        if "figure" in str(src).lower() or src.name.startswith("fig"):
            candidates.append(src)
    seen = set()
    for src in candidates:
        if not src.is_file() or src.name in seen:
            continue
        seen.add(src.name)
        shutil.copy2(src, dest / src.name)
        copied.append(src.name)
        if len(copied) >= 12:
            break
    return copied


def write_gpu_jobs(missing: list[dict]) -> None:
    gdir = ROOT / "gpu_jobs"
    gdir.mkdir(exist_ok=True)
    if not missing:
        note = "All registered checkpoints exist. No formal retrain required for RadarMind v0.1.\n"
        (gdir / "train_missing_models.ps1").write_text("# " + note, encoding="utf-8")
        (gdir / "train_missing_models.sh").write_text("#!/usr/bin/env bash\necho '" + note.strip() + "'\n", encoding="utf-8")
        return
    lines_ps = [
        "# Run on RTX 4060 / local CUDA. Do not train on 01 Linux CPU.",
        "conda activate matgl-cu128",
        "Set-Location (Split-Path $PSScriptRoot -Parent)",
    ]
    lines_sh = ["#!/usr/bin/env bash", "set -euo pipefail", "cd \"$(dirname \"$0\")/..\"", "# GPU only"]
    for m in missing:
        lines_ps.append(f"Write-Host 'MISSING {m['name']} {m['checkpoint']}'")
        lines_sh.append(f"echo MISSING {m['name']} {m['checkpoint']}")
        if m["name"] == "radhar_voxel_cnn_gru":
            lines_ps.append("python scripts/pointcloud/train_voxel.py")
            lines_sh.append("python scripts/pointcloud/train_voxel.py")
    (gdir / "train_missing_models.ps1").write_text("\n".join(lines_ps) + "\n", encoding="utf-8")
    (gdir / "train_missing_models.sh").write_text("\n".join(lines_sh) + "\n", encoding="utf-8")


def write_report(demos, smoke_rows, missing) -> None:
    gen = generalization_table()
    temp = temporal_rows()
    abl_mri = ablation_block("mRI", "subject_independent")
    abl_rad = ablation_block("RadHAR", "official_train_test")
    n_ok = sum(1 for r in smoke_rows if r["ok"])
    lines = [
        "# RadarMind v0.1 report",
        "",
        "本阶段目标是把已有 Phase 1A/1B 的真实 cache、checkpoint 和指标接到统一回放链路。没有重跑 Phase 1，没有进 MiliPoint。",
        "",
        "## 1. 当前支持哪些 dataset？",
        "",
        "- CI4R 77 GHz（652 usable records）",
        "- mRI（20 subjects / 13 actions / 273 segments）",
        "- RadHAR（5 actions，official train/test）",
        "",
        "## 2. 当前支持哪些 radar representation？",
        "",
        "- CI4R：`micro_doppler`（AIRHAR Range-FFT → range gate → STFT → 224×224 z-score）",
        "- mRI / RadHAR：`point_cloud`（`[x,y,z,Doppler/velocity,intensity]`，P/T 见 Phase 1B）",
        "- RadHAR voxel occupancy 有训练结果；Replay 默认走点云。voxel checkpoint 若缺失则不加载。",
        "",
        "## 3. 每种 representation 使用哪个 Sensor Adapter？",
        "",
        "- `micro_doppler` → `SpectrogramAdapter` → `MotionInput.tensor` 为 224×224",
        "- `point_cloud` → `PointCloudAdapter` → `tensor` + `mask` + `temporal_mask`",
        "- v0.1 没有 shared token 训练，只统一软件接口。",
        "",
        "## 4. 每个 dataset 使用哪个 model/checkpoint？",
        "",
        f"- CI4R within_distribution：`ci4r_resnet18` ← `{CI4R_RESNET_RANDOM}`",
        f"- CI4R prefix_group_disjoint：`ci4r_resnet18_prefix` ← `{CI4R_RESNET_PREFIX}`",
        f"- mRI subject_independent：`mri_pointnet_gru` XYZ ← `{MRI_GRU_SI_XYZ}`",
        f"- RadHAR official_split：`radhar_pointnet_gru` XYZV ← `{RADHAR_GRU_XYZV}`",
        f"- RadHAR voxel（可选）：`{RADHAR_VOXEL}` exists={RADHAR_VOXEL.is_file()}",
        "",
        "## 5. MotionPrediction schema 是什么？",
        "",
        "字段：`activity`, `confidence`, `class_probabilities`, `embedding`（ResNet avgpool / GRU last hidden）, `model_name`, `dataset`, `protocol`, `prediction_entropy`, `uncertainty`（entropy / log C）, `logits`, `sample_id`。",
        "",
        "## 6. MotionIntelligence 哪些字段是真实可计算的？",
        "",
        "- 全部 dataset：`activity`/`confidence`/`uncertainty`（MODEL）；`dataset`/`radar`（METADATA）；点云的 Doppler/velocity 统计（PHYSICS_DERIVED）",
        "- mRI：`subject`（METADATA）；`duration`（T / 10 Hz）",
        "- RadHAR：`duration`（T / 30 Hz）；velocity 单位 m/s（README）",
        "- CI4R：spectrogram 列能量统计（PHYSICS_DERIVED，不是 m/s）",
        "",
        "## 7. 哪些字段仍然 UNKNOWN？",
        "",
        "- `direction`：未验证正负 Doppler 与 toward/away 的对应",
        "- CI4R `mean_velocity` / `peak_velocity`：谱图像素没有可靠 Hz→m/s",
        "- mRI 速度单位：`UNIT_UNKNOWN`",
        "- RadHAR / CI4R `subject`；全部 `environment`",
        "- `periodicity`：v0.1 不做周期估计",
        "",
        "## 8. Timeline 如何生成？",
        "",
        "每个 window 一次 `predict`。对标签做多数投票（k=3）和 confidence 阈值（0.35，低于则沿用上一窗）。相邻相同动作合并为 `MotionEvent`。不训练分割模型。CI4R 一条 record 一个事件；mRI/RadHAR 按段/txt 的窗序列回放。",
        "",
        "## 9. LLM 收到什么输入？",
        "",
        "只收 `MotionIntelligence` JSON 和精简后的 `MotionTimeline` events。不收 IQ、点云、spectrogram。",
        "",
        "## 10. 如何防止 LLM hallucination？",
        "",
        "system prompt 禁止描述 JSON 没有的动作/速度/方向/人员。低置信度必须用「可能 / 置信度较低」。UNKNOWN 不得补数。无 `DEEPSEEK_API_KEY` 或 API 失败时走确定性模板。",
        "",
        "## 11. Replay App 是否真正调用模型 inference？",
        "",
        "是。`ReplayEngine.load` 对每个 window 调 `ModelRegistry.predict`，读的是 Phase 1 的 `best.pt`。Application 不直接调 dataset-specific 训练脚本。",
        "",
        "## 12. 哪些 Demo 来自真实 test data？",
        "",
    ]
    for d in demos:
        lines.append(f"- `{d['demo_id']}` {d['title']}  split=test sample=`{d.get('sample_id')}`")
    lines += [
        "",
        "禁止用训练样本冒充 test demo。选取脚本只扫 `split=test`。",
        "",
        "## 13. 当前哪些 generalization protocol 已经有真实结果？",
        "",
        "- within_distribution：CI4R random-record、mRI random-segment",
        "- prefix_group_disjoint：CI4R",
        "- subject_independent：mRI",
        "- official_split：RadHAR",
        "",
        "读到的对照：",
        "",
        "```json",
        json.dumps(gen, indent=2, ensure_ascii=False),
        "```",
        "",
        "Temporal：",
        "",
        "```json",
        json.dumps(temp, indent=2, ensure_ascii=False),
        "```",
        "",
        "## 14. 哪些 protocol 只是代码接口、尚无真实结果？",
        "",
        "- `cross_radar`：available=false，无跨雷达训练模型",
        "- `cross_environment`：available=false，无已校验环境字段",
        "- `cross_dataset`：available=false，标签空间与表征未对齐",
        "",
        json.dumps(ProtocolRegistry().unavailable(), ensure_ascii=False, indent=2),
        "",
        "## 15. RadarMind v0.1 当前最主要限制是什么？",
        "",
        "- 三个 dataset 仍是各自模型，没有 shared motion representation。",
        "- CI4R 没有 subject id，prefix-group-disjoint 不能当成 cross-subject。",
        "- RadHAR 没有 subject / environment。",
        "- 点云坐标系和单位在 mRI 与 RadHAR 之间不一致，不能直接数值迁移。",
        "- Timeline 是窗级平滑，不是连续分割模型。",
        "- LLM 只解释结构化结果，不看雷达图。",
        "",
        "## 16. 下一步进入 cross-radar 需要补什么？",
        "",
        "优先补一套**同一动作标签空间、可对齐时间窗**的双雷达数据，而不是再扩单数据集 baseline。CI4R 已有 10 GHz 未纳入 v0.1，但它和 77 GHz 是同一套动作词、同一批采集前缀；这是当前仓库里最接近 cross-radar 的真实条件。同时要先写清：谱图像素不能当 m/s、点云轴向不能混。Shared Motion Representation 应建立在已经对齐的 MotionInput / 物理统计之上，而不是先上大模型搜索。",
        "",
        "## Ablation text（由 training_runs.csv 生成）",
        "",
        abl_mri["text"],
        "",
        abl_rad["text"],
        "",
        f"## Smoke  {n_ok}/{len(smoke_rows)}",
        "",
        f"## Missing checkpoints  {len(missing)}",
        "",
        json.dumps(missing, indent=2),
        "",
    ]
    (V01 / "radarmind_v01_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    V01.mkdir(parents=True, exist_ok=True)
    (V01 / "demo_timelines").mkdir(exist_ok=True)
    (V01 / "figures").mkdir(exist_ok=True)
    dump(V01 / "artifact_registry.json", artifact_registry())
    missing = ModelRegistry().missing()
    write_gpu_jobs(missing)
    print("[v01] picking demos", flush=True)
    demos = pick_demos()
    dump(V01 / "demo_registry.json", {"n": len(demos), "rule": "test split only", "demos": demos})
    print("[v01] writing timelines", flush=True)
    tls, examples = write_demo_outputs(demos)
    dump(V01 / "motion_intelligence_examples.json", examples)
    copy_figures()
    print("[v01] smoke", flush=True)
    smoke = run_smoke()
    write_report(demos, smoke, missing)
    print(json.dumps({"demos": len(demos), "smoke_ok": sum(1 for r in smoke if r["ok"]), "missing": missing}, indent=2), flush=True)


if __name__ == "__main__":
    main()
