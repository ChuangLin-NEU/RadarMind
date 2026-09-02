# RadarMind

雷达人体动作识别的泛化评测与回放系统。冻结版本：v0.2（2026-08）。

当前能复现的是：CI4R / mRI / RadHAR 上的 HAR、mRI cross-subject、CI4R cross-radar、Shared Motion v0.1、labeled-interval continuous replay、open-set recognition MVP。  
不能写成已完成：sensor-invariant、open-vocabulary、open-world、cross-environment、Live Radar。

## 启动

环境：`airhar`（Python 3.11，torch 2.7.1+cu128）。不要用 cgcnn。

```
conda activate airhar
python run_radarmind.py
```

`http://localhost:8502`

现场无网：

```
python run_radarmind.py --offline-demo
```

或 `python run_radarmind_offline.py`。说明层走模板；点云、分类器和结果文件仍是本机真实产物。

## 目录

- `radarmind/` Application、adapter、engine、LLM
- `scripts/` 各 Phase 训练与评测（已跑完的不要覆盖结果）
- `data_splits/` 划分表
- `results/phase1_ci4r/` … `results/phase5_open_set/` 冻结实验结果
- `results/frozen_2026_08/` 本阶段收口清单
- `results/paper_package/` 论文表/图/故事线
- `results/presentation_package/` 演示稿与 PPT 修订说明

## Datasets

| 数据集 | 表示 | 本仓库用法 |
|---|---|---|
| CI4R 77 GHz | micro-Doppler cache | random-record；prefix-group-disjoint（≠ cross-subject） |
| CI4R 10/24/77 official PNG | PNG | within / zero-shot / CORAL / Shared Motion |
| mRI | point cloud T=32 P=64 | SI、LOAO、labeled-interval |
| RadHAR | point cloud T=60 | official Train/Test，无 subject |

原始大包不进仓库。Application 读 `results/*/cache`。

## Preprocessing

已完成，读 cache。不要为了涨数字重切 test。mRI 无 raw CSV 时不能做 UNLABELED 全流。

## Training / Evaluation

数字以各 Phase 的 `metrics.json` / `training_runs.csv` 为准。需要重训时才跑对应脚本，本冻结阶段不要重训已有 checkpoint。

- Phase 1：`scripts/pointcloud/phase1b_train.py`（GPU）
- Phase 2 / 3：`results/phase2_cross_radar/`、`results/phase3_shared_motion/`
- Phase 4：`python run_continuous_demo.py`（CPU）
- Phase 5：`python run_open_set_experiments.py`（GPU 训练，已完成 13-fold）

## Application

页面：首页、回放、连续运动、Open-Set Recognition、泛化对照、表征消融、Cross-Radar、Shared Motion、RadarMind Pipeline、研究进度。

DeepSeek 可选：`.env` 里 `DEEPSEEK_API_KEY`（已 gitignore）。没有 key 或 `--offline-demo` 时用模板。

## Result reproduction（只读）

```
python scripts/freeze/build_package.py
python scripts/paper/build_tables.py
python scripts/paper/build_figures.py
python smoke_test_all.py
```

## GPU / CPU

- 训练：GPU，`airhar`
- 回放、评测表、Application：CPU 可跑（点云推理有 CUDA 更快）
- DeepSeek：可选外网

## Known limitations

见 `results/frozen_2026_08/limitations.md`。主要：CI4R 无可靠 subject ID；continuous 不是 full-stream；Radar-ID probe=1.0；open-set ≠ open-world；多为单 seed。
