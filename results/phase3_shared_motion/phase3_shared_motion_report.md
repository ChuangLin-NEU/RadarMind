# Phase 3 Shared Motion Representation

数据：CI4R 三频动作一致的 327 个 acquisition。未纳入 133 条 action-conflict。
划分：与 Phase 2 相同的 timestamp / acquisition split（seed=42）。同一 timestamp 的 10/24/77 GHz 同属 train / val / test。

## Split

- acquisitions: 327
- train / val / test: 229 / 49 / 49
- file: `data_splits/ci4r_shared_motion_acquisition_split.csv`

按动作：

| action | train | val | test |
|---|---:|---:|---:|
| Away | 37 | 8 | 8 |
| Bend | 35 | 7 | 7 |
| Crawl | 31 | 7 | 7 |
| Kneel | 32 | 7 | 7 |
| Pick | 39 | 8 | 8 |
| Sit | 31 | 7 | 7 |
| Towards | 24 | 5 | 5 |

## 协议区分

- Phase 2B `77→10`：single-source unseen-radar transfer。
- Phase 3 `10+24→77`：multi-source leave-one-radar-out domain generalization。held-out 雷达的 train/val/test 图像、feature、统计、contrastive pair 均不可用。
- Task A `10+24+77` 训练：multi-radar shared representation，不是 unseen-radar DG。

lambda（val 一次）：0.1  run=M1_l01

## Task A 分类

- M0 source-only：Acc 0.769 / F1 0.749
- M1 contrastive：Acc 0.932 / F1 0.937
- M2 Shared Motion v0.1：Acc 0.878 / F1 0.884

## Representation probes（线性探针，train embedding 拟合，test 评估）

- Activity probe：M0 0.918；M1 0.952；M2 0.952
- Radar-ID probe：M0 1.000；M1 1.000；M2 1.000

## Cross-radar retrieval（6 个有向 pair 平均）

- exact acquisition Top-1 / Top-5：M0 0.126 / 0.520；M1 0.190 / 0.755；M2 0.231 / 0.697
- same-action Top-1 / Top-5：M0 0.718 / 0.918；M1 0.935 / 0.990；M2 0.864 / 0.983

## Task B Leave-One-Radar-Out

- 10+24 → 77：M0 Acc 0.347 / F1 0.215；M2 Acc 0.204 / F1 0.102
- 10+77 → 24：M0 Acc 0.143 / F1 0.036；M2 Acc 0.143 / F1 0.036
- 24+77 → 10：M0 Acc 0.367 / F1 0.222；M2 Acc 0.367 / F1 0.265

## 对照：Phase 2B single-source（不重训）

- 77→10 source-only Acc 0.275
- 77→10 CORAL Acc 0.196（negative baseline，本阶段未再调）

## 24 GHz 域统计（官方 PNG，未先验下结论）

| radar | n | native_w | native_h | gray_mean | gray_std | row_energy_std | col_energy_std |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10GHz | 346 | 1178.0 | 628.0 | 25.79 | 38.31 | 22.58 | 9.66 |
| 24GHz | 327 | 128.0 | 128.0 | 60.11 | 78.75 | 75.26 | 5.91 |
| 77GHz | 334 | 1475.0 | 804.0 | 39.46 | 54.49 | 39.08 | 16.69 |

从实测 PNG 看：
- 原生宽度：最小 24GHz=128.000，最大 77GHz=1475.000
- 原生高度：最小 24GHz=128.000，最大 77GHz=804.000
- 灰度均值：最小 10GHz=25.790，最大 24GHz=60.112
- 灰度标准差：最小 10GHz=38.313，最大 24GHz=78.749
- 行能量标准差（Doppler 轴变化）：最小 10GHz=22.579，最大 24GHz=75.261
- 列能量标准差（时间轴变化）：最小 24GHz=5.908，最大 77GHz=16.693

## 必须回答的问题

1. baseline embedding 是否强烈编码 radar identity？Radar-ID probe M0=1.000。高则说明仍能从 embedding 读出 10/24/77。
2. paired contrastive 是否降低 Radar-ID probe？M0→M1：1.000 → 1.000（变化不到 0.01；此处“下降”才是目标）。
3. activity 信息是否保持？Activity probe M0=0.918，M1=0.952，M2=0.952。
4. physics-aware sensor randomization 是否进一步改善？对照 M1 与 M2 的 probe / retrieval / LORO。
5. exact paired acquisition retrieval 是否提高？M0 0.126 → M1 0.190 → M2 0.231。
6. same-action cross-radar retrieval 是否提高？M0 0.718 → M1 0.935 → M2 0.864。
7. 10+24→77 相对 source-only LORO：M0 0.347，M2 0.204。这与 Phase 2B 单源 24→77 / 10→77 不是同一协议。
8. 10+77→24：M0 0.143，M2 0.143。
9. 24+77→10：M0 0.367，M2 0.367。
10. 哪个 radar 最难泛化：24 GHz。LORO 10+77→24 Acc=0.143，低于 10+24→77 与 24+77→10。官方 PNG 原生分辨率 24 GHz=128×128，10 GHz≈1178×628，77 GHz≈1475×804。
11. Shared Motion v0.1 与 CORAL：CORAL 77→10=0.196（UDA，用了 target unlabeled）。LORO 24+77→10 M2=0.367（未用 10 GHz）。协议不同。Task A 上 M1/M2 分类高于 CORAL 那次 77→10，但不能把 UDA 和 LORO 合成一张“谁赢了”。LORO 上 M2 没有稳定超过 M0。
12. 是否支持 Learn Motion, Not Sensor：部分成立。Activity probe 保持/提高（M0 0.918 → M1/M2 0.952），exact retrieval 上升（M0 0.126 → M2 0.231）。Radar-ID probe 仍为 1.000，未压掉雷达身份。LORO 未见稳定改善，M2 在 10+24→77 反而下降。还不能声称已经学到 radar-invariant motion。
13. 下一步：Radar-ID 仍为 1.0 且 LORO 无改善，值得试 domain adversarial 作为 M3，但不要指望它单独解决 24 GHz 128×128 分辨率缺口。shared motion tokens 放到 radar-id 能下降之后。

## Failed runs

无。

表征图：本机未装 umap-learn，用的是 t-SNE（seed=42），文件在 `results/phase3_shared_motion/figures/tsne_*`。

## 文件

- `results/phase3_shared_motion/training_runs.csv`
- `results/phase3_shared_motion/representation_metrics.csv`
- `results/phase3_shared_motion/radar_domain_statistics.csv`
- `results/phase3_shared_motion/augmentation_spec.json`
