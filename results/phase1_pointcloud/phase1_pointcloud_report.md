# Phase 1B — mRI / RadHAR 点云管线与第一轮 baseline

只处理 mRI 与 RadHAR。未打开 CI4R / MM-Fi / FAU / MiliPoint。原始数据只读。

## 1. mRI 的真实数据层级是什么？

```
subject（JSON 的 subject，subject1–subject20）
  └── recording：一条 raw CSV `raw_data/radar/{subject}_{MMDD}.csv`
        └── action segment：videolabels JSON `labels[name] = [start, end]`
              └── frame：该 Frame # 的全部行
                    └── point：X, Y, Z, Doppler, Intensity
```

每个 subject 一条雷达 CSV、一条 JSON。`_MMDD` 只作为文件名里的 session 记号，不解释成日历。

20 人，273 条 JSON 段，frame manifest 153077 行。详见 `mri_schema.md`、`mri_manifest.csv`、`mri_segments.csv`。

## 2. mRI JSON 标签如何与 radar frame 对齐？

JSON 里的整数是 **Frame #**，不是 `Time`。

- `Time` 是墙钟 unix（约 1.65e9）。JSON 区间（如 98–139）不会落在 Time 上。
- DAP-Net `mRI2csv.py` 对 `t ∈ [a, b]`（闭区间）取 `Frame # == t`。
- subject1：区间 98–139 中 42/43 个 Frame # 在 CSV 里；缺的是帧号空隙，不是对错轴。
- `# Obj` 与该帧行数一致（抽样 20 人，mismatch=0）。
- Frame # 不连续（每人约 160–380 个空隙）。空隙处无点。
- 相邻段常共享端点帧（如 640）。这些帧在 manifest 里 `action` 用 `|` 拼接。
- 段外、空隙处标 `UNLABELED`。
- `T pose 2`：13/20 人结尾的短 T pose。训练用规范名 `T pose`，`action_raw` 保留原键。

帧率：`Time` 跨度 / 独立帧数约 **9.6–9.6 Hz**，按 10 Hz 计。不是 RadHAR 的 30 Hz。

## 3. RadHAR 一个 txt 的真实数据结构是什么？

一个 txt = **一段连续点云**，不是单帧。官方 `voxels.py` 与 DAP-Net：`---` 分块，**`point_id == 0` 开新帧**。

每点字段（README）：`x,y,z` 米，`velocity` m/s，`intensity` dB；另有 range / doppler_bin / bearing，本阶段统一 5 维时不用后三个。

多数文件约 600 帧（约 20 s @ 30 Hz）。少数更长（如 `boxing_184s.txt` 5548 帧）。

文件名带下划线和 `20`，**不能恢复 subject / session**。`subject` 填 `UNKNOWN`。官方 `Data/Train`、`Data/Test` 只能叫 **official train/test split**，不能叫 cross-subject。

142 条 Train txt，58 条 Test txt。从 Train 按动作分层划 20% 文件做 val（seed=0），不改官方 Test。

## 4. 两套数据的 x/y/z/v/intensity 是否单位和尺度一致？

**不一致。不要混在同一个数值空间里做 transfer。**

| 特征 | mRI | RadHAR |
|---|---|---|
| x/y/z | README 只写 3D coordinates，**UNIT_UNKNOWN**。数值大约 ±3.3，Y 均值约 2.22 | 米。X 向前，均值约 2.11；Y 右正 |
| velocity | 列名 Doppler，**UNIT_UNKNOWN**，约 −5.3～5.0 | m/s；多数在 ±5，存在 111 m/s 离群点 |
| intensity | “reflected power”，1～1407 | dB，约 15～37 |

坐标系：mRI 的 Y 更像径向距离；RadHAR 的径向是 X。不能假设同一原点、同一轴向。

`point_feature_schema.csv`。

## 5. 最合理的 point-cloud sample unit 是什么？

**一条带标签的动作段（mRI JSON interval / RadHAR 一个 txt），再在段内滑窗。**

禁止先切窗再随机划分。窗继承段（或 txt）的 split。

单帧不是 HAR 样本：mRI 一帧中位数 31 点，RadHAR 21 点，轮廓不完整。

## 6. 每 frame 应保留多少 point？

按实测分布，不是拍脑袋：

- mRI：中位 31，最大 64 → **P=64**（少则 pad+mask，满则随机下采样）
- RadHAR：中位 21 → **P=32**

## 7. 最合理的 temporal window 是多少？

- mRI：官方 DAP-Net **T=32、stride=16**，约 3.2 s（10 Hz）。主段长度约 400–600 帧，够切。
- RadHAR：官方 voxel **T=60、stride=10**，README 写 2 s / 0.33 s（30 Hz）。

两套帧率不同，**不共用同一个 T**。

本阶段窗口数：mRI 7891，RadHAR 15715。

## 8. velocity 是否显著提高识别？

| 数据 | XYZ Acc / Macro-F1 | XYZV | 差值 |
|---|---|---|---|
| mRI subject-independent | 0.478 / 0.420 | 0.471 / 0.416 | 略降 |
| RadHAR official test | 0.868 / 0.869 | **0.949 / 0.949** | Acc **+0.082** |

mRI 上 Doppler 没有抬主指标。RadHAR 上 velocity 明显有用。

## 9. intensity 是否提供额外收益？

XYZVI 相对 XYZV：mRI SI Acc 0.465（再降一点）；RadHAR 0.926（低于 XYZV 的 0.949）。

**本轮看不到 intensity 的稳定增益。** 保留通道做消融即可，不要当成主判别量。

## 10. temporal modeling 是否优于 single-frame？

是。

| 数据 | T=1 pool XYZVI | 序列 GRU XYZVI |
|---|---|---|
| mRI SI | 0.254 / 0.210 | 0.465 / 0.418 |
| RadHAR official | 0.607 / 0.596 | 0.926 / 0.925 |

## 11. mRI random-segment 表现如何？

PointNet-GRU，XYZVI，T=32：Accuracy **0.511**，Macro-F1 **0.459**（test 1208 窗）。  
这是段级分层 70/15/15，同一人可同时出现在 train/test。只作 sanity。

## 12. mRI subject-independent 表现如何？

同一模型 XYZVI：Accuracy **0.465**，Macro-F1 **0.418**（test subject10/3/20，1207 窗）。

三个特征里 SI 最好的是 **XYZ：0.478 / 0.420**。

train/val/test 人为 14/3/3，seed=0。LOSO 20 折文件已生成，本阶段未训。

## 13. cross-subject gap 多大？

同一表征 XYZVI、PointNet-GRU：

| | random-segment | subject-independent | gap（random − SI） |
|---|---|---|---|
| Accuracy | 0.511 | 0.465 | **0.046** |
| Macro-F1 | 0.459 | 0.418 | **0.041** |

这里可以正式叫 **cross-subject generalization gap**（mRI 有 participant ID）。

和 CI4R Phase 1A 的 **prefix-group-disjoint**（当时 Acc 差 0.294）不是同一件事：CI4R 没有官方 subject 字段，不得写成 subject-independent。mRI 上随机段划分与按人划分的差距小得多。只报数字。

## 14. RadHAR official test 表现如何？

PointNet-GRU，T=60：

| 特征 | Accuracy | Macro-F1 |
|---|---|---|
| XYZ | 0.868 | 0.869 |
| XYZV | **0.949** | **0.949** |
| XYZVI | 0.926 | 0.925 |

n_test=3565 窗。这是 official train/test，**不是** cross-subject。

同一划分上，官方 10×32×32 occupancy + 本阶段轻量 Voxel-CNN-GRU：Accuracy **0.744**，Macro-F1 **0.741**。低于 PointNet-GRU XYZV（0.949）。这是简单 CNN，不是仓库里的 TD-CNN-LSTM；不能当成论文 90–92% 的复现。

## 15. PointNet 与 voxel 谁更合适？

统一 benchmark 的点云分支优先 **采样后的 T×P×C + PointNet（+GRU）**：

- 一张窗：mRI 约 41 KB，RadHAR 约 38 KB
- 全量 pack：mRI 34 MB，RadHAR 26 MB
- 官方 voxel 窗是 60×10×32×32 occupancy，稠密约 **9.6 GB**，且网格按**当前帧 min/max** 自适应，不能和 mRI 共用同一套 bounds

`voxels_official.py` 已按官方 10×32×32、T=60、stride=10 实现。稠密窗约 9.0 GB，压缩 pack 42 MB。Voxel CNN 作为 RadHAR 对照，不取代点集表征。mRI 与 RadHAR 空间尺度不同，**不强制同一 voxel 盒。**

## 16. mRI 与 RadHAR 的 domain gap 多大？

主要来自（有数据的项）：

- **硬件 / 点密度**：mRI 中位 31 点/帧，RadHAR 21
- **帧率**：约 10 Hz vs 30 Hz
- **坐标系**：径向轴不同（mRI 的 Y vs RadHAR 的 X）
- **intensity 定义**：未归一化功率 vs dB
- **动作规程**：pose/free_form vs 健身五类
- **划分**：mRI 可按人；RadHAR 只有目录划分

图：`figures/mri_vs_radhar_*.png`。

## 17. 两套数据有哪些真正可以匹配的 action？

只有 **walk**，confidence **medium**。名字都是走，距离、速度、场地未对齐。

pose_1–10、T pose、free_form 对不上 boxing/jack/jump/squats。见 `action_taxonomy.csv`。

## 18. 是否值得进入 cross-dataset transfer Phase？

**现在不够。** 只有一类可对，单位和坐标还对不上。先不要训跨库分类器。

## 19. 下一阶段应继续哪一个？

**优先 MiliPoint**（把 `x`/`y` 和 `action_label.npy` 对齐，仍走点云分支）。  
CI4R 10 GHz 是另一条 IQ 线，等点云分支稳定后再接。  
跨库 transfer 等有更多对得上的动作再开。

---

## 预处理与 cache

- A：每帧固定 P，pad+mask。本阶段主路径。
- C：PointNet 吃无序点。
- B：官方 voxel，独立表征。

窗文件逐条 npz，坏样本进 `failed_samples.csv`，不中断整批。pack：`cache/mri_windows.npz`、`cache/radhar_windows.npz`。

磁盘：窗 cache 未压缩约 0.29+0.56 GB；01 当时剩余约 3052 GB。未全量解任何原始大包。

## 训练控制

seed=0，early stopping patience=6，按 val 选模型，test 只评一次。  
`results/phase1_pointcloud/training_runs.csv`。
