# Phase 9 RadarBridge v2

RadarBridge v2：Known-Radar Sensor-Adapter Framework。  
选 Adapter 必须知道 radar_id。不是 Universal Radar Encoder。

本报告只记录已跑完的数字。未重训，未改 λ，未改 split，未换模型。  
负结果全部保留。

## Physics Gate

| 项 | 结论 |
|---|---|
| XeThru effective fc | **UNRESOLVED** |
| R2-V | **未建立** |
| 本阶段表示 | **R2-D** `S(normalized_time, Doppler_Hz)` 1×224×224 |

依据见 `xethru_frequency_audit.md`。

`XeThru_mDopp_bulk.m` 没有 `fc`、没有中心频率、没有 `v = c f_D / (2 fc)`。  
Doppler 轴是 `PRF=512` 的 **Hz**。

`23.328e9` 出现在

```
bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
```

这是测距采样 `fs`，用来算距离元间隔。**不得解释为 carrier frequency。**  
没有官方证据支持的单一有效载频，因此不建三频统一速度轴 R2-V。  
24 GHz 头里的 24.75 GHz、77 GHz 脚本里的 77.56365 GHz 不能单独凑成共同 R2-V。

R2-D 去掉的是作图/像素坐标，不是按径向速度对齐，也不是 sensor-invariant。

## 和 Phase 3 M1 的差别

- Phase 3 M1：同一套图进同一个 encoder，再做 contrastive。
- 本阶段：每个雷达先过自己的轻量 Adapter，再进 Shared Motion Encoder；同 timestamp 三频 embedding 用 cosine distance 对齐。
- 假设：低层雷达统计不该全部丢给 shared encoder 自己消化。
- 本轮结果不支持该假设带来 HAR 或 invariance 提升。

## 参数量

| 模块 | 参数量 |
|---|---|
| 单个 Adapter | 9537 |
| 三个 Adapter | 28611 |
| Shared encoder | 11173831 |

运动建模仍在 shared encoder。

## Known-radar 联合 HAR

三频 TRAIN / VAL / TEST。表示：R2-D。seed=42。CE。A2 另加 exact-pair。  
λ ∈ {0, 0.1, 0.5, 1.0} 只按 VAL 选过一次，选定 **λ=0.5**。不再搜索。

| 模型 | acc | Macro-F1 | 10 | 24 | 77 |
|---|---|---|---|---|---|
| A0 shared encoder only, CE | **0.946** | 0.948 | 0.959 | 0.918 | 0.959 |
| A1 adapters + CE | **0.878** | 0.877 | 0.878 | 0.939 | 0.816 |
| A2 adapters + CE + pair (λ=0.5) | **0.782** | 0.782 | 0.796 | 0.816 | 0.735 |

**A0 0.946 > A1 0.878 > A2 0.782。**  
radar-specific adapters 没有带来性能提升。  
加上 exact-pair 后联合 HAR 继续下降。  
这些负结果保留。

## Radar-ID

同一套 linear probe（train 拟合，test 评估）。

| 位置 | A0 | A1 | A2 |
|---|---|---|---|
| Input R2-D 16×16 | 0.993 | 0.993 | 0.993 |
| Adapter output | 0.993（A0=输入） | 1.000 | 0.986 |
| Shared embedding | 0.993 | 1.000 | **1.000** |

A2 embedding Radar-ID 仍为 **1.000**。  
Adapter 没有逐层去掉雷达身份。Radar invariance: **NO**。

## Activity probe

| 位置 | A0 | A1 | A2 |
|---|---|---|---|
| Adapter output | 0.864 | 0.871 | 0.837 |
| Shared embedding | 0.946 | 0.932 | 0.925 |

动作信息在 embedding 上仍在。雷达信息也仍在。

## 同 acquisition 三频距离

TEST，cosine distance，越小越近。147 对（49 条 × 3 对雷达）。

| 模型 | 同 acquisition | 同动作不同 acquisition | 不同动作 |
|---|---|---|---|
| A0 | **0.086** | 0.100 | 0.379 |
| A1 | 0.098 | 0.115 | 0.346 |
| A2 | **0.141** | 0.157 | 0.330 |

exact-pair alignment 没有使同 acquisition embedding 更接近。A2 比 A0 更远。

## 三频预测一致

known-radar TEST，49 条。不是 unseen-radar DG。

| 模型 | all-three agree | at-least-two | all-three correct | KL |
|---|---|---|---|---|
| A0 | **0.878** | 0.980 | 0.878 | 0.385 |
| A1 | 0.735 | 0.959 | 0.735 | 0.701 |
| A2 | **0.592** | 0.980 | 0.592 | 0.776 |

A2 三频一致率下降。

## LORO adapter transfer

held-out 雷达没有自己的 Adapter。不用 unlabeled target、不拟合 target 归一化、不微调 target Adapter。

| 协议 | identity | 源 Adapter A | 源 Adapter B |
|---|---|---|---|
| 10+24→77 | 0.204 | Adapter_10 0.143 | Adapter_24 **0.286** |
| 10+77→24 | 0.184 | Adapter_10 0.143 | Adapter_77 **0.224** |
| 24+77→10 | 0.163 | Adapter_24 **0.224** | Adapter_77 0.122 |

7 类随机约 0.143。最好一格 0.286，仍不能当活动分类用。  
24+77→10 最好 0.224，低于 Phase 8C R2-D CE 的 0.306。  
**LORO 未改善。** Unseen-radar generalization: **NOT IMPROVED**。

A2 本身不能证明 unseen-radar DG：没见过的雷达没有对应 Adapter。

## 图

| 文件 | 内容 |
|---|---|
| `figures/fig1_architecture.png` | Adapter → Shared Encoder |
| `figures/fig2_r2_triplet.png` | 同 acquisition 三频 R2-D 输入 |
| `figures/fig3_adapter_features.png` | A2 Adapter 输出 |
| `figures/fig4_embed_action.png` | embedding，颜色=动作 |
| `figures/fig5_embed_radar.png` | embedding，颜色=雷达 |
| `figures/fig6_radar_id_flow.png` | Radar-ID：输入 / Adapter / embedding |
| `figures/fig7_activity_flow.png` | Activity probe |
| `figures/fig8_paired_distance.png` | 同 acquisition 距离分布 |
| `figures/fig9_prediction_agreement.png` | 三频预测一致 |

## 结论

- Known-radar interoperability: **NOT IMPROVED**
- Radar invariance: **NO**
- Unseen-radar generalization: **NOT IMPROVED**
- Learn Motion, Not Sensor: **NO**
- Sensor Adapter 实际解决的是: **两者都没有**

A0 0.946 > A1 0.878 > A2 0.782。  
radar-specific adapters 没有带来性能提升。  
exact-pair alignment 没有使同 acquisition embedding 更接近。  
Radar-ID 仍为 1.000。  
LORO 未改善。  
R2-V 因 XeThru effective fc 缺乏官方证据而没有构建。  
`23.328e9` 不是 carrier frequency。

主论文 frozen 表未改。本阶段是独立分支。不进入 Phase 10。
