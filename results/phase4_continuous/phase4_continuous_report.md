# Phase 4 Continuous Motion Understanding MVP

不是 open-world recognition。分类器仍是 mRI 13 类 subject-independent PointNet-GRU，未重训。

本阶段是 **labeled-interval continuous replay**：把 Phase 1B 已有 labeled-segment windows 按原始 Frame # 顺序串起来。UNLABELED 区间只有 GT metadata，**没有**在这些区间上做雷达推理。

不要写成对整条录音做 full-stream 推理。true full-stream inference including unlabeled intervals 需要原始 mRI radar CSV。本机没有该 CSV。

数字未改。

## 1. mRI 是否真正包含 continuous multi-action recordings？

是。20 条 recording（一人一 CSV）。其中 20/20 条有不少于 2 个 labeled 动作段。
例如 subject10 一条 `subject10_0504.csv` 内含 T pose、pose_1…pose_10、free_form、walk。
没有把训练窗手工拼成假连续流。

## 2. 完整 recording 的层级

subject → 一条 radar CSV → videolabels JSON 的多个 [start,end]（Frame #）→ 帧 → 点。
段外、段间空隙为 UNLABELED。相邻段共用边界帧时，后一段从该帧起算，不把空隙贴给邻段。

## 3. sliding-window 如何定义

T=32，stride=16（Phase 1B 已缓存）。绝对帧 = segment_start + 窗偏移。
本机没有 raw CSV，不能重切 stride=8，也不能在 UNLABELED 上滑窗。
评测只走原 SI **test subjects**：subject10, subject20, subject3。val 只用于选平滑阈值。

## 4. smoothing 是否改善

val 选定 confidence_aware conf_min=0.45（val macro-F1=0.571）。
- raw window Acc 0.480 / F1 0.417
- isolated_flip Acc 0.496 / F1 0.426
- confidence_aware Acc 0.494 / F1 0.419

## 5–7. Mode A 指标（只评 labeled）

- frame Acc 0.489 / F1 0.416
- window Acc 0.494 / F1 0.419
- segment F1 0.152（P 0.100 / R 0.321）
- median boundary error 190.8 frames / 20.63 s

## 8. 连续混淆

见 `figures/fig4_confusion.png`。pose_* 之间最容易连成一段。

## 9. velocity / motion-energy

来自窗内 Doppler 的 PHYSICS_DERIVED 统计，单位未经标定。可画曲线，不能单独当分割器。

## 10–11. LLM

只吃 Timeline JSON + series 摘要。pose_* 不得意译。低置信度必须写「模型低置信度地判断为 …」。
uncertainty 指示器 ≠ open-set。

## 12. Continuous Replay

ReplayEngine mode=continuous：按绝对 Frame # seek，点云、当前预测、MI、双轨时间线共用同一游标。

## 13. 能不能称为 open-world？

**不能。** 没有 open-set / open-vocabulary 评测。当前是 Continuous Motion Understanding MVP。

## 14. cross-environment 能不能做？

**不能。** 见 `cross_environment_feasibility.md`。ProtocolRegistry.cross_environment.available=false。

## 15. 离 PPT 完整目标还缺什么

- sensor-invariant representation（Phase 3 Radar-ID probe 仍为 1.0）
- 真正 open-set / open-vocabulary
- 有环境标注后的 cross-environment
- 新雷达集（MiliPoint / MM-Fi / FAU）与 Live Radar
- 在 UNLABELED 上的 background detector（本阶段明确不做）

## Demos

- Demo A subject20_0513 (subject20): highest window accuracy among test recordings
- Demo B subject10_0504 (subject10): largest median boundary error among test recordings
- Demo C subject3_0501 (subject3): lowest window macro-F1 among test recordings

指标文件：continuous_runs.csv / segment_metrics.csv / boundary_metrics.csv
