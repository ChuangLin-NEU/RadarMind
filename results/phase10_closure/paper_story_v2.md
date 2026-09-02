# Paper story v2

主线不要塞 RadarMind 全部功能。
Continuous / open-set / LLM / 桌面回放：系统扩展，不抢主文。

## 建议题目

Cross-Radar Human Activity Recognition:
When High Within-Radar Accuracy Fails to Generalize

## 结构

1. Benchmark setup — CI4R 327 对，7 动作，10/24/77；acquisition split 229/49/49。
2. Within vs cross-radar gap — within 高，cross 塌。
3. Preprocessing confounds — R0 整图 PNG → R1 裁边；Radar-ID 仍 1。
4. Numerical and velocity-domain reconstruction — R2-D；24/77 R2-V。10 GHz 速度轴未决。
5. Representation fingerprint — 输入 / 边际 / 因子 / embedding 仍能识别雷达。
6. Failure of generic alignment/calibration — 统一 STFT、MTI、C0–C4、M1–M3。
7. Motion-dependent measurement shift — D(v|q) 可预测；二维低秩天花板低。
8. Limits of current public data — 无距离/方位/姿态；不能验证 geometry-aware。
9. Implications — 下一资源是可拆因子的数据，不是更深的 translator。

## 核心表

Cross-Radar Failure Ladder（见 `cross_radar_failure_ladder.csv`）。
让读者看到：这是系统拆解 domain shift，不是试一个模型没过。

## 核心图

From Apparent Generalization to Measurement-Model Limitation
（`figures/summary_figure.png`）

左：within-radar 高。
中：cross-radar 塌。
右：PNG → 数值 → 速度 → 统一 STFT，Radar-ID 仍高。
最后：运动相关测量差 + 缺失空间隐变量。

## 不写进主文的句子

- Universal encoder
- Learn Motion, Not Sensor 已实现
- geometry 已证实是残差原因
- 10 GHz 已速度对齐
