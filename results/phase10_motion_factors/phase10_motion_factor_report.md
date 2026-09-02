# Phase 10A Motion Factor Audit

未训深度网络。未改 Phase 1–9。未使用径向速度 / m/s。
XeThru effective fc 仍 UNRESOLVED。因子只基于时间、Doppler Hz、谱能量、时间动态。

## 定义

见 `motion_factor_definitions.md`。
预注册频带（共同支撑 ±256 Hz，不用 test 调）：low |f_D|<32.0 Hz；mid [32.0,128.0)；high ≥128.0 Hz。

能量：把 R2-D 的 TRAIN z-score 还原为 clipped dB，再 `W=10^(dB/10)`。

stability_score = eta²_activity / (intra-acquisition CV + 1e-6)。
eta² 与 intra-CV 都只在 TRAIN acquisition 上算。
per-radar 归一化只作 known-radar 诊断，不声称 unseen generalization。

## 因子数：38

最雷达稳定（intra-CV 最低）：
sparsity, spectral_entropy, temporal_entropy, dominant_temporal_freq, autocorr_peak

最依赖雷达（eta²_radar 最高）：
peak_doppler_magnitude, n_temporal_peaks, sparsity, temporal_energy_mean, doppler_envelope_roughness

最能区分动作（eta²_activity 最高）：
abs_doppler_centroid_hz, low_doppler_energy_frac, mid_doppler_energy_frac, high_doppler_energy_frac, doppler_std_hz

同时偏高 activity / 偏低 radar（eta²_act≥0.08 且 eta²_rad≤0.15）：
dominant_temporal_freq, temporal_energy_range, temporal_energy_std, high_doppler_energy_frac

## Probe

- all_raw: Activity 0.687 (F1 0.693)  Radar-ID 1.000
- all_global: Activity 0.687 (F1 0.693)  Radar-ID 1.000
- all_per_radar: Activity 0.769 (F1 0.775)  Radar-ID 0.333
- stable_global: Activity 0.490 (F1 0.455)  Radar-ID 0.959
- activity_top_global: Activity 0.524 (F1 0.521)  Radar-ID 0.925
- radar_specific_global: Activity 0.463 (F1 0.437)  Radar-ID 0.986

对照（Phase 9，未重训）：
- R2-D 16×16: Activity 0.8639455782312925  Radar-ID 0.9931972789115646
- A0 embedding: Activity 0.9455782312925171  Radar-ID 0.9931972789115646
- A2 embedding: Activity 0.9251700680272109  Radar-ID 1.0

## 同 acquisition 距离（cosine distance，越小越近）

- R2-D_16x16: 0.645297937247218
- A0_embedding: 0.08585922420024872
- A2_embedding: 0.14124566316604614
- MotionFactors_global: 1.1813566272727225
- MotionFactors_stable: 1.1228836312594164

## 动作稳定性（TRAIN 平均 intra-CV）

- 最稳定: Kneel
- 最不稳定: Towards

- Kneel: 0.7168198216314509
- Pick: 0.8009918755427637
- Bend: 0.9539133372414383
- Sit: 1.247599715643262
- Away: 3.1387462824766583
- Crawl: 3.1710127356902738
- Towards: 12.530514006914693

## 频段对位移（TRAIN 平均 |Δfactor|，跨因子平均）

- 10 vs 24: 1.0881
- 10 vs 77: 1.2027
- 24 vs 77: 1.2451

24 vs 77 最大（1.245），其次 10 vs 77（1.203），10 vs 24 最小（1.088）。
24 GHz 仍在最大位移的一对里。Towards / Away 的 intra-CV 很高，和「同一径向速度在不同载频上对应不同 Doppler Hz」一致。

per-radar TRAIN 归一化后 Radar-ID = 0.333（三类随机）。这是把各雷达均值减掉后的已知雷达诊断，不能当成 unseen-radar invariance。

## Motion Intelligence

可对接、且不得写成 m/s 的量：
- periodicity_score / dominant_temporal_freq：周期，单位是「每次采集的周期数」，不是速度。
- temporal_energy_std / peak_to_mean_ratio：运动强弱的相对量。
- doppler_std_hz / doppler_bandwidth_hz：Doppler 展宽，单位 Hz。
- active_duration_frac：归一化时长上的有效占比。
mean_velocity / peak_velocity 仍保持 UNKNOWN。

## CASE

**CASE B**
下一步：**SENSOR CALIBRATION**

CASE A：Activity 可用且 Radar-ID 明显低于 0.993。
CASE B：Activity 与 Radar-ID 都高，因子仍偏 sensor。
CASE C：Radar-ID 降了但 Activity 接近随机。

未训练新 encoder。未进入后续深度训练。
