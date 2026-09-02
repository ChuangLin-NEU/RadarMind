# Motion factor definitions

表示：R2-D `S(τ, f_D)`，1×224×224。  
τ ∈ [0,1]。f_D ∈ [−256, +255.875] Hz（TRAIN 三频共同支撑）。  
**不用速度。** `23.328e9` 不是载波。XeThru fc 仍 UNRESOLVED。

## 能量

R2-D 是 clipped dB 的 TRAIN z-score。先还原：

```
dB = S · std_train + mean_train
W = 10^(dB / 10)
```

`std_train`、`mean_train` 来自 `results/phase8_physical_representation/r2_normalization.json`。  
W 是线性功率代理，不是 m/s。

E = Σ W。  
e(f) = Σ_τ W。  
e(τ) = Σ_f W。  
p(f) = e(f)/E，p(τ) = e(τ)/E。

## 预注册频带（TRAIN 共同支撑，不用 test 调）

| 带 | 定义 |
|---|---|
| low | \|f_D\| < 32 Hz |
| mid | 32 ≤ \|f_D\| < 128 Hz |
| high | \|f_D\| ≥ 128 Hz |

32 = 256/8，128 = 256/2。在写代码时固定。

## A. Doppler energy

| 名 | 定义 |
|---|---|
| doppler_centroid_hz | Σ p(f) f_D |
| abs_doppler_centroid_hz | \|centroid\| |
| doppler_std_hz | sqrt(Σ p(f) (f_D − centroid)²) |
| fd_q10/25/50/75/90_hz | 以 e(f) 为质量的 f_D 分位数 |
| doppler_bandwidth_hz | q95 − q05 |
| pos/neg_doppler_energy_frac | f_D>0 / f_D<0 的能量比 |
| low/mid/high_doppler_energy_frac | 预注册带内能量 / E |

## B. Temporal dynamics

| 名 | 定义 |
|---|---|
| temporal_energy_mean/std/range | e(τ) 的均值、标准差、max−min |
| peak_to_mean_ratio | max e(τ) / mean e(τ) |
| n_temporal_peaks | 局部极大且 ≥ 0.25·max e(τ) 的个数 |
| dominant_temporal_freq | e(τ) 去均值后 rFFT 主峰，单位：每次采集的周期数 |
| periodicity_score | 该主峰幅度 / 其余正频幅度和 |
| autocorr_peak | 标准化 e(τ) 自相关在 lag≥4 的最大值 |
| temporal_entropy | p(τ) 的归一化熵 −Σp log p / log N |

## C. Spectral shape

| 名 | 定义 |
|---|---|
| spectral_entropy | p(f) 归一化熵 |
| sparsity | Hoyer： (√N − ‖W‖₁/‖W‖₂) / (√N − 1) |
| occupied_doppler_frac | (q95−q05) / (fmax−fmin) |
| peak_doppler_magnitude | max e(f) |
| n_doppler_peaks | 同 n_temporal_peaks，作用在 e(f) |
| doppler_envelope_roughness | mean\|Δe(f)\| / mean e(f) |

## D. Onset / offset（τ 已归一化）

| 名 | 定义 |
|---|---|
| energy_onset_tau | 累计能量首次达到 10% 的 τ |
| energy_offset_tau | 累计能量达到 90% 的 τ |
| active_duration_frac | offset − onset |
| early/middle/late_energy_frac | τ 三段 [0,1/3)、[1/3,2/3)、[2/3,1] 的能量比 |

## E. 2D moments

| 名 | 定义 |
|---|---|
| time_centroid | Σ p(τ) τ |
| time_spread | sqrt(Σ p(τ) (τ − μ_τ)²) |
| time_doppler_covariance | Σ W (f−μ_f)(τ−μ_τ) / E |

doppler 一阶/二阶矩与 A 共用，不重复列。

## 归一化

- RAW：上表直接值。
- GLOBAL：TRAIN 全体（三频）均值方差。用于跨雷达比较和 unseen 讨论。
- PER-RADAR：各雷达自己的 TRAIN 均值方差。只作 known-radar 诊断。

## Stability

只在 TRAIN 的 229 条 paired acquisition 上算。

- intra-acquisition CV = std(v10,v24,v77) / (|mean| + ε)
- radar_domain_variation = 各条 CV 的平均
- eta²_activity、eta²_radar：单因素 ANOVA，SS_between / SS_total
- motion_signal_strength = eta²_activity
- stability_score = eta²_activity / (intra-CV + 1e-6)

排序：intra-CV 升序 = 更雷达稳定。

子集大小 K=8，按 TRAIN 指标预注册，不用 test 调。
