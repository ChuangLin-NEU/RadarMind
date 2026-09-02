# Conditioning factors q (source-only)

公式复用 Phase 10A（能量 = 反 z-score 后 10^(dB/10)），轴改为径向速度。
低/中/高速带预注册：|v|<1.0、[1.0,2.0)、≥2.0 m/s。与 10E 低速门 |v|<1 对齐。不按 TEST 调。

- 只从 **当前 source radar** 的这一条 U1 谱提取
- 不用 action 标签，不用对端雷达

- `abs_velocity_centroid`
- `velocity_spread`
- `low_v_energy_frac`
- `mid_v_energy_frac`
- `high_v_energy_frac`
- `pos_velocity_frac`
- `neg_velocity_frac`
- `spectral_entropy`
- `sparsity`
- `temporal_energy_mean`
- `temporal_energy_std`
- `temporal_energy_range`
- `dominant_temporal_freq`
- `periodicity_score`
- `autocorr_peak`
- `active_duration_frac`
- `velocity_envelope_roughness`
