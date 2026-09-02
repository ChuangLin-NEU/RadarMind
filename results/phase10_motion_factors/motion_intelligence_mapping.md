# Motion Intelligence 对接

CASE B：不把 MotionFactorVector 升成正式中间 schema。  
下面只标哪些量可以进 Motion Intelligence，以及不能写什么。

| Motion Intelligence 字段 | 可接的 factor | 单位 / 标签 | 不能写成 |
|---|---|---|---|
| periodicity | periodicity_score, dominant_temporal_freq | 每次采集的周期数 | 速度、m/s |
| motion_intensity | temporal_energy_std, peak_to_mean_ratio | 相对量 | m/s、动能 |
| duration | active_duration_frac | 归一化 τ 占比 | 秒（除非另有 PRF） |
| （无对应速度场） | doppler_std_hz, doppler_bandwidth_hz | Hz | radial velocity |
| mean_velocity | 无 | UNKNOWN | 任何由 f_D 猜的 m/s |
| peak_velocity | 无 | UNKNOWN | 同上 |
| direction | pos/neg_doppler_energy_frac 只作符号线索 | 未接 | toward/away |

`23.328e9` 不是载波。XeThru fc UNRESOLVED。
