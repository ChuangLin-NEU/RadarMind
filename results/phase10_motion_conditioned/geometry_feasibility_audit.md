# Geometry feasibility audit

逐项只标 SUPPORTED / UNRESOLVED / UNAVAILABLE。没有字段就不猜。

| 量 | 状态 | 证据 |
|---|---|---|
| subject distance | **UNAVAILABLE** | `raw_mapping.csv` / split / 文件名无距离字段 |
| radar-target distance | **UNAVAILABLE** | 24/77 header 只有 fc、Bw、NTS，无逐条距离 |
| azimuth | **UNAVAILABLE** | MATLAB / README / mapping 无该字段 |
| elevation | **UNAVAILABLE** | 同上 |
| body orientation | **UNAVAILABLE** | 无姿态或朝向记录 |
| approach/recede direction | **UNRESOLVED** | Towards/Away 是动作标签，不是测得几何；禁止作 q |
| range trajectory | **UNAVAILABLE** | 未保存逐 chirp 距离轨迹 |
| angle-of-arrival | **UNAVAILABLE** | 无多通道测角处理 |
| antenna geometry | **UNRESOLVED** | README 写 co-located，无基线/阵面坐标 |
| target range bin in meters | **UNAVAILABLE** | 24 脚本无 `rng_axis`；77 AIRHAR `rp[23:31]` 未接到米 |

查过：`third_party/CI4R-MULTI3/README.md`、`datToImage_Anchortech.m`、`datToImage_77.m`、
`XeThru_mDopp_bulk.m`、`results/phase8_physical_representation/raw_mapping.csv`（timestamp/action/radar/路径）、
`data_splits/ci4r_shared_motion_acquisition_split.csv`（timestamp/action/split）、
Phase 10D `range_gate_audit.md`。文件名形如 `06050010_1573836822.dat`，前缀是被试/动作编码，不是距离或角度。

README：`All activities were conducted along the radar line-of-sight`。这是采集说明，不是每条 acquisition 的几何量。
文中六名被试有年龄/身高/体重描述，**没有**逐条 subject id 对上 327 timestamp。

## True geometry-aware model

**BLOCKED**

本阶段只称：**motion-conditioned measurement response**。
不能叫 geometry-aware。
