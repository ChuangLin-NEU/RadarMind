# Phase 7 — Cross-radar representation fairness + Radar-invariant encoder v0.3

未改 frozen / paper / presentation / Phase 1–5 数字。R0 对照读 Phase 2/3 旧表。

## Gate

```json
{
  "r1_constructed": true,
  "r1_source": "PLOTTED_PNG",
  "physical_alignment": false,
  "within_radar_reasonable": true,
  "enter_part_d": true,
  "reason": "R1 is fair-PNG crop + common post; within-radar Acc>=0.70"
}
```

## R1 来源

- 10 GHz: PLOTTED_PNG，官方 628×1178，裁 1 像素边
- 24 GHz: PLOTTED_PNG，官方 128×128，裁上下 8 行亮条
- 77 GHz: PLOTTED_PNG，官方 804×1475，裁 top 55 / bottom 58
- 物理速度轴：未做

## R0 vs R1 单源

| protocol | R0 | R1 | Δ |
|---|---:|---:|---:|
| 10→10 | 0.843 | 0.961 | 0.118 |
| 24→24 | 0.898 | 0.857 | -0.041 |
| 77→77 | 0.922 | 0.824 | -0.098 |
| 77→10 | 0.275 | 0.176 | -0.099 |
| 10→77 | 0.373 | 0.294 | -0.079 |
| 77→24 | 0.163 | 0.020 | -0.143 |
| 24→77 | 0.039 | 0.137 | 0.098 |
| 10→24 | 0.143 | 0.510 | 0.367 |
| 24→10 | 0.353 | 0.333 | -0.020 |

## 必须回答的 12 问

### 1. 官方 PNG 是否存在明显 radar-specific preprocessing fingerprint？

**是。**

Representation-only Radar-ID R0=1.000，R1=1.000。三频原生分辨率 628×1178 / 128×128 / 804×1475。

### 2. 24 GHz 128×128 是否是显著 confound？

**是尺寸 confound，但是否单独造成 24→77 失败要看 R1 对照。**

SUPPORTED：全部 24 GHz 官方 PNG 为 128×128。R1 裁掉上下 8 行亮条后仍是 112×128 再放大到 224。24→77 R0=0.039 R1=0.137。

### 3. R1 是否成功建立？

**YES（PLOTTED_PNG 公平裁边）**

不是 RAW_RECOMPUTED。本机无 raw_data.zip，无三频 .mat STFT。禁止把 R1 写成物理速度对齐。

### 4. representation fairness 后 cross-radar 如何变化？

**均值 Δ=0.021**

24→77 0.039→0.137；77→24 0.163→0.020；10→77 0.373→0.294；77→10 0.275→0.176；10→24 0.143→0.510；24→10 0.353→0.333

### 5. 24 GHz transfer 是否仍然最差？

**R1 最差方向：77→24**

24→77=0.137，77→24=0.020，10→24=0.510。

### 6. R1 baseline Radar-ID probe 是否仍接近 1？

**是**

R1 M0 probe=1.000；R0 Phase 3=1.000。

### 7. M3 能否降低 Radar-ID probe？

**没有稳定降到可用程度**

M3=1.000。

### 8. Activity Probe 是否保持？

**保持**

R1 M0=0.925，M3=0.946。

### 9. Retrieval 是否改善？

**有**

action top1 M0=0.840 M3=0.925。

### 10. LORO 是否改善？

**没有**

10+24→77 0.224→0.163；10+77→24 0.347→0.347；24+77→10 0.633→0.531。

### 11. preprocessing vs sensor-domain 各有哪些证据？

**preprocessing=MODERATE；sensor-domain=STRONG**

preprocessing：分辨率/色条/整图缩放、像素 Radar-ID、R0 vs R1 Δ。sensor-domain：R1 后 within 仍高、cross 仍低、运动 embedding Radar-ID 仍高、LORO 仍差。

### 12. 当前是否可以支持 Learn Motion, Not Sensor？

**NO**

只有 Radar-ID 明显下降、Activity 保持、检索/LORO 同时变好才能答 YES。本轮按上列数字判定。

## 应用页三问

- Q1 R0 整图 PNG Radar-ID（像素）=1.000。R1 裁边后=1.000。单源 cross 均值变化 0.021。24→77 0.039→0.137；77→24 0.163→0.020。 个别方向（尤其涉及 24 GHz 色条）变化大，但六条 cross 没有一起抬升，像素 Radar-ID 仍为 1。
- Q2 R1 运动编码器 Radar-ID probe=1.000（R0 Phase 3 为 1.000）。 统一 representation 后仍能按雷达分开。
- Q3 M3 Radar-ID 1.000→1.000；Activity Probe 0.925→0.946；LORO 10+24→77 0.224→0.163；10+77→24 0.347→0.347；24+77→10 0.633→0.531。 未同时满足去雷达信息且保住动作/泛化。

## 结论标签

- Preprocessing confound: **MODERATE**
- Sensor-domain shift: **STRONG**
- Radar invariance improved: **NO**
- Learn Motion, Not Sensor: **NO**

未进入 Phase 8。
