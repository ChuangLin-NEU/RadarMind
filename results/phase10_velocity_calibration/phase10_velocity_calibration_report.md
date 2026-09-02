# Phase 10C — Two-radar velocity-domain calibration

未覆盖 Phase 1–10B。未开新 Adapter / 对抗 / pair loss。10 GHz 不进 R2-V。

## PHYSICS

- 24 fc: **24.75 GHz**（raw header / `datToImage_Anchortech.m:8`）
- 77 fc: **77.56365 GHz** = (77.1799e9+77.9474e9)/2
- evidence: `third_party/CI4R-MULTI3/datToImage_77.m` lines 8–10
- R2-V: **AVAILABLE**
- 24 velocity: [−3.0282, 3.0267] m/s
- 77 velocity: [−3.0921, 3.0906] m/s
- common velocity: **[−3.0282, 3.0267] m/s**

共同速度区间只由两路 native Doppler support 换算后求交，未按 HAR 调。公式：`v = c f_D / (2 fc)`，`c = 299792458`。源是 Phase 8B `cache_native`，不是 224×224 R2-D。

## REPRESENTATION（24/77 两分类 Radar-ID，不要和三分类 0.993 横比）

- R2-D 24/77 Radar-ID: **1.000**
- R2-V Radar-ID: **0.939**
- R2-D Activity: **0.878**
- R2-V Activity: **0.857**
- R2-D Activity PCA32: 0.827
- R2-V Activity PCA32: 0.847

16×16 logistic；TRAIN 458 / TEST 98（各 24+77）。

## PAIR ALIGNMENT（TEST，24↔77）

- R2-D same cosine 0.460  different 0.374  sep 0.086
- R2-V same cosine 0.546  different 0.459  sep 0.087
- R2-D same corr 0.600  R2-V same corr 0.711

同对相似度升了，不同对也升了，间隔几乎没变。

## MOTION GEOMETRY（TEST 对，24 vs 77）

- envelope correlation: R2-D −0.031  R2-V 0.095
- spread correlation: R2-D −0.104  R2-V 0.227

几何相关有升高，未到预注册门限。

## GATE: **FAIL**

预注册：A ΔRadar-ID≥0.10 或 B Δ(same−diff)≥0.05 或 C Δgeometry≥0.10。

- A rid_drop=0.061 → False
- B sep_gain=0.002 → False
- C geo_gain=0.100 → False（0.09988 < 0.10，不凑整过门）

HAR: **NOT TRAINED**

## MEASUREMENT OPERATOR

Top remaining differences:

1. **STFT_window_seconds** [B_sampling_resolution] 0.128 s vs 0.080 s — 24 GHz 窗更长，速度轴对齐消不掉。SUPPORTED
2. **MTI_clutter** [D_signal_conditioning] 24 GHz Butterworth 高通 vs Phase 8B 77 GHz 无 MTI（AIRHAR 段被注释）。SUPPORTED
3. **range_preprocessing** [C_spatial_measurement] 24 GHz 单距离元 19 vs 77 GHz 距离元 23:31 求和 — 门和距离分辨都不同。SUPPORTED

置信度只标 SUPPORTED / LIKELY / UNKNOWN。以上三条都能在脚本里对上，标 SUPPORTED。

## SCIENTIFIC CONCLUSION

- Doppler scaling contribution: **WEAK**
- Measurement-model shift: **STRONG**
- NEXT STEP: **UNIFIED STFT / MEASUREMENT MODEL**

H1（跨雷达差主要来自载频 Doppler 缩放）：不成立为主要原因。换成径向速度后 Radar-ID 只从 1.000 降到 0.939，同对间隔几乎不变。几何相关有一点改善，不够过门。

未自动进入下一阶段。未训 ResNet。
