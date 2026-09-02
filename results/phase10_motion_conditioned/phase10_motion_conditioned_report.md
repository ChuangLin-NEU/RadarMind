# Phase 10F — Motion-conditioned measurement response

不是 geometry-aware。不是 unseen-radar DG。配对 TRAIN 上学跨雷达响应。未训 ResNet。

## Geometry

range / distance / aspect / orientation：**UNAVAILABLE** 或 UNRESOLVED。
True geometry-aware model: **BLOCKED**

## Predictability

### 24to77 (ridge_1.0)
- B0: R² 0.219  MAE 1.265  corr 0.470
- B1: R² 0.744  MAE 0.666  corr 0.868

### 77to24 (ridge_0.1)
- B0: R² 0.219  MAE 1.265  corr 0.470
- B1: R² 0.575  MAE 0.876  corr 0.758

## Velocity regions

- 24to77 low (energy_mask_10E): n=69  B0 R² 0.219 MAE 1.265  B1 R² 0.744 MAE 0.666
- 24to77 mid (energy_mask_10E): n=0 UNAVAILABLE
- 24to77 high (energy_mask_10E): n=0 UNAVAILABLE
- 77to24 low (energy_mask_10E): n=69  B0 R² 0.219 MAE 1.265  B1 R² 0.575 MAE 0.876
- 77to24 mid (energy_mask_10E): n=0 UNAVAILABLE
- 77to24 high (energy_mask_10E): n=0 UNAVAILABLE
- 24to77 low (all_bins_diagnostic): n=74  B0 R² 0.214 MAE 1.258  B1 R² 0.744 MAE 0.658
- 24to77 mid (all_bins_diagnostic): n=74  B0 R² -0.007 MAE 0.931  B1 R² 0.766 MAE 0.410
- 24to77 high (all_bins_diagnostic): n=76  B0 R² -0.010 MAE 0.896  B1 R² 0.796 MAE 0.390
- 77to24 low (all_bins_diagnostic): n=74  B0 R² 0.214 MAE 1.258  B1 R² 0.570 MAE 0.875
- 77to24 mid (all_bins_diagnostic): n=74  B0 R² -0.007 MAE 0.931  B1 R² 0.267 MAE 0.762
- 77to24 high (all_bins_diagnostic): n=76  B0 R² -0.010 MAE 0.896  B1 R² 0.250 MAE 0.746

10E energy mask: 69 bins, v in [-0.8831849472405131, 0.9631627931723146] m/s。
官方 valid bin **全部**落在 |v|<1。中/高速 D(v) 在该 mask 下没有定义。

## Top conditioners

两方向平均 |coef|：
1. sparsity (mean |coef| 0.9422)
2. velocity_envelope_roughness (mean |coef| 0.9359)
3. spectral_entropy (mean |coef| 0.8807)
24→77: velocity_spread, mid_v_energy_frac, abs_velocity_centroid
77→24: velocity_envelope_roughness, sparsity, spectral_entropy

## Gate: **PASS**

Gate 只看 D(v) 的 MAE/corr，不看 Radar-ID。
这是 paired known-radar calibration，不是 unseen-radar DG。

## Conditional calibration (paired known-radar)
- Radar-ID 0.939 → 1.000
- vel-marginal 1.000 → 1.000
- Activity 0.867 → 0.857
- sep 0.105 → 0.079

## Low-|v| mask (diagnostic only)

- none: Radar-ID 0.939  Activity 0.867
- remove |v|<0.1: Radar-ID 0.949  Activity 0.847
- keep only |v|<0.1: Radar-ID 0.776  Activity 0.735
- remove |v|<0.25: Radar-ID 0.939  Activity 0.857
- keep only |v|<0.25: Radar-ID 0.796  Activity 0.776
- remove |v|<0.5: Radar-ID 0.949  Activity 0.714
- keep only |v|<0.5: Radar-ID 0.939  Activity 0.827

Action residual 24→77: best **Bend**  worst **Towards**

## CASE **MC-A**
NEXT STEP: **CONDITIONAL MEASUREMENT MODEL**

未自动进入 Phase 11。
