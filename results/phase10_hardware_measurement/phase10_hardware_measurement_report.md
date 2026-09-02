# Phase 10E — Empirical measurement-response audit

工作模型，不是 exact forward model。输入 U1。未用 U2。未训 ResNet。

## Transfer

- valid bins: 69 / 224（TRAIN 能量第 20 分位）
- transfer_stability_score: **0.395**  (mean std D = 1.531)
- action η² (mean over valid v): **0.558** → B_action_dependent

## Calibration (TEST)

- M0: Radar-ID 0.939  vel-marginal 1.000  Activity 0.867  sep 0.105
- M1: Radar-ID 0.959  vel-marginal 1.000  Activity 0.857  sep 0.082
- M2: Radar-ID 0.949  vel-marginal 0.929  Activity 0.857  sep 0.072
- M3: Radar-ID 1.000  vel-marginal 1.000  Activity 0.857  sep 0.082

VAL 选定：**M2**。Gate: **FAIL**

## Residual

- higher-order Radar-ID M0 **0.990** / best **1.000**

Towards 残差高于 Kneel（14.20 vs 11.00），方向和 Phase 10A 一致。最稳是 Pick，最不稳是 Crawl。

## Action residual norm

- Pick: 10.719
- Kneel: 11.000
- Sit: 11.004
- Bend: 11.539
- Away: 13.355
- Towards: 14.197
- Crawl: 14.871

## Regions (M0 velocity-marginal Radar-ID)

- low_abs_v: 1.000
- high_abs_v: 0.929
- positive_v: 1.000
- negative_v: 1.000

## CASE **HW-C**
NEXT STEP: **PAIR-CONDITIONAL / GEOMETRY-AWARE MODEL**

未自动进入下一阶段。
