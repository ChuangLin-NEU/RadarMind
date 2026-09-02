# Phase 10D — Unified measurement pipeline

未覆盖 Phase 1–10C。未训 ResNet。10 GHz 不参与。
每个 U 用自己的 TRAIN 24+77 全局 clip 1–99 + z-score。不是 per-radar。

明显下降预注册为 Radar-ID 降幅 ≥ 0.10（与 10C 同一尺度）。

## OPERATOR AUDIT

STFT:
- 24: Hamming，窗 0.128 s，hop 0.028 s，nfft 4096
- 77: Hann（AIRHAR），窗 0.080 s，hop 0.0175 s，nfft 4096

MTI:
- 24: Butterworth 4 阶，Wn=0.01（PRF 1000 → 物理截止 5 Hz），因果 `filter`
- 77 Phase 8B: 无

Range:
- 24: 半剖面 MATLAB bin 19。脚本无 `rng_axis`，bin 的米 **UNKNOWN**
- 77: AIRHAR 对半剖面 bins 23:31 复数求和。官方 MATLAB 门是第 8 行，不是这一套。AIRHAR 行号的米 **UNKNOWN**

## U0 — Native operators（Phase 10C R2-V）

- Radar-ID: **0.939**
- Activity: **0.857**（macro-F1 0.863）
- Pair separation: **0.095**（same 0.546，diff 0.451）
- Geometry: mean **0.136**，envelope **0.095**，spread **0.227**

## U1 — UNIFIED STFT

available: **YES**
窗 0.100 s（24 实际 0.100 s / 100 点；77 实际 0.100 s / 320 点），overlap 75%，hop 两边都是 0.025 s，Hamming，nfft 4096。官方 range / MTI 不动。

- Radar-ID: **0.939**
- Activity: **0.867**
- Pair separation: **0.105**（same 0.507，diff 0.401）
- Geometry: mean **0.134**，envelope **0.088**
- delta vs U0 Radar-ID: **0.000**

## U2 — + UNIFIED MTI

available: **YES**
24 官方 Butterworth 按 5 Hz 物理截止移植到 77。24 从 pre-MTI 重滤，避免叠两次。

- Radar-ID: **0.980**
- Activity: **0.867**
- Pair separation: **0.063**（same −0.068，diff −0.130）
- Geometry: mean **0.140**，envelope **0.098**
- delta vs U1 Radar-ID: **−0.041**（身份更好认，不是更差认）

U2 之后 TRAIN 均值 24 为 −0.545、77 为 +0.545，全局统计指纹变强。

## U3 — + UNIFIED RANGE

available: **NO**
physical range axis: **NO**
不编 ±Δr。

## FINGERPRINT（各 U 的 Radar-ID）

| 特征 | U0 | U1 | U2 |
|---|---|---|---|
| Global stats | 0.765 | 0.878 | 0.949 |
| Velocity marginal | 1.000 | 1.000 | 1.000 |
| Temporal | 0.847 | 0.847 | 0.980 |
| Full 16×16 | 0.939 | 0.939 | 0.980 |

速度边际始终 1.000。统一 STFT / MTI 没有拆掉这条指纹。

## SCIENTIFIC CASE

**MEAS-D**

MAIN DOMAIN-SHIFT SOURCE: **HARDWARE/MEASUREMENT OPERATOR**

统一窗长和 hop 几乎不改变 Radar-ID。把 24 的 Butterworth 接到 77 上，雷达身份更明显。剩下的差不是“两边 STFT 窗差 48 ms”或“77 没做 MTI”能解释的。

NEXT STEP: **HARDWARE-AWARE MEASUREMENT MODEL**

未训 ResNet。未自动进入 Phase 11。
