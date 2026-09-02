# Empirical measurement-response model

这是工作模型，**不是** exact radar forward model。

```
S_r(tau, v) ≈ G_r(v) * M(tau, v) + B_r(v) + noise
```

- `M`：未观测的人体运动谱
- `G_r(v)`：雷达相关的速度响应（经验）
- `B_r(v)`：雷达相关的背景/偏移（经验）

输入：Phase 10D **U1**（统一 STFT 后的 R2-V）。不用 U2。
先把 U1 z-score 反回 clip 后的 dB，再 `E=10^(dB/10)`。

## Energy mask（TRAIN only）

对 TRAIN 的 `P_r(v)=mean_tau E_r(tau,v)` 取两路平均，阈值 = 该 224 点曲线的第 **20** 分位。
两路均值都高于阈值的 bin 才解释 transfer / 才写入 D(v)。

## Transfer

`D_i(v) = log(P77_i(v)+eps) - log(P24_i(v)+eps)`，只在 mask 上。

- `transfer_stability_score = 1 / (1 + mean std_D(v) on valid bins)`
- 越接近 1 越稳。不按 TEST 调。

## Models（拟合不用 action 标签）

- M0：不校正
- M1：每雷达一个 scale+offset，把 TRAIN dB 的全局均值/方差对齐到两雷达平均（对称 canonical）
- M2：每个速度 bin 一个 gain/offset，对齐该 bin 的 TRAIN 均值/方差到两雷达平均
- M3：TRAIN 成对平均 D(v)；`log E24' = log E24 + 0.5 D`，`log E77' = log E77 - 0.5 D`

VAL 选模型。TEST 只评一次。

## Gate

同时：TEST 速度边际 Radar-ID ≤ 0.90；Activity 相对 M0 下降 ≤ 0.05；pair sep 下降 ≤ 0.02。
