# Known-radar sensor calibration

Phase 1–10A 只读。本目录是新审计。  
表示仍是 R2-D `S(τ, f_D)`。**不是速度。**  
`radar_id` 已知时才能用该雷达 TRAIN 统计。这不是 unseen-radar DG。

## 三种说法

| 名称 | 可用数据 | 不能叫 |
|---|---|---|
| Known-radar calibration | 该雷达 TRAIN 有标签或至少知道是哪部雷达 | Universal / zero-shot DG |
| Strict unseen-radar DG | held-out 雷达任何 train/val 统计都不用 | 见 Phase 8C LORO |
| Unlabeled target calibration / UDA | 目标雷达无标签样本估 mean/std 或 Doppler 剖面 | DG |

## C0

无校准。当前 R2-D。

## C1 Radar-wise Train Z-score

每个雷达只用 TRAIN 全体像素：

```
μ_r, σ_r = mean, std  (TRAIN, that radar)
S' = (S − μ_r) / (σ_r + ε)
```

VAL/TEST 用同一套 μ_r, σ_r。不用 TEST 统计。

## C2 Doppler-bin calibration

每个雷达、每个 Doppler 行（对 τ 和 TRAIN 样本）：

```
μ_r(f), σ_r(f)
S'(τ,f) = (S(τ,f) − μ_r(f)) / (σ_r(f) + ε)
```

用来去掉雷达固有 Doppler 包络。

## C3 Temporal + Doppler separable

TRAIN：

```
μ_r = 总均值
d_r(f) = 对 τ、样本平均
t_r(τ) = 对 f、样本平均
S' = (S − d_r(f) − t_r(τ) + μ_r) / σ_resid
```

`σ_resid` 是减去可分偏置后的 TRAIN 残差标准差。无神经网络。

## C4 Paired affine → 77 GHz

77 GHz 只是工程参考域，**不是物理真值**。  
TRAIN 同 timestamp 配对。每个 Doppler 行最小二乘：

```
S_77(f) ≈ a_r(f) S_r(f) + b_r(f)
```

77：a=1, b=0。映射在 VAL/TEST 上固定，不再拟合。

禁止 GAN / CycleGAN / UNet / diffusion。

## Gate（工程，不是科学阈值）

VAL 上：Radar-ID < 0.80 且 Activity Probe > 0.75 才训 ResNet。  
否则 STOP，不训新网络。
