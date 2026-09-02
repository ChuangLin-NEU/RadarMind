# Phase 10G — Motion-conditioned 2D measurement model

经验 paired difference，不是正演/反演，不是 geometry-aware，不是 unseen-radar DG。
未训 ResNet / translator。未改 Phase 1–10F。

Selected K: {'24to77': 20, '77to24': 20}
TRAIN explained variance: {'24to77': 0.48969790078241354, '77to24': 0.48969790078241354}

## Oracle (TEST 投影到 TRAIN 基，不是部署模型)

- 24to77: R² 0.418  MAE 1.318  corr 0.647
- 77to24: R² 0.418  MAE 1.318  corr 0.647

## Transfer (true Delta vs hat)

### 24to77
- B0: R² 0.049  MAE 1.683  corr 0.227
- B1: R² -0.027  MAE 1.752  corr 0.231
- B2: R² 0.300  MAE 1.428  corr 0.555

### 77to24
- B0: R² 0.049  MAE 1.683  corr 0.227
- B1: R² -0.055  MAE 1.779  corr 0.166
- B2: R² 0.173  MAE 1.571  corr 0.418

## PC 结构（只据图，不写肢体）

- 24to77 PC1: spectral widening / rough velocity envelope
- 24to77 PC2: near-zero velocity energy; spectral widening / rough velocity envelope; late-time; mid-time
- 24to77 PC3: near-zero velocity energy; stronger on -v; spectral widening / rough velocity envelope; late-time; mid-time
- 77to24 PC1: spectral widening / rough velocity envelope
- 77to24 PC2: near-zero velocity energy; spectral widening / rough velocity envelope; late-time; mid-time
- 77to24 PC3: near-zero velocity energy; stronger on -v; spectral widening / rough velocity envelope; late-time; mid-time

## Oracle gap 解释: **BOTH**

K=20 的 TRAIN 解释方差只有 0.490。TEST Oracle R²=0.418，不是“很好”，也还没差到完全无结构。
B2 相对 Oracle 仍有缺口（24→77 ΔR²=0.118，77→24 ΔR²=0.245）。
残差不再被 source q 解释（vel-marginal R²<0）。
不能把残差写成 geometry 造成：几何元数据 UNAVAILABLE。候选只列：range / distance / aspect / orientation / spatial scattering。

B1（10F D(v|q) 沿 τ 平铺）在二维 Delta 上不如 B0：速度边际校正不是二维场模型。

## Target reconstruction（cosine）

24→77 source 0.987 / B1 0.986 / B2 0.990 / Oracle 0.991
77→24 source 0.987 / B1 0.986 / B2 0.989 / Oracle 0.990
Prediction gain 相对 source 只有约 0.002–0.003。log-energy 上两雷达本来就很近。

## Radar-domain fidelity（不是 invariance）

24→77：source 判成 24 的比例 0.898；B2 预测判成 77 = 1.000。
77→24：source 判成 77 的比例 0.980；B2 预测判成 24 = 0.898。
预测图会往目标雷达域靠，但动作信息 24→77 从 0.898 掉到 0.735。

## Gate: **FAIL**

B2 相对 B1 的 2D MAE 有下降，但 reconstruction cosine 增益 <0.05，且 24→77 Activity 掉超过 0.05。
不改 gate。

CASE **C2D-C**
NEXT: **STOP EMPIRICAL SENSOR TRANSLATION**

未训 deep model。未自动进入 Phase 11。
