# Phase 10B Sensor Calibration Audit

未覆盖 Phase 1–10A。未使用 GAN / 像素翻译网络。
C1–C4 都是 **KNOWN-RADAR CALIBRATION**：要用该雷达 TRAIN 统计。
不能叫 Universal，也不能叫 unseen-radar DG。
Phase 8C LORO 仍是 strict DG 对照，本阶段不改。

## 雷达指纹来自哪里

- global_stats: Radar-ID 0.959  Activity 0.313
- doppler_marginal: Radar-ID 1.000  Activity 0.707
- temporal_marginal: Radar-ID 0.871  Activity 0.395
- motion_factors: Radar-ID 1.000  Activity 0.687
- r2_16x16: Radar-ID 0.993  Activity 0.864

TRAIN 全局统计：

- 10GHz: mean=-0.217 std=0.766 doppler_profile_std=0.304 pos_mass=0.498
- 24GHz: mean=-0.161 std=0.927 doppler_profile_std=0.421 pos_mass=0.506
- 77GHz: mean=0.377 std=1.157 doppler_profile_std=0.941 pos_mass=0.511

## Calibration probes（16×16 logistic，TRAIN 拟合）

- C0: TEST Activity 0.864  Radar-ID 0.993  VAL Activity 0.803 Radar-ID 0.966  gate=FAIL
- C1: TEST Activity 0.864  Radar-ID 0.986  VAL Activity 0.816 Radar-ID 0.946  gate=FAIL
- C2: TEST Activity 0.837  Radar-ID 0.952  VAL Activity 0.816 Radar-ID 0.925  gate=FAIL
- C3: TEST Activity 0.864  Radar-ID 0.980  VAL Activity 0.857 Radar-ID 0.959  gate=FAIL
- C4: TEST Activity 0.782  Radar-ID 0.966  VAL Activity 0.762 Radar-ID 0.952  gate=FAIL

## Pair similarity（TEST，16×16 cosine）

- C0: same 0.3547019870558969  diff 0.34067610829916534
- C1: same 0.42380362784578685  diff 0.39682943305494833
- C2: same 0.07885556602153648  diff 0.029499343868826522
- C3: same 0.04256875827579069  diff -0.0009105600619397196
- C4: same 0.9274256756516541  diff 0.9259540430542563

## Gate（工程：VAL Radar-ID < 0.80 且 Activity > 0.75）

**FAIL**  诊断用 C_best = C3
HAR trained: **NO**

校准后 38 factors（C3）：Activity 0.803  Radar-ID 0.993

## Unlabeled target calibration budget

C2 用目标雷达 TRAIN 无标签样本估剖面。源雷达用自己的满 TRAIN C2。**不是 DG。**

- target 10GHz budget=1 n=1  target HAR probe 0.143  Radar-ID 0.993
- target 10GHz budget=5 n=5  target HAR probe 0.163  Radar-ID 0.932
- target 10GHz budget=10 n=10  target HAR probe 0.143  Radar-ID 0.939
- target 10GHz budget=20 n=20  target HAR probe 0.143  Radar-ID 0.946
- target 10GHz budget=all n=229  target HAR probe 0.143  Radar-ID 0.952
- target 24GHz budget=1 n=1  target HAR probe 0.122  Radar-ID 0.993
- target 24GHz budget=5 n=5  target HAR probe 0.163  Radar-ID 0.993
- target 24GHz budget=10 n=10  target HAR probe 0.102  Radar-ID 0.973
- target 24GHz budget=20 n=20  target HAR probe 0.082  Radar-ID 0.959
- target 24GHz budget=all n=229  target HAR probe 0.082  Radar-ID 0.952
- target 77GHz budget=1 n=1  target HAR probe 0.122  Radar-ID 0.952
- target 77GHz budget=5 n=5  target HAR probe 0.286  Radar-ID 0.959
- target 77GHz budget=10 n=10  target HAR probe 0.367  Radar-ID 0.959
- target 77GHz budget=20 n=20  target HAR probe 0.388  Radar-ID 0.952
- target 77GHz budget=all n=229  target HAR probe 0.388  Radar-ID 0.952

C4 的 same-pair 相似度升到 0.927，但不同 acquisition 也是 0.926：配对仿射把图都往 77 GHz 靠，过拟合了测量匹配，不是动作对齐。
C2/C3 去掉 Doppler/时间包络后，同 acquisition 相似度反而掉到 0.08 / 0.04。
77 GHz TRAIN Doppler 剖面标准差 0.941，明显高于 10（0.304）和 24（0.421）。

## CASE **CAL-C**
下一步：**MEASUREMENT-MODEL CALIBRATION**

CAL-A：简单校准就能把 Radar-ID 压下去且动作还在。
CAL-B：Radar-ID 降一点，动作也掉。
CAL-C：Radar-ID 基本不降。

未自动开始 Calibrated Shared Motion Encoder。
