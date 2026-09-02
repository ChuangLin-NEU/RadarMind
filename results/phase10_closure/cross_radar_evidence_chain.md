# Cross-radar evidence chain

只读 Phase 7–10G 已冻结数字。本文件不改那些目录。
不是 geometry-aware 结论集。经验传感器翻译到此停止。

## A. Official PNG (R0)

Within-radar HAR 高：10/24/77 Accuracy 0.843 / 0.898 / 0.922（Phase 7）。
Cross-radar 塌：例如 24→77 0.039，77→24 0.163。
像素 Radar-ID = 1.000。

## B. R1 plotted-image normalization

裁边 + 统一后处理。preprocessing confound 存在（个别方向数字会动）。
Radar-ID 仍为 1.000。六条 cross 没有一起抬升。
结论：图框/色条不是主因。

## C. R2-D numerical Doppler

去掉绘图，轴为归一化时间 × Doppler Hz。
Radar-ID 0.993。Activity Probe 0.946。
Within：10/24/77 = 0.776 / 1.000 / 0.878。
Cross / LORO 仍差（Phase 8C case D）。
结论：画图伪影去掉以后，雷达身份还在。

## D. R2-V 24/77 velocity alignment

10 GHz：`EFFECTIVE_FC_UNRESOLVED`，不进速度轴。
24/77 Radar-ID：1.000 → 0.939。Activity 0.878 → 0.857。
Pair sep：0.086 → 0.087，基本没变。
Doppler scaling：**WEAK**。Measurement-model shift：**STRONG**。

## E. Unified STFT (U1)

相对 U0，Radar-ID 0.939 → 0.939（Δ=0）。Activity 0.857 → 0.867。
统一窗长/hop 没有消掉指纹。

## F. Unified MTI (U2)

Radar-ID 0.939 → 0.980，更强。Pair sep 下降。
不能把 MTI 差当成主缺口去“对齐”。

## G. Simple sensor calibration (C0–C4 / M1–M3)

Phase 10B：VAL RID gate 全 FAIL。CASE CAL-C。
Phase 10E M2：Radar-ID 0.949，速度边际 0.929，Activity 0.857。Gate FAIL。
一阶/按速度仿射校正不够。

## H. Motion factors (10A)

38 个可解释因子。Activity probe 0.687，Radar-ID 仍 1.000（稳定子集 0.959）。
动作信息在，雷达身份也在。CASE B。

## I. Fixed empirical G_r(v) (10E)

Transfer stability 0.395。mean std D(v) 1.531。动作 η² 0.558。
固定、与动作无关的 G_r(v) 不稳。CASE HW-C。

## J. Motion-conditioned D(v|q) (10F)

24→77 B1：R² 0.744，MAE 0.666，corr 0.868（相对 B0 明显好）。
77→24 B1：R² 0.575，MAE 0.876，corr 0.758。
校正后 Radar-ID 0.939 → 1.000。速度边际仍 1.000。
速度边际可预测，校正解释不了整张谱的传感器差。
真 geometry-aware：**BLOCKED**。

## K. Low-rank 2D Δ(τ,v|q) (10G)

B2 二维 Δ：24→77 R² 0.300；77→24 R² 0.173。
Oracle 天花板：R² 0.418（K=20 TRAIN 解释方差 0.490）。
24→77 预测图会被判成 77，但 Activity 0.898 → 0.735。
Gate FAIL。CASE C2D-C。

## Final

**STOP EMPIRICAL SENSOR TRANSLATION.**
当前 CI4R 327 对不支持再堆经验翻译 / 不变编码器。
下一资源是带几何与采集元数据的新数据，不是新网络。
