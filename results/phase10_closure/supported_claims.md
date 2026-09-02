# Supported claims

依据 Phase 7–10G。不把未做的写成已做。

## STRONG CLAIM

- Cross-radar HAR 明显弱于 within-radar。Within 约 0.82–1.00；多条 cross 接近或低于有用水平（24→77 R0 0.039）。
- 去掉绘图伪影后，雷达身份仍然容易恢复：R0/R1 = 1.000，R2-D = 0.993。
- 仅做 24/77 速度轴对齐不能消掉域差：Radar-ID 仍 0.939，pair sep 几乎不变。
- 统一 STFT / 简单 MTI / 一阶雷达校正都不够。U1 ΔRID=0；U2 更差；C0–C4 与 M2 gate 未过。
- 当前公开 CI4R 327 对无法验证 geometry-aware 模型：距离、方位、姿态字段不存在。
- 10 GHz 不能可靠地放到与 24/77 相同的速度轴上（有效载频未决）。
- 经验传感器翻译应停止：10G Oracle R² 仅 0.418，Activity 保不住。

## SUPPORTED BUT LIMITED

- 运动状态能解释一部分成对响应差：10F D(v|q) 24→77 R² 0.744。
- 跨雷达测量差里有可结构、随运动变化的成分：10E 动作 η² 0.558；10G B2 好于 B0。
- 低秩二维能抓住一部分未见 acquisition 的差，但不是大部分：Oracle 0.418，K=20 只解释 TRAIN 0.490。
- 载频 Doppler 缩放是弱贡献，不是零。
- Phase 9 Adapter 在已知雷达上也不稳定（A0>A1>A2），不能写成“已知雷达已对齐”。

## NOT SUPPORTED

- Learn Motion, Not Sensor 已经做到。
- 存在 Universal / sensor-invariant radar encoder。
- geometry 造成了当前残差（元数据 UNAVAILABLE，只能列候选隐变量）。
- 10 GHz 可转到可靠公共速度轴。
- 已证明 cross-environment 泛化。
- 已做成可用的 24↔77 传感器翻译或不变表示。
- unseen-radar DG 已被校准或 Adapter 解决（Phase 8C LORO / Phase 9 LORO 未改善）。
