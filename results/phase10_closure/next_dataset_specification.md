# Next dataset specification

目标不是“再多录一些动作”。
目标是：把 sensor / motion / geometry / subject / environment 拆开。

## Minimum viable

雷达：先做 **24 GHz + 77 GHz**。两路 fc 与速度轴已经能恢复。
10 GHz UWB：第二批再加；第一批不要让未决载频拖死速度轴。

同步：两部雷达必须同时看同一个人、同一次动作。禁止分开各做一遍再事后配对。

标识：每条记录统一
`acquisition_id, timestamp, subject_id, action_id, repetition_id, environment_id`。

几何（每条至少）：
subject–radar distance，aspect，body orientation，radar position / height / orientation，动作起止位置。
能记 reference trajectory 就记。

最小 factorial（第一批只控 sensor × subject × geometry × motion）：

- distance：2 m / 3 m / 4 m
- aspect：0° / 45° / 90°
- subject：5–10 人
- action：沿用现有 7 类（Away, Bend, Crawl, Kneel, Pick, Sit, Towards）
- repetition：每条件至少 3 次
- environment：先固定一个房间

environment（房间 / LOS / 遮挡）放到第二阶段，第一轮不要混。

必须能恢复：range [m]、velocity [m/s]、time [s]。
禁止只存 PNG。必须 RAW IQ + metadata。中间量（range-time / range-Doppler / micro-Doppler）可选。
尽量保留全部 RX；有 MIMO 就留虚拟阵。

标定会话与正式动作分开标记：空室、静止反射、站立、简单周期运动。

运动真值：能同步视频就同步；时间戳必须对齐。专业 MoCap 不作第一阶段硬性要求。

模型：数据问题清楚以前，不预定 Transformer / GNN / Mamba / Diffusion。
有几何之后才考虑 M(t, range, velocity, angle) 或 latent motion + sensor operator。
当前 CI4R 不支持。
