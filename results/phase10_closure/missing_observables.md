# Missing observables

只列当前真正缺、且有科学价值的量。
这些是 **candidate latent variables**。
**不能**写成：它们一定解释了现在的 residual。

| 量 | 状态 | 为何可能有用 |
|---|---|---|
| physical target range | MISSING | 24 无米轴；77 AIRHAR 行号未接到米 |
| subject-to-radar distance | MISSING | 文件名/mapping/split 无逐条距离 |
| azimuth / aspect angle | MISSING | 无测角、无逐条方位 |
| elevation angle | MISSING | 无 |
| body orientation | MISSING | 无姿态字段 |
| radar pose | MISSING | 无雷达位姿 |
| antenna geometry | UNRESOLVED | README 只写 co-located，无基线坐标 |
| channel-level IQ | MISSING | 现用链不是全通道保留后分析 |
| range trajectory | MISSING | 未存逐 chirp 距离轨迹 |
| spatial scattering distribution | MISSING | 无多通道/角度维 |
| subject identity | UNRESOLVED | 文中 6 人，未对上 327 timestamp |
| session identity | MISSING | 无 session 字段 |
| environment identity | MISSING | 单实验室说明，无房间 ID |
| approach/recede as measured geometry | UNRESOLVED | Towards/Away 是动作标签，不是几何量 |

AVAILABLE（与几何无关、已经在用）：
- 雷达种类、动作标签、acquisition timestamp、24/77 的 fc 与公共速度轴、U1 数值谱。

下一轮若要做 hardware-aware 模型，必须先能记录 range / aspect / orientation 或阵列空间量。
现在不能从动作标签代替几何。
