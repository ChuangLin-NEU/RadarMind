# Research backlog（不在本阶段执行）

## P0 真正 sensor-invariant representation

- Question: 能否在 Radar-ID probe 明显低于 1.0 的同时保持活动分类？
- Data: 现有 CI4R official PNG triples；不要先上新数据集。
- Code: 表征约束 / 规范化；不是再调 CORAL。
- Experiment: Task A + Radar-ID + LORO。
- Success: Radar-ID 下降且 LORO 不低于 M0。

## P1 24 GHz representation normalization

- Question: 24 GHz 难迁移是预处理还是传感器？
- Data: 同一批 327 triples；核对 native PNG 尺寸。
- Code: 统一 resample / 频谱审计。
- Experiment: 重做 within-24 与 24↔77，不改标签。
- Success: 能分离 confound 与真实频段差。

## P2 Cross-environment

- Question: 同一动作换房间/布置后还能否保持？
- Data: 需要带 environment 标签的采集。
- Code: environment split。
- Experiment: within-env vs cross-env。
- Success: 有可引用的环境划分和数字。

## P3 Full mRI raw continuous inference

- Question: UNLABELED 区间上模型会输出什么？
- Data: 原始 mRI radar CSV。
- Code: 全流滑窗。
- Experiment: labeled + unlabeled 时间线。
- Success: 不再只是 labeled-interval replay。

## P4 Open-set improvement

- Question: 相似 pose_* 能否被稳定拒识？
- Data: 现有 mRI LOAO。
- Code: 更好的 embedding / score；不改 test 调阈值。
- Experiment: 13-fold mean AUROC / FPR95。
- Success: FPR95 明显低于当前 Prototype 0.791，且不偷看 unseen。

## P5 Open-vocabulary motion representation

- Question: 能否用文本或属性描述未见过的动作，而不是只输出 UNKNOWN？
- Data: 带文本/属性的动作集。
- Code: 对齐层。
- Experiment: unseen-class 名称检索。
- Success: 才可称 open-vocabulary，仍不是 open-world。

## P6 Live Radar

- Question: 在线雷达流上闭环是否稳定？
- Data: 实时设备。
- Code: 采集适配。
- Experiment: 现场延迟与掉线。
- Success: 不依赖预缓存窗口。
