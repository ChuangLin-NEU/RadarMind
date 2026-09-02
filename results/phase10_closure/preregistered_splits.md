# Preregistered splits

采集前定死。禁止按 TEST 结果改划分。

| split | 定义 | 用来回答 |
|---|---|---|
| within-subject | 同一 subject 的 rep 划分 train/val/test | 个人复现 |
| cross-subject | 若干 subject 全持出 | 人是否可迁移 |
| cross-geometry | 持出某个 distance 或 aspect | 几何是否可迁移 |
| cross-radar known-sensor | 两雷达都在训练里出现，但是成对/不成对要写明 | 已知雷达校准上限 |
| leave-one-radar-out | 一部雷达的全部 activity 不进训练 | unseen-radar DG |
| cross-environment | 第二阶段才启用；持出 room / clutter | 环境 |

规则：

- 同一 `acquisition_id` 的 24/77 必须落在同一 split。
- calibration session 不进 HAR 的 train/test，只进测量响应估计。
- 主文表格只报预注册 split。探索性切分标 `exploratory`，不进主表。
