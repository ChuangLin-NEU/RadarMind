# Phase 10 closure

Phase 10A–10G 只读。本阶段无新训练、无新模型、无 Phase 11。

## 四个判断

A. 现有 CI4R 是否还值得继续大量调模型？ **NO**
B. 是否继续在这 327 对上追求 radar-invariant encoder？ **NO**
C. 下一科研资源： **DATA**
D. 下一批最关键新增变量：subject–radar distance，aspect，body orientation（外加 RAW + 可恢复 range 轴）
E. Learn Motion, Not Sensor： **NO**

## 主瓶颈

MISSING SPATIAL / ACQUISITION METADATA

不是“项目失败”。是：在现有公开成对谱上，经验对齐/校准/低秩翻译已经做到可写进论文的上限。
再堆网络不会把未记录的几何变成可观测量。

## 主文

题目走 within 高、cross 塌、拆解 representation 与测量差。
系统功能（连续、open-set、说明层）不进主线。

## 下一数据

双雷达同步 24/77；最小几何网格；预注册 split；RAW IQ。
有几何之前不写 hardware-aware 深度模型。
