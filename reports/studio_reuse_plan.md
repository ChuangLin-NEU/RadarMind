# RadarMind Studio-0：复用计划

审计日期：2026-09-02  
依据：`reports/studio_existing_system_audit.md`

---

## 1. 一句话

科学核已经齐，缺的是展示层和一层只读服务。  
不要重写训练、评测、ReplayEngine、Motion Intelligence、open-set、cross-radar、continuous 的计算。

---

## 2. 推荐架构

```
RadarMind Scientific Core          原样保留
  radarmind/engine.py
  radarmind/replay.py
  radarmind/models/*
  radarmind/intelligence/*
  radarmind/llm/*
  radarmind/data/*
  radarmind/protocols.py
  radarmind/health.py
  scripts/paper_v2/registry.py
  results/**（只读）
        │
        ▼
Thin Python services（新建，只包装，不复制公式）
  services/replay_service.py      → RadarMindEngine.replay_*
  services/inference_service.py   → infer / intelligence
  services/result_service.py      → get_metric(key) 读 snapshot_full / 回源文件
  services/intelligence_service.py→ build_intelligence（转调）
  services/language_service.py    → LLMService.explain
  services/sample_service.py      → adapter.get + 可视化切片
  services/health_service.py      → health.check 扩展
        │
        ▼
studio/api  FastAPI
  GET  /health
  GET  /metrics/{key}
  GET  /claims
  GET  /demos
  GET  /samples?...
  GET  /input/{dataset}/{sample_id}     元数据 + 当前帧点/谱图
  POST /replay/load                     Run Inference
  GET  /replay/state
  POST /replay/seek
  GET  /frozen/intelligence/{sample_id} Replay Frozen Result
  GET  /paired/{timestamp}
  GET  /continuous/{recording}
  GET  /openset/...
  POST /explain
        │
        ▼
studio/frontend  React + Vite + Tailwind + Plotly
  01 Overview … 07 Research Evidence
```

旧界面：

```
radarmind/app/streamlit_app.py     改称 RadarMind Research Console
  调试、对表、查 checkpoint
  与 Studio 共用同一套 Engine / 结果文件
```

禁止：Studio 与 Console 各维护一套 0.039。

---

## 3. 技术选型（按现有仓库）

| 层 | 选择 | 理由 |
|---|---|---|
| 前端 | 新建 React + Vite | 仓库无 React；PPT 构建用的 node 不能当应用栈 |
| 样式 | Tailwind | 按 Studio 规格，不要 SaaS dashboard 模板 |
| 图 | Plotly（前端 plotly.js） | 现应用已用 Plotly 画谱图和点云；颜色跟 Studio token |
| 后端 | FastAPI 薄层 | 现无 HTTP；`RadarMindEngine` 已是 Python API |
| 读数 | `metric_snapshot_full.json` + `paper_v2.registry` | 110 项的 v2 snapshot 缺 HAR/连续/open-set |
| 环境 | 继续 `airhar` | torch / 推理已在此环境；Streamlit+Plotly 在 lock 里 |

`requirements.txt` 未列出 streamlit / plotly / fastapi。Studio-1 起单独加 `studio` 依赖，不要改科研环境的训练依赖。

---

## 4. 目录（建议，Studio-1 再创建）

```
services/                 薄封装
studio/
  api/                    FastAPI
  frontend/               React
configs/
  demo_samples.json
  scientific_claims.yaml
  studio_manifest.json    由 snapshot 生成，禁止手写数字
radarmind/app/            Research Console，不动科学
```

不搬 `radarmind/` 内部包结构。用户建议的 `radarmind/core/` 重排 **不做**：那是为干净而搬家，违反本轮原则。

---

## 5. 两套 UI 的分工

| | Research Console（保留） | RadarMind Studio（新建） |
|---|---|---|
| 对象 | 自己对结果、改协议、查缺文件 | 导师 / 评审 / 非本方向听众 |
| 启动 | `python run_radarmind.py` | 日后 `START_STUDIO.bat` / `.sh` |
| 数字 | 已有 `results_io` + 页内 CSV | 只走 `get_metric` |
| 推理 | `ReplayEngine.load`（现场） | 同一 Engine；按钮写清 Run Inference / Replay Frozen Result |
| 美化 | 只修坏页和错误措辞 | 本轮视觉重点 |

Console 本阶段不花时间改皮肤。

---

## 6. 按 Studio 页面：复用什么、新写什么

### Overview

- 复用：`research_status()` 判定依据；四条状态从 claims + 已有结论映射  
  Closed-set HAR = SUPPORTED；Cross-Radar generalization = UNRESOLVED；Continuous = PARTIAL；Open-Set = PARTIAL
- 新写：横向 pipeline、那句核心问句、不要 KPI
- 不要：把 `health` 的 CPU/文件数放首页

### Live Demo（Studio-1 唯一页面）

- 复用：`DatasetRegistry`、`ReplayEngine.load`、`build_intelligence`、`_view()`、v01 demo_5 boxing
- 新写：三列 INPUT / RADARMIND / OUTPUT、管线状态、Plotly 点云播放
- 按钮：现场调 checkpoint 写 **Run Inference**；读 `motion_intelligence_examples.json` 写 **Replay Frozen Result**
- 解释：默认不调 LLM，点 Explain this motion
- 窗级概率保持窗级；滑帧只更新点云，不伪造逐帧 class

### Cross-Radar

- 复用：manifest + band adapter + 官方 PNG + `R0_*` 矩阵 + phase2 demo timestamps
- 新写：三雷达并排、交互 heatmap、点 24→77 时的旁注
- 0.039 只从 `get_metric("R0_24_TO_77")` 取，并标明 protocol-level

### Radar-ID / Representation Lab

- 复用：snapshot 键 `RID_R0_3CLASS` `RID_R1_3CLASS` `RID_R2D_3CLASS` `RID_R2D_2477` `RID_R2V_2477` `U1_RID` `U2_RID`
- 新写：3-RADAR PROBE 与 24/77 PROBE 两块，禁止画成一条连续下降曲线
- Representation Lab：R0 可用官方 PNG。R1/R2-D/R2-V **没有** 运行时逐样本数组。  
  决策：Lab 用已有静态 figure（如 `phase10_velocity_calibration/figures/fig3_r2v_pair.png`）+ 冻结指标。  
  **不要** 为了拖动 stage 重算 STFT。缺数组就写未接入 / 无逐样本重建。

### Shared Representation

- 复用：`representation_metrics.csv` → M0/M1/M2
- 文案：Better cross-radar correspondence does not imply sensor invariance.
- Radar-ID 保持 1.000，与 activity acc 分列

### Continuous

- 复用：`load_continuous`、demo `subject20_0513`、三层数据在 GT JSON + pred JSON
- 新写：三层时间轴、play/scrub、UNLABELED hatch
- 指标只读 snapshot / CSV，不现场重算 segment F1
- 页上固定：LABELED-INTERVAL REPLAY

### Open-Set

- 复用：`openset.py`、`mri_openset`、`loao_*`、`composite_demo.json`（须标明拼接）
- 主输出：UNKNOWN MOTION；nearest 标 Diagnostic only
- 禁止写成 open-world

### Motion Intelligence + Language

- 复用：schema、extract、prompt、boxing 样本
- 新写：provenance 卡片、Evidence Available / Unavailable、LLM INPUT 对照
- LANGUAGE 不是第五种 source

### Research Evidence / Status / Health

- 复用：snapshot 211 项、`supported_claims.md`、`health.check`
- 新写：Research Map、SUPPORTED/PARTIAL/UNRESOLVED 页、Health 抽屉
- 映射：DONE→SUPPORTED（仅当 claims 允许）；现有 PARTIAL 保持；BLOCKED/PENDING 中「未做到」→ UNRESOLVED

---

## 7. 读数规则

```
get_metric("R0_24_TO_77")
  → metric_snapshot_full.json
  → 若缺：按 registry 回源文件
  → 仍缺：FAIL CLOSED，页面 MISSING，禁止 fallback 数字
```

前端不得出现 `const accuracy = 0.039`。  
页文案里必须出现的数字，从 `studio_manifest.json` 生成。

两套 snapshot 的用法：

| 文件 | 用途 |
|---|---|
| `results/final_delivery/metric_snapshot_full.json` | Studio 主源（211） |
| `results/paper_package_v2/metric_snapshot.json` | 交叉核对 cross-radar / 物理项（110） |
| `scripts/paper_v2/registry.py` | 键、protocol、源路径的定义 |

---

## 8. 按钮与实时感

| 行为 | 按钮 | 实际 |
|---|---|---|
| `ReplayEngine.load` / `infer` | Run Inference | 读 cache + 冻结权重前向 |
| 读 `motion_intelligence_examples.json` 或 Phase 4/5 JSON | Replay Frozen Result / Replay Experiment | 不跑网 |
| 点 Explain | 现有 `explain()` | 只收 JSON |

不要用「RUN RADARMIND」同时表示两种。

---

## 9. 分阶段（确认后才开工）

| 阶段 | 做 | 不做 |
|---|---|---|
| **Studio-0** | 本三份报告 | 前端、删旧代码 |
| **Studio-1** | Live Demo 一条真实链路：RadHAR boxing → 点云 → 推理 → MI → 可选语言；`tests/studio_live_demo_test.*` | 其它页、调阴影 |
| **Studio-2** | 配对 CI4R 三列 + 矩阵 + Radar-ID 分栏 | 重算 R2 |
| **Studio-3** | 连续时间轴 + LOAO | 补全流、open-world |
| **Studio-4** | MI 页 + 语言 | ChatGPT 式首页 |
| **Studio-5** | 视觉、Presenter、Playwright 截图 | 改科学数字 |

---

## 10. Studio-1 最小切口（下一步，现在不写代码）

1. 新建 `services/replay_service.py`：调用 `RadarMindEngine.replay_load(dataset="radhar", sample_id="radhar___20_boxing_6_w0000", protocol="official_split")`
2. 新建 `studio/api`：返回 JSON（点云当前帧、prediction、MI）
3. 一个 React 页：左点云、中管线、右 boxing / 0.999 / MI
4. 测试：sample_id → activity/confidence/peak_velocity 与 `motion_intelligence_examples.json` demo_5 在容差内一致
5. 不一致时以 **现场推理** 为准，更新 UI，不改冻结 JSON 去迁就截图

---

## 11. 与 PPT 对齐

PPT 色在 `src/paper/visual_style.py`（NAVY `#3E5F7D` 等）。  
Studio 规格是另一套（`#10202E` / `#176B78` / `#C99A3D` / `#C76052`）。

本轮：**状态词、框架、矩阵、时间轴、MI  provenance 与 PPT 一一对应**；色值按 Studio 规格。Studio-5 再出 token 对照表，避免页和 slide 各说各的。

禁止 PPT 与软件互相抄硬编码数字。都回同一 snapshot。

---

## 12. 明确不做

- 重训、重切 test、重算 AUROC / segment F1 / Radar-ID
- 为 Representation Lab 重建 IQ→STFT
- 把 prefix_group 写成 subject
- 把 labeled-interval 写成 full-stream
- 把 LOAO 写成 open-world
- 把 LLM 当证据源
- 删除 Streamlit
- 为了「目录好看」拆 `radarmind/` 科学包
