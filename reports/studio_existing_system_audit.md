# RadarMind Studio-0：现有系统审计

审计日期：2026-09-02  
仓库：`C:\Users\13919\Desktop\RadarMind`  
范围：只读扫描。未删代码、未换 checkpoint、未重训、未写 Studio 前端。

原则：REUSE SCIENCE / REBUILD PRESENTATION。

---

## 结论（先看这段）

当前仓库已经有一套可运行的科研应用，不是空项目。

已有：

- Python 科学核：`RadarMindEngine` → `ReplayEngine` / `ModelRegistry` / Motion Intelligence / Timeline / LLM
- Streamlit 研究界面：12 个侧栏页，入口 `python run_radarmind.py`（8502）
- 冻结 cache、checkpoint、Phase 1–5 / 7–10 结果文件
- 论文数字注册表：`scripts/paper_v2/registry.py` + `results/final_delivery/metric_snapshot_full.json`（211 项）
- manuscript 中 boxing 示例可定位到真实样本 `radhar___20_boxing_6_w0000`

没有：

- `studio/`、`configs/`、`services/`
- FastAPI / React / Vite
- `scientific_claims.yaml`
- 面向展示的 Presenter Mode / Demo Mode 1–5 统一入口

旧 UI 能核验结果，不能当现场展示软件。科学计算不必重写。

---

## 1. 组件清单

| Component | Existing path | Current function | Scientific status | Reusable? | Needs refactor? | Needs redesign? | Decision |
|---|---|---|---|---|---|---|---|
| RadarMindEngine | `radarmind/engine.py` | 统一入口：`infer` / `intelligence` / `replay_load` / `explain` / `demo_registry` / `missing_checkpoints`。注释写明给 Streamlit 和后续前端用 | 冻结协议 + 现场推理 | 是 | 否 | 否 | KEEP |
| ReplayEngine | `radarmind/replay.py` | `load()` 读 cache、按窗跑 checkpoint、建 MI 与 timeline；`load_continuous()` 叠 Phase 4 冻结时间线；`_view()` 出谱图/点云当前帧 | 样本模式=现场推理；连续模式=冻结事件 + 再推理窗口 | 是 | 否（科学） | 否 | KEEP |
| ReplayState | `radarmind/replay.py` | `to_dict()`：prediction、MI、timeline、view | view 里含 numpy，不能直接 JSON | 是 | 序列化需抽一层 | 否 | KEEP + 薄封装 |
| DatasetRegistry / RadarSample | `radarmind/data/base.py` | 统一样本：`image` / `points` / `point_mask` / metadata | 冻结 cache | 是 | 否 | 否 | KEEP |
| CI4R 77 GHz adapter | `radarmind/data/ci4r_adapter.py` | Phase 1 micro-Doppler 224×224 | 冻结 npz | 是 | 否 | 否 | KEEP |
| CI4R 10/24/77 adapter | `radarmind/data/ci4r_band_adapter.py` | 官方 PNG 缓存；`metadata["timestamp"]` | 冻结 npz；本机另有 PNG 树 | 是 | 否 | 否 | KEEP |
| mRI adapter | `radarmind/data/mri_adapter.py` | T×P×5 点云；SI / random-segment | 冻结；velocity 单位 `UNIT_UNKNOWN` | 是 | 否 | 否 | KEEP |
| RadHAR adapter | `radarmind/data/radhar_adapter.py` | T×P 点云；official Train/Test；无 subject | 冻结；velocity 单位 m/s | 是 | 否 | 否 | KEEP |
| ModelRegistry | `radarmind/models/registry.py` | 按 dataset/protocol 解析 checkpoint，`predict()` | 冻结权重 | 是 | 否 | 否 | KEEP |
| MotionPrediction | `radarmind/models/prediction.py` | logits → 概率、entropy、uncertainty | 现场从 logits 算 | 是 | 否 | 否 | KEEP |
| Open-set scoring | `radarmind/models/openset.py` | MSP / Energy / Prototype；`display_activity()` → `UNKNOWN_MOTION` | 阈值来自 `app_pack.json` | 是 | 否 | 否 | KEEP |
| Source / Tagged | `radarmind/schema.py` | `MODEL` / `PHYSICS_DERIVED` / `METADATA` / `UNKNOWN`；禁止补 0 | 冻结契约 | 是 | 否 | 否 | KEEP |
| MotionIntelligence | `radarmind/intelligence/schema.py` | 带 provenance 的中间层 | 冻结 schema | 是 | 否 | 否 | KEEP |
| build_intelligence | `radarmind/intelligence/extract.py` | 点云运动学 / 谱图列能量；谱图速度保持 UNKNOWN | 现场从样本算，不编方向 | 是 | 否 | 否 | KEEP |
| Timeline | `radarmind/intelligence/timeline.py` | `smooth_labels` + `merge_adjacent`；无学习分段器 | 连续页用冻结 events 覆盖 | 是 | 否 | 否 | KEEP |
| LLMService | `radarmind/llm/provider.py` | `explain(task, intelligence, timeline, question)` | DeepSeek 或模板 | 是 | 否 | 否 | KEEP |
| Prompt contract | `radarmind/llm/prompt.py` | 只收 MI + 瘦 Timeline JSON；禁止 raw radar | 冻结 | 是 | 否 | 否 | KEEP |
| Template fallback | `radarmind/llm/fallback.py` | 无 key / 离线 / 调用失败 | 可用 | 是 | 否 | 否 | KEEP |
| ProtocolRegistry | `radarmind/protocols.py` | 协议是否可用；CI4R prefix ≠ subject；cross-env 未接入 | 冻结措辞 | 是 | 否 | 否 | KEEP |
| results_io | `radarmind/results_io.py` | Phase 1 泛化表、消融、`research_status()` | 只覆盖 Phase 1 + 状态列表 | 是 | 需扩展读数，不重算 | 否 | KEEP + 扩展读接口 |
| health.check | `radarmind/health.py` | 3 cache + 4 主 checkpoint；Phase 2–5 / DeepSeek 可选 | 缺 required 不该继续演示 | 是 | Studio 要加 demo/LLM/结果文件项 | 否 | KEEP + 扩展检查项 |
| paths | `radarmind/paths.py` | cache / checkpoint 路径 | 冻结 | 是 | 否 | 否 | KEEP |
| Frequency adapters | `radarmind/adapters/radar_freq.py` | 10/24/77 编码包装 | 不共享 token | 是 | 否 | 否 | KEEP |
| Sensor adapt_sample | `radarmind/sensors/base.py` | 样本 → 模型张量 | 推理用 | 是 | 否 | 否 | KEEP |
| Paper metric registry | `scripts/paper_v2/registry.py` | 每个论文数字 → 源文件；缺文件则失败 | 冻结；Studio 应走这里 | 是 | 抽成可 import 的 get_metric | 否 | KEEP |
| Snapshot（完整） | `results/final_delivery/metric_snapshot_full.json` | 211 项，含 HAR / 连续 / open-set / Radar-ID | 与 manuscript 一致 | 是 | 否 | 否 | KEEP（Studio 读数主源） |
| Snapshot（v2 子集） | `results/paper_package_v2/metric_snapshot.json` | 110 项，偏 cross-radar / 物理诊断 | 子集，缺 mRI/RadHAR/连续/open-set | 是 | 不要当唯一源 | 否 | KEEP（对照用） |
| Claims 文稿 | `results/paper_package_v2/supported_claims.md` | STRONG / LIMITED / UNSUPPORTED | 无 YAML registry | 是 | 需做成 `scientific_claims.yaml` | 否 | KEEP 内容；新建 YAML |
| Streamlit 主应用 | `radarmind/app/streamlit_app.py` | 12 页研究控制台；回放三列已有 | UI 耦合部分 CSV 直读 | 界面否；调用 Engine 是 | 研究页 CSV 应抽出 | 展示层是 | REBUILD UI ONLY（Studio）；Console 保留 |
| RadarBridge Explorer | `demo/app.py` | 旧演示：Explorer / Cross-Radar / Story | 已被主应用替代 | 部分交互可参考 | 否 | 否 | DEPRECATE（不删，不维护） |
| 采集工具 UI | `radarmind_acquisition/ui.py` | MOCK 采集 | 与推理核无关 | 工具本身可留 | 否 | 否 | 不进 Studio 导航 |
| 论文配色 | `src/paper/visual_style.py` | 图/PPT 色板 | 与本次 Studio 指定色不完全相同 | 术语/状态语义可对齐 | Studio 用指定 token | 展示层 | 色值以 Studio 规格为准；术语对齐 PPT |
| 活动色 | `src/paper/activity_color_map.py` | 活动 → 色 | 论文图用 | 可参考 | 否 | 否 | KEEP |
| Service layer | （不存在） | — | — | — | 新建薄封装，禁止复制公式 | 否 | 新建（抽，不重写） |
| FastAPI / React | （不存在） | — | — | — | 新建 | 是（展示） | 新建 |
| `configs/` | （不存在） | — | — | — | 新建 demo + claims | 否 | 新建 |

---

## 2. 分类汇总

### A. KEEP（科学逻辑原样保留）

- `radarmind/engine.py`
- `radarmind/replay.py`
- `radarmind/schema.py`（UNKNOWN 契约）
- `radarmind/intelligence/{schema,extract,timeline}.py`
- `radarmind/models/{registry,prediction,openset}.py`
- `radarmind/data/*` adapters 与 cache
- `radarmind/llm/*`（prompt 硬约束）
- `radarmind/protocols.py`
- `radarmind/health.py`（检查逻辑）
- `radarmind/paths.py`
- `scripts/paper_v2/registry.py`
- 全部 `results/phase*/` 冻结 CSV / JSON / checkpoint / cache
- `results/final_delivery/metric_snapshot_full.json`
- `results/radarmind_v01/demo_registry.json` 及 `motion_intelligence_examples.json`
- Streamlit 作为 **RadarMind Research Console**（不删）

### B. REFACTOR（抽接口，不改计算公式）

- 把 Streamlit 里直读 `results/phase2…phase10` 的表格逻辑收到 `result_service` / 扩展 `results_io`
- `ReplayState.view` 的 numpy → 按帧 PNG / 点列表，供 HTTP
- `health.check()` 增加：demo sample、语言服务、结果文件、ReplayEngine
- `research_status()` 的 DONE/PARTIAL/BLOCKED 映射到 Studio 的 SUPPORTED/PARTIAL/UNRESOLVED（映射层，不改判定依据）
- 从 `supported_claims.md` + registry 生成 `configs/scientific_claims.yaml`
- 汇总已有 demo registry → `configs/demo_samples.json`（只引用真实 sample_id）
- 可选：adapter 内 `CLASS_NAMES` 收拢到一处（低优先级，不要先动 adapter）

### C. REBUILD UI ONLY

- Overview（现在是能力清单，不是 graphical abstract）
- Live Demo（回放页已有三列，但是 Streamlit 控件堆叠）
- Cross-Radar（表 + 三张 PNG，没有交互矩阵）
- Continuous（表 + 折线，没有三层时间轴播放）
- Open-Set（指标表 + 直方图）
- Motion Intelligence 独立页（现在嵌在回放中栏）
- Research Evidence / Research Status
- System Health 抽屉
- Presenter Mode、Demo Mode、Inspect Input/Output

### D. DEPRECATE（不删文件，Studio 不再当主入口）

- `demo/app.py` RadarBridge Explorer（独立旧应用）
- `demo/story.py`
- Streamlit「采集」进入 Studio 主导航
- 把 `paper_package_v2/metric_snapshot.json`（110 项）当成全量数字源
- 在前端写死 `0.039` / `0.949` 等

---

## 3. 十五个必答问题

### 1. Streamlit 在哪里？

`radarmind/app/streamlit_app.py`（1151 行，单文件多页）。  
启动：`python run_radarmind.py` → `http://localhost:8502`。  
离线说明：`python run_radarmind.py --offline-demo` 或 `run_radarmind_offline.py`。  
脚本：`scripts/run_radarmind.sh`。没有 `START_STUDIO.bat`。

### 2. 现有哪些页面？

侧栏（`streamlit_app.py` 1093–1108）：

| 页 | 函数 | 对应 Studio |
|---|---|---|
| 首页 | `page_home` | Overview（能力矩阵，不是 pipeline abstract） |
| 回放 | `page_replay` | Live Demo 雏形 |
| 连续运动 | `page_continuous` | Continuous |
| Open-Set Recognition | `page_open_set` | Open-Set |
| 泛化对照 | `page_generalization` | Research Evidence 一节 |
| 表征消融 | `page_ablation` | Research Evidence 一节 |
| Cross-Radar | `page_cross_radar` | Cross-Radar |
| Shared Motion | `page_shared_motion` | Shared Representation |
| Radar Invariance | `page_radar_invariance` | Representation Lab / Radar-ID（长滚动） |
| RadarMind Pipeline | `page_pipeline` | Overview 的静态清单 |
| 研究进度 | `page_status` | Research Status |
| 采集 | `radarmind_acquisition.ui.page` | 不进 Studio |

没有独立 Motion Intelligence 页、没有 Presenter Mode、没有统一 Demo 1–5。

### 3. ReplayEngine 还能工作吗？

代码完整：`ReplayEngine.load` / `load_continuous` / `seek_frame` / `_view`。  
测试：无单独 `test_replay_engine.py`；`radarmind/tests/smoke.py` 的 `test_app_backend` 调 `eng.replay_load()`；`test_continuous.py` 测连续 state。  
本机 cache 与 4 个主 checkpoint 均在盘上。  
未在本轮启动进程做运行时冒烟（审计不跑推理）。代码路径与产物齐全，Studio-1 应用现有 smoke 验证。

### 4. 模型 inference API 在哪里？

- `RadarMindEngine.infer(sample, model_name, protocol)` → `radarmind/engine.py:38`
- `ModelRegistry.predict(sample, …)` → `radarmind/models/registry.py:317`
- 回放：`replay_load()` 对每个 window 调 `predict()`

没有 HTTP API。

### 5. Frozen checkpoint 在哪里？

本机均存在：

| 模型 | 路径 |
|---|---|
| CI4R random ResNet18 | `results/phase1_ci4r/baselines/random_record/resnet18/best.pt` |
| CI4R prefix ResNet18 | `results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/best.pt` |
| mRI SI PointNet-GRU XYZ | `results/phase1_pointcloud/baselines/mri_si_gru_xyz/best.pt` |
| RadHAR official XYZV | `results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/best.pt` |
| Shared Motion M2 | `results/phase3_shared_motion/runs/M2_shared/best.pt` |
| Open-set | `results/phase5_open_set/app_pack.json` 内 checkpoint 字段 |

### 6. Results CSV / JSON 在哪里？

| 内容 | 路径 |
|---|---|
| 全量论文数字 | `results/final_delivery/metric_snapshot_full.json`（211） |
| Cross-radar 矩阵源 | `results/phase2_cross_radar/runs/within_*/metrics.json`、`cross_*/metrics.json`、`training_runs.csv` |
| 配对表 | `results/phase2_cross_radar/ci4r_cross_radar_manifest.csv`（1790 行） |
| M0/M1/M2 | `results/phase3_shared_motion/representation_metrics.csv` |
| 连续 | `results/phase4_continuous/{continuous_runs,segment_metrics,boundary_metrics}.csv` |
| Open-set | `results/phase5_open_set/{eval_summary,loao_*_per_action,app_pack,composite_demo}.json` |
| Radar-ID R0/R1 | `results/phase7_radar_invariance/representation_radar_id.json` |
| R2-D | `results/phase8_physical_representation/answers.json` |
| 24/77 velocity | `results/phase10_velocity_calibration/answers.json`（snapshot 键 `RID_R2V_2477`） |
| Boxing MI 冻结例 | `results/radarmind_v01/motion_intelligence_examples.json`（demo_5） |

### 7. Motion Intelligence schema 在哪里？

- 结构：`radarmind/intelligence/schema.py` `class MotionIntelligence`
- 构造：`radarmind/intelligence/extract.py` `build_intelligence()`
- 标签：`radarmind/schema.py` `Source`
- 冻结示例：`results/radarmind_v01/motion_intelligence_examples.json`

字段：activity, confidence, direction, mean_velocity, peak_velocity, motion_intensity, periodicity, duration, motion_state, uncertainty, dataset, radar, subject, environment, evidence。

### 8. Timeline schema 在哪里？

`radarmind/intelligence/timeline.py`：`MotionEvent`、`MotionTimeline`、`build_timeline()`。  
连续冻结：`results/phase4_continuous/predicted_timelines/{recording}.json`、`ground_truth_timelines/{recording}.json`。

Event 已有：start_time, end_time, activity, confidence, n_windows, low_confidence。  
没有单独的 boundary_offset 字段；边界误差在 `boundary_metrics.csv` 的 protocol 级统计，不要现场现算。

### 9. Open-set result 在哪里？

`results/phase5_open_set/`：13-fold LOAO（pose_1…pose_10, T_pose, walk, free_form）。  
`loao_{msp,energy,prototype}_per_action.json`；`app_pack.json`；`composite_demo.json`（拼接 TEST 窗，不是真实连续 recording）。

Snapshot：`PROTO_AUROC` 0.628 ± 0.125。

### 10. Continuous result 在哪里？

`results/phase4_continuous/`。  
冻结预测时间线（6 条）：subject5_0502, subject3_0501, subject20_0513, subject17_0510, subject15_0508, subject10_0504。  
GT 时间线更多。  
Demo 推荐：`subject20_0513`（test 里 window accuracy 最高）。

Snapshot（confidence_aware，3 条 test 未加权均值）：

- CONT_WIN_ACC 0.494
- CONT_WIN_F1 0.419
- CONT_FRAME_ACC 0.489
- CONT_FRAME_F1 0.416
- CONT_SEG_F1 0.152
- CONT_BND_SEC 20.6 s

每行 CSV 都写了：`Mode A: labeled intervals only`。

### 11. Cross-radar paired sample 怎么定位？

1. `ci4r_cross_radar_manifest.csv`：`pairing_confidence == exact_timestamp`，按 `timestamp` 聚 10/24/77。
2. `pairing_stats.json`：三频动作一致 327 条；冲突 133 条已排除。
3. Adapter：`CI4RBandAdapter.get()` → `metadata["timestamp"]`。
4. 现成 demo：`results/phase2_cross_radar/demo_registry.json`  
   - `1573837254` Away（三频 within 都对 / 77→10 错）  
   - `1574194391` Away（source-only 错，CORAL 对；UDA ≠ DG）
5. 官方 PNG 本机存在：`data/CI4R_MULTI3/` 下 1959 张 PNG。manifest 第一条 PNG 已核对存在。
6. 交互谱图也可用 band cache：`results/phase2_cross_radar/cache/ci4r_{10,24,77}ghz_png224.npz`。

配对键是 **acquisition timestamp**，不是文件名前缀，也不是 subject。

### 12. Language layer 怎么调用？

```
eng.explain(task="current"|"summary"|"uncertainty"|"qa",
            intelligence=dict, timeline=dict, question=str|None)
```

实现：`LLMService.explain()`。  
输入：`build_user_payload()` 只放 Motion Intelligence + 瘦 Timeline。  
模式：`ui_mode()` → `DeepSeek Online` 或 `Template Fallback`。  
Streamlit 回放页每次渲染会自动打 current/summary/uncertainty 三次；Studio 应改为按钮触发。

### 13. 哪些旧 UI 可以删？

本阶段 **不删**。Studio 上线后可停止维护：

- `demo/app.py`（RadarBridge）
- Streamlit 主导航里的「采集」

不可删：`radarmind/app/streamlit_app.py`（Research Console）。

### 14. 哪些 scientific component 必须原样保留？

见上文 A. KEEP。尤其不要重写：训练脚本、open-set 计分、timeline 平滑、cross-radar 数字、UNKNOWN 契约、LLM prompt 禁令。

### 15. 哪些数据缺少真实样本可视化？

| 缺口 | 说明 |
|---|---|
| R1 / R2-D / R2-V 逐样本数组 | 运行时 adapter 只有 R0 官方 PNG 与 Phase 2 PNG-npz。Phase 8 无 R2 `*.npz`。Representation Lab **不能**现场重算 Doppler 网格 |
| mRI 原始 CSV / 未标注 gap 点云 | 连续只能 labeled-interval；UNLABELED 不得补成 background |
| CI4R 谱图 → m/s | `extract.py` 明确 UNKNOWN |
| mRI 速度单位 | `UNIT_UNKNOWN`，不得标 m/s |
| Open-set composite | 拼接窗，不是一条 recording |
| 逐帧 class probability | 模型是 window-level，禁止伪造帧级类别曲线 |
| Event.boundary_offset | 无此字段则显示 UNKNOWN |
| 坐标 Forward/Left/Height | 未建立约定，点云轴只用 X/Y/Z |

已有可视化原料：

- RadHAR / mRI：cache 内 `points` + `mask`；`ReplayEngine._view()` 已出 xyz + velocity
- CI4R：npz 224×224 与官方 PNG
- 连续：冻结 GT / pred events + 标注窗点云
- Boxing：真实样本 + 冻结 MI 与现场推理路径都在

---

## 4. 关键数字核对（只对照冻结文件，未另造真值）

| 声称 | 冻结值 | 源 |
|---|---|---|
| 24→77 | 0.0392156862745098 → 0.039 | `phase2_cross_radar/runs/cross_24_to_77/metrics.json`；snapshot `R0_24_TO_77` |
| CI4R within 10/24/77 | 0.843 / 0.898 / 0.922 | `within_10/24/77/metrics.json` |
| 矩阵其余 | 10→24 0.143；10→77 0.373；24→10 0.353；77→10 0.275；77→24 0.163 | snapshot R0_* |
| mRI SI | Acc 0.478 / F1 0.420 | `mri_si_gru_xyz/metrics.json` |
| RadHAR XYZV | Acc 0.949 / F1 0.949 | `radhar_official_gru_xyzv/metrics.json` |
| M0 / M1 / M2 Acc | 0.769 / 0.932 / 0.878 | `representation_metrics.csv` |
| M0 / M1 / M2 retrieval | 0.718 / 0.935 / 0.864 | 同上 `retrieval_action_top1` |
| M0 / M1 / M2 Radar-ID | 1.000 / 1.000 / 1.000 | 同上 `radar_probe_accuracy`（known-radar joint，不是 zero-shot） |
| Radar-ID R0 / R1 | 1.000 / 1.000 | `representation_radar_id.json`（3-radar probe） |
| R2-D Radar-ID | 0.993 | `phase8/.../answers.json` `fingerprint.R2_D` |
| 24/77 velocity align | 0.939 | snapshot `RID_R2V_2477`（**两雷达 probe，不能和 3-radar 1.000 画成一条单调轨**） |
| Unified STFT / MTI | 0.939 / 0.980 | snapshot `U1_RID` / `U2_RID` |
| Prototype AUROC | 0.628 ± 0.125 | snapshot `PROTO_AUROC` |
| Boxing MI | act boxing；conf 0.999；unc 0.004；mean v −0.035 m/s；peak \|v\| 3.073 m/s；energy 0.176 | `motion_intelligence_examples.json` demo_5；sample `radhar___20_boxing_6_w0000` |

`paper_package_v2/metric_snapshot.json` **没有** MRI_SI / RADHAR / CONT / PROTO 键。Studio 读数必须用 `metric_snapshot_full.json` 或现场按 `scripts/paper_v2/registry.py` 回源文件。

---

## 5. 现有 Demo 与 Studio Demo 1–5 的对应

| Studio Demo | 现成条目 | 样本 | 注意 |
|---|---|---|---|
| 1 Closed-set PC HAR | `radarmind_v01` demo_5 | `radhar___20_boxing_6_w0000`（24 窗） | 与 manuscript boxing 一致 |
| 2 Cross-radar failure | phase2 demo_A/B | timestamp `1573837254` Away | 现成失败例是 **77→10**，不是 24→77 单样本。24→77 **0.039 是 protocol 准确率**，不得写成该窗 confidence |
| 3 Continuous | phase4 demo_a | `subject20_0513` | labeled-interval；why=最高 window acc |
| 4 Unknown rejection | `composite_demo.json` held_out=`pose_4` | 拼接 TEST 窗 | 必须标明 spliced windows。更好：用 fold `pose_4` 的真实窗 + `mri_openset` |
| 5 MI + LLM | 同 Demo 1 | boxing 同上 | 已定位，不必伪造 |

v01 demo_1/2 是 CI4R 谱图；demo_3/4 是 mRI SI 对/错。可作 Sample Gallery，不必当现场主路径。

---

## 6. 旧 UI 能做什么 / 缺什么

回放页已经是三列：Radar View | Motion Intelligence | Timeline + 下方 AI。  
缺：INPUT→RADARMIND→OUTPUT 的处理管线动画、Inspect 抽屉、点云时间轴与 MI 字段联动、点「Explain」才调 LLM、Presenter Mode。

Cross-Radar 页：表 + 三列官方 PNG。缺：交互 transfer heatmap、24→77 旁注、3-radar / 24-77 两套 Radar-ID 分栏、Representation Lab 拖 stage。

Continuous 页：滑条 + GT/pred 表 + 置信度折线。缺：三层时间轴、hatch gap、播放头。

Open-Set 页：指标 + 直方图 + composite 按钮。缺：UNKNOWN MOTION 作为主输出、nearest 仅 diagnostic。

---

## 7. 本轮未做

- 未启动 Streamlit / 未跑 GPU 推理
- 未改任何科学代码
- 未进入 Studio-1
