# RadarMind Studio-0：数据流

审计日期：2026-09-02  
说明：下面每条路径都在仓库里对过文件。箭头表示现有调用，不是规划中的新计算。

---

## 1. 总览

```
原始大数据（多数不在 Application 热路径）
        │ 已完成预处理
        ▼
results/*/cache/*.npz          应用只读 cache
data/CI4R_MULTI3/**.png        官方谱图（本机 1959 张，Cross-Radar 页在用）
        │
        ▼
DatasetRegistry.get(sample_id) → RadarSample
        │
        ├─ ModelRegistry.predict()     冻结 checkpoint 前向
        │         ▼
        │   MotionPrediction
        │         ▼
        ├─ build_intelligence()        现场从样本+预测打 provenance
        │         ▼
        │   MotionIntelligence
        │         ▼
        ├─ build_timeline()            窗级合并；连续模式改用冻结 events
        │         ▼
        │   MotionTimeline
        │         ▼
        └─ LLMService.explain()        只收 MI JSON + 瘦 Timeline JSON

数字页（矩阵、AUROC、连续均值）不走上面这条推理链，
只读 results/phase* 与 metric_snapshot_full.json。
```

两条线必须分开：

| 线 | 用途 | 可否现场算 |
|---|---|---|
| A. 样本推理 | Live Demo：这个窗是什么、MI 是什么 | 可以：cache + checkpoint |
| B. Protocol 指标 | 24→77=0.039、mRI SI=0.478 | 不可以：只读冻结结果 |

---

## 2. 样本怎么进系统

### 2.1 RadHAR（Studio-1 首选）

```
results/phase1_pointcloud/cache/radhar_windows.npz
        │
        ▼
radarmind/data/radhar_adapter.py
  get("radhar___20_boxing_6_w0000")
        │
        ▼
RadarSample
  representation = point_cloud
  points         = T×P×C（T=60）
  point_mask
  action_gt      = boxing
  frame_rate     = 30
  subject_id     = UNKNOWN
  metadata.velocity_units = m/s
```

划分：`data_splits/radhar_official.csv`（official Train/Test，无 subject）。  
默认模型：`radhar_pointnet_gru` ← `results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/best.pt`。

### 2.2 mRI

```
results/phase1_pointcloud/cache/mri_windows.npz
        │
        ▼
mri_adapter.py
  sample_id 例：subject10_01_pose_1_w0000
  points T=32 P≤64  通道 x,y,z,doppler,intensity
  subject_id 有官方 ID
  velocity_units = UNIT_UNKNOWN     ← 界面不得写 m/s
```

协议：`subject_independent` → `data_splits/mri_subject_independent.csv`。  
默认模型：`mri_pointnet_gru` ← `mri_si_gru_xyz/best.pt`。

### 2.3 CI4R 77 GHz（Phase 1 HAR）

```
results/phase1_ci4r/cache/ci4r77_airhar_spec_zscore.npz
        │
        ▼
ci4r_adapter.py
  image 224×224  micro_doppler
  11 类
```

协议：`within_distribution` 或 `prefix_group_disjoint`（**不是** subject-independent）。

### 2.4 CI4R 三频配对（Cross-Radar）

```
results/phase2_cross_radar/cache/ci4r_{10,24,77}ghz_png224.npz
        │
        ▼
ci4r_band_adapter.py   注册名 ci4r10 / ci4r24 / ci4r77
  metadata.timestamp
  metadata.prefix_group     ← acquisition group，不是 subject
  7 类：Away Bend Crawl Kneel Pick Sit Towards
```

并排官方图：

```
ci4r_cross_radar_manifest.csv
  filter pairing_confidence == exact_timestamp
  group by timestamp
        │
        ▼
source_file → data/CI4R_MULTI3/spectrograms/.../*.png
```

配对统计：`pairing_stats.json`  
三频动作一致 327；冲突 133（已排除）。

---

## 3. 推理与 Motion Intelligence

```
RadarSample
    │ adapt_sample()     radarmind/sensors/base.py
    ▼
tensor (+ mask)
    │ checkpoint forward
    ▼
logits, embedding
    │ from_logits()
    ▼
MotionPrediction
  activity, confidence, class_probabilities
  prediction_entropy, uncertainty
  （open-set 时再 apply_open_set）
    │
    ▼
build_intelligence(sample, pred)
    │
    ├─ MODEL：activity, confidence, uncertainty
    ├─ PHYSICS_DERIVED：点云 mean/peak velocity, motion_energy
    │                   谱图只有 column-energy std，速度 UNKNOWN
    ├─ METADATA：dataset, radar, subject（有才写）, duration（有 frame_rate）
    └─ UNKNOWN：direction, periodicity, environment；
                无 subject / 无标定速度时对应字段
```

Open-set：`open_set_status == UNKNOWN` 时 activity 显示 `UNKNOWN_MOTION`，`nearest_known_class` 只进 evidence，不替代预测。

Boxing 冻结对照（demo_5，勿手写另一套）：

| 字段 | 值 | source |
|---|---|---|
| activity | boxing | MODEL |
| confidence | 0.9992529… | MODEL |
| uncertainty | 0.0039714… | MODEL |
| mean_velocity | −0.03487… m/s | PHYSICS_DERIVED |
| peak_velocity | 3.07334… m/s | PHYSICS_DERIVED |
| motion_intensity | 0.17618… | PHYSICS_DERIVED |
| direction / periodicity / subject / environment | UNKNOWN | UNKNOWN |

Studio-1 测试：同一 `sample_id` 现场推理应落在这些数附近；以现场为准，不改 JSON 去贴截图。

---

## 4. ReplayEngine 两条模式

### 4.1 sample（Live Demo）

```
replay.load(dataset, sample_id | sequence_id | window_ids)
  adapter.get 每个窗
  models.predict 每个窗          ← Run Inference
  build_intelligence 每个窗
  build_timeline
  state()._view()
    micro_doppler: 全图 + cursor 列
    point_cloud:   当前帧 xyz + velocity
```

RadHAR boxing demo 一次载入 24 个窗。窗间切换更新预测；帧滑条只改点云。

### 4.2 continuous（Continuous 页）

```
predicted_timelines/{recording}.json
ground_truth_timelines/{recording}.json
        │
        ▼
load_continuous()
  取出 mode["windows"] 的 sample_id
  再走 load() 推理这些窗
  用冻结 mode["events"] 覆盖 timeline.events
  seek_frame 在标注区间内移动
  GT 命中不到 → gt_activity = UNLABELED
```

禁止：把 UNLABELED 填成某一类。  
禁止：对 gap 补点云或补预测。

推荐 recording：`subject20_0513`（phase4 demo_a）。

---

## 5. 语言层

```
UI 按钮 Explain / Summarize / Uncertainty / Q&A
        │
        ▼
build_user_payload()
  task
  motion_intelligence   完整 Tagged 字典
  motion_timeline       仅 events 摘要（无 raw 窗、无点云）
        │
        ▼
有 DEEPSEEK_API_KEY 且非 offline？
  是 → DeepSeek
  否或失败 → fallback.render()
        │
        ▼
{ text, backend, ui_mode }
```

**不提供：** raw IQ、spectrogram 像素、point cloud、hidden features。  
页面应始终标明：ONLINE LLM / TEMPLATE FALLBACK。

Streamlit 现状：回放页每次刷新打 3 次 explain。Studio 改为点击才调用。

---

## 6. Protocol 指标怎么到页面

```
scripts/paper_v2/registry.py     键 → 源文件 + protocol 说明
        │ snapshot.py 已跑过
        ▼
results/final_delivery/metric_snapshot_full.json
        │
        ▼
Studio result_service.get_metric(key)
        │
        ▼
Cross-Radar / Evidence / Status
```

缺文件：FAIL CLOSED，状态 MISSING。  
不要：静默换成另一 protocol 的数。

### 6.1 Cross-radar 矩阵

```
Train\Test     10      24      77
10            0.843   0.143   0.373
24            0.353   0.898   0.039
77            0.275   0.163   0.922
```

每个格子 ← `results/phase2_cross_radar/runs/{within|cross}_*/metrics.json` 的 `accuracy`。  
点 24→77：只展示 protocol accuracy 0.039，旁边写 zero-shot、target 未参与训练。  
不要把当前窗 softmax 写成 0.039。

### 6.2 Radar-ID（两套 probe）

```
3-RADAR PROBE（10/24/77）
  Official Plot     RID_R0_3CLASS   1.000
  Crop / Normalize  RID_R1_3CLASS   1.000
  Numerical Doppler RID_R2D_3CLASS  0.993

24/77 PROBE（不可与上一列连成一条轴）
  Numerical Doppler RID_R2D_2477    1.000
  Velocity Align    RID_R2V_2477    0.939
  Unified STFT      U1_RID          0.939
  MTI               U2_RID          0.980
```

### 6.3 Shared representation

```
     Acc     Retrieval  Radar-ID
M0   0.769   0.718      1.000
M1   0.932   0.935      1.000
M2   0.878   0.864      1.000
```

来源：`representation_metrics.csv`，protocol=`multi_radar_shared`（known-radar joint encoder，不是 unseen-radar）。

### 6.4 Continuous 汇总

`continuous_runs.csv` 中 `mode=confidence_aware` 对 subject10/20/3 三行做未加权均值 → snapshot `CONT_*`。  
单条 recording 的 window_accuracy 用该行，不要和 0.494 混用。

### 6.5 Open-set

`loao_*_per_action.json` 对 13 fold 做 mean/std → `MSP_AUROC` 等。  
chance 0.500 是约定对照，不是实验文件里的第三种方法。

---

## 7. 可视化数据是否够现场看

| 需要看到 | 有没有 | 从哪取 | Studio 注意 |
|---|---|---|---|
| RadHAR / mRI 点云动画 | 有 | cache `points`[t] + mask；`_view()` | 限点数；按帧 API，不把整库下发 |
| CI4R 谱图 heatmap | 有 | band/phase1 npz `image` | 轴：time bin / freq bin；无标定不标 m/s |
| 官方 PNG 三频对照 | 有 | `data/CI4R_MULTI3` + manifest | Cross-Radar 主视觉可用 PNG |
| R1 裁剪图逐样本 | **无运行时数组** | 仅有指标 | Lab 用冻结 figure 或写未接入 |
| R2-D / R2-V 逐样本 | **无 `*.npz`** | 仅 answers.json + 少量 figure | 禁止现场重做 STFT |
| 连续三层时间轴 | 有 JSON | GT events + pred events + series | gap = UNLABELED |
| 连续 gap 点云 | 无 | — | hatch，不补类 |
| Open-set 输入点云 | 有 | 真实 TEST 窗 | composite 须写 spliced |
| Boxing 速度着色 | 有 | 第 4 通道 m/s | hover 可用；不声称 attribution |
| 窗级 top-5 概率 | 有 | `class_probabilities` | 可以 |
| 帧级类别曲线 | 无 | — | 不画 |

---

## 8. Demo 载入路径（现场 4–6 分钟）

建议写入日后的 `configs/demo_samples.json`（本阶段不建文件）：

```
Demo 1  dataset=radhar
        sample_id=radhar___20_boxing_6_w0000
        protocol=official_split
        why=manuscript boxing；点云密、预测对

Demo 2  timestamps=1573837254
        protocol=zero-shot cross-radar
        metric=R0_24_TO_77
        why=真实三频 pair；0.039 用 protocol 数

Demo 3  recording=subject20_0513
        protocol=subject_independent
        why=phase4 demo_a；labeled-interval

Demo 4  held_out=pose_4
        model=mri_openset
        source=fold 真实窗 或 composite_demo（须标明拼接）
        why=UNKNOWN MOTION

Demo 5  同 Demo 1 + explain(task=current)
```

---

## 9. Health / Fail-closed

现有 `health.check()`：

Required：CI4R/mRI/RadHAR cache；CI4R random/prefix；mRI SI；RadHAR XYZV。  
Optional：Phase 2–5 结果文件；DeepSeek key。

Studio 还应报：

| 项 | READY 条件 | 缺了怎么办 |
|---|---|---|
| Demo 1 样本 | adapter.get boxing 成功 | Live Demo 不可开，不编样本 |
| 矩阵文件 | `cross_24_to_77/metrics.json` | Cross-Radar 格子 MISSING |
| 连续 JSON | `predicted_timelines/subject20_0513.json` | Continuous 不可播 |
| Open-set pack | `app_pack.json` + checkpoint | 不输出假 UNKNOWN |
| 语言 | key 或 fallback | 标明模式，不装成在线 |

---

## 10. 给 API 的序列化约束

`ReplayState.to_dict()` 里 `view.image` / `view.xyz` 是 numpy。  
HTTP 不要一次塞整窗 T×P：

- 点云：当前帧 `xyz` + `velocity` + `n_points`（RadHAR 单帧约几十点）
- 谱图：png / 下采样数组 + cursor_t
- MI / prediction：已是 JSON 友好结构（`jsonable`）
- embedding：默认不进展示 payload；Output Inspector 需要再给

---

## 11. 与 Research Console 的共用点

```
                    ┌─ Streamlit page_replay
RadarMindEngine  ───┤
                    └─ studio/api /replay
                         两者同一 cache、同一 checkpoint

                    ┌─ Streamlit page_cross_radar 直读 CSV
results/phase2  ────┤
                    └─ studio get_metric("R0_24_TO_77")
```

Console 现在有一部分 CSV 写在 `streamlit_app.py` 里。Studio **不要抄第二份解析**。先抽 `result_service`，Console 以后再改调用（本阶段不必改 Console）。

---

## 12. 已知措辞陷阱（数据流里就要挡住）

| 数据 | 允许说法 | 禁止说法 |
|---|---|---|
| mRI SI split | subject-independent | 套到 CI4R |
| CI4R prefix_group | acquisition group | subject |
| Phase 4 | labeled-interval replay | full-stream continuous recognition |
| Phase 5 | open-set rejection / LOAO | open-world / open-vocabulary achieved |
| R0 矩阵 | unseen-radar generalization unresolved | radar-invariant recognition |
| M1 Radar-ID 1.000 | known-radar 上身份仍可分 | 学到了不变表示 |
| LLM | semantic interpretation of MI | 模型看过雷达图 |
| 连续 gap | UNLABELED | background |
| 24→77 0.039 | protocol accuracy | 当前样本 confidence |
