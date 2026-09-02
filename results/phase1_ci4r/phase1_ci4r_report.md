# Phase 1A — CI4R 77 GHz 预处理与第一轮 baseline

只处理 `Cross-frequency/77ghz/` 下动作目录中的 `*_Raw_0.bin`。未打开 10 GHz，未全量解压 `raw_data.zip`。

## 1. CI4R 77 GHz raw IQ 的真实数据结构是什么？

磁盘是 DCA1000 `int16`，长度 65536000（文件 131072000 字节）。按 AIRHAR `bin2np` / MATLAB `datToImage_77.m` 列优先 reshape 成 `(8, 8192000)`：前 4 行 I、后 4 行 Q，对应 4 路 RX。

AIRHAR 只取第 4 路：`lane[3] + 1j*lane[7]`，再 Fortran reshape 成 `(256, 32000)` `complex64`。

12 条不同动作 / 文件名前缀的 QC：shape 均为 `[256, 32000]`，无 NaN / Inf，幅度零值比例约 1e-6。见 `raw_signal_qc.csv`。

## 2. `(256, 32000)` 每个维度的物理含义是什么？

- **256**：每个 chirp 的 ADC 采样数（NTS）。Range FFT 沿此轴。AIRHAR 与 MATLAB 注释一致。
- **32000**：该条记录、选定 1 路 RX 之后的 chirp / pulse 数。`8192000 / 256 = 32000`，与文件长度一致。

官方 reshape 链（本阶段采用）：

```
int16[65536000] → (8, 8192000) F → 4 RX 复数流 → AIRHAR 第 4 路 → (256 ADC, 32000 chirps) F
```

**UNRESOLVED**

- 注释写 `NPpF=128`、`NoF=500` → 64000 pulse / 约 20 s；实文件是 32000 pulse。是否 `128×250` 未在 loader 中写死，不按帧维拆。
- Tx 天线数：两份官方 loader 都没写。
- 不把数据说成已存好的 `[frame, chirp, rx, adc]`。

详见 `ci4r_signal_schema.md`。

## 3. 官方 preprocessing 到底做了什么？

分两套，不要混。

**A. AIRHAR `RGB_mD`（本阶段复现并用于训练）**

1. Range FFT + fftshift（沿 256）
2. 保留正距离半边 `[128:]`
3. 距离门 `23:31` 求和
4. STFT：Hann 256，overlap 200（hop 56），nfft 4096
5. `|fftshift|` 后上下翻转
6. 相对峰值 `20 log10`，截断 `[-45, 0]`
7. `cv2` cubic 缩到 224×224
8. 建库时再对每张图做 z-score

AIRHAR 源码里 MTI 只有空注释，**没有执行**。

**B. MATLAB `datToImage_77.m`（archive 里 PNG / `sx1` 的来源）**

- 4 路 RX 都做 Range FFT，之后只用 RX1
- MTI：`conv([1,-2,3,-2,1])`
- 对距离门 **第 8 行** 做 spectrogram，overlap **197** → 时间维 538
- 保存复数 `sx1`，再用 jet + getframe 出 PNG

立方插值后 AIRHAR 图会略低于 −45 dB（抽样 36 条常见到约 −47）。这是 resize，不是另加一步预处理。

## 4. 我们能否从 raw IQ 基本复现官方 spectrogram？

能复现 **AIRHAR 核心链**（距离 FFT → 选门求和 → STFT → 对数截断 → 224）。抽样 36 条均得到 `(224, 224)`。

**不能**与 archive PNG 逐像素对齐。8 条 Towards 对照：把 PNG 转灰度再 resize 到 224 后，与 AIRHAR z-score 图相关系数约 **0.37–0.45**。原因已核对：

- PNG 是带坐标轴和 colorbar 的绘图（976×770），不是 224 特征图
- MATLAB 用了 MTI、距离门 8、overlap 197；AIRHAR 不用 MTI、门 23:31、overlap 200
- jet 伪彩 vs 灰度

结论：物理过程对得上两套公开代码；archive 图不是 AIRHAR 输出的拷贝。对照图：`figures/preprocess_validation_*.png`。

## 5. MAT 和 PNG 分别保存了什么？

跨 11 个动作抽查（`mat_schema_diverse.csv`）：

| 文件 | 内容 |
|---|---|
| `*_matrix.mat` 键 `sx1` | `complex128`，shape `(4096, 538)`。模值约 1e-2～1e7，**不是** log，也未归一化到 [0,1] |
| `*_FWCC.mat` 键 `C_DCT2` | `(538, 10)` 复系数，是 HWCC/DCT，不是谱图 |
| `*_Raw_0.png` | 带坐标的可视化，约 976×770 |

MAT 是未上色的官方 STFT 复数矩阵，比 PNG 更适合当数值输入。本轮 baseline **没有**用 MAT，而用 AIRHAR 224 z-score，因为那是已发表 HAR 管线的训练输入，且一张约 200 KB。

## 6. CI4R 有多少 subject、action 和 usable samples？

- archive 中 77 GHz 动作目录 `*_Raw_0.bin`：**653** 条（`ci4r_77ghz_manifest.csv`）
- 其中 **1** 条文件大小为 0：`04190014_1574699363_Raw_0.bin`（Scissor / tail4=0014）。无法解析 IQ，已写入 `excluded_samples.csv`，不进 cache / split / 训练
- **usable IQ：652**
- action：**11**（Towards / Away / Pick / Bend / Sit / Kneel / Crawl / Toes / Limp / SStep / Scissor）
- `subject` / `session` / `repetition`：路径和文件名都**不能可靠解析**，manifest 填 `UNKNOWN`
- 文件名 `{8位}_{unix}_Raw_0.bin` 的后 4 位只作为分组记号（`name_token_tail4`），**不称为 subject ID**：`0000`(10)、`0010`(116)、`0014`(91，已去掉空文件)、`0029`(221)、`0030`(214)。文献写 6 人，这里只有 5 个记号。

## 7. 是否存在 class imbalance？

usable 后类别计数 56–62，相对均衡（Towards 60 … Pick 56 … Kneel 62 … Scissor 60）。见 `class_distribution.csv`、`figures/fig2_class_distribution.png`。

## 8. 是否存在 subject-action coverage bias？

有，而且会伤到按人划分：

- `0000`：**只有 Towards**（10 条）
- `0014`：10 类有样本，**缺 Towards**
- `0010` / `0029` / `0030`：11 类都有

`session` 无法统计。见 `subject_action_counts.csv`、`figures/fig3_subject_action_matrix.png`。

## 9. 最合理的 sample unit 是什么？

**一条 `*_Raw_0.bin` = 一次动作记录 = 一个 sample。**  
标签来自动作目录。若以后把一条切成多段，所有段必须跟这条记录走同一个 split，禁止按段再随机划分。

## 10. 最合理的 train/test protocol 是什么？

主结果用 **记号不相交**（`prefix_tail4_disjoint`，文件仍叫 `ci4r_subject_independent.csv`，protocol 字段写的是前者）：

- train：`0029`（221）
- val：`0030`（214）
- test：`0010`（116，11 类都有）
- heldout：`0014` + `0000`（101；`0014` 缺 Towards，不进本轮评测）

这不是官方 subject ID，只是目前能做到的、避免同一记号进 train/test 的划分。

`ci4r_random.csv`：按动作分层 70/15/15（457/98/97），seed=0。同一记号可同时出现在 train 和 test。**只作 sanity check，不作跨人泛化。**

LOSO 四个文件已生成（`0010/0014/0029/0030`），本阶段未训练。

AIRHAR 原脚本是按类文件夹最后 12 个文件当 test，也不是按人。

## 11. random split baseline 表现如何？

输入：AIRHAR 224×224 z-score，1 通道。seed=0。early stopping（patience=6，监控 val acc），test 只评一次。n_train/val/test = 457/98/97。

| 模型 | Accuracy | Macro-F1 | Weighted-F1 | best val acc |
|---|---|---|---|---|
| Simple CNN | 0.474 | 0.458 | 0.456 | 0.551 |
| ResNet-18（conv1 改为 1 通道） | **0.742** | **0.735** | 0.733 | 0.857 |
| CNN+GRU | 0.351 | 0.273 | 0.277 | 0.408 |

这是 within-distribution / random-record。ResNet-18 在 Away / Crawl 上较高，Scissor F1 只有 0.17。完整逐类数字：`baselines/random_record/*/metrics.json`。混淆矩阵：`figures/fig5_cm_random_record.png`（ResNet-18）。

## 12. subject-independent baseline 表现如何？

协议名：`prefix_tail4_disjoint`（train `0029` / val `0030` / test `0010`）。n=221/214/116。同一文件名前缀不跨集合。**不是官方 subject ID。**

| 模型 | Accuracy | Macro-F1 | Weighted-F1 | best val acc |
|---|---|---|---|---|
| Simple CNN | 0.198 | 0.133 | 0.144 | 0.238 |
| ResNet-18 | **0.448** | **0.409** | 0.405 | 0.621 |
| CNN+GRU | 0.233 | 0.163 | 0.158 | 0.318 |

ResNet-18 在 Away（F1=1.00）、SStep（0.87）、Crawl（0.77）上仍高；Pick / Sit / Limp 的 F1 为 0。混淆矩阵：`figures/fig5_cm_prefix_tail4_disjoint.png`。

## 13. 两种 split 的差距多大？

以本轮最强模型 ResNet-18 计：

| | random-record | prefix-tail4-disjoint | 差值（random − disjoint） |
|---|---|---|---|
| Accuracy | 0.742 | 0.448 | **0.294** |
| Macro-F1 | 0.735 | 0.409 | **0.326** |

Simple CNN：Accuracy 0.474 → 0.198（差 0.276）；Macro-F1 0.458 → 0.133（差 0.326）。

11 类均匀瞎猜约 0.091。disjoint 上 Simple CNN 只略高于该线。本表只报数字，不把 random-record 写成跨人结果。对照图：`figures/fig6_split_comparison.png`。

## 14. 哪一种 radar representation 最适合作为后续统一 benchmark 的输入？

本阶段建议：**Representation B = AIRHAR 224×224 z-score micro-Doppler**。

- 与已发表 CI4R HAR 代码一致
- 一张约 200 KB；653 张未压缩约 0.122 GB
- 本轮三种模型都吃这一种

三种表征的定位：

- **A**：`(256, 32000)` 复数 IQ。保留最多原始信息，但不宜整条直接训 2D CNN。若用，按 chirp 切窗，窗继承记录级 split。
- **B**：上段。本轮采用。
- **C**：archive `sx1` `(4096, 538)` 复数，分辨率高于 224，但是 MATLAB 链（含 MTI）。PNG 不当 ground-truth 特征。

## 15. 下一阶段是否值得继续 CI4R 10 GHz，还是优先接入 mRI / RadHAR？

**先接 mRI / RadHAR。** 工位 01 上这两套已经是点云/轨迹，和 77 GHz 谱图能形成第二条模态。CI4R 10 GHz 目录结构不同，本阶段刻意未混；MM-Fi 当前 zip 只有 RGB。FAU 下载未完成。10 GHz 留到 77 GHz 基线数字稳定之后。

## 磁盘与读取方式

| 方案 | 估算 | 结论 |
|---|---|---|
| A. zip 内按需读 raw | 每次打开约 125 MB/条 | 可复现，训练时太慢 |
| B. 全量解开 raw | 约 79.7 GB | 不必要，01 上 `/home` 当时剩余约 3057 GB，但仍不解开 |
| C. 预处理后数值 cache | 224×224 float32 × 652 ≈ **0.122 GB** 未压缩；写出的 `npz` 约 **27 MB** | **采用** |

推荐：zip 只读 + `cache/ci4r77_airhar_spec_zscore.npz`。禁止先转成成千上万张 PNG。空文件不进 cache。

## 代码

- `scripts/ci4r/raw_reader.py` — `load_ci4r_77ghz_raw`
- `scripts/ci4r/preprocess_77ghz.py` — AIRHAR `RGB_mD`
- `scripts/ci4r/representations.py` — A/B/C 说明
- `scripts/ci4r/phase1a_build.py` — 清单、QC、划分、cache
- `scripts/ci4r/phase1a_train.py` — 三个 baseline，seed=0，early stopping，test 只评一次

ResNet-18：`conv1` 改成 1 通道，**没有**把单通道复制成伪 RGB。
