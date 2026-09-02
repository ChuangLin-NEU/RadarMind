# RadarMind Phase 2B — DeepSeek Online + CI4R Cross-Radar MVP

未改 Phase 1 cache/checkpoint。未进 MiliPoint / MM-Fi / FAU。未做超参搜索。seed=42。文件名前缀是 acquisition group，不是 subject。

## 1. CI4R 中实际存在几个可用 radar frequency？

三个：10 GHz（XeThru UWB）、24 GHz（Ancortek FMCW）、77 GHz（TI IWR1443）。官方 README 写明三雷达 **synched / co-located**。本地官方谱图 PNG 各 653 张。

## 2. 10 / 24 / 77 GHz 的真实文件格式分别是什么？

| 频率 | raw | 不要混用 |
|---|---|---|
| 10 GHz | `.dat`：uint32 头 + 182 float/帧。同目录 616 字节 `.bin` **不是** 77 GHz 8-lane IQ | `XeThru_mDopp_bulk.m` |
| 24 GHz | 文本 `.dat`：fc, Tsweep, NTS, Bw + I/Q | `datToImage_Anchortech.m` |
| 77 GHz | `*_Raw_0.bin` int16 8-lane | AIRHAR `bin2np` / `datToImage_77.m` |

本阶段训练用官方 released spectrogram PNG，未全量解压 `raw_data.zip`。

## 3. preprocessing 分别是什么？

- 10 GHz：XeThru IQ → MTI → STFT（PRF=512）→ 去噪 → 20log。
- 24 GHz：range FFT → MTI → 指定距离元 STFT → 去噪。
- 77 GHz：AIRHAR Range FFT → 门 23:31 → STFT Hann 256/overlap 200 → 20log/−45 → 224。Phase 1A 已复现。

Cross-radar 实验把官方 PNG 转灰度、缩到 224×224、逐图 z-score。统一形状 **不等于** 物理分辨率相同。

## 4. 是否可以恢复同步 acquisition pairing？

可以按文件名 10 位 unix timestamp 做 **exact timestamp pair**。这是证据，不是编的同步轴。

同 timestamp 但动作目录不一致（常见：24 GHz `Toes` vs 10/77 `Limp`）共 133 组，标 `timestamp_action_conflict`，**不进公平三频子集**。

## 5. 三雷达共同 usable subset 有多大？

| 统计 | N |
|---|---|
| 官方 PNG 各频 | 653 |
| timestamp 三频都有 | 460 |
| 其中动作一致 | **327** |
| 动作冲突 | 133 |

训练用 327 个 timestamp，按 timestamp 划 train/val/test（seed=42）。

## 6. 哪些 action 可以公平比较？

7 类：Away, Bend, Crawl, Kneel, Pick, Sit, Towards。

Limp / Toes / SStep / Scissor 在三频 timestamp 上对不齐或冲突，本轮不进公平子集。

## 7. within-radar performance 如何？

同一 7 类、同一 timestamp 划分，ResNet18：

| | Accuracy | Macro-F1 | n_test |
|---|---|---|---|
| 10 → 10 | 0.843 | 0.855 | 51 |
| 24 → 24 | 0.898 | 0.897 | 49 |
| 77 → 77 | 0.922 | 0.926 | 51 |

这是 cross-radar 的 reference ceiling。

## 8. zero-shot cross-radar performance 如何？

test 频段样本未进训练。source-only：

| | Acc | Macro-F1 |
|---|---|---|
| 77 → 10 | 0.275 | 0.221 |
| 10 → 77 | 0.373 | 0.288 |
| 77 → 24 | 0.163 | 0.040 |
| 24 → 77 | 0.039 | 0.026 |
| 10 → 24 | 0.143 | 0.036 |
| 24 → 10 | 0.353 | 0.222 |

## 9. cross-radar generalization gap 多大？

以 within 为上界：

- 77 within 0.922 − 77→10 0.275 = **0.647**
- 77 within 0.922 − 77→24 0.163 = **0.758**
- 10 within 0.843 − 10→77 0.373 = **0.470**
- 24 within 0.898 − 24→77 0.039 = **0.859**

## 10. 哪个方向最难？

**24 → 77**（Acc 0.039，低于 7 类随机 0.143）。其次 10→24 / 77→24。10↔77 相对没那么糟，但仍远低于 within。

## 11. baseline embedding 更按 action 聚类还是 radar 聚类？

77 编码器在三频 test embedding 上：Radar-ID probe **1.00**。t-SNE 见 `figures/tsne_within_77_radar.png` 与 `tsne_within_77_action.png`。当前更按 **radar** 分开。

## 12. Radar-ID probe accuracy 多高？

within_77 编码器：Radar-ID = 1.00（n=151，三频 test 拼在一起的诊断拟合，不是另开的 held-out probe 声明）。

Activity probe（同样诊断）：0.980。

## 13. RadarBridge v0.1 是否降低 radar information？

方法：CORAL **unsupervised domain adaptation**（源有标签 + 目标 train 无标签）。不是 domain generalization。

CORAL 后 Radar-ID probe 仍是 **1.00**。没有降低 radar 信息。

## 14. 是否改善 unseen-radar HAR？

没有改善主指标。

| | Acc | F1 |
|---|---|---|
| 77→10 source-only | 0.275 | 0.221 |
| 77→10 CORAL UDA | 0.196 | 0.108 |
| 77→24 source-only | 0.163 | 0.040 |
| 77→24 CORAL UDA | 0.143 | 0.036 |

个别 test timestamp 上 CORAL 能改对（demo_C），总体变差。按原 protocol 记录，不改 test。

## 15. 当前结果是否已经足以支持 Learn Motion, Not Sensor？

足以把问题摆出来：within 已经能分 7 类，zero-shot 跨频掉一大截，embedding 能完美读出雷达频率。还不足以说已经学到 domain-invariant motion。

## 16. 下一阶段是否值得进入真正 Shared Motion Representation？

值得，但不要先上大模型。先处理：官方 PNG 是不同物理 STFT；24 GHz 与 10/77 有 timestamp 动作冲突；CORAL 不够。下一步应在 **已对齐的 10↔77 327 子集** 上做带约束的 shared encoder，而不是再扩数据集。
