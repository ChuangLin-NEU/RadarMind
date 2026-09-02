# Phase 8 — Physical motion representation reconstruction

未改 Phase 1–7 / frozen / paper。未重做 CORAL / M3。未进入 Phase 9。未改 ±256 Hz 网格。

## Phase 8C 结论

R2-D = S(τ, f_D) 能做 HAR（within 仍可用），但 **没有改善跨雷达泛化**。
这是 Doppler 频率网格，不是径向速度对齐。

共同带宽 ±256 Hz 的能量保留：10 = 1.000，24 = 0.994±0.009，77 = 0.995±0.006。
10 GHz within 下降与裁频带无关（该频段本来就是 ±256 Hz；错样本保留能量仍为 1）。
24/77 错对与保留能量也对不上。不把 within 变化归因于 clipping。

**CASE D**（部分方向升、部分降），同时带 B 的特征：跨雷达仍差，Radar-ID 仍高。

下一瓶颈：**BOTH**。输入层仍能读出雷达身份（0.993）；纯 CE encoder 也没有做成不变表示（跨雷达/LORO 仍差）。本轮不加新 loss。

### 8C 数字

Within（Accuracy）：10 R0 0.843 / R1 0.961 / R2 0.776；24 0.898 / 0.857 / 1.000；77 0.922 / 0.824 / 0.878。

Cross：10→24 0.143 / 0.510 / 0.143；10→77 0.373 / 0.294 / 0.122；24→10 0.353 / 0.333 / 0.347；24→77 0.039 / 0.137 / 0.143；77→10 0.275 / 0.176 / 0.245；77→24 0.163 / 0.020 / 0.184。

LORO CE：10+24→77 R1 0.224 / R2 0.143；10+77→24 0.347 / 0.102；24+77→10 0.633 / 0.306。

Probe：Representation Radar-ID R0/R1/R2 = 1.000 / 1.000 / 0.993；Embedding Radar-ID 1.000 / 1.000 / 0.993；Activity Probe 0.918 / 0.925 / 0.946。

Retrieval R2-D 六向平均：exact Top-1 0.136，same-action Top-1 0.915。

R2-NATIVE 未建（保留能量已说明 clipping 不是主因）。未改 frozen paper。

## Phase 8B 结论

在 01（`303vasp-share-01`）对流式读取 `raw_data.zip` 的 327×3 个成员，重建了 **R2-D = S(τ, f_D)**。
没有把 86.6 GB 拉回本机，没有全量解压。

R2-D 去掉了官方 PNG 的图框/色条，但 **Radar-ID 仍为 0.993**。
same-pair 与 diff-pair 几乎一样。这不是 encoder 的问题。

## RAW ACCESS

- archive: **FOUND**
- path: `/home/radar_har/ci4r/raw_data.zip`
- size: **92968614387** bytes（86.583769 GB），与 GCS HEAD 一致
- zip readable: **YES**（`zipfile` 流式，4315 个成员）
- SHA256：未对整包计算（86 GB，禁止为校验再读一遍）
- 本机：仍然没有这份 zip

327 mapping（沿用 `raw_mapping.csv`，未重匹配 timestamp）：

| 频段 | raw 成员对上 | 本次数值重建 |
|---|---:|---:|
| 10 GHz | 327/327 | 327/327 |
| 24 GHz | 327/327 | 327/327 |
| 77 GHz | 327/327 | 327/327 |

split 仍是 229 / 49 / 49。

## SMOKE（3 个不同 action × 3 雷达）

Towards `1573836822`，Away `1573837114`，Pick `1573837435`。

| 频段 | Gate | numerical shape | Doppler Hz | 物理时间 | 与官方 PNG 形态相关（time / Doppler） |
|---|---|---|---|---|---|
| 10 GHz | **PASS** | 4096 × 231 | YES，±256 Hz | YES，PRF=512 | 0.77 / 0.94（Towards） |
| 24 GHz | **PASS** | 4096 × 352 | YES，±500 Hz | YES，10.0 s | 0.51 / 0.98（Towards） |
| 77 GHz | **PASS** | 4096 × 566 | YES，±1600 Hz | YES，10.0 s | 0.66 / 0.71（Towards）；Pick 时间轮廓弱（AIRHAR 无 MTI vs 官方 MATLAB PNG） |

三个都 PASS，进入全量。

## 24 GHz 真实文件头（不是 Phase 7 推断）

从 `01050010_1573836822.dat` 原文：

```
24750000000.000000    fc = 24.75 GHz
1.000000              Tsweep = 1.0 ms
128                   NTS = 128
1500000000.000000     Bw = 1.5 GHz
1942+2152i            其后每行一个 MATLAB 复数
```

10000 sweep × 1 ms = **10.0 s**。`textscan('%f')` 能读 `I+Qi`；Python 按同样规则拆成复数再 `reshape(NTS, nc)`。

### 5.8 是什么

`datToImage_Anchortech.m` 第 99–104 行：

```
imagesc(MD.TimeAxis, MD.DopplerAxis, 20*log10(...))
ylim([-6/(3e8/2/5.8e9) 6/(3e8/2/5.8e9)])
```

- 含义：把遗留的 ±6 m/s 用 **5.8 GHz** 换成频率轴，去裁 **绘图 ylim**（约 ±232 Hz）
- 不影响本次 STFT 数值
- 不是载波；载波以文件头 **24.75 GHz** 为准
- 不是 raw 头里的字段

## MISSING HELPER

`myspecgramnew`：仓库、zip 文件名、三份官方 `.m` 均 **没有源码**。

按调用 `myspecgramnew(x, window, nfft, shift)` 建立兼容实现，标记 **REIMPLEMENTED_MISSING_HELPER**（不是 OFFICIAL）：

- Hamming 窗（MATLAB `specgram` 在 window 为长度时的默认）
- `n_frames = floor((N - window) / shift)`
- 与 77 GHz 官方 `sx1` 宽度 ~538（window=256, shift=59, N≈32000）一致

10 GHz 官方 PNG 形态相关足够，未因 helper 停掉。

77 GHz R2 用的是 **AIRHAR** Hann STFT，不经过 `myspecgramnew`。

## 全量重建

processed：10 **327/327**，24 **327/327**，77 **327/327**  
failures：0（`failed_samples.csv` 空）

native 在 01：`/home/radar_har/work/phase8b/results/phase8_physical_representation/cache_native/`  
本机只保留 smoke 图和 **cache_r2**（173 MB tar / 981 npz）。未传 86.6 GB zip。

## R2-D

- available: **YES**
- type: `S(normalized_time, Doppler_Hz)`
- shape: **1×224×224**
- paired samples: **327**
- 共同 Doppler：TRAIN 交集 **[−256, +255.875] Hz**（受 10 GHz PRF/2 限制）
- τ ∈ [0,1]，按 acquisition 时长归一
- 归一化：TRAIN 1%/99% clip，再减均值除标准差；val/test 只用这套数

## R2-V

- available: **PARTIAL / NO（共同网格未建）**
- reason: 10 GHz UWB 没有单一可靠 fc。24 GHz 可用头 `fc=24.75e9`，77 GHz 可用脚本 `fc=(77.1799e9+77.9474e9)/2`。共同 R2 本轮只用 f_D。

## Radar fingerprint（16×16 logreg，与 Phase 7 同一套像素探针，不是 ResNet）

| 表示 | Radar-ID |
|---|---:|
| R0 | 1.000 |
| R1 | 1.000 |
| R2-D | **0.993** |
| R2-V | 未建 |

## Pair similarity（16×16 cosine）

| 表示 | same acquisition / 不同雷达 | 不同 acquisition / 不同雷达 |
|---|---:|---:|
| R1 | 0.774 | 0.738 |
| R2-D | 0.917 | 0.913 |
| R2-V | — | — |

R2-D 把三频都插到同一 (τ, f_D) 网格后，整体更像，但 **same 与 diff 几乎无差**。配对没有变清楚。

## 限制（必须写明）

相同人体径向速度，在 10 / 24.75 / 77.56 GHz 上对应 **不同** Doppler Hz。
R2-D 去掉的是作图和像素网格混淆，**还不是 sensor-invariant motion space**。
10 GHz 进共同网格时，24/77 的 |f_D|>256 Hz 被裁掉。

## Gate 2

```json
{
  "A_numerical_source": true,
  "B_physical_axes_evidence": true,
  "C_common_grid": true,
  "D_sanity_plots": true,
  "enter_min_train": false,
  "reason": "R2-D built. No encoder training in Phase 8B."
}
```

## SCIENTIFIC STATUS

- plotting/preprocessing confound：R0/R1→R2-D 后 Radar-ID 从 1.000 到 0.993，只动了一点。图框不是主因。
- physical sensor difference：**仍强**。共同轴是 f_D 不是 v；三频 Nyquist 不同。
- next step：**TRAIN R2**（单独分支，不换 frozen 表）。本轮停在 QC。不自动训 ResNet，不进入 Phase 9。
