# Phase 7 preprocessing audit

只记录仓库里能核对的事实。缺文件、缺脚本、缺字段一律写 UNKNOWN。不把像素坐标写成 m/s。

本机没有 `raw_data.zip`。未全量解压、未按需抽出 327 条 raw。R1 不能走 `RAW_RECOMPUTED`。

公平子集与划分（只读，未改）：

- 327 个 timestamp，三频动作一致
- 7 类：Away, Bend, Crawl, Kneel, Pick, Sit, Towards
- acquisition split seed=42：229 train / 49 val / 49 test（`data_splits/ci4r_shared_motion_acquisition_split.csv`）
- Phase 2 缓存按文件计：10 GHz 346、24 GHz 327、77 GHz 334（同 timestamp 多文件时 Phase 2 全收；Phase 3 `views_for` 每个 timestamp 每频只取第一条）

---

## 10 GHz（XeThru UWB）

| 字段 | 记录 | 出处 |
|---|---|---|
| raw format | 二进制 `.dat`：每帧 `uint32 ContentId`, `uint32 Info`, `uint32 Length`, 随后 **182 float** | `third_party/CI4R-MULTI3/XeThru_mDopp_bulk.m` |
| 182 float 含义 | 前半 I、后半 Q，合成 91 点复向量 | 同上 `Data(1:end/2)+1i*Data(end/2+1:end)` |
| sample rate / ADC fs | 公式里出现 `23.328e9`（`bin_length = 8 * 1.5e8 / 23.328e9`） | 同上；是否等于采集 ADC 采样率：脚本未再写，按公式记 |
| frame rate / PRF | **512 Hz** | 同上 `PRF=512` |
| carrier frequency | README 写 7–10 GHz UWB。脚本无单一 `fc` | README + 脚本。标称 10 GHz 是数据集标签，不是测得载频 |
| bandwidth | UNKNOWN（UWB 脉冲；脚本未写 Bw） | — |
| chirp/frame organization | 不是 FMCW chirp。按帧堆叠 91-bin 距离向 | 脚本 |
| range axis | `frame_start=0.3`, `frame_stop=20`；`bin_length` 如上 | 脚本；单位按 XeThru 惯例是米，脚本未写 unit 字符串 |
| FFT | 无独立 Range FFT。距离向已是 XeThru 输出 | — |
| range gate | 距离元 **27**（注释：18 front, 27 corner） | `myspecgramnew(Data_RTI_complex_MTIFilt(27,:), ...)` |
| MTI | 4 阶 Butterworth 高通，`Wn=0.01`，对每个距离元沿慢时间 | 脚本 |
| STFT | `window=150`, `nfft=4096`, `noverlap=128` | 脚本 |
| STFT 实现 | `myspecgramnew` | **不在本仓库**。窗函数、是否 Hilbert、单边/双边：UNKNOWN |
| Doppler processing | `fftshift`；`.mat` 裁 `sx1(1408:3688,:)`（2281/4096 bin） | 脚本 |
| dynamic range | 显示：`20*log10` 相对峰值，`caxis([-45 0])` | 脚本 |
| normalization | Isodata 阈值；低于阈值置 0；再按 100 段能量，低于 `mean/200` 的段置 0；`rescale` 到 0–255 后再 log | 脚本 |
| cropping | 数值矩阵裁 1408:3688；显示 `axis([0 max(Time) -PRF/2 PRF/2])` → 意图 **±256 Hz** | 脚本 |
| interpolation | UNKNOWN | — |
| plotting | `imagesc` + `jet(256)` + `axis xy`；`imwrite` **被注释** | 脚本 |
| output resolution | 本机官方 PNG：**628×1178**（高×宽），1 像素灰边 | 实测 |
| MAT / 数值谱 | 脚本 `save ... sx_scaled`。本机 PNG 树与仓库内 **没有** 对应 `.mat` | 未找到 |
| published PNG 如何从 figure 落到 628×1178 | UNKNOWN（`imwrite` 未启用；像整图导出，不是 `getframe`） | 脚本 vs 磁盘 |

**representation_source（本机 R1）:** `PLOTTED_PNG`

---

## 24 GHz（Ancortek FMCW）

| 字段 | 记录 | 出处 |
|---|---|---|
| raw format | 文本 `.dat`：`fc, Tsweep, NTS, Bw`，其后一列 `%f` | `datToImage_Anchortech.m` |
| I/Q | 注释写 “raw data in I+j*Q”。实现是 `reshape(Data,[NTS nc])` 后直接 `fft`，未见显式 I/Q 拆列 | 脚本。磁盘上 I/Q 是否交错：本机无 raw，UNKNOWN |
| sample rate | `fs = NTS/Tsweep`（Tsweep 先从 ms 除 1000） | 脚本；**逐文件**，不在 PNG 上 |
| frame / sweep | `Tsweep` 来自文件头；`PRF=1/Tsweep` | 脚本 |
| carrier frequency | 文件头 `fc` | 逐文件；PNG 无此字段 → 本机 UNKNOWN |
| bandwidth | 文件头 `Bw` | 同上，本机 UNKNOWN |
| NTS | 文件头 | 本机 UNKNOWN |
| chirp organization | `reshape` 成 `[NTS, n_chirps]` | 脚本 |
| FFT | `fftshift(fft(Data_time),1)`，保留正距离半边 `NTS/2` | 脚本 |
| range gate（作图用） | 距离元 **19** | `myspecgramnew(Data_range_MTI(19,:), ...)` |
| 另一套 STFT（MD.*） | window 200, overlap 0.95, pad 4 | **算了但未用于 `imagesc`** |
| MTI | 4 阶 Butterworth 高通 `Wn=0.01` | 脚本 |
| STFT（作图） | `window=128`, `nfft=4096`, `noverlap=100` | 脚本 |
| STFT 实现 | `myspecgramnew` | **不在本仓库** → UNKNOWN |
| Doppler processing | `flipud(fftshift)`；`.mat` 裁 `sx1(1408:3688,:)` | 脚本 |
| dynamic range | `20*log10` 相对峰值；`CLim = clim(2)+[-45,0]` | 脚本 |
| normalization | Isodata；再 `<= median(nonzero)/2` 置 0 | 脚本 |
| cropping / ylim | `ylim([-6/(3e8/2/5.8e9) 6/(3e8/2/5.8e9)])` ≈ **±232 Hz**，换算用的是 **5.8 GHz**，不是 24 GHz | 脚本原文 |
| interpolation | UNKNOWN | — |
| plotting | `imagesc` + jet；保存 `.mat` / `.fig`；**无 `imwrite` PNG** | 脚本 |
| output resolution | 本机官方 PNG：**128×128**。上下约 8 行亮条（色条/边框） | 实测，40 张稳定 |
| 128×128 如何得到 | UNKNOWN（不像原生 figure；与仓库里把谱图缩到 128 的习惯一致，但发布脚本未写 `imresize`） | — |
| MAT | 脚本写 `sx_scaled`。本机无这些 `.mat` | 未找到 |

**representation_source（本机 R1）:** `PLOTTED_PNG`

---

## 77 GHz（TI IWR1443 / DCA1000）

两条链不要混。

### A. 官方 PNG（`datToImage_77.m`）— Phase 2/3/R0 用的图

| 字段 | 记录 | 出处 |
|---|---|---|
| raw format | `*_Raw_0.bin`，`int16`，8 行：4 路 I + 4 路 Q | 脚本 `reshape(..., numLanes*2, [])` |
| sample rate | `sampleRate = 10e6` | 脚本 |
| frame | `SweepTime=40e-3`, `NPpF=128` → PRF **3200 Hz** | `prf=1/(SweepTime/NPpF)` |
| carrier | `fstart=77.1799e9`, `fstop=77.9474e9`, `fc=(fstart+fstop)/2` | 脚本 |
| bandwidth | `Bw = fstop-fstart` = 0.7675e9 | 由上式 |
| chirp/frame | `NTS=256` ADC / pulse；脚本里 `NoF=500` 是注释常量，实际脉冲数按文件长度 | 脚本 |
| FFT | 每 RX：reshape `(256, n_pulses)` → `fftshift(fft)`，正距离半边 | 脚本 |
| MTI | 卷积核 `[1 -2 3 -2 1]`，用 RX1 的 range profile | 脚本。Butterworth 段被注释，未跑 |
| range gate | 距离行 **8**（半边 range profile 上） | `myspecgramnew(rngpro(8,:), ...)` |
| STFT | `window=256`, `nfft=4096`, `noverlap=197` | 脚本 |
| STFT 实现 | `myspecgramnew` | **不在本仓库** → UNKNOWN |
| Doppler | `fftshift`；显示 `flipud` 后再 `axis xy` | 脚本 |
| dynamic range | `20*log10` 相对峰值，`caxis([-45 0])` | 脚本 |
| normalization | 除峰值 | 脚本 |
| cropping | `axis([0 20 -500 500])`，意图 **±500 Hz**（PRF/2=1600 Hz，显示裁过） | 脚本 |
| interpolation | UNKNOWN（`getframe` 光栅） | — |
| plotting | 全屏 figure、标题、坐标轴；`getframe(gca)` + `imwrite` | 脚本 |
| output resolution | 本机官方 PNG：**804×1475**。四周仍有暗边 | 实测 |
| MAT | 未 `save` STFT 矩阵；只存 `.fig` + PNG | 脚本 |

### B. AIRHAR 数值链（Phase 1A，**不是** 官方 PNG）

| 字段 | 记录 | 出处 |
|---|---|---|
| raw | 同 `.bin`，但只取 lane I=3、Q=7 | `scripts/ci4r/raw_reader.py` / AIRHAR `bin2np` |
| Range FFT | `fftshift` along ADC，正距离半边 | `scripts/ci4r/preprocess_77ghz.py` |
| range gate | 求和 bins **23:31** | 同上 |
| STFT | Hann 256，hop 56，overlap 200，nfft 4096 | 同上 |
| MTI | AIRHAR 源码里注释掉，**未跑** | 同上 docstring |
| dynamic range | `20 log10`，clip **[-45, 0]** | 同上 |
| output | cubic resize **224×224** | 同上 |
| 与 327 三频子集 | Phase 1 cache 是 11 类、另一套划分，不能直接当本阶段 R1 | — |

本机无 zip，不能对 327 条重算 AIRHAR 谱并与 10/24 对齐。若只对 77 用数值、10/24 用 PNG，会把“来源不同”和“公平化”搅在一起。Phase 7 R1 **三频都走官方 PNG 裁边**。

**representation_source（本机 R1）:** `PLOTTED_PNG`（三频相同来源等级）

---

## 官方训练图（R0）怎么进网络

`scripts/cross_radar/pack_cache.py` `load_gray224`：

1. 整张 PNG 转灰度（坐标轴、色条、标题都留着）
2. BICUBIC 缩到 224×224
3. 逐图 z-score

没有重做 STFT，没有物理轴。

---

## 数值表征是否存在

| 对象 | 本机 |
|---|---|
| 10 GHz `.mat` `sx_scaled` | 未找到 |
| 24 GHz `.mat` `sx_scaled` | 未找到 |
| 77 GHz 官方 STFT 矩阵 | 未保存 |
| 77 GHz AIRHAR `ci4r77_airhar_spec_zscore.npz` | 有，但是 Phase 1A 11 类 / 另一划分 |
| complex STFT / range-Doppler 阵列（三频对齐） | 未找到 |

因此 Fair Representation v1 **不能** 标 `RAW_RECOMPUTED` 或 `NUMERICAL_PRECOMPUTED`。

---

## 物理 Doppler / 速度轴

要做 `v = c f_D / (2 f_c)` 需要：可靠的 `f_D(row)`、可靠的 `f_c`、以及三频同一显示约定。

| 条件 | 10 | 24 | 77 PNG |
|---|---|---|---|
| `f_c` | UWB，非单频 | 在 raw 头，本机无 raw | 脚本有 |
| `f_D`↔行号 | PNG 是 figure；`myspecgramnew` 缺失 | 同上；ylim 还用了 5.8 GHz | `getframe` 含轴；暗边不固定 |
| 本机 raw | 无 | 无 | 无 |

**不做物理速度对齐。** R1 纵轴是 **normalized Doppler coordinate**（像素行，缩到 224），不是 m/s。

---

## R1 实际做了什么（Fair PNG crop）

固定裁边，一次锁定，不对单张图按能量自适应裁（自适应会吃掉信号，不是去图框）。

| 频段 | top | bottom | left | right | 依据 |
|---|---:|---:|---:|---:|---|
| 10 GHz | 1 | 1 | 1 | 1 | 实测 1 像素灰边；与 `experiments/png_physics_probe/audit/png_axis_audit.md` 一致 |
| 24 GHz | 8 | 8 | 0 | 0 | 40 张样本上下约 8 行亮条稳定（色条/边） |
| 77 GHz | 55 | 58 | 0 | 0 | 同上 audit：去标题/刻度暗边。**不用** 按行方差找谱峰（40 张 top 111–357，随信号变） |

随后三频同一后处理：

- 灰度
- 不翻转（保持官方 `axis xy` 方向）
- BICUBIC → 224×224
- 逐图 z-score
- 1 通道

保存 `native_shape`、裁后 `native_time_resolution` / `native_doppler_resolution`（像素/轴，**不是 Hz 或 m/s**）、标称 `carrier_frequency`、`representation_source=PLOTTED_PNG`。

R1 消除的是：整图缩放到 224 时，色条/灰边/标题占像素的差异。  
R1 **不**声称：距离门、STFT、去噪、显示 Hz 范围已经对齐。
