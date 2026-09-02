# 缺的不是“原始数据”四个字，是这些文件

本机已有：327 条官方 PNG；Phase 0 zip 中央目录清单；77 GHz Phase 1 AIRHAR 224 cache。  
GCS 上 `raw_data.zip` HEAD 为 **92968614387** 字节（与 Phase 0 86.58 GB 一致）。  
01 号机路径（Phase 1 实际用过）：`/home/radar_har/ci4r/raw_data.zip`。

不要全量解压。按 `raw_mapping.csv` **按需抽出** 327×3 个成员。

## 必须拿到的文件

### 10 GHz

- 成员：`Cross-frequency/10ghz/<动作夹>/<8位前缀>_<unix>.dat`
- 格式：二进制 XeThru 帧：`uint32 ContentId, Info, Length` + **182 float/帧**
- 327 条清单已对齐 timestamp
- 为什么：没有它不能重放 range-bin 27 的 STFT，不能建 `S(t, f_D)`
- zip 里没有现成 10 GHz `.mat`

### 24 GHz（最高优先）

- 成员：`Cross-frequency/24ghz/<动作夹>/<8位前缀>_<unix>.dat`
- 格式：文本，头 4 个数 **`fc, Tsweep, NTS, Bw`**，其后一列 `%f`（脚本当 I/Q 后 `reshape [NTS, nc]`）
- 未压缩大小常见 **15360054** 字节
- 为什么：要读真实 `fc`/`Tsweep` 才能谈 Doppler Hz 和 `v=c f_D/(2 fc)`；官方 128×128 PNG 和 ylim 5.8 GHz **不能**代替文件头
- zip 里没有现成 24 GHz `.mat`

### 77 GHz

- 成员：`Cross-frequency/77ghz/<动作夹>/<8位>_<unix>_Raw_0.bin`（约 131072000 字节）
- 或同 timestamp 的 `activity_spect_matrices/.../*_matrix.mat`（`sx1` complex）
- 为什么：R2 要用数值矩阵，不要再用官方 PNG。Phase 1 的 224 z-score cache **没有物理轴**，不能当 R2
- 本机 327 条 timestamp 已在 Phase 1 manifest / AIRHAR cache 对上，**映射已在**

## 还需要的实现（不是数据文件）

- `myspecgramnew.m`：10/24/77 官方 MATLAB 都调用，**本仓库没有**。没有它，10/24 的官方 STFT 不能逐 bin 复现。AIRHAR 77 GHz 用 numpy STFT，不依赖该 mex/m 文件

## 不要做的

- 不要从 PNG 反推 I/Q
- 不要为凑三频伪造 24 GHz 头
- 不要在这台 Windows 上从外网拉完整 86 GB（01 上已经有包）
