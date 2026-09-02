# 10 GHz 链（XeThru）

## raw → I/Q

`XeThru_mDopp_bulk.m`：

1. 循环 `fread`：`uint32 ContentId`，`uint32 Info`，`uint32 Length`，**182 float**
2. `iq_mat(:,n) = Data(1:end/2) + 1i*Data(end/2+1:end)` → 每帧 **91** 个复距离元
3. 182 不是 Doppler FFT，不是 FMCW chirp 采样。它是 XeThru 给出的 **range profile 基带 I/Q**

`myspecgramnew` 不在仓库和 zip 里。Phase 8B 用 Hamming + `floor((N-window)/shift)` 的兼容实现，标记 REIMPLEMENTED_MISSING_HELPER。对官方 PNG 做了形态对照，未逐 bin 宣称 OFFICIAL。

## preprocessing → STFT

1. MTI：4 阶 Butterworth 高通 `Wn=0.01`，沿慢时间
2. 取距离元 **27**
3. `myspecgramnew(..., window=150, nfft=4096, noverlap=128)`
4. `fftshift`；`.mat` 裁 `sx1(1408:3688,:)`
5. Isodata + 分段能量门限
6. `20*log10` 相对峰值，`caxis([-45 0])`
7. 显示 `axis([0 T -PRF/2 PRF/2])`，PRF=512 → **±256 Hz**

## plotted PNG

`imwrite` **被注释**。本机官方图 628×1178，像整图导出。不能从 PNG 还原 182 路 I/Q。

## 本机 / 归档

- 官方 PNG：327/327 在 `data/CI4R_MULTI3/.../Xethru/`
- raw `.dat`：327/327 在 `raw_data.zip` 的 `Cross-frequency/10ghz/.../{prefix}_{timestamp}.dat`（Phase 0 清单）
- zip 内 **没有** 10 GHz `.mat`
- 本机 **没有** `raw_data.zip`
