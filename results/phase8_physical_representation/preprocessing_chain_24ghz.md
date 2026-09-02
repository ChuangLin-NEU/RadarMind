# 24 GHz 链（Ancortek）

## raw → I/Q

`datToImage_Anchortech.m`：

```
fc = radarData(1)
Tsweep = radarData(2)/1000   % 文件里是 ms
NTS = radarData(3)
Bw = radarData(4)
Data = radarData(5:end)
Data_time = reshape(Data, [NTS nc])
```

文件已打开。头四行是文本：`fc=24.75e9`，`Tsweep=1.0`（ms），`NTS=128`，`Bw=1.5e9`。
其后每行一个 MATLAB 复数 `I+Qi`（例如 `1942+2152i`）。`textscan('%f')` 把它们读成复数，再 `reshape([NTS nc])`。
10000 sweep × 1 ms = 10.0 s。5.8 GHz 只出现在 `ylim`，见第 103 行，不进 STFT。

## preprocessing → STFT

1. Range FFT，`fftshift`，留正距离半边
2. MTI：Butterworth 高通 `Wn=0.01`
3. 作图用距离元 **19**，`myspecgramnew` window=128, nfft=4096, noverlap=100
4. 另一套 `MD.*`（window 200）**算了但不用**
5. Isodata + `<= median(nonzero)/2` 置 0
6. `20*log10` 相对峰值，`CLim = clim(2)+[-45,0]`

## 为什么 ylim 用 5.8 GHz？

脚本原文：

```
imagesc(MD.TimeAxis, MD.DopplerAxis, 20*log10(...))
ylim([-6/(3e8/2/5.8e9) 6/(3e8/2/5.8e9)])
```

`MD.DopplerAxis` 是 **频率（Hz）**，不是 m/s。  
`3e8/2/5.8e9` 是用 **5.8 GHz** 把速度换到频率的系数。  
`±6` 按注释里被划掉的 `ylim([-6 6])` 和 `ylabel Velocity`，像是 **±6 m/s** 的遗留，再误用 5.8 GHz 换成约 **±232 Hz** 去裁频率轴。

结论（只根据源码）：

- 这是 **绘图参数**（`ylim`）
- 不是写进文件的速度轴
- 不是 24 GHz 载波
- 不是 raw 头里的 `fc`

## plotted PNG

脚本只 `save` `.mat`/`fig`，**没有 `imwrite`**。本机发布图 **128×128**，上下约 8 行亮条。128 如何得到：UNKNOWN。

## 本机 / 归档

- PNG：327/327
- raw `.dat`：327/327 在 zip `Cross-frequency/24ghz/.../{prefix}_{timestamp}.dat`，未压缩大小常见 **15360054**
- zip 内 **没有** 24 GHz `.mat`
- 本机没有 zip，头里的 `fc/Tsweep/NTS/Bw` **未读到**
