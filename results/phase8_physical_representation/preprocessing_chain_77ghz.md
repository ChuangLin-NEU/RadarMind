# 77 GHz 链

两条链不要混。

## A. 官方 PNG（`datToImage_77.m`）

`*_Raw_0.bin` int16 → 8 行（4 I + 4 Q）→ Range FFT → 卷积 MTI `[1 -2 3 -2 1]` → 距离行 8 → `myspecgramnew` 256/4096/197 → `20*log10` clip −45 → `getframe` PNG（本机 804×1475）。

显示 `axis([0 20 -500 500])`。Phase 1 实测文件约 **10 s / 32000 chirp**，20 s 是脚本常数，与文件不符。

## B. AIRHAR 数值（Phase 1A）

`bin2np`：只取 lane I=3、Q=7 → `(256, n_chirps)` → Range FFT → 门 **23:31 求和** → Hann STFT overlap 200 → clip −45 → **cubic 224×224**。MTI 在 AIRHAR 里注释掉，未跑。

本机 `results/phase1_ci4r/cache/ci4r77_airhar_spec_zscore.npz` 覆盖这 327 个 unix timestamp。这是 **224 已缩放 z-score**，没有保存 Hz / m/s 轴，**不能当 R2**。

## C. zip 内数值 MAT

`Cross-frequency/77ghz/activity_spect_matrices/.../*_Raw_0_matrix.mat`  
Phase 1 读过 key `sx1`，complex `[4096, ~538]`。327 条在清单里都有。本机未抽出。

## 327 paired → raw

`raw_mapping.csv`：327/327 能对到动作目录里的 `*_Raw_0.bin`（约 131072000 字节）。  
不是“文件名没对上”。差的是 **本机没有打开 zip 成员**。

Phase 1 默认 zip：`/home/radar_har/ci4r/raw_data.zip`（01）。
