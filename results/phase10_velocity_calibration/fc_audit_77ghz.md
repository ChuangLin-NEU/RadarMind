# 77 GHz carrier-frequency audit

本阶段只问：77 GHz 的 `fc` 能否从官方处理链可靠读出。
不把数据集名字「77 GHz」写成 `fc = 77e9`。
AIRHAR 不提供独立载频。raw `.bin` 无文本头。

## Verdict

**CONFIRMED_FROM_OFFICIAL_MATLAB**

- `fstart = 77.1799e9`
- `fstop = 77.9474e9`
- `fc = (fstart + fstop) / 2 = 77563650000 Hz = 77.56365 GHz`

## Evidence

| source | location | what it says | usable as fc |
|---|---|---|---|
| Official MATLAB | `third_party/CI4R-MULTI3/datToImage_77.m` lines 8–10 | `fstart=77.1799e9`; `fstop=77.9474e9`; `fc=(fstart+fstop)/2` | **YES** |
| Official MATLAB | same file line 17 | `Bw = fstop - fstart` = 0.7675e9 | bandwidth, not fc |
| Official MATLAB | same file lines 13–20 | `SweepTime=40e-3`, `NPpF=128` → PRF 3200 Hz | timing |
| AIRHAR | `third_party/AIRHAR/datasets/data_prepare_CI4R.py` | `RGB_mD` / `bin2np` 无 `fc`、无 `v=c f_D/(2 fc)` | NO independent fc |
| AIRHAR | `third_party/AIRHAR/datasets/CI4R/spec.json` | 只有网络训练超参 | NO |
| Dataset README | `third_party/CI4R-MULTI3/README.md` line 9 | `TI IWR1443 BOOST (77 GHz FMCW radar with 4 GHz bandwidth)` | label + chip BW, **not** chirp fc |
| Raw `*_Raw_0.bin` | DCA1000 int16 | 无 `fc` 文本头 | NO |
| mmWave Studio `.cfg` | not in this repository | — | UNKNOWN |

## What is not used

- `fc = 77e9`（仅因数据集名叫 77 GHz）
- README 的 4 GHz bandwidth（芯片能力；与脚本 chirp `Bw=0.7675 GHz` 不一致，不以 README 覆盖脚本）
- AIRHAR 数值链：它做 Range FFT + Hann STFT，不换算速度

## 24 GHz (already confirmed, restated)

- `fc = 24750000000 Hz = 24.75 GHz`
- source: raw `.dat` header line 1, `datToImage_Anchortech.m` line 8 `fc = radarData(1)`
- Phase 8B `header_24ghz.json` and native manifest `fc_hz=24750000000`

## Decision

R2-V 24↔77：**PROCEED**。
10 GHz 仍不进入本阶段速度域。
