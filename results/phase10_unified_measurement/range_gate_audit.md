# Range gate audit (24 / 77)

问的是：两套官方/Phase 8B pipeline 各自看哪一段距离。不能把 bin 19 改成 bin 19 跨雷达。

## 24 GHz

| 量 | 值 | 证据 |
|---|---|---|
| Bw | 1.5e9 Hz | raw header / `header_24ghz.json` |
| range resolution | c/(2 Bw) ≈ 0.0999 m | FMCW 公式；脚本未写出 `dr` 字符串 |
| 距离轴 | **UNKNOWN（脚本未写）** | `datToImage_Anchortech.m` 无 `rng_axis` |
| 选择 | 半剖面 MATLAB bin **19**（0-based 18） | `myspecgramnew(Data_range_MTI(19,:), ...)` |
| bin 19 的米 | **UNKNOWN** | 没有官方把第 19 行标成米 |

若额外假设 fftshift 后半剖面第 1 行是 DC=0 m（脚本没写），则 18×0.0999 ≈ 1.80 m。这是推导，不是源码字段。**不算可靠 physical axis。**

## 77 GHz

| 量 | 值 | 证据 |
|---|---|---|
| Bw | 0.7675e9 Hz | `datToImage_77.m:8-9,17` |
| range resolution | c/(2 Bw) ≈ 0.1954 m | 同上 |
| 官方 MATLAB 轴 | `([1:256]-1)*c/2/Bw` | `datToImage_77.m:72`，针对 **全 NTS** |
| 官方 MATLAB 门 | 半剖面第 **8** 行 | `myspecgramnew(rngpro(8,:), ...)` |
| Phase 8B / AIRHAR 门 | `rp[23:31]` 复数求和 | `data_prepare_CI4R.py` RGB_mD |
| AIRHAR 23:31 的米 | **UNKNOWN** | AIRHAR 未引用 `rng_axis` |

官方 MATLAB 和第 8 行、AIRHAR 23:31 不是同一距离策略。本阶段 77 必须跟 Phase 8B/10C 一致，用 AIRHAR 求和门。

## 是否在看同一人体距离区

**不能可靠回答。** 24 没有米轴；77 的米轴没有接到 AIRHAR 行号上。
两路官方门的物理重叠：**UNKNOWN**。

## U3

**NOT AVAILABLE**。不编 ±Δr。
