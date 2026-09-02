# Unified range definition (U3)

**U3 = NOT AVAILABLE**

bin index 不能跨雷达当同一物理距离。要统一必须两路都有可核对的 `range_axis_m`。

- 24 GHz：`datToImage_Anchortech.m` **没有** `rng_axis`。只有 bin 19。
- 77 GHz 官方 MATLAB：有 `rng_axis=([1:NTS]-1)*c/2/Bw`，但是用在全 256 点，且官方门是半剖面 **第 8 行**，不是 AIRHAR 的 23:31。
- Phase 8B / AIRHAR：对半剖面 `rp[23:31]` 复数求和，**没有** 把这些行标成米。

距离分辨可用 `c/(2 Bw)` 计算，但官方目标距离区间不能在不额外假设 FFT 零距对准的情况下写成可靠米制门。
按规则：不造统一 physical range gate。本阶段停在 U2。
