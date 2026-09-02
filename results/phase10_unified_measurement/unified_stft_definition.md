# Unified STFT definition (U1)

参数按两路共同支持的物理时间预注册，不按分类结果选。

- window family: **hamming**（24 官方 MATLAB / Phase 8B 已是 Hamming；77 官方 MATLAB 也是 `myspecgramnew`。AIRHAR 的 Hann 不采用，避免给 24 引入新窗。）
- target window duration: **0.100 s**
- 24: 100 samples, actual duration **0.100000 s**
- 77: 320 samples, actual duration **0.100000 s**
- overlap fraction: **0.75**（不统一 sample count）
- 24 hop: 25 samples = **0.025000 s**
- 77 hop: 80 samples = **0.025000 s**
- nfft: **4096**
- 比较轴: velocity，不是 FFT bin index
- common velocity: [-3.028207, 3.026728] m/s（沿用 Phase 10C，不改）
- 各雷达仍用自己的官方 range selection 和官方 MTI 状态
