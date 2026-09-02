# Unified MTI definition (U2)

只复现 24 GHz 官方 Butterworth，再按物理截止频率移植到 77。
不跑 mean-subtraction / SVD / 其它高通。

- type: Butterworth highpass, causal `lfilter`（对应 MATLAB `filter`，不是 `filtfilt`）
- order: **4**
- 24 official: `butter(4, 0.01, 'high')` at PRF=1000
- physical cutoff: Wn * (PRF/2) = 0.01 * 500 = **5.0000 Hz**
- 24 normalized cutoff: **0.010000** (PRF=1000, Nyquist=500)
- 77 normalized cutoff: **0.003125** (PRF=3200, Nyquist=1600) so cutoff stays 5.0000 Hz
- apply on official-gate slow-time (complex). LTI: 77 的 bin 求和后再滤，等于先滤再求和。
- 24 U2 从 **pre-MTI** bin 19 用同一 5 Hz 滤波器重算，避免和官方链叠滤两次。
