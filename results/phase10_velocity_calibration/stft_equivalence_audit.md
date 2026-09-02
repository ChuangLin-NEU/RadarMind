# STFT equivalence audit (24 vs 77, Phase 8B numerical chain)

只审计。不重新生成 R3。

## Phase 8B chains used for native STFT

| | 24 GHz | 77 GHz |
|---|---|---|
| source script | `datToImage_Anchortech.m` numerical | AIRHAR `RGB_mD` numerical |
| helper | `myspecgramnew` reimplementation, Hamming | `np.hanning`, AIRHAR loop |
| window (samples) | 128 | 256 |
| window (seconds) | **0.128** | **0.080** |
| overlap (samples) | 100 | 200 |
| hop (samples) | 28 | 56 |
| hop (seconds) | **0.028** | **0.0175** |
| nfft | 4096 | 4096 |
| PRF | 1000 Hz | 3200 Hz |
| Doppler bin (Hz) | 0.244 | 0.781 |
| Doppler bin (m/s) | see operator table | see operator table |

## Official 77 GHz MATLAB (not the Phase 8B 77 chain)

`datToImage_77.m:73`: window=256, nfft=4096, noverlap=197, hop=59.
与 AIRHAR hop=56 不同；且 MATLAB 有卷积 MTI，AIRHAR/Phase8B 77 **无 MTI**。

## Can native time-domain be re-unified?

**YES, in principle.** Both radars still have slow-time range-gate series before STFT.
A later R3 could fix:

1. window duration in seconds (same physical time)
2. overlap fraction
3. frequency / velocity bin width
4. window family (Hamming vs Hann)

本阶段不自动重算。

## Verdict

**NOT EQUIVALENT.** Velocity remapping does not equalize window seconds or hop seconds.
If H1 is weak, STFT mismatch is a SUPPORTED remaining operator difference.
