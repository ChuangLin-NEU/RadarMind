# Physical schema（只写脚本/清单能核对的量）

本机没有打开 `raw_data.zip`。10/24 的 `fc`/`Tsweep`/`NTS`/`Bw` 在文件头里，未读到样本则 **UNKNOWN**。公式来自仓库 MATLAB / AIRHAR，不是从 PNG 反推。

## 10 GHz（XeThru UWB）

| 量 | 值 | 证据 |
|---|---|---|
| 标称载频 | 数据集标签 10 GHz；README：7–10 GHz UWB impulse | `third_party/CI4R-MULTI3/README.md` |
| 单一 `fc` | 脚本未写 | UNKNOWN（UWB 不是单频） |
| bandwidth | 脚本未写 | UNKNOWN |
| frame rate / PRF | **512 Hz** | `XeThru_mDopp_bulk.m` `PRF=512` |
| ADC / 采样 | 公式 `23.328e9` 出现在 `bin_length` | 是否等于 ADC fs：脚本未再声明 |
| 每帧载荷 | **182 float** = 91 I + 91 Q | `Data(1:end/2)+1i*Data(end/2+1:end)` |
| 182 的物理含义 | **距离向复剖面（range-bin 基带）**，不是 chirp ADC，也不是已经算好的 Doppler FFT | 同上；`range_vector` 用 `bin_length` 生成 |
| range-bin spacing | `bin_length = 8 * 1.5e8 / 23.328e9` | 脚本；单位未写字符串，按 XeThru 惯例是米 |
| native temporal resolution | `1/512` s / 帧 | PRF |
| native Doppler resolution | STFT `nfft=4096`，PRF=512 → bin 宽 512/4096 Hz（若 `myspecgramnew` 按此 nfft） | `myspecgramnew` **不在仓库**，窗实现 UNKNOWN |
| maximum unambiguous velocity | 需要 `v=c f_D/(2 fc)` 且单一 `fc` | **不能可靠写成 m/s** |
| available physical axes | 时间（帧/PRF）；距离（bin_length）；Doppler **频率** ±PRF/2 = ±256 Hz（显示） | 速度轴：NO |

## 24 GHz（Ancortek FMCW）

| 量 | 值 | 证据 |
|---|---|---|
| `fc` | 每个 `.dat` 第 1 个数 | `datToImage_Anchortech.m`；本机未打开文件 → **UNKNOWN** |
| `Tsweep` | 第 2 个数，单位 ms，脚本 `/1000` 成秒 | 本机 UNKNOWN |
| `NTS` | 第 3 个数 | 本机 UNKNOWN |
| `Bw` | 第 4 个数 | 本机 UNKNOWN |
| README bandwidth | 1500 MHz | README；与文件头是否一致：未读文件 |
| PRF | `1/Tsweep` | 依赖文件头 |
| sampling | `fs=NTS/Tsweep` | 依赖文件头 |
| native temporal / Doppler | 依赖 `Tsweep`、STFT 128/4096/100 | 未打开 raw → UNKNOWN |
| max unambiguous velocity | 需 `fc` 与 PRF | UNKNOWN |
| ylim 5.8 GHz | 见 `preprocessing_chain_24ghz.md` | **作图参数**，不是 24 GHz 物理轴 |
| available physical axes | 打开 `.dat` 后：时间、Doppler Hz、可用 `fc` 算 v | 本机未打开：**不能声称已恢复** |

清单中 327 条 24 GHz `.dat` 压缩前大小多为 **15360054** 字节（固定长度记录）。未打开，不由此反推 NTS。

## 77 GHz（IWR1443 / DCA1000）

| 量 | 值 | 证据 |
|---|---|---|
| `fstart` / `fstop` | 77.1799e9 / 77.9474e9 | `datToImage_77.m` |
| `fc` | `(fstart+fstop)/2` = 77.56365e9 | 同上 |
| bandwidth | 0.7675e9 | 由上式 |
| README 写 4 GHz | 与脚本 0.7675 GHz **不一致** | 两处都记下，不以 README 覆盖脚本 |
| ADC fs | 10e6 | 脚本 |
| NTS | 256 | 脚本 + AIRHAR + Phase 1 实测 reshape |
| SweepTime / NPpF | 40e-3 / 128 → PRF **3200 Hz** | 脚本 |
| 文件内 chirp 数 | **32000**（约 10 s） | Phase 1 打开 `*_Raw_0.bin` |
| MATLAB `NoF=500` | 会给出 64000 chirp / 20 s | **与实测文件不符** |
| native temporal | `dT=SweepTime/NPpF` = 0.3125 ms / pulse | 脚本 |
| native range | `c/2/Bw` | 脚本有 `rng_axis` |
| native Doppler | STFT nfft=4096，PRF=3200 | MATLAB 链；AIRHAR overlap 不同 |
| display Doppler | `axis([0 20 -500 500])` → ±500 Hz | 作图裁切，不是 PRF/2 |
| max unambiguous velocity | `v_max = c * (PRF/2) / (2 fc)` 若用 PRF=3200、脚本 fc | 公式可用；**未在本机对 327 条算出数值网格** |
| available physical axes | 时间、距离、Doppler Hz；速度需采用脚本 fc | 本机 IQ 未抽出，R2 未建 |

77 GHz 另有 zip 内 `activity_spect_matrices/*_matrix.mat`，Phase 1 读过 `sx1` 为 complex `[4096, ~538]`。那是数值 STFT，轴标定仍要靠脚本，且 **不在本机**。
