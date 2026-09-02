# XeThru 10 GHz effective-frequency audit

Verdict: **EFFECTIVE_FC_UNRESOLVED**

Phase 9 does **not** build R2-V. Main training stays on **R2-D**
`S(normalized_time, Doppler_Hz)`.

No official source in this repository gives a single effective carrier
frequency, center frequency, or velocity-conversion convention that can
be used in `v = c f_D / (2 f_c)` for the 10 GHz XeThru recordings.

## What was searched

| Location | Result |
|---|---|
| `third_party/CI4R-MULTI3/XeThru_mDopp_bulk.m` | No `fc`, no center frequency, no `v = c f_D / (2 fc)` |
| `third_party/CI4R-MULTI3/datToImage_Anchortech.m` | 24 GHz only; `fc = radarData(1)` |
| `third_party/CI4R-MULTI3/datToImage_77.m` | 77 GHz only; `fc = (fstart+fstop)/2` |
| `third_party/CI4R-MULTI3/README.md` | Label only: “XeThru (10 GHz UWB impulse radar)” |
| MATLAB notebooks under `third_party/CI4R-MULTI3/` | No XeThru `fc` / velocity axis |
| Raw 10 GHz `.dat` header (Phase 8B) | XeThru frame: `ContentId`, `Info`, `Length`, 182 float I/Q. No frequency field |
| Hardware / model datasheet in this repo | Not present |

No other README, config, or metadata file in the repo states an XeThru
center frequency for this dataset.

## Evidence from `XeThru_mDopp_bulk.m`

The official 10 GHz chain never converts Doppler to velocity.

1. Doppler axis is frequency in Hz, from PRF:

```
PRF=512;
DopplerAxis=linspace(-PRF/2,PRF/2,size(iq_mat,2));
```

2. The displayed spectrogram uses that frequency axis:

```
imagesc(TimeAxis,DopplerAxis,20*log10(sx2./max(max(sx2))));
axis([0 max(TimeAxis) -PRF/2 PRF/2])
```

The commented ylabel is `Frequency (Hz)`, not velocity.

3. The only large frequency constant in the file is **not** a carrier:

```
bin_length = 8 * 1.5e8/23.328e9; % range_decimation_factor * (c/2) / fs.
```

`23.328e9` is the sampling term `fs` used to space **range bins**.
It is not assigned to `fc`, not used in a Doppler-to-velocity formula,
and is not described as center frequency.

4. Range-profile recovery is baseband I/Q (`91` I + `91` Q). The script
does not introduce a carrier for velocity scaling.

## Why the README “10 GHz” label is not an effective fc

README line:

> XeThru (10 GHz UWB impulse radar)

That is a band / product label. UWB impulse radar occupies a wide
spectrum. The official MATLAB chain does not pick a single number from
that label and does not apply `v = c f_D / (2 fc)`.

Using 10e9, 7.29e9, 8.7e9, or any Novelda X4 datasheet default would be
an external guess. This audit forbids that.

## What is available on the other two radars

These values are **not** used for a three-radar R2-V, because 10 GHz
blocks a common velocity axis.

| Radar | fc | Source |
|---|---|---|
| 24 GHz Ancortek | 24.75 GHz | raw `.dat` header, Phase 8B `header_24ghz.json` |
| 77 GHz IWR1443 | `(77.1799e9 + 77.9474e9)/2 = 77.56365 GHz` | `datToImage_77.m` lines 8–10 |
| 10 GHz XeThru | none | this audit |

## Decision

- Result: **B. EFFECTIVE_FC_UNRESOLVED**
- R2-V: **not built**
- Representation for Phase 9: **R2-D**
- R2-D remains a physical Doppler-frequency grid. It is not velocity
  aligned and is not sensor-invariant.
