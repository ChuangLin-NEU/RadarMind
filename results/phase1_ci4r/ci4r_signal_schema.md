# CI4R 77 GHz signal schema

Evidence: `third_party/AIRHAR/datasets/data_prepare_CI4R.py` (`bin2np`, `RGB_mD`) and `third_party/CI4R-MULTI3/datToImage_77.m`, plus Phase 1A reads of `*_Raw_0.bin`.

## On-disk layout

- dtype: `int16` (DCA1000 two's complement). MATLAB: `fread(..., 'int16')`. AIRHAR: `np.fromfile(..., dtype=np.int16)`.
- Interleave: reshape to `(8, N)` Fortran / MATLAB column-major.
- `datToImage_77.m`: `numLanes = 4`; `Data(1:4,:) + 1j*Data(5:8,:)`. Four RX, I then Q.
- AIRHAR `bin2np`: keeps **only the fourth lane** (`data[3,:] + 1j*data[7,:]`).

Observed file size for usable action-folder `*_Raw_0.bin`: **131072000** bytes = 65536000 int16.

```
int16[65536000]
  -> reshape (8, 8192000) order F
  -> 4 RX complex streams of length 8192000
  -> AIRHAR lane 4 -> complex[8192000]
  -> reshape (256, 32000) order F
```

## What is 256?

**ADC samples per chirp (NTS).** Both AIRHAR (`NTS = 256`) and MATLAB (`NTS = 256`) say this. Range FFT is along this axis.

## What is 32000?

**Number of chirps / pulses in this recording**, after one RX is selected: `8192000 / 256 = 32000`. This is measured from the files, and AIRHAR also writes `N = 32000`.

## Comments that do **not** match these files

MATLAB / AIRHAR also write:

- `NPpF = 128` pulses per frame
- `NoF = 500` frames
- `SweepTime = 40e-3`

`128 * 500 = 64000` pulses, which would be **20 s** at `dT = SweepTime/NPpF`. The `*_Raw_0.bin` we opened have **32000** pulses (about **10 s** at that `dT`). Plot code still does `axis([0 20 ...])`.

So:

- `256` = NTS: **resolved**
- `32000` = chirps in file: **resolved from data**
- frame packing `32000 = frames × pulses_per_frame`: **UNRESOLVED** (128×250 would fit; 128×500 does not)
- Tx count: **UNRESOLVED** (not in these two loaders)
- Rx count: **4** in MATLAB; AIRHAR uses 1 of 4

Official reshape used in this phase (AIRHAR):

`raw int16 -> [8, N] -> IQ lane 4 -> [256 ADC, 32000 chirps]`

Not claimed: `[frame, chirp, rx, adc]` as a stored tensor. That would require a frame size that matches 32000, which the comments do not.
