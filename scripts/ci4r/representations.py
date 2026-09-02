"""Three CI4R 77 GHz representations kept for later work.

A — raw / minimal: one record = AIRHAR IQ (256 ADC, 32000 chirps), complex64.
    Do not feed the full 32000-chirp tensor to a 2D CNN.
    If a later stage uses A: slice chirps into non-overlapping windows
    (e.g. 512 chirps → (256, 512) complex, or |IQ| + phase as 2 ch).
    Per-window z-score. Every window inherits the parent record split.
    This file does not implement training on A.

B — official AIRHAR physics map (Phase 1A baseline input):
    range FFT → sum bins 23:31 → STFT → 20log clip → 224x224 → z-score.
    See preprocess_77ghz.airhar_micro_doppler.

C — archive precomputed:
    MAT key sx1: complex128 STFT (4096, 538), MATLAB hop, not log-mapped.
    PNG: plotted figure ~976x770 with axes/colorbar. Not a numeric feature.
    FWCC C_DCT2 (538, 10): cepstral coefficients, not the spectrogram.
"""
