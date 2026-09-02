"""Official AIRHAR 77 GHz raw reader. Do not invent a second bin layout."""
from __future__ import annotations

from typing import Any, Dict, Union

import numpy as np

# From third_party/AIRHAR/datasets/data_prepare_CI4R.py:bin2np
NTS = 256
N_LANES = 8
I_LANE = 3  # "fourth Lanes" (0-based)
Q_LANE = 7

# From third_party/CI4R-MULTI3/datToImage_77.m (same 8-row I/Q packing)
MATLAB_NUM_LANES = 4  # 4 RX; I in rows 1-4, Q in rows 5-8


def load_ci4r_77ghz_raw(path_or_bytes: Union[str, bytes]) -> Dict[str, Any]:
    """Load one 77 GHz DCA1000 .bin the way AIRHAR bin2np does.

    Returns
    -------
    iq : complex64 array shaped (256, n_chirps) after AIRHAR reshape
    metadata : dict of constants and observed sizes
    """
    if isinstance(path_or_bytes, (bytes, bytearray)):
        raw = np.frombuffer(path_or_bytes, dtype=np.int16).copy()
        source = "bytes"
    else:
        raw = np.fromfile(str(path_or_bytes), dtype=np.int16)
        source = str(path_or_bytes)

    n_int16 = int(raw.size)
    meta: Dict[str, Any] = {
        "source": source,
        "dtype_on_disk": "int16",
        "n_int16": n_int16,
        "n_lanes": N_LANES,
        "nts": NTS,
        "i_lane_index": I_LANE,
        "q_lane_index": Q_LANE,
        "reshape_order": "F",
        "reader": "AIRHAR.bin2np",
        "ok": True,
        "error": "",
    }
    if n_int16 == 0 or n_int16 % N_LANES != 0:
        meta["ok"] = False
        meta["error"] = "int16 length not divisible by 8"
        return {"iq": np.zeros((NTS, 0), dtype=np.complex64), "metadata": meta}

    lanes = raw.reshape((N_LANES, -1), order="F")
    iq_1d = lanes[I_LANE, :].astype(np.float32) + 1j * lanes[Q_LANE, :].astype(np.float32)
    if iq_1d.size % NTS != 0:
        meta["ok"] = False
        meta["error"] = "selected-lane length not divisible by NTS=256"
        return {"iq": iq_1d.astype(np.complex64), "metadata": meta}

    iq = iq_1d.reshape((NTS, -1), order="F").astype(np.complex64)
    meta["n_chirps"] = int(iq.shape[1])
    meta["shape"] = list(iq.shape)
    meta["matlab_lane_layout"] = (
        "datToImage_77.m: reshape(int16, 8, []) then lanes 1:4 + 1j*lanes 5:8; "
        "AIRHAR keeps only lane 4 (index 3/7)"
    )
    return {"iq": iq, "metadata": meta}


def iq_qc(iq: np.ndarray) -> Dict[str, Any]:
    re = np.real(iq)
    im = np.imag(iq)
    amp = np.abs(iq)
    return {
        "shape": list(iq.shape),
        "dtype": str(iq.dtype),
        "real_min": float(re.min()) if iq.size else None,
        "real_max": float(re.max()) if iq.size else None,
        "imag_min": float(im.min()) if iq.size else None,
        "imag_max": float(im.max()) if iq.size else None,
        "amp_min": float(amp.min()) if iq.size else None,
        "amp_max": float(amp.max()) if iq.size else None,
        "nan_count": int(np.isnan(re).sum() + np.isnan(im).sum()),
        "inf_count": int(np.isinf(re).sum() + np.isinf(im).sum()),
        "zero_fraction": float(np.mean(amp == 0)) if iq.size else None,
    }
