"""Vectorized occupancy voxel matching RadHAR DataPreprocessing/voxels.py.

Official grid is 10 x 32 x 32, bounds = min/max of the *current frame*.
together_frames=1, then temporal windows of 60 frames, slide 10.
"""
from __future__ import annotations

from typing import List

import numpy as np


def voxelize_frame(pts: np.ndarray, nx: int = 10, ny: int = 32, nz: int = 32) -> np.ndarray:
    pix = np.zeros((nx, ny, nz), dtype=np.uint8)
    if pts is None or pts.size == 0:
        return pix
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    xmin, xmax = float(x.min()), float(x.max())
    ymin, ymax = float(y.min()), float(y.max())
    zmin, zmax = float(z.min()), float(z.max())
    x_res = (xmax - xmin) / nx if xmax > xmin else 1.0
    y_res = (ymax - ymin) / ny if ymax > ymin else 1.0
    z_res = (zmax - zmin) / nz if zmax > zmin else 1.0
    xi = np.floor((x - xmin) / x_res).astype(np.int32)
    yi = np.floor((y - ymin) / y_res).astype(np.int32)
    zi = np.floor((z - zmin) / z_res).astype(np.int32)
    xi = np.clip(xi, 0, nx - 1)
    yi = np.clip(yi, 0, ny - 1)
    zi = np.clip(zi, 0, nz - 1)
    # occupancy count, saturate uint8
    np.add.at(pix, (xi, yi, zi), 1)
    return pix


def voxel_windows(frames: List[np.ndarray], t_len: int = 60, stride: int = 10) -> np.ndarray:
    n = len(frames)
    if n < t_len:
        return np.zeros((0, t_len, 10, 32, 32), dtype=np.uint8)
    voxels = [voxelize_frame(fr) for fr in frames]
    out = []
    i = 0
    while i + t_len <= n:
        out.append(np.stack(voxels[i : i + t_len], axis=0))
        i += stride
    if not out:
        return np.zeros((0, t_len, 10, 32, 32), dtype=np.uint8)
    return np.stack(out, axis=0)
