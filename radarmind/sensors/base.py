"""Sensor adapters. Shared interface, not a shared learned token space."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from radarmind.data.base import RadarSample
from radarmind.schema import UNKNOWN


@dataclass
class MotionInput:
    tensor: np.ndarray
    mask: np.ndarray | None
    temporal_mask: np.ndarray | None
    representation_type: str
    physical_metadata: dict = field(default_factory=dict)


class SpectrogramAdapter:
    representation_type = "micro_doppler"

    def encode(self, sample: RadarSample) -> MotionInput:
        if sample.image is None:
            raise ValueError(f"{sample.sample_id} has no spectrogram image")
        img = np.asarray(sample.image, dtype=np.float32)
        if img.ndim != 2:
            raise ValueError(f"expected 2D spectrogram, got {img.shape}")
        t = img.shape[1]
        temporal_mask = np.ones((t,), dtype=bool)
        return MotionInput(
            tensor=img,
            mask=None,
            temporal_mask=temporal_mask,
            representation_type=self.representation_type,
            physical_metadata={
                "dataset": sample.dataset,
                "radar_type": sample.radar_type,
                "shape": list(img.shape),
                "velocity_calibration": sample.metadata.get("velocity_calibration", "absent"),
            },
        )


class PointCloudAdapter:
    representation_type = "point_cloud"

    def encode(self, sample: RadarSample, n_ch: int | None = None) -> MotionInput:
        if sample.points is None:
            raise ValueError(f"{sample.sample_id} has no point cloud")
        pts = np.asarray(sample.points, dtype=np.float32)
        mask = sample.point_mask
        if mask is None:
            mask = np.any(np.isfinite(pts) & (np.abs(pts).sum(axis=-1, keepdims=False) > 0), axis=-1)
        else:
            mask = np.asarray(mask, dtype=bool)
        if n_ch is not None:
            pts = pts[..., : int(n_ch)]
        t = pts.shape[0]
        temporal_mask = mask.any(axis=-1) if mask.ndim == 2 else np.ones((t,), dtype=bool)
        return MotionInput(
            tensor=pts,
            mask=mask,
            temporal_mask=temporal_mask,
            representation_type=self.representation_type,
            physical_metadata={
                "dataset": sample.dataset,
                "radar_type": sample.radar_type,
                "shape": list(pts.shape),
                "channels": sample.metadata.get("channels", UNKNOWN),
                "coordinate_units": sample.metadata.get("coordinate_units", UNKNOWN),
                "velocity_units": sample.metadata.get("velocity_units", UNKNOWN),
            },
        )


def adapt_sample(sample: RadarSample, n_ch: int | None = None) -> MotionInput:
    if sample.representation in ("micro_doppler", "iq_derived"):
        return SpectrogramAdapter().encode(sample)
    if sample.representation == "point_cloud":
        return PointCloudAdapter().encode(sample, n_ch=n_ch)
    raise ValueError(f"no sensor adapter for representation={sample.representation}")
