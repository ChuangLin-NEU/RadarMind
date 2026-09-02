"""Radar-specific spectrogram adapters. Same interface, separate objects."""
from __future__ import annotations

from radarmind.data.base import RadarSample
from radarmind.sensors.base import MotionInput, SpectrogramAdapter


class _FreqAdapter:
    frequency_ghz: float
    radar_name: str

    def __init__(self):
        self.inner = SpectrogramAdapter()

    def encode(self, sample: RadarSample) -> MotionInput:
        inp = self.inner.encode(sample)
        inp.physical_metadata = {
            **inp.physical_metadata,
            "radar_frequency_ghz": self.frequency_ghz,
            "radar_name": self.radar_name,
            "shared_token_learning": False,
        }
        return inp


class Radar10Adapter(_FreqAdapter):
    frequency_ghz = 10.0
    radar_name = "XeThru_UWB"


class Radar24Adapter(_FreqAdapter):
    frequency_ghz = 24.0
    radar_name = "Ancortek_FMCW"


class Radar77Adapter(_FreqAdapter):
    frequency_ghz = 77.0
    radar_name = "TI_IWR1443"
