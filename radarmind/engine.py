"""RadarMindEngine — Python API used by Streamlit and later frontends."""
from __future__ import annotations

from pathlib import Path

from radarmind.data.base import DatasetRegistry, RadarSample
from radarmind.intelligence.extract import build_intelligence
from radarmind.intelligence.schema import MotionIntelligence
from radarmind.llm.provider import LLMService
from radarmind.models.prediction import MotionPrediction
from radarmind.models.registry import ModelRegistry
from radarmind.paths import V01
from radarmind.protocols import ProtocolRegistry
from radarmind.replay import ReplayEngine, ReplayState


class RadarMindEngine:
    def __init__(self, offline_demo: bool | None = None):
        self.datasets = DatasetRegistry
        self.models = ModelRegistry()
        self.protocols = ProtocolRegistry()
        self.llm = LLMService(force_offline=offline_demo)
        # Public UI contract: ui_mode() is required. Rebuild if a stale class was imported.
        if not callable(getattr(self.llm, "ui_mode", None)):
            from radarmind.llm import provider as _llm_mod
            import importlib

            importlib.reload(_llm_mod)
            self.llm = _llm_mod.LLMService()
        self.replay = ReplayEngine(self.models)

    def load_dataset(self, name: str):
        return self.datasets.load_dataset(name)

    def load_sample(self, dataset: str, sample_id: str) -> RadarSample:
        return self.load_dataset(dataset).get(sample_id)

    def infer(self, sample: RadarSample, model_name: str | None = None, protocol: str | None = None) -> MotionPrediction:
        return self.models.predict(sample, model_name=model_name, protocol=protocol)

    def intelligence(self, sample: RadarSample, pred: MotionPrediction | None = None) -> MotionIntelligence:
        if pred is None:
            pred = self.infer(sample)
        return build_intelligence(sample, pred)

    def replay_load(self, **kwargs) -> ReplayState:
        if kwargs.get("mode") == "continuous" or kwargs.get("recording"):
            rec = kwargs.get("recording") or kwargs.get("sequence_id")
            return self.replay.load_continuous(
                recording=rec,
                protocol=kwargs.get("protocol") or "subject_independent",
                model_name=kwargs.get("model_name"),
                smooth_mode=kwargs.get("smooth_mode") or "confidence_aware",
            )
        return self.replay.load(**{k: v for k, v in kwargs.items() if k not in ("mode", "recording", "smooth_mode")})

    def replay_step(self, action: str, **kwargs) -> ReplayState:
        fn = {
            "play": self.replay.play,
            "pause": self.replay.pause,
            "next": self.replay.next_frame,
            "previous": self.replay.prev_frame,
            "next_window": self.replay.next_window,
            "prev_window": self.replay.prev_window,
            "seek": lambda: self.replay.seek(**kwargs),
            "seek_frame": lambda: self.replay.seek_frame(kwargs.get("frame", 0)),
        }.get(action)
        if fn is None:
            raise KeyError(action)
        return fn()

    def explain(self, timeline: dict | None = None, intelligence: dict | None = None, task: str = "summary", question: str | None = None) -> dict:
        return self.llm.explain(task, intelligence=intelligence, timeline=timeline, question=question)

    def default_model(self, dataset: str) -> str:
        return self.models.resolve(dataset)

    def demo_registry(self) -> dict:
        path = V01 / "demo_registry.json"
        if not path.is_file():
            return {"demos": []}
        import json

        return json.loads(path.read_text(encoding="utf-8"))

    def missing_checkpoints(self) -> list[dict]:
        return self.models.missing()
