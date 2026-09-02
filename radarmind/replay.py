"""ReplayEngine: load sequence, step frames/windows, expose current view state."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np

from radarmind.data.base import DatasetRegistry, RadarSample
from radarmind.intelligence.extract import build_intelligence
from radarmind.intelligence.timeline import MotionTimeline, build_timeline
from radarmind.models.openset import display_activity
from radarmind.models.prediction import MotionPrediction
from radarmind.models.registry import ModelRegistry
from radarmind.schema import UNKNOWN, jsonable


@dataclass
class ReplayState:
    dataset: str
    sequence_id: str
    window_index: int
    frame_index: int
    n_windows: int
    n_frames: int
    playing: bool
    sample: RadarSample | None
    prediction: MotionPrediction | None
    intelligence: dict
    timeline: dict
    view: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = {
            "dataset": self.dataset,
            "sequence_id": self.sequence_id,
            "window_index": self.window_index,
            "frame_index": self.frame_index,
            "n_windows": self.n_windows,
            "n_frames": self.n_frames,
            "playing": self.playing,
            "sample_id": self.sample.sample_id if self.sample else UNKNOWN,
            "prediction": self.prediction.to_dict() if self.prediction else None,
            "motion_intelligence": self.intelligence,
            "timeline": self.timeline,
            "view": self.view,
        }
        return jsonable(d)


class ReplayEngine:
    def __init__(self, models: ModelRegistry | None = None):
        self.models = models or ModelRegistry()
        self.dataset = "ci4r"
        self.protocol = "within_distribution"
        self.model_name: str | None = None
        self.sequence_id = ""
        self.window_ids: list[str] = []
        self.samples: list[RadarSample] = []
        self.preds: list[MotionPrediction] = []
        self.intelligences: list[dict] = []
        self.timeline: MotionTimeline | None = None
        self.window_index = 0
        self.frame_index = 0
        self.playing = False
        self.mode = "sample"
        self.cursor_frame = 0
        self.gt_timeline: dict = {}
        self.pred_pack: dict = {}
        self.smooth_mode = "confidence_aware"
        self.pred_windows: list[dict] = []
        self.series: list[dict] = []

    def load(
        self,
        dataset: str,
        sequence_id: str | None = None,
        sample_id: str | None = None,
        protocol: str | None = None,
        model_name: str | None = None,
        window_ids: list[str] | None = None,
    ) -> ReplayState:
        self.mode = "sample"
        self.dataset = dataset
        self.protocol = protocol or {
            "ci4r": "within_distribution",
            "mri": "subject_independent",
            "radhar": "official_split",
        }[dataset]
        self.model_name = model_name
        adapter = DatasetRegistry.load_dataset(dataset)
        if window_ids:
            ids = list(window_ids)
            seq = sequence_id or ids[0]
        elif sample_id:
            ids = [sample_id]
            seq = sequence_id or sample_id
        elif sequence_id:
            ids = adapter.windows_in_sequence(sequence_id)
            seq = sequence_id
        else:
            raise ValueError("need sample_id, sequence_id, or window_ids")
        if not ids:
            raise ValueError(f"empty sequence {sequence_id}")
        self.sequence_id = seq
        self.window_ids = ids
        self.samples = [adapter.get(s) for s in ids]
        self.preds = [self.models.predict(s, model_name=self.model_name, protocol=self.protocol) for s in self.samples]
        self.intelligences = [build_intelligence(s, p).to_dict() for s, p in zip(self.samples, self.preds)]
        rows = []
        t_cursor = 0.0
        time_unit = "s"
        for s, p, intel in zip(self.samples, self.preds, self.intelligences):
            fr = s.frame_rate if isinstance(s.frame_rate, (int, float)) else None
            n = max(s.n_frames(), 1)
            if fr:
                dur = n / float(fr)
            else:
                dur = float(n)
                time_unit = "bin"
            rows.append(
                {
                    "sample_id": s.sample_id,
                    "activity": display_activity(p),
                    "confidence": p.confidence,
                    "start_time": t_cursor,
                    "end_time": t_cursor + dur,
                    "time_unit": time_unit,
                    "motion_intelligence": intel,
                }
            )
            t_cursor += dur
        self.timeline = build_timeline(dataset, seq, self.protocol, self.preds[0].model_name, rows)
        self.timeline.smoothing["time_unit"] = time_unit
        self.window_index = 0
        self.frame_index = 0
        self.playing = False
        return self.state()

    def load_continuous(
        self,
        recording: str,
        protocol: str = "subject_independent",
        model_name: str | None = None,
        smooth_mode: str = "confidence_aware",
    ) -> ReplayState:
        from radarmind.paths import ROOT

        pred_p = ROOT / "results" / "phase4_continuous" / "predicted_timelines" / f"{recording}.json"
        gt_p = ROOT / "results" / "phase4_continuous" / "ground_truth_timelines" / f"{recording}.json"
        pred = json.loads(pred_p.read_text(encoding="utf-8"))
        gt = json.loads(gt_p.read_text(encoding="utf-8"))
        mode = pred["modes"][smooth_mode]
        ids = [w["sample_id"] for w in mode["windows"]]
        st = self.load(dataset="mri", sequence_id=recording, protocol=protocol, model_name=model_name, window_ids=ids)
        self.mode = "continuous"
        self.gt_timeline = gt
        self.pred_pack = pred
        self.smooth_mode = smooth_mode
        self.pred_windows = mode["windows"]
        self.series = mode.get("series") or []
        # overlay predicted events (absolute frames / seconds)
        if self.timeline is not None:
            self.timeline.events = mode["events"]
            self.timeline.smoothing = {
                **(self.timeline.smoothing or {}),
                **(pred.get("smoothing") or {}),
                "mode": smooth_mode,
                "time_unit": "s",
            }
        if self.pred_windows:
            self.cursor_frame = int(self.pred_windows[0]["start_frame"])
            self._sync_continuous()
        return self.state() if st else self.state()

    def _sync_continuous(self) -> None:
        if not self.pred_windows:
            return
        hit = None
        for i, w in enumerate(self.pred_windows):
            if int(w["start_frame"]) <= self.cursor_frame <= int(w["end_frame"]):
                hit = i
                break
        if hit is None:
            hit = min(range(len(self.pred_windows)), key=lambda i: abs(int(self.pred_windows[i]["center_frame"]) - self.cursor_frame))
        self.window_index = hit
        w = self.pred_windows[hit]
        self.frame_index = int(np.clip(self.cursor_frame - int(w["start_frame"]), 0, max(self.samples[hit].n_frames() - 1, 0)))

    def seek_frame(self, frame: int) -> ReplayState:
        if self.mode != "continuous" or not self.pred_windows:
            return self.seek(frame_index=frame)
        lo = min(int(w["start_frame"]) for w in self.pred_windows)
        hi = max(int(w["end_frame"]) for w in self.pred_windows)
        self.cursor_frame = int(np.clip(frame, lo, hi))
        self._sync_continuous()
        return self.state()

    def play(self) -> ReplayState:
        self.playing = True
        return self.state()

    def pause(self) -> ReplayState:
        self.playing = False
        return self.state()

    def next_frame(self) -> ReplayState:
        if self.mode == "continuous" and self.pred_windows:
            return self.seek_frame(self.cursor_frame + 1)
        s = self.samples[self.window_index]
        n = max(s.n_frames(), 1)
        if self.frame_index + 1 < n:
            self.frame_index += 1
        elif self.window_index + 1 < len(self.samples):
            self.window_index += 1
            self.frame_index = 0
        return self.state()

    def prev_frame(self) -> ReplayState:
        if self.mode == "continuous" and self.pred_windows:
            return self.seek_frame(self.cursor_frame - 1)
        if self.frame_index > 0:
            self.frame_index -= 1
        elif self.window_index > 0:
            self.window_index -= 1
            self.frame_index = max(self.samples[self.window_index].n_frames() - 1, 0)
        return self.state()

    def next_window(self) -> ReplayState:
        if self.window_index + 1 < len(self.samples):
            self.window_index += 1
            self.frame_index = 0
        return self.state()

    def prev_window(self) -> ReplayState:
        if self.window_index > 0:
            self.window_index -= 1
            self.frame_index = 0
        return self.state()

    def seek(self, window_index: int | None = None, frame_index: int | None = None) -> ReplayState:
        if window_index is not None:
            self.window_index = int(np.clip(window_index, 0, max(len(self.samples) - 1, 0)))
        n = max(self.samples[self.window_index].n_frames(), 1)
        if frame_index is not None:
            self.frame_index = int(np.clip(frame_index, 0, n - 1))
        return self.state()

    def _view(self, sample: RadarSample) -> dict:
        t = self.frame_index
        if sample.representation == "micro_doppler" and sample.image is not None:
            img = np.asarray(sample.image)
            t = int(np.clip(t, 0, img.shape[1] - 1))
            return {
                "kind": "micro_doppler",
                "image": img,
                "cursor_t": t,
                "column": img[:, t],
            }
        if sample.points is not None:
            pts = np.asarray(sample.points)
            mask = sample.point_mask
            if mask is None:
                mask = np.ones(pts.shape[:2], dtype=bool)
            t = int(np.clip(t, 0, pts.shape[0] - 1))
            valid = pts[t][mask[t]]
            return {
                "kind": "point_cloud",
                "frame_index": t,
                "xy": valid[:, :2] if valid.size else np.zeros((0, 2)),
                "xz": valid[:, [0, 2]] if valid.size else np.zeros((0, 2)),
                "yz": valid[:, [1, 2]] if valid.size else np.zeros((0, 2)),
                "xyz": valid[:, :3] if valid.size else np.zeros((0, 3)),
                "velocity": valid[:, 3] if valid.size and valid.shape[1] >= 4 else np.zeros((0,)),
                "n_points": int(valid.shape[0]),
            }
        return {"kind": "empty"}

    def state(self) -> ReplayState:
        if not self.samples:
            return ReplayState("", "", 0, 0, 0, 0, False, None, None, {}, {}, {})
        s = self.samples[self.window_index]
        p = self.preds[self.window_index]
        intel = self.intelligences[self.window_index]
        tl = self.timeline.to_dict() if self.timeline else {}
        view = self._view(s)
        if self.mode == "continuous":
            view["cursor_frame"] = self.cursor_frame
            view["mode"] = "continuous"
            gt_act = "UNLABELED"
            for e in self.gt_timeline.get("events", []):
                if int(e["start_frame"]) <= self.cursor_frame <= int(e["end_frame"]):
                    gt_act = e["activity"]
                    break
            view["gt_activity"] = gt_act
            view["uncertain"] = float(p.confidence) < 0.5
            tl["gt_events"] = self.gt_timeline.get("events")
            tl["series"] = self.series
        return ReplayState(
            dataset=self.dataset,
            sequence_id=self.sequence_id,
            window_index=self.window_index,
            frame_index=self.frame_index,
            n_windows=len(self.samples),
            n_frames=max(s.n_frames(), 1),
            playing=self.playing,
            sample=s,
            prediction=p,
            intelligence=intel,
            timeline=tl,
            view=view,
        )
