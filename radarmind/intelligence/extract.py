"""Physics and metadata for MotionIntelligence. Do not invent direction or SI units."""
from __future__ import annotations

import numpy as np

from radarmind.data.base import RadarSample
from radarmind.intelligence.schema import MotionIntelligence, empty_intelligence
from radarmind.models.prediction import MotionPrediction
from radarmind.schema import UNKNOWN, tagged, unknown


def _finite(v) -> np.ndarray:
    a = np.asarray(v, dtype=np.float64).ravel()
    return a[np.isfinite(a)]


def point_cloud_kinematics(sample: RadarSample) -> dict:
    """Doppler/velocity stats on valid points only. Units follow dataset metadata."""
    if sample.points is None:
        return {}
    pts = np.asarray(sample.points, dtype=np.float64)
    mask = sample.point_mask
    if mask is None:
        mask = np.ones(pts.shape[:2], dtype=bool)
    else:
        mask = np.asarray(mask, dtype=bool)
    vel = pts[..., 3][mask]
    vel = _finite(vel)
    n_frame = mask.sum(axis=-1).astype(np.float64)
    out = {
        "n_valid_points": int(mask.sum()),
        "mean_points_per_frame": float(n_frame.mean()) if n_frame.size else 0.0,
        "n_empty_frames": int((n_frame == 0).sum()),
    }
    if vel.size == 0:
        out.update(
            {
                "mean_velocity": UNKNOWN,
                "median_velocity": UNKNOWN,
                "peak_abs_velocity": UNKNOWN,
                "positive_velocity_fraction": UNKNOWN,
                "negative_velocity_fraction": UNKNOWN,
                "velocity_std": UNKNOWN,
                "motion_energy": UNKNOWN,
            }
        )
        return out
    out.update(
        {
            "mean_velocity": float(vel.mean()),
            "median_velocity": float(np.median(vel)),
            "peak_abs_velocity": float(np.max(np.abs(vel))),
            "positive_velocity_fraction": float((vel > 0).mean()),
            "negative_velocity_fraction": float((vel < 0).mean()),
            "velocity_std": float(vel.std()),
            "motion_energy": float(np.mean(vel**2)),
        }
    )
    if pts.shape[-1] >= 5:
        inten = _finite(pts[..., 4][mask])
        if inten.size:
            out["mean_intensity"] = float(inten.mean())
            out["peak_intensity"] = float(inten.max())
        else:
            out["mean_intensity"] = UNKNOWN
            out["peak_intensity"] = UNKNOWN
    return out


def spectrogram_temporal_stats(sample: RadarSample) -> dict:
    """Column energy of the z-scored spectrogram. Not converted to m/s."""
    if sample.image is None:
        return {}
    img = np.asarray(sample.image, dtype=np.float64)
    col = img.mean(axis=0)
    return {
        "spec_time_energy_mean": float(col.mean()),
        "spec_time_energy_std": float(col.std()),
        "spec_time_energy_peak": float(np.max(np.abs(col))),
        "n_time_bins": int(img.shape[1]),
        "n_freq_bins": int(img.shape[0]),
        "velocity_note": "no reliable Hz→m/s calibration on this spectrogram; velocity left UNKNOWN",
    }


def _duration(sample: RadarSample) -> tuple:
    fr = sample.frame_rate
    n = sample.n_frames()
    if isinstance(fr, (int, float)) and fr > 0 and n > 0:
        return float(n / float(fr)), "METADATA"
    if sample.representation == "micro_doppler":
        return UNKNOWN, "UNKNOWN"
    return UNKNOWN, "UNKNOWN"


def build_intelligence(sample: RadarSample, pred: MotionPrediction | None = None) -> MotionIntelligence:
    mi = empty_intelligence()
    mi.dataset = tagged(sample.dataset, "METADATA")
    mi.radar = tagged(sample.radar_id, "METADATA")
    if sample.subject_id and sample.subject_id != UNKNOWN:
        mi.subject = tagged(sample.subject_id, "METADATA")
    else:
        mi.subject = unknown("no subject id on this sample")
    mi.environment = unknown("no validated environment metadata")
    mi.periodicity = unknown("v0.1 does not estimate period")
    mi.direction = unknown(
        "Doppler/velocity sign is not mapped to toward/away; coordinate convention not validated for this field"
    )

    dur, dsrc = _duration(sample)
    if dur != UNKNOWN:
        unit = "s"
        note = f"T={sample.n_frames()} / frame_rate={sample.frame_rate}"
        mi.duration = tagged(dur, dsrc, unit=unit, note=note)
    else:
        mi.duration = unknown("frame_rate not available for physical duration")

    evidence = {
        "sample_id": sample.sample_id,
        "representation": sample.representation,
        "action_gt": sample.action_gt,
    }

    if pred is not None:
        status = getattr(pred, "open_set_status", UNKNOWN)
        if status == "UNKNOWN":
            mi.activity = tagged("UNKNOWN_MOTION", "MODEL", note="open-set reject; not a closed-set label")
            evidence["nearest_known_class"] = pred.activity
            evidence["open_set_note"] = "unknown motion; physics fields below are still from the radar window"
        else:
            mi.activity = tagged(pred.activity, "MODEL")
        mi.confidence = tagged(pred.confidence, "MODEL", unit="probability")
        if isinstance(pred.uncertainty, float):
            mi.uncertainty = tagged(pred.uncertainty, "MODEL", unit="normalized_entropy", note="entropy / log(C)")
        evidence["model_name"] = pred.model_name
        evidence["protocol"] = pred.protocol
        evidence["prediction_entropy"] = pred.prediction_entropy
        evidence["class_probabilities"] = pred.class_probabilities
        evidence["open_set_status"] = status
        evidence["open_set_score"] = pred.open_set_score
        evidence["open_set_method"] = pred.open_set_method
        evidence["open_set_threshold"] = pred.threshold

    if sample.representation == "point_cloud":
        kin = point_cloud_kinematics(sample)
        evidence["kinematics"] = kin
        units = str(sample.metadata.get("velocity_units", UNKNOWN))
        if kin.get("mean_velocity") != UNKNOWN and kin.get("mean_velocity") is not None:
            mi.mean_velocity = tagged(kin["mean_velocity"], "PHYSICS_DERIVED", unit=units)
            mi.peak_velocity = tagged(kin["peak_abs_velocity"], "PHYSICS_DERIVED", unit=units, note="peak |v|")
            energy = kin.get("motion_energy")
            if energy != UNKNOWN:
                mi.motion_intensity = tagged(energy, "PHYSICS_DERIVED", unit=f"{units}^2" if units != UNKNOWN else UNKNOWN, note="mean v^2")
            if kin.get("mean_points_per_frame", 0) > 0:
                mi.motion_state = tagged("points_present", "PHYSICS_DERIVED", note="at least one valid point in the window")
            else:
                mi.motion_state = tagged("empty_window", "PHYSICS_DERIVED")
        else:
            mi.mean_velocity = unknown("no valid Doppler/velocity points")
            mi.peak_velocity = unknown("no valid Doppler/velocity points")
            mi.motion_intensity = unknown("no valid Doppler/velocity points")
    elif sample.representation == "micro_doppler":
        stats = spectrogram_temporal_stats(sample)
        evidence["spectrogram_stats"] = stats
        mi.mean_velocity = unknown("no Doppler calibration from spectrogram pixels to m/s")
        mi.peak_velocity = unknown("no Doppler calibration from spectrogram pixels to m/s")
        mi.motion_intensity = tagged(
            stats.get("spec_time_energy_std", UNKNOWN),
            "PHYSICS_DERIVED",
            unit="zscore_column_std",
            note="spectrogram column-energy std; not m/s",
        )
        mi.motion_state = tagged("record_spectrogram", "METADATA", note="one CI4R record is one labeled action")

    extra = {}
    if sample.representation == "point_cloud":
        kin = evidence.get("kinematics", {})
        extra = {
            "mean_velocity": kin.get("mean_velocity", UNKNOWN),
            "median_velocity": kin.get("median_velocity", UNKNOWN),
            "peak_abs_velocity": kin.get("peak_abs_velocity", UNKNOWN),
            "positive_velocity_fraction": kin.get("positive_velocity_fraction", UNKNOWN),
            "negative_velocity_fraction": kin.get("negative_velocity_fraction", UNKNOWN),
            "mean_points_per_frame": kin.get("mean_points_per_frame", UNKNOWN),
            "motion_energy": kin.get("motion_energy", UNKNOWN),
            "velocity_std": kin.get("velocity_std", UNKNOWN),
        }
    evidence["motion_statistics"] = extra
    mi.evidence = evidence
    return mi
