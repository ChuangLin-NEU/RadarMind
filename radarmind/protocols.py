"""Cross-domain protocol API. Unavailable protocols stay available=false."""
from __future__ import annotations

PROTOCOLS = {
    "within_distribution": {
        "available": True,
        "datasets": ["ci4r", "mri"],
        "aliases": ["random_record", "random_segment"],
        "result_files": {
            "ci4r": "results/phase1_ci4r/baselines/random_record/resnet18/metrics.json",
            "mri": "results/phase1_pointcloud/baselines/mri_random_gru_xyzvi/metrics.json",
        },
        "reason": None,
        "note": "CI4R random-record; mRI random-segment. Not subject-independent.",
    },
    "subject_independent": {
        "available": True,
        "datasets": ["mri"],
        "aliases": [],
        "result_files": {
            "mri": "results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json",
        },
        "reason": None,
        "note": "mRI only. CI4R has no official subject IDs; do not call prefix-group-disjoint subject-independent.",
    },
    "prefix_group_disjoint": {
        "available": True,
        "datasets": ["ci4r"],
        "aliases": ["prefix_tail4_disjoint"],
        "result_files": {
            "ci4r": "results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json",
        },
        "reason": None,
        "note": "Filename tail4 acquisition-group split. Not subject-independent. Not cross-subject.",
    },
    "official_split": {
        "available": True,
        "datasets": ["radhar"],
        "aliases": ["official_train_test"],
        "result_files": {
            "radhar": "results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json",
        },
        "reason": None,
        "note": "RadHAR official Train/Test folders. No subject metadata.",
    },
    "cross_radar": {
        "available": True,
        "datasets": ["ci4r10", "ci4r24", "ci4r77"],
        "aliases": [],
        "result_files": {
            "training_runs": "results/phase2_cross_radar/training_runs.csv",
        },
        "reason": None,
        "note": "CI4R official spectrogram PNG, 7-class timestamp-agreed subset, seed=42. prefix_group is not subject.",
    },
    "cross_environment": {
        "available": False,
        "datasets": [],
        "aliases": [],
        "result_files": {},
        "reason": "No validated environment metadata in current local radar datasets.",
        "note": "interface only; not a failed experiment",
    },
    "cross_dataset": {
        "available": False,
        "datasets": [],
        "aliases": [],
        "result_files": {},
        "reason": "label spaces and radar representations differ; no shared motion model trained",
        "note": "interface only",
    },
}


class ProtocolRegistry:
    def names(self) -> list[str]:
        return list(PROTOCOLS)

    def get(self, name: str) -> dict:
        key = name.strip().lower()
        if key in PROTOCOLS:
            return {"name": key, **PROTOCOLS[key]}
        for canon, spec in PROTOCOLS.items():
            if key in spec.get("aliases", []):
                return {"name": canon, **spec}
        raise KeyError(name)

    def available(self) -> list[str]:
        return [k for k, v in PROTOCOLS.items() if v["available"]]

    def unavailable(self) -> list[dict]:
        return [{"name": k, "reason": v["reason"]} for k, v in PROTOCOLS.items() if not v["available"]]

    def as_dict(self) -> dict:
        return {k: dict(v) for k, v in PROTOCOLS.items()}
