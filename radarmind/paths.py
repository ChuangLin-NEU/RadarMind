"""Resolved paths. Application reads processed cache only."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PHASE1A = RESULTS / "phase1_ci4r"
PHASE1B = RESULTS / "phase1_pointcloud"
SPLITS = ROOT / "data_splits"
V01 = RESULTS / "radarmind_v01"

CI4R_CACHE = PHASE1A / "cache" / "ci4r77_airhar_spec_zscore.npz"
MRI_CACHE = PHASE1B / "cache" / "mri_windows.npz"
RADHAR_CACHE = PHASE1B / "cache" / "radhar_windows.npz"
RADHAR_VOXEL_CACHE = PHASE1B / "cache" / "radhar_voxels.npz"

CI4R_RANDOM_SPLIT = SPLITS / "ci4r_random.csv"
CI4R_PREFIX_SPLIT = SPLITS / "ci4r_subject_independent.csv"
MRI_RANDOM_SPLIT = SPLITS / "mri_random_segment.csv"
MRI_SI_SPLIT = SPLITS / "mri_subject_independent.csv"
RADHAR_SPLIT = SPLITS / "radhar_official.csv"

CI4R_RESNET_RANDOM = PHASE1A / "baselines" / "random_record" / "resnet18" / "best.pt"
CI4R_RESNET_PREFIX = PHASE1A / "baselines" / "prefix_tail4_disjoint" / "resnet18" / "best.pt"
MRI_GRU_SI_XYZ = PHASE1B / "baselines" / "mri_si_gru_xyz" / "best.pt"
MRI_GRU_RANDOM_XYZVI = PHASE1B / "baselines" / "mri_random_gru_xyzvi" / "best.pt"
RADHAR_GRU_XYZV = PHASE1B / "baselines" / "radhar_official_gru_xyzv" / "best.pt"
RADHAR_VOXEL = PHASE1B / "baselines" / "radhar_official_voxel_cnn_gru" / "best.pt"

PHASE1B_RUNS = PHASE1B / "training_runs.csv"
PHASE1A_SUMMARY = PHASE1A / "baselines" / "baseline_summary.json"
PHASE1A_RANDOM_METRICS = PHASE1A / "baselines" / "random_record" / "resnet18" / "metrics.json"
PHASE1A_PREFIX_METRICS = PHASE1A / "baselines" / "prefix_tail4_disjoint" / "resnet18" / "metrics.json"
