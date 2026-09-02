# Checkpoints

`*.pt` / `*.pth` / `*.ckpt` files are **not included in git**.

SHA256 and byte sizes for the v0.2 freeze are in `results/frozen_2026_08/artifact_manifest.csv`. Download URLs are not defined.

| Group | Experiment | Expected file (local) | Approx. size | Training? | Inference? | Figure reproduction? | Distribution |
|---|---|---|---|---|---|---|---|
| CI4R ResNet18 random | 11-class random-record | `results/phase1_ci4r/baselines/random_record/resnet18/best.pt` | ~43 MB | rebuild only | app / eval | no (metrics suffice) | NOT IN GIT; PRIVATE RELEASE / EXTERNAL STORAGE TBD |
| CI4R ResNet18 prefix | prefix-group-disjoint | `results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/best.pt` | ~43 MB | rebuild only | app / eval | no | same |
| mRI PointNet-GRU SI | subject-independent | `results/phase1_pointcloud/baselines/mri_si_gru_xyz/best.pt` | ~0.5 MB | rebuild only | app / continuous / LOAO | no | same |
| RadHAR PointNet-GRU | official XYZV | `results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/best.pt` | ~0.5 MB | rebuild only | app | no | same |
| RadHAR voxel CNN-GRU | official voxels | `results/phase1_pointcloud/baselines/radhar_official_voxel_cnn_gru/best.pt` | small / optional | rebuild only | optional app | no | same |
| Phase 2 within/cross | R0 HAR | `results/phase2_cross_radar/runs/*/best.pt` | ~43 MB each | rebuild only | optional | no | same |
| Phase 3 M0/M1/M2 | shared representation | `results/phase3_shared_motion/runs/*/best.pt` | ~43 MB each | rebuild only | probes | no | same |
| Phase 5 LOAO | 13 folds | `results/phase5_open_set/folds/*/best.pt` | ~0.5 MB each | rebuild only | open-set eval | no | same |
| Phase 7 M3 / R1 | GRL / R1 HAR | `results/phase7_radar_invariance/runs/*/best.pt` | ~43 MB each | rebuild only | probes | no | same |
| Phase 9 A0–A2 | adapters | `results/phase9_radarbridge_v2/checkpoints/*/best.pt` | ~43 MB each | rebuild only | adapter eval | no | same |

Table and figure rebuilders read JSON/CSV metrics, not these weights.

Creating a `checkpoints/` directory is not required and would not match the paths above. Keep weights in the existing `results/**` locations on the machines that have them.
