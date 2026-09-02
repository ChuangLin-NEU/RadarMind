# Metric registry trace

Stored snapshot: `results/final_delivery/metric_snapshot_full.json`
Live rebuild: `scripts/paper_v2/snapshot.py` + `scripts/paper_v2/registry.py`

Manuscript keys checked: 28. Failures: 0.

| Key | Source file | Source exists | Stored display | Live display | Match |
|---|---|---|---|---|---|
| `MRI_SI_ACC` | `results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json` | True | 0.478 | 0.478 | PASS |
| `MRI_SI_F1` | `results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json` | True | 0.420 | 0.420 | PASS |
| `RADHAR_XYZV_ACC` | `results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json` | True | 0.949 | 0.949 | PASS |
| `RADHAR_XYZV_F1` | `results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json` | True | 0.949 | 0.949 | PASS |
| `R0_10_WITHIN` | `results/phase2_cross_radar/runs/within_10/metrics.json` | True | 0.843 | 0.843 | PASS |
| `R0_24_WITHIN` | `results/phase2_cross_radar/runs/within_24/metrics.json` | True | 0.898 | 0.898 | PASS |
| `R0_77_WITHIN` | `results/phase2_cross_radar/runs/within_77/metrics.json` | True | 0.922 | 0.922 | PASS |
| `R0_10_TO_24` | `results/phase2_cross_radar/runs/cross_10_to_24/metrics.json` | True | 0.143 | 0.143 | PASS |
| `R0_10_TO_77` | `results/phase2_cross_radar/runs/cross_10_to_77/metrics.json` | True | 0.373 | 0.373 | PASS |
| `R0_24_TO_10` | `results/phase2_cross_radar/runs/cross_24_to_10/metrics.json` | True | 0.353 | 0.353 | PASS |
| `R0_24_TO_77` | `results/phase2_cross_radar/runs/cross_24_to_77/metrics.json` | True | 0.039 | 0.039 | PASS |
| `R0_77_TO_10` | `results/phase2_cross_radar/runs/cross_77_to_10/metrics.json` | True | 0.275 | 0.275 | PASS |
| `R0_77_TO_24` | `results/phase2_cross_radar/runs/cross_77_to_24/metrics.json` | True | 0.163 | 0.163 | PASS |
| `M0_ACC` | `results/phase3_shared_motion/representation_metrics.csv` | True | 0.769 | 0.769 | PASS |
| `M1_ACC` | `results/phase3_shared_motion/representation_metrics.csv` | True | 0.932 | 0.932 | PASS |
| `M2_ACC` | `results/phase3_shared_motion/representation_metrics.csv` | True | 0.878 | 0.878 | PASS |
| `M1_RET` | `results/phase3_shared_motion/representation_metrics.csv` | True | 0.935 | 0.935 | PASS |
| `M1_RID` | `results/phase3_shared_motion/representation_metrics.csv` | True | 1.000 | 1.000 | PASS |
| `RID_R2D_3CLASS` | `results/phase8_physical_representation/answers.json` | True | 0.993 | 0.993 | PASS |
| `RID_R2V_2477` | `results/phase10_velocity_calibration/answers.json` | True | 0.939 | 0.939 | PASS |
| `CONT_FRAME_ACC` | `results/phase4_continuous/continuous_runs.csv` | True | 0.489 | 0.489 | PASS |
| `CONT_FRAME_F1` | `results/phase4_continuous/continuous_runs.csv` | True | 0.416 | 0.416 | PASS |
| `CONT_WIN_ACC` | `results/phase4_continuous/continuous_runs.csv` | True | 0.494 | 0.494 | PASS |
| `CONT_WIN_F1` | `results/phase4_continuous/continuous_runs.csv` | True | 0.419 | 0.419 | PASS |
| `CONT_SEG_F1` | `results/phase4_continuous/segment_metrics.csv` | True | 0.152 | 0.152 | PASS |
| `CONT_BND_SEC` | `results/phase4_continuous/boundary_metrics.csv` | True | 20.6 | 20.6 | PASS |
| `PROTO_AUROC` | `results/phase5_open_set/loao_prototype_per_action.json` | True | 0.628 | 0.628 | PASS |
| `PROTO_AUROC_STD` | `results/phase5_open_set/loao_prototype_per_action.json` | True | 0.125 | 0.125 | PASS |

Status: PASS
