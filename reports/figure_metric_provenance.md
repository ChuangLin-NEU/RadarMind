# Figure metric provenance

Every plotted scalar traces `figure → plotted item → metric key → source file → source field`.
Schematic geometry (box positions, arrows) is not a scientific metric.
Hard-coded scientific metrics in plotting code: forbidden.

| figure | plotted item | metric key | value | display | source file | source field | protocol | kind |
| ------ | ------------ | ---------- | ----- | ------- | ----------- | ------------ | -------- | ---- |
| Fig. 1 | retained framework |  |  | Fig. 1 not redrawn | C:/Users/13919/Desktop/RadarBridge/figures/main/fig01_radarmind_framework.png | unchanged | Fig. 1 not redrawn | frozen_file |
| Fig. 2 | paired acquisitions | N_PAIRED_ACQ | 327 | 327 | results/phase2_cross_radar/pairing_stats.json | paired_10_24_77_action_agreed | timestamp-paired 10/24/77 with agreed action | snapshot |
| Fig. 2 | mRI subjects | MRI_N_SUBJECTS | 20 | 20 | results/phase1_pointcloud/mri_summary.json | n_subjects | mRI point-cloud cache | snapshot |
| Fig. 2 | mRI actions | MRI_N_ACTIONS | 13 | 13 | results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json | per_class | normalized 13-class mRI taxonomy | snapshot |
| Fig. 2 | RadHAR actions | RADHAR_N_ACTIONS | 5 | 5 | results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json | per_class |  | snapshot |
| Fig. 2 | CI4R-77 random | CI4R77_RAND_ACC | 0.742268 | 0.742 | results/phase1_ci4r/baselines/random_record/resnet18/metrics.json | accuracy | CI4R 77 GHz 11-class random-record; not the 7-class paired split | snapshot |
| Fig. 2 | CI4R-77 prefix | CI4R77_PREFIX_ACC | 0.448276 | 0.448 | results/phase1_ci4r/baselines/prefix_tail4_disjoint/resnet18/metrics.json | accuracy | CI4R 77 GHz prefix-group-disjoint; not subject-independent | snapshot |
| Fig. 3 | mRI SI XYZ T=32 | MRI_SI_ACC | 0.478045 | 0.478 | results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json | accuracy | mRI subject-independent PointNet-GRU XYZ T=32 | snapshot |
| Fig. 3 | RadHAR official XYZV | RADHAR_XYZV_ACC | 0.949229 | 0.949 | results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json | accuracy | RadHAR official PointNet-GRU XYZV T=60 | snapshot |
| Fig. 3 | CI4R 10 GHz within | R0_10_WITHIN | 0.843137 | 0.843 | results/phase2_cross_radar/runs/within_10/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 3 | CI4R 24 GHz within | R0_24_WITHIN | 0.897959 | 0.898 | results/phase2_cross_radar/runs/within_24/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 3 | CI4R 77 GHz within | R0_77_WITHIN | 0.921569 | 0.922 | results/phase2_cross_radar/runs/within_77/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 3 | mRI T=1 | MRI_T1_ACC | 0.25435 | 0.254 | results/phase1_pointcloud/baselines/mri_si_pool_t1_xyzvi/metrics.json | accuracy | mRI SI T=1 PointNet-pool XYZVI | snapshot |
| Fig. 3 | mRI T=32 | MRI_SI_ACC | 0.478045 | 0.478 | results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json | accuracy | mRI subject-independent PointNet-GRU XYZ T=32 | snapshot |
| Fig. 3 | RadHAR T=1 | RADHAR_T1_ACC | 0.607293 | 0.607 | results/phase1_pointcloud/baselines/radhar_official_pool_t1_xyzvi/metrics.json | accuracy | RadHAR official T=1 PointNet-pool XYZVI | snapshot |
| Fig. 3 | RadHAR T=60 | RADHAR_XYZV_ACC | 0.949229 | 0.949 | results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json | accuracy | RadHAR official PointNet-GRU XYZV T=60 | snapshot |
| Fig. 4 | R0_10_WITHIN | R0_10_WITHIN | 0.843137 | 0.843 | results/phase2_cross_radar/runs/within_10/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_10_TO_24 | R0_10_TO_24 | 0.142857 | 0.143 | results/phase2_cross_radar/runs/cross_10_to_24/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_10_TO_77 | R0_10_TO_77 | 0.372549 | 0.373 | results/phase2_cross_radar/runs/cross_10_to_77/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_24_TO_10 | R0_24_TO_10 | 0.352941 | 0.353 | results/phase2_cross_radar/runs/cross_24_to_10/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_24_WITHIN | R0_24_WITHIN | 0.897959 | 0.898 | results/phase2_cross_radar/runs/within_24/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_24_TO_77 | R0_24_TO_77 | 0.0392157 | 0.039 | results/phase2_cross_radar/runs/cross_24_to_77/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_77_TO_10 | R0_77_TO_10 | 0.27451 | 0.275 | results/phase2_cross_radar/runs/cross_77_to_10/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_77_TO_24 | R0_77_TO_24 | 0.163265 | 0.163 | results/phase2_cross_radar/runs/cross_77_to_24/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 4 | R0_77_WITHIN | R0_77_WITHIN | 0.921569 | 0.922 | results/phase2_cross_radar/runs/within_77/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 5 | M0 activity | M0_ACC | 0.768707 | 0.769 | results/phase3_shared_motion/representation_metrics.csv | activity_accuracy | known-radar joint encoder; CE only | snapshot |
| Fig. 5 | M1 activity | M1_ACC | 0.931973 | 0.932 | results/phase3_shared_motion/representation_metrics.csv | activity_accuracy | known-radar joint encoder; CE+InfoNCE | snapshot |
| Fig. 5 | M2 activity | M2_ACC | 0.877551 | 0.878 | results/phase3_shared_motion/representation_metrics.csv | activity_accuracy | known-radar joint encoder; CE+InfoNCE+physics aug | snapshot |
| Fig. 5 | M0 retrieval | M0_RET | 0.717687 | 0.718 | results/phase3_shared_motion/representation_metrics.csv | retrieval_action_top1 | same-action retrieval top-1; mean over 6 directed pairs | snapshot |
| Fig. 5 | M1 retrieval | M1_RET | 0.935374 | 0.935 | results/phase3_shared_motion/representation_metrics.csv | retrieval_action_top1 |  | snapshot |
| Fig. 5 | M2 retrieval | M2_RET | 0.863946 | 0.864 | results/phase3_shared_motion/representation_metrics.csv | retrieval_action_top1 |  | snapshot |
| Fig. 5 | M0 Radar-ID | M0_RID | 1 | 1.000 | results/phase3_shared_motion/representation_metrics.csv | radar_probe_accuracy | 3-radar Radar-ID on shared embeddings | snapshot |
| Fig. 5 | M1 Radar-ID | M1_RID | 1 | 1.000 | results/phase3_shared_motion/representation_metrics.csv | radar_probe_accuracy |  | snapshot |
| Fig. 5 | M2 Radar-ID | M2_RID_SHARE | 1 | 1.000 | results/phase3_shared_motion/representation_metrics.csv | radar_probe_accuracy |  | snapshot |
| Fig. 6 | R0 Radar-ID | RID_R0_3CLASS | 1 | 1.000 | results/phase7_radar_invariance/representation_radar_id.json | 0.accuracy | 3-radar representation probe (10/24/77); logistic 16x16; not a HAR system | snapshot |
| Fig. 6 | R1 Radar-ID | RID_R1_3CLASS | 1 | 1.000 | results/phase7_radar_invariance/representation_radar_id.json | 1.accuracy | 3-radar representation probe (10/24/77); logistic 16x16; not a HAR system | snapshot |
| Fig. 6 | R2-D Radar-ID | RID_R2D_3CLASS | 0.993197 | 0.993 | results/phase8_physical_representation/answers.json | fingerprint.R2_D | 3-radar representation probe (10/24/77); logistic 16x16; not a HAR system | snapshot |
| Fig. 6 | R1 activity | ACT_R1_REP | 0.92517 | 0.925 | results/phase7_radar_invariance/answers.json | activity_R1_baseline | representation-level activity probe (same feature family as Radar-ID on that map) | snapshot |
| Fig. 6 | R2-D activity | ACT_R2D_REP | 0.945578 | 0.946 | results/phase8_physical_representation/answers.json | activity_probe_R2 | representation-level activity probe (same feature family as Radar-ID on that map) | snapshot |
| Fig. 6 | R2-D Radar-ID | RID_R2D_2477 | 1 | 1.000 | results/phase10_velocity_calibration/answers.json | REPRESENTATION.R2D_radar_id | 2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID | snapshot |
| Fig. 6 | R2-V Radar-ID | RID_R2V_2477 | 0.938776 | 0.939 | results/phase10_velocity_calibration/answers.json | REPRESENTATION.R2V_radar_id | 2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID | snapshot |
| Fig. 6 | U1 STFT Radar-ID | U1_RID | 0.938776 | 0.939 | results/phase10_unified_measurement/answers.json | U1.radar_id | 2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID; unified STFT | snapshot |
| Fig. 6 | U2 MTI Radar-ID | U2_RID | 0.979592 | 0.980 | results/phase10_unified_measurement/answers.json | U2.radar_id | 2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID; transplanted MTI | snapshot |
| Fig. 6 | D(v/q) cal. Radar-ID | CAL_RID_AFTER | 1 | 1.000 | results/phase10_motion_conditioned/answers.json | calibration.after.radar_id | 2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID; after applying D(v/q) as calibration | snapshot |
| Fig. 6 | R2-D activity | ACT_R2D_2477 | 0.877551 | 0.878 | results/phase10_velocity_calibration/answers.json | REPRESENTATION.R2D_activity | 24/77 activity probe on R2-D | snapshot |
| Fig. 6 | R2-V activity | ACT_R2V_2477 | 0.857143 | 0.857 | results/phase10_velocity_calibration/answers.json | REPRESENTATION.R2V_activity | 24/77 activity probe on R2-V | snapshot |
| Fig. 6 | U1 STFT activity | U1_ACT | 0.867347 | 0.867 | results/phase10_unified_measurement/answers.json | U1.activity | 24/77 activity probe; unified STFT | snapshot |
| Fig. 6 | U2 MTI activity | U2_ACT | 0.867347 | 0.867 | results/phase10_unified_measurement/answers.json | U2.activity | 24/77 activity probe; transplanted MTI | snapshot |
| Fig. 6 | D(v/q) cal. activity | CAL_ACT_AFTER | 0.857143 | 0.857 | results/phase10_motion_conditioned/answers.json | calibration.after.activity | 24/77 activity probe after calibration | snapshot |
| Fig. 7 | D(v/q), 24→77 | D2477_B1_R2 | 0.744187 | 0.744 | results/phase10_motion_conditioned/answers.json | 24to77.B1.r2 | velocity-marginal D(v/q) Ridge; 24→77; not a HAR score | snapshot |
| Fig. 7 | D(v/q), 77→24 | D7724_B1_R2 | 0.575155 | 0.575 | results/phase10_motion_conditioned/answers.json | 77to24.B1.r2 | velocity-marginal D(v/q) Ridge; 77→24 | snapshot |
| Fig. 7 | 2D TEST oracle | ORACLE_R2_2D | 0.418307 | 0.418 | results/phase10_conditional_2d_measurement/answers.json | oracle.24to77.r2 | TEST true Δ projected onto TRAIN PCA basis; oracle upper bound, not deployment | snapshot |
| Fig. 7 | B2 predicted, 24→77 | B2_R2_2477 | 0.300152 | 0.300 | results/phase10_conditional_2d_measurement/answers.json | transfer.24to77.B2.r2 | q-predicted 2D Δ; 24→77 | snapshot |
| Fig. 7 | B2 predicted, 77→24 | B2_R2_7724 | 0.172932 | 0.173 | results/phase10_conditional_2d_measurement/answers.json | transfer.77to24.B2.r2 | q-predicted 2D Δ; 77→24 | snapshot |
| Fig. 8 | GT |  |  |  | results/phase4_continuous/ground_truth_timelines/subject20_0513.json | events |  | frozen_file |
| Fig. 8 | raw windows |  |  |  | results/phase4_continuous/predicted_timelines/subject20_0513.json | modes.raw.windows |  | frozen_file |
| Fig. 8 | smoothed |  |  |  | results/phase4_continuous/predicted_timelines/subject20_0513.json | modes.confidence_aware.events |  | frozen_file |
| Fig. 8 | Raw | CONT_SEG_F1_RAW | 0.0261284 | 0.026 | results/phase4_continuous/segment_metrics.csv | segment_f1 |  | snapshot |
| Fig. 8 | Isolated flip | CONT_SEG_F1_FLIP | 0.103628 | 0.104 | results/phase4_continuous/segment_metrics.csv | segment_f1 |  | snapshot |
| Fig. 8 | Confidence-aware | CONT_SEG_F1 | 0.152252 | 0.152 | results/phase4_continuous/segment_metrics.csv | segment_f1 | labeled-interval segment F1; unweighted mean | snapshot |
| Fig. 8 | n windows | CONT_N_WINDOWS | 1207 | 1207 | results/phase4_continuous/continuous_runs.csv | n_windows | labeled-interval continuous replay; test subjects only | snapshot |
| Fig. 8 | boundary error s | CONT_BND_SEC | 20.6345 | 20.6 | results/phase4_continuous/boundary_metrics.csv | median_seconds |  | snapshot |
| Fig. 9 | MSP AUROC |  |  | 13 folds | results/phase5_open_set/loao_msp_per_action.json | AUROC | 13 folds | frozen_file |
| Fig. 9 | Energy AUROC |  |  | 13 folds | results/phase5_open_set/loao_energy_per_action.json | AUROC | 13 folds | frozen_file |
| Fig. 9 | Prototype AUROC |  |  | 13 folds | results/phase5_open_set/loao_prototype_per_action.json | AUROC | 13 folds | frozen_file |
| Fig. 9 | MSP_AUROC | MSP_AUROC | 0.475708 | 0.476 | results/phase5_open_set/loao_msp_per_action.json | AUROC | 13-fold LOAO + SI; MSP | snapshot |
| Fig. 9 | ENERGY_AUROC | ENERGY_AUROC | 0.539582 | 0.540 | results/phase5_open_set/loao_energy_per_action.json | AUROC | 13-fold LOAO + SI; Energy | snapshot |
| Fig. 9 | PROTO_AUROC | PROTO_AUROC | 0.627959 | 0.628 | results/phase5_open_set/loao_prototype_per_action.json | AUROC | 13-fold LOAO + SI; prototype distance | snapshot |
| Fig. 9 | MSP_FPR | MSP_FPR | 0.903084 | 0.903 | results/phase5_open_set/loao_msp_per_action.json | FPR95 |  | snapshot |
| Fig. 9 | ENERGY_FPR | ENERGY_FPR | 0.850177 | 0.850 | results/phase5_open_set/loao_energy_per_action.json | FPR95 |  | snapshot |
| Fig. 9 | PROTO_FPR | PROTO_FPR | 0.791049 | 0.791 | results/phase5_open_set/loao_prototype_per_action.json | FPR95 |  | snapshot |
| Fig. 9 | MSP FPR95 | MSP_FPR | 0.903084 | 0.903 | results/phase5_open_set/loao_msp_per_action.json | FPR95 |  | snapshot |
| Fig. 9 | Energy FPR95 | ENERGY_FPR | 0.850177 | 0.850 | results/phase5_open_set/loao_energy_per_action.json | FPR95 |  | snapshot |
| Fig. 9 | Prototype FPR95 | PROTO_FPR | 0.791049 | 0.791 | results/phase5_open_set/loao_prototype_per_action.json | FPR95 |  | snapshot |
| Fig. 10 | MI JSON |  |  |  | results/radarmind_v01/llm_online_test.json | records.0 |  | frozen_file |
| Fig. 10 | BOX_CONF | BOX_CONF | 0.999253 | 0.999 | results/radarmind_v01/llm_online_test.json | records.0.input.motion_intelligence.confidence.value | RadHAR official test boxing demo | snapshot |
| Fig. 10 | BOX_UNC | BOX_UNC | 0.00397142 | 0.004 | results/radarmind_v01/llm_online_test.json | records.0.input.motion_intelligence.uncertainty.value |  | snapshot |
| Fig. 10 | BOX_NWIN | BOX_NWIN | 24 | 24 | results/radarmind_v01/llm_online_test.json | records.0.input.motion_timeline.events.0.n_windows |  | snapshot |
| Fig. 10 | BOX_PEAK_V | BOX_PEAK_V | 3.07334 | 3.073 | results/radarmind_v01/llm_online_test.json | records.0.input.motion_intelligence.peak_velocity.value |  | snapshot |
| Fig. 10 | BOX_ENERGY | BOX_ENERGY | 0.176181 | 0.176 | results/radarmind_v01/llm_online_test.json | records.0.input.motion_intelligence.motion_intensity.value |  | snapshot |
| Fig. 10 | LLM_N | LLM_N | 4 | 4 | results/radarmind_v01/llm_online_test.json | n | DeepSeek online tasks on structured MI+Timeline | snapshot |
| Fig. 10 | LLM_N_OK | LLM_N_OK | 4 | 4 | results/radarmind_v01/llm_online_test.json | n_ok |  | snapshot |
| Fig. 11 | mRI | MRI_SI_ACC | 0.478045 | 0.478 | results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json | accuracy | mRI subject-independent PointNet-GRU XYZ T=32 | snapshot |
| Fig. 11 | RadHAR | RADHAR_XYZV_ACC | 0.949229 | 0.949 | results/phase1_pointcloud/baselines/radhar_official_gru_xyzv/metrics.json | accuracy | RadHAR official PointNet-GRU XYZV T=60 | snapshot |
| Fig. 11 | SI | MRI_SI_ACC | 0.478045 | 0.478 | results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json | accuracy | mRI subject-independent PointNet-GRU XYZ T=32 | snapshot |
| Fig. 11 | 24to77 | R0_24_TO_77 | 0.0392157 | 0.039 | results/phase2_cross_radar/runs/cross_24_to_77/metrics.json | accuracy | official plotted PNG; source-only 7-class HAR; acquisition-timestamp split | snapshot |
| Fig. 11 | seg | CONT_SEG_F1 | 0.152252 | 0.152 | results/phase4_continuous/segment_metrics.csv | segment_f1 | labeled-interval segment F1; unweighted mean | snapshot |
| Fig. 11 | proto | PROTO_AUROC | 0.627959 | 0.628 | results/phase5_open_set/loao_prototype_per_action.json | AUROC | 13-fold LOAO + SI; prototype distance | snapshot |
| Fig. 11 | ok | LLM_N_OK | 4 | 4 | results/radarmind_v01/llm_online_test.json | n_ok |  | snapshot |
| Fig. 11 | n | LLM_N | 4 | 4 | results/radarmind_v01/llm_online_test.json | n | DeepSeek online tasks on structured MI+Timeline | snapshot |
