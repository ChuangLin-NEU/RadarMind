"""Metric registry: every paper number must resolve from an existing result file."""
from __future__ import annotations

from typing import Any

# kind:
#   json          source_field dotted path
#   csv_row       match dict + field
#   csv_count     count rows where field==value
#   json_len      len(dotted path)
#   derived_sub   value = metrics[a] - metrics[b]
#   derived_div   value = metrics[a] / metrics[b]  (Hz → GHz etc.)


def spec(
    key: str,
    source_file: str,
    source_field: str,
    *,
    kind: str = "json",
    match: dict | None = None,
    format_rule: str = "f3",
    protocol: str = "",
    compare_group: str = "",
    note: str = "",
    a: str = "",
    b: str = "",
    scale: float | None = None,
) -> dict[str, Any]:
    return {
        "key": key,
        "source_file": source_file,
        "source_field": source_field,
        "kind": kind,
        "match": match or {},
        "format_rule": format_rule,
        "protocol": protocol,
        "compare_group": compare_group,
        "note": note,
        "a": a,
        "b": b,
        "scale": scale,
    }


P1PC = "results/phase1_pointcloud"
P1CI = "results/phase1_ci4r"
P2 = "results/phase2_cross_radar"
P3 = "results/phase3_shared_motion"
P4 = "results/phase4_continuous"
P5 = "results/phase5_open_set"
P7 = "results/phase7_radar_invariance"
P8 = "results/phase8_physical_representation"
P9 = "results/phase9_radarbridge_v2"
P10A = "results/phase10_motion_factors"
P10B = "results/phase10_sensor_calibration"
P10C = "results/phase10_velocity_calibration"
P10D = "results/phase10_unified_measurement"
P10E = "results/phase10_hardware_measurement"
P10F = "results/phase10_motion_conditioned"
P10G = "results/phase10_conditional_2d_measurement"
RM01 = "results/radarmind_v01"
RM02 = "results/radarmind_v02"
SPLIT = "data_splits/ci4r_shared_motion_acquisition_split.csv"

R0_HAR = "official plotted PNG; source-only 7-class HAR; acquisition-timestamp split"
R1_HAR = "cropped plotted PNG; source-only 7-class HAR; acquisition-timestamp split"
RID3 = "3-radar representation probe (10/24/77); logistic 16x16; not a HAR system"
RID2 = "2-radar 24/77 representation probe; not comparable to 3-radar Radar-ID"
ACT_REP = "representation-level activity probe (same feature family as Radar-ID on that map)"
ACT_ENC = "known-radar joint encoder HAR accuracy; not cross-radar zero-shot"
ACT_FAC = "linear probe on interpretable motion-factor vectors"
UDA = "unsupervised domain adaptation (CORAL); not domain generalization"
DG = "leave-one-radar-out domain generalization"

METRICS: list[dict[str, Any]] = [
    spec("N_PAIRED_ACQ", f"{P2}/pairing_stats.json", "paired_10_24_77_action_agreed", format_rule="int", protocol="timestamp-paired 10/24/77 with agreed action"),
    spec("N_CONFLICT", f"{P2}/pairing_stats.json", "paired_10_24_77_action_conflict", format_rule="int", protocol="timestamp-paired triples with conflicting action labels; excluded"),
    spec("N_ACTIONS", f"{P2}/pairing_stats.json", "common_actions_agreed", kind="json_len", format_rule="int"),
    spec("SPLIT_TRAIN", SPLIT, "split", kind="csv_count", match={"value": "train"}, format_rule="int", protocol="acquisition-timestamp shared-motion split"),
    spec("SPLIT_VAL", SPLIT, "split", kind="csv_count", match={"value": "val"}, format_rule="int"),
    spec("SPLIT_TEST", SPLIT, "split", kind="csv_count", match={"value": "test"}, format_rule="int"),
    spec("N_FACTORS", f"{P10A}/answers.json", "n_factors", format_rule="int"),
    spec("K_SELECTED", f"{P10G}/answers.json", "selected_k.24to77", format_rule="int"),
    spec("FC_24_HZ", f"{P8}/answers.json", "header_24ghz.fc_hz", format_rule="sci"),
    spec("FC_24_GHZ", "", "", kind="derived_div", a="FC_24_HZ", b="", scale=1e9, format_rule="f2", note="Hz/1e9 from header"),
    spec("FC_77_GHZ", f"{P10C}/answers.json", "PHYSICS.fc_77", format_rule="str"),
    spec("VEL_LO", f"{P10C}/answers.json", "PHYSICS.common_velocity.0", format_rule="f3"),
    spec("VEL_HI", f"{P10C}/answers.json", "PHYSICS.common_velocity.1", format_rule="f3"),
    # Official representation benchmark (R0)
    spec("R0_10_WITHIN", f"{P2}/runs/within_10/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_24_WITHIN", f"{P2}/runs/within_24/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_77_WITHIN", f"{P2}/runs/within_77/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_10_TO_24", f"{P2}/runs/cross_10_to_24/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_10_TO_77", f"{P2}/runs/cross_10_to_77/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_24_TO_10", f"{P2}/runs/cross_24_to_10/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_24_TO_77", f"{P2}/runs/cross_24_to_77/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_77_TO_10", f"{P2}/runs/cross_77_to_10/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("R0_77_TO_24", f"{P2}/runs/cross_77_to_24/metrics.json", "accuracy", protocol=R0_HAR, compare_group="r0_har"),
    spec("CORAL_77_TO_24", f"{P2}/runs/coral_uda_77_to_24/metrics.json", "accuracy", protocol=UDA, compare_group="coral_uda"),
    spec("CORAL_77_TO_10", f"{P2}/runs/coral_uda_77_to_10/metrics.json", "accuracy", protocol=UDA, compare_group="coral_uda"),
    # Representation normalization (R1)
    spec("R1_10_WITHIN", f"{P7}/runs/r1_within_10/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_24_WITHIN", f"{P7}/runs/r1_within_24/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_77_WITHIN", f"{P7}/runs/r1_within_77/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_10_TO_24", f"{P7}/runs/r1_cross_10_to_24/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_10_TO_77", f"{P7}/runs/r1_cross_10_to_77/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_24_TO_10", f"{P7}/runs/r1_cross_24_to_10/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_24_TO_77", f"{P7}/runs/r1_cross_24_to_77/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_77_TO_10", f"{P7}/runs/r1_cross_77_to_10/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("R1_77_TO_24", f"{P7}/runs/r1_cross_77_to_24/metrics.json", "accuracy", protocol=R1_HAR, compare_group="r1_har"),
    spec("RID_R0_3CLASS", f"{P7}/representation_radar_id.json", "0.accuracy", protocol=RID3, compare_group="rid_3class", note="index 0 is R0"),
    spec("RID_R1_3CLASS", f"{P7}/representation_radar_id.json", "1.accuracy", protocol=RID3, compare_group="rid_3class", note="index 1 is R1"),
    spec("ACT_R1_REP", f"{P7}/answers.json", "activity_R1_baseline", protocol=ACT_REP, compare_group="act_r1_rep"),
    spec("LORO_1024_TO_77", f"{P7}/runs/R1_M0_loro_10_24_to_77/metrics.json", "activity_accuracy", protocol=DG, compare_group="loro"),
    spec("LORO_1077_TO_24", f"{P7}/answers.json", "loro.10+77→24.baseline", protocol=DG, compare_group="loro"),
    spec("LORO_2477_TO_10", f"{P7}/answers.json", "loro.24+77→10.baseline", protocol=DG, compare_group="loro"),
    # Numerical Doppler
    spec("RID_R2D_3CLASS", f"{P8}/answers.json", "fingerprint.R2_D", protocol=RID3, compare_group="rid_3class"),
    spec("ACT_R2D_REP", f"{P8}/answers.json", "activity_probe_R2", protocol=ACT_REP, compare_group="act_r2d_rep"),
    spec("R2D_SAME_COS", f"{P8}/answers.json", "similarity.R2_D_same", protocol="paired cosine on R2-D maps"),
    spec("R2D_DIFF_COS", f"{P8}/answers.json", "similarity.R2_D_diff", protocol="unpaired cosine on R2-D maps"),
    spec("R2D_W10", f"{P8}/r2_training_runs.csv", "accuracy", kind="csv_row", match={"run_id": "r2_within_10"}, protocol="R2-D source-only 7-class HAR", compare_group="r2d_har"),
    spec("R2D_W24", f"{P8}/r2_training_runs.csv", "accuracy", kind="csv_row", match={"run_id": "r2_within_24"}, protocol="R2-D source-only 7-class HAR", compare_group="r2d_har"),
    spec("R2D_W77", f"{P8}/r2_training_runs.csv", "accuracy", kind="csv_row", match={"run_id": "r2_within_77"}, protocol="R2-D source-only 7-class HAR", compare_group="r2d_har"),
    spec("NATIVE_N_10", f"{P8}/answers.json", "native.10GHz", format_rule="int"),
    spec("NATIVE_N_24", f"{P8}/answers.json", "native.24GHz", format_rule="int"),
    spec("NATIVE_N_77", f"{P8}/answers.json", "native.77GHz", format_rule="int"),
    # Learned alignment (known-radar)
    spec("A0_ACC", f"{P9}/answers.json", "A0_acc", protocol=ACT_ENC, compare_group="adapter_har"),
    spec("A1_ACC", f"{P9}/answers.json", "A1_acc", protocol=ACT_ENC, compare_group="adapter_har"),
    spec("A2_ACC", f"{P9}/answers.json", "A2_acc", protocol=ACT_ENC, compare_group="adapter_har"),
    spec("RID_A0_INPUT", f"{P9}/answers.json", "radar_id_input", protocol="Radar-ID on known-radar encoder input maps", compare_group="rid_encoder"),
    spec("RID_A1_ADAPTER", f"{P9}/answers.json", "radar_id_adapter", protocol="Radar-ID after radar-specific adapters", compare_group="rid_encoder"),
    spec("RID_A2_EMBED", f"{P9}/answers.json", "radar_id_embedding", protocol="Radar-ID on A2 embeddings", compare_group="rid_encoder"),
    spec("ACT_A1_PROBE", f"{P9}/answers.json", "activity_adapter", protocol=ACT_ENC),
    spec("ACT_A2_PROBE", f"{P9}/answers.json", "activity_embedding", protocol=ACT_ENC),
    # Motion-factor audit
    spec("FAC_RID", f"{P10A}/answers.json", "probe_all_radar", protocol=ACT_FAC + "; Radar-ID", compare_group="rid_factors"),
    spec("FAC_ACT", f"{P10A}/answers.json", "probe_all_activity", protocol=ACT_FAC, compare_group="act_factors"),
    spec("FAC_RID_STABLE", f"{P10A}/answers.json", "probe_stable_radar", protocol=ACT_FAC + "; stable subset Radar-ID", compare_group="rid_factors"),
    spec("FAC_ACT_STABLE", f"{P10A}/answers.json", "probe_stable_activity", protocol=ACT_FAC + "; stable subset", compare_group="act_factors"),
    # Sensor calibration
    spec("C0_RID", f"{P10B}/answers.json", "c0_radar", protocol="known-radar calibration; 3-radar Radar-ID", compare_group="rid_3class"),
    spec("C0_ACT", f"{P10B}/answers.json", "c0_activity", protocol="known-radar calibration activity probe"),
    spec("CBEST_RID", f"{P10B}/answers.json", "best_radar", protocol="best of C0–C4 on VAL gate; still known-radar"),
    spec("CBEST_ACT", f"{P10B}/answers.json", "best_activity", protocol="best of C0–C4 activity probe"),
    # Velocity-domain alignment (24/77 only)
    spec("RID_R2D_2477", f"{P10C}/answers.json", "REPRESENTATION.R2D_radar_id", protocol=RID2, compare_group="rid_2class"),
    spec("RID_R2V_2477", f"{P10C}/answers.json", "REPRESENTATION.R2V_radar_id", protocol=RID2, compare_group="rid_2class"),
    spec("ACT_R2D_2477", f"{P10C}/answers.json", "REPRESENTATION.R2D_activity", protocol="24/77 activity probe on R2-D", compare_group="act_2477"),
    spec("ACT_R2V_2477", f"{P10C}/answers.json", "REPRESENTATION.R2V_activity", protocol="24/77 activity probe on R2-V", compare_group="act_2477"),
    spec("PAIR_R2D_SAME", f"{P10C}/answers.json", "PAIR_ALIGNMENT.R2D.same", protocol="24/77 pair cosine"),
    spec("PAIR_R2D_DIFF", f"{P10C}/answers.json", "PAIR_ALIGNMENT.R2D.different", protocol="24/77 unpaired cosine"),
    spec("PAIR_R2V_SAME", f"{P10C}/answers.json", "PAIR_ALIGNMENT.R2V.same", protocol="24/77 pair cosine"),
    spec("PAIR_R2V_DIFF", f"{P10C}/answers.json", "PAIR_ALIGNMENT.R2V.different", protocol="24/77 unpaired cosine"),
    spec("SEP_R2D", "", "", kind="derived_sub", a="PAIR_R2D_SAME", b="PAIR_R2D_DIFF", protocol="same minus different cosine"),
    spec("SEP_R2V", "", "", kind="derived_sub", a="PAIR_R2V_SAME", b="PAIR_R2V_DIFF", protocol="same minus different cosine"),
    # Signal-processing harmonization
    spec("U0_RID", f"{P10D}/answers.json", "U0.radar_id", protocol=RID2, compare_group="rid_2class"),
    spec("U0_ACT", f"{P10D}/answers.json", "U0.activity", protocol="24/77 activity probe", compare_group="act_2477"),
    spec("U1_RID", f"{P10D}/answers.json", "U1.radar_id", protocol=RID2 + "; unified STFT", compare_group="rid_2class"),
    spec("U1_ACT", f"{P10D}/answers.json", "U1.activity", protocol="24/77 activity probe; unified STFT", compare_group="act_2477"),
    spec("U1_SEP", f"{P10D}/answers.json", "U1.sep_cosine", protocol="24/77 pair separation"),
    spec("U2_RID", f"{P10D}/answers.json", "U2.radar_id", protocol=RID2 + "; transplanted MTI", compare_group="rid_2class"),
    spec("U2_ACT", f"{P10D}/answers.json", "U2.activity", protocol="24/77 activity probe; transplanted MTI", compare_group="act_2477"),
    spec("U2_SEP", f"{P10D}/answers.json", "U2.sep_cosine", protocol="24/77 pair separation"),
    # First-order measurement maps
    spec("M2_RID", f"{P10E}/answers.json", "M2.radar_id", protocol=RID2, compare_group="rid_2class"),
    spec("M2_VEL_RID", f"{P10E}/answers.json", "M2.vel_rid", protocol="velocity-marginal 24/77 Radar-ID"),
    spec("M2_ACT", f"{P10E}/answers.json", "M2.activity", protocol="24/77 activity probe after M2", compare_group="act_2477"),
    spec("TRANSFER_STAB", f"{P10E}/answers.json", "transfer_stability_score", protocol="action-independence of empirical D(v)"),
    spec("ETA2_ACTION", f"{P10E}/answers.json", "mean_eta2_action", protocol="action eta-squared of D(v)"),
    spec("N_VALID_BINS", f"{P10E}/answers.json", "n_valid_bins", format_rule="int"),
    # Motion-conditioned response
    spec("D2477_B0_R2", f"{P10F}/answers.json", "24to77.B0.r2", protocol="velocity-marginal D(v); train-mean baseline; 24→77"),
    spec("D2477_B1_R2", f"{P10F}/answers.json", "24to77.B1.r2", protocol="velocity-marginal D(v|q) Ridge; 24→77; not a HAR score"),
    spec("D2477_B1_MAE", f"{P10F}/answers.json", "24to77.B1.mae", protocol="velocity-marginal D(v|q) Ridge; 24→77"),
    spec("D2477_B1_CORR", f"{P10F}/answers.json", "24to77.B1.corr", protocol="velocity-marginal D(v|q) Ridge; 24→77"),
    spec("D7724_B0_R2", f"{P10F}/answers.json", "77to24.B0.r2", protocol="velocity-marginal D(v); train-mean baseline; 77→24"),
    spec("D7724_B1_R2", f"{P10F}/answers.json", "77to24.B1.r2", protocol="velocity-marginal D(v|q) Ridge; 77→24"),
    spec("D7724_B1_MAE", f"{P10F}/answers.json", "77to24.B1.mae", protocol="velocity-marginal D(v|q) Ridge; 77→24"),
    spec("D7724_B1_CORR", f"{P10F}/answers.json", "77to24.B1.corr", protocol="velocity-marginal D(v|q) Ridge; 77→24"),
    spec("CAL_RID_BEFORE", f"{P10F}/answers.json", "calibration.before.radar_id", protocol=RID2 + "; before applying D(v|q)"),
    spec("CAL_RID_AFTER", f"{P10F}/answers.json", "calibration.after.radar_id", protocol=RID2 + "; after applying D(v|q) as calibration"),
    spec("CAL_ACT_BEFORE", f"{P10F}/answers.json", "calibration.before.activity", protocol="24/77 activity probe before calibration"),
    spec("CAL_ACT_AFTER", f"{P10F}/answers.json", "calibration.after.activity", protocol="24/77 activity probe after calibration"),
    spec("CAL_SEP_BEFORE", f"{P10F}/answers.json", "calibration.sep_before", protocol="pair separation before calibration"),
    spec("CAL_SEP_AFTER", f"{P10F}/answers.json", "calibration.sep_after", protocol="pair separation after calibration"),
    # Low-rank 2D
    spec("K20_TRAIN_VAR", f"{P10G}/answers.json", "train_explained_var.24to77", protocol="TRAIN PCA explained variance of 2D Δ; not TEST performance"),
    spec("ORACLE_R2_2D", f"{P10G}/answers.json", "oracle.24to77.r2", protocol="TEST true Δ projected onto TRAIN PCA basis; oracle upper bound, not deployment"),
    spec("B2_R2_2477", f"{P10G}/answers.json", "transfer.24to77.B2.r2", protocol="q-predicted 2D Δ; 24→77"),
    spec("B2_R2_7724", f"{P10G}/answers.json", "transfer.77to24.B2.r2", protocol="q-predicted 2D Δ; 77→24"),
    spec("B0_2D_2477", f"{P10G}/answers.json", "transfer.24to77.B0.r2", protocol="train-mean 2D Δ; 24→77"),
    spec("B1_2D_2477", f"{P10G}/answers.json", "transfer.24to77.B1.r2", protocol="tiled D(v) as 2D field; 24→77"),
    spec("B2_ACT_SRC", f"{P10G}/activity_preservation.csv", "activity", kind="csv_row", match={"direction": "24to77", "item": "source"}, protocol="24/77 activity probe on source maps"),
    spec("B2_ACT_PRED", f"{P10G}/activity_preservation.csv", "activity", kind="csv_row", match={"direction": "24to77", "item": "B2_predicted"}, protocol="24/77 activity probe on B2-predicted maps"),
    # mRI / RadHAR / CI4R-77 baselines
    spec("MRI_N_SUBJECTS", f"{P1PC}/mri_summary.json", "n_subjects", format_rule="int", protocol="mRI point-cloud cache"),
    spec("MRI_N_SEGMENTS", f"{P1PC}/mri_summary.json", "n_segments", format_rule="int"),
    spec("MRI_N_FRAMES", f"{P1PC}/mri_summary.json", "n_frames_manifest", format_rule="int"),
    spec("MRI_T", f"{P1PC}/mri_summary.json", "selected_T", format_rule="int"),
    spec("MRI_P", f"{P1PC}/mri_summary.json", "selected_P", format_rule="int"),
    spec("MRI_N_ACTIONS", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "per_class", kind="json_len", format_rule="int", protocol="normalized 13-class mRI taxonomy"),
    spec("MRI_SI_N_TRAIN", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "n_train", format_rule="int", protocol="true subject-independent; official subject IDs"),
    spec("MRI_SI_N_VAL", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "n_val", format_rule="int"),
    spec("MRI_SI_N_TEST", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "n_test", format_rule="int"),
    spec("MRI_SI_ACC", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "accuracy", protocol="mRI subject-independent PointNet-GRU XYZ T=32"),
    spec("MRI_SI_F1", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "macro_f1", protocol="mRI subject-independent PointNet-GRU XYZ T=32"),
    spec("MRI_TPOSE_F1", f"{P1PC}/baselines/mri_si_gru_xyz/metrics.json", "per_class.0.f1", protocol="mRI SI per-class F1; class 0 is T pose"),
    spec("MRI_RAND_ACC", f"{P1PC}/baselines/mri_random_gru_xyzvi/metrics.json", "accuracy", protocol="mRI random-segment PointNet-GRU XYZVI T=32; not subject-independent"),
    spec("MRI_RAND_F1", f"{P1PC}/baselines/mri_random_gru_xyzvi/metrics.json", "macro_f1", protocol="mRI random-segment PointNet-GRU XYZVI T=32"),
    spec("MRI_T1_ACC", f"{P1PC}/baselines/mri_si_pool_t1_xyzvi/metrics.json", "accuracy", protocol="mRI SI T=1 PointNet-pool XYZVI"),
    spec("MRI_T1_F1", f"{P1PC}/baselines/mri_si_pool_t1_xyzvi/metrics.json", "macro_f1", protocol="mRI SI T=1 PointNet-pool XYZVI"),
    spec("MRI_XYZV_ACC", f"{P1PC}/baselines/mri_si_gru_xyzv/metrics.json", "accuracy", protocol="mRI SI PointNet-GRU XYZV T=32"),
    spec("MRI_XYZVI_ACC", f"{P1PC}/baselines/mri_si_gru_xyzvi/metrics.json", "accuracy", protocol="mRI SI PointNet-GRU XYZVI T=32"),
    spec("MRI_XYZVI_F1", f"{P1PC}/baselines/mri_si_gru_xyzvi/metrics.json", "macro_f1", protocol="mRI SI PointNet-GRU XYZVI T=32"),
    spec("RADHAR_N_REC", f"{P1PC}/radhar_summary.json", "n_recordings", format_rule="int"),
    spec("RADHAR_T", f"{P1PC}/radhar_summary.json", "selected_T", format_rule="int"),
    spec("RADHAR_P", f"{P1PC}/radhar_summary.json", "selected_P", format_rule="int"),
    spec("RADHAR_N_ACTIONS", f"{P1PC}/baselines/radhar_official_gru_xyzv/metrics.json", "per_class", kind="json_len", format_rule="int"),
    spec("RADHAR_N_TEST", f"{P1PC}/baselines/radhar_official_gru_xyzv/metrics.json", "n_test", format_rule="int", protocol="RadHAR official train/test; not subject-independent"),
    spec("RADHAR_XYZ_ACC", f"{P1PC}/baselines/radhar_official_gru_xyz/metrics.json", "accuracy", protocol="RadHAR official PointNet-GRU XYZ T=60"),
    spec("RADHAR_XYZV_ACC", f"{P1PC}/baselines/radhar_official_gru_xyzv/metrics.json", "accuracy", protocol="RadHAR official PointNet-GRU XYZV T=60"),
    spec("RADHAR_XYZV_F1", f"{P1PC}/baselines/radhar_official_gru_xyzv/metrics.json", "macro_f1", protocol="RadHAR official PointNet-GRU XYZV T=60"),
    spec("RADHAR_XYZVI_ACC", f"{P1PC}/baselines/radhar_official_gru_xyzvi/metrics.json", "accuracy", protocol="RadHAR official PointNet-GRU XYZVI T=60"),
    spec("RADHAR_T1_ACC", f"{P1PC}/baselines/radhar_official_pool_t1_xyzvi/metrics.json", "accuracy", protocol="RadHAR official T=1 PointNet-pool XYZVI"),
    spec("RADHAR_T1_F1", f"{P1PC}/baselines/radhar_official_pool_t1_xyzvi/metrics.json", "macro_f1", protocol="RadHAR official T=1"),
    spec("RADHAR_VOXEL_ACC", f"{P1PC}/baselines/radhar_official_voxel_cnn_gru/metrics.json", "accuracy", protocol="RadHAR official voxel CNN-GRU"),
    spec("RADHAR_VOXEL_F1", f"{P1PC}/baselines/radhar_official_voxel_cnn_gru/metrics.json", "macro_f1", protocol="RadHAR official voxel CNN-GRU"),
    spec("CI4R77_RAND_ACC", f"{P1CI}/baselines/random_record/resnet18/metrics.json", "accuracy", protocol="CI4R 77 GHz 11-class random-record; not the 7-class paired split"),
    spec("CI4R77_RAND_F1", f"{P1CI}/baselines/random_record/resnet18/metrics.json", "macro_f1", protocol="CI4R 77 GHz 11-class random-record"),
    spec("CI4R77_PREFIX_ACC", f"{P1CI}/baselines/prefix_tail4_disjoint/resnet18/metrics.json", "accuracy", protocol="CI4R 77 GHz prefix-group-disjoint; not subject-independent"),
    spec("CI4R77_PREFIX_F1", f"{P1CI}/baselines/prefix_tail4_disjoint/resnet18/metrics.json", "macro_f1", protocol="CI4R 77 GHz prefix-group-disjoint"),
    spec("CI4R77_RAND_NTEST", f"{P1CI}/baselines/random_record/resnet18/metrics.json", "n_test", format_rule="int"),
    spec("CI4R77_PREFIX_NTEST", f"{P1CI}/baselines/prefix_tail4_disjoint/resnet18/metrics.json", "n_test", format_rule="int"),
    # Shared motion (paired CE / InfoNCE)
    spec("M0_ACC", f"{P3}/representation_metrics.csv", "activity_accuracy", kind="csv_row", match={"model": "M0", "protocol": "multi_radar_shared"}, protocol="known-radar joint encoder; CE only"),
    spec("M1_ACC", f"{P3}/representation_metrics.csv", "activity_accuracy", kind="csv_row", match={"model": "M1", "protocol": "multi_radar_shared"}, protocol="known-radar joint encoder; CE+InfoNCE"),
    spec("M2_ACC", f"{P3}/representation_metrics.csv", "activity_accuracy", kind="csv_row", match={"model": "M2", "protocol": "multi_radar_shared"}, protocol="known-radar joint encoder; CE+InfoNCE+physics aug"),
    spec("M0_ACTP", f"{P3}/representation_metrics.csv", "activity_probe_accuracy", kind="csv_row", match={"model": "M0", "protocol": "multi_radar_shared"}, protocol="linear activity probe on M0 embeddings"),
    spec("M1_ACTP", f"{P3}/representation_metrics.csv", "activity_probe_accuracy", kind="csv_row", match={"model": "M1", "protocol": "multi_radar_shared"}),
    spec("M2_ACTP", f"{P3}/representation_metrics.csv", "activity_probe_accuracy", kind="csv_row", match={"model": "M2", "protocol": "multi_radar_shared"}),
    spec("M0_RID", f"{P3}/representation_metrics.csv", "radar_probe_accuracy", kind="csv_row", match={"model": "M0", "protocol": "multi_radar_shared"}, protocol="3-radar Radar-ID on shared embeddings"),
    spec("M1_RID", f"{P3}/representation_metrics.csv", "radar_probe_accuracy", kind="csv_row", match={"model": "M1", "protocol": "multi_radar_shared"}),
    spec("M2_RID_SHARE", f"{P3}/representation_metrics.csv", "radar_probe_accuracy", kind="csv_row", match={"model": "M2", "protocol": "multi_radar_shared"}),
    spec("M0_RET", f"{P3}/representation_metrics.csv", "retrieval_action_top1", kind="csv_row", match={"model": "M0", "protocol": "multi_radar_shared"}, protocol="same-action retrieval top-1; mean over 6 directed pairs"),
    spec("M1_RET", f"{P3}/representation_metrics.csv", "retrieval_action_top1", kind="csv_row", match={"model": "M1", "protocol": "multi_radar_shared"}),
    spec("M2_RET", f"{P3}/representation_metrics.csv", "retrieval_action_top1", kind="csv_row", match={"model": "M2", "protocol": "multi_radar_shared"}),
    spec("M3_RID", f"{P7}/answers.json", "radar_id_M3", protocol="3-radar Radar-ID after domain-adversarial M3"),
    spec("M3_ACC", f"{P7}/runs/R1_M3_shared/metrics.json", "activity_accuracy", protocol="known-radar M3 CE+InfoNCE+GRL"),
    spec("M3_LORO_77", f"{P7}/answers.json", "loro.10+24→77.M3", protocol=DG),
    spec("SMOOTH_CONF", f"{P4}/smoothing_choice.json", "conf_min", protocol="val-only threshold for confidence-aware smoothing"),
    spec("SMOOTH_VAL_F1", f"{P4}/smoothing_choice.json", "val_macro_f1", protocol="val macro-F1 at selected conf_min"),
    spec("CONT_N_WINDOWS", f"{P4}/continuous_runs.csv", "n_windows", kind="csv_sum", match={"mode": "confidence_aware"}, format_rule="int", protocol="labeled-interval continuous replay; test subjects only"),
    spec("CONT_FRAME_ACC", f"{P4}/continuous_runs.csv", "frame_accuracy", kind="csv_mean", match={"mode": "confidence_aware"}, protocol="labeled-interval; unweighted mean over 3 test recordings"),
    spec("CONT_FRAME_F1", f"{P4}/continuous_runs.csv", "frame_macro_f1", kind="csv_mean", match={"mode": "confidence_aware"}),
    spec("CONT_WIN_ACC", f"{P4}/continuous_runs.csv", "window_accuracy", kind="csv_mean", match={"mode": "confidence_aware"}),
    spec("CONT_WIN_F1", f"{P4}/continuous_runs.csv", "window_macro_f1", kind="csv_mean", match={"mode": "confidence_aware"}),
    spec("CONT_WIN_ACC_RAW", f"{P4}/continuous_runs.csv", "window_accuracy", kind="csv_mean", match={"mode": "raw"}),
    spec("CONT_WIN_F1_RAW", f"{P4}/continuous_runs.csv", "window_macro_f1", kind="csv_mean", match={"mode": "raw"}),
    spec("CONT_WIN_ACC_FLIP", f"{P4}/continuous_runs.csv", "window_accuracy", kind="csv_mean", match={"mode": "isolated_flip"}),
    spec("CONT_WIN_F1_FLIP", f"{P4}/continuous_runs.csv", "window_macro_f1", kind="csv_mean", match={"mode": "isolated_flip"}),
    spec("CONT_SEG_F1", f"{P4}/segment_metrics.csv", "segment_f1", kind="csv_mean", match={"mode": "confidence_aware"}, protocol="labeled-interval segment F1; unweighted mean"),
    spec("CONT_SEG_F1_RAW", f"{P4}/segment_metrics.csv", "segment_f1", kind="csv_mean", match={"mode": "raw"}),
    spec("CONT_SEG_F1_FLIP", f"{P4}/segment_metrics.csv", "segment_f1", kind="csv_mean", match={"mode": "isolated_flip"}),
    spec("CONT_BND_MED", f"{P4}/boundary_metrics.csv", "median_frames", kind="csv_mean", match={"mode": "confidence_aware"}, format_rule="f1", protocol="mean of per-recording median boundary errors"),
    spec("CONT_BND_SEC", f"{P4}/boundary_metrics.csv", "median_seconds", kind="csv_mean", match={"mode": "confidence_aware"}, format_rule="f1"),
    spec("OPEN_N_FOLDS", f"{P5}/loao_msp_per_action.json", "", kind="json_len", format_rule="int", protocol="leave-one-action-out; 12 known + 1 unseen"),
    spec("MSP_AUROC", f"{P5}/loao_msp_per_action.json", "AUROC", kind="json_mean", protocol="13-fold LOAO + SI; MSP"),
    spec("MSP_AUROC_STD", f"{P5}/loao_msp_per_action.json", "AUROC", kind="json_std"),
    spec("MSP_AUPR", f"{P5}/loao_msp_per_action.json", "AUPR", kind="json_mean"),
    spec("MSP_AUPR_STD", f"{P5}/loao_msp_per_action.json", "AUPR", kind="json_std"),
    spec("MSP_FPR", f"{P5}/loao_msp_per_action.json", "FPR95", kind="json_mean"),
    spec("MSP_FPR_STD", f"{P5}/loao_msp_per_action.json", "FPR95", kind="json_std"),
    spec("ENERGY_AUROC", f"{P5}/loao_energy_per_action.json", "AUROC", kind="json_mean", protocol="13-fold LOAO + SI; Energy"),
    spec("ENERGY_AUROC_STD", f"{P5}/loao_energy_per_action.json", "AUROC", kind="json_std"),
    spec("ENERGY_AUPR", f"{P5}/loao_energy_per_action.json", "AUPR", kind="json_mean"),
    spec("ENERGY_AUPR_STD", f"{P5}/loao_energy_per_action.json", "AUPR", kind="json_std"),
    spec("ENERGY_FPR", f"{P5}/loao_energy_per_action.json", "FPR95", kind="json_mean"),
    spec("ENERGY_FPR_STD", f"{P5}/loao_energy_per_action.json", "FPR95", kind="json_std"),
    spec("PROTO_AUROC", f"{P5}/loao_prototype_per_action.json", "AUROC", kind="json_mean", protocol="13-fold LOAO + SI; prototype distance"),
    spec("PROTO_AUROC_STD", f"{P5}/loao_prototype_per_action.json", "AUROC", kind="json_std"),
    spec("PROTO_AUPR", f"{P5}/loao_prototype_per_action.json", "AUPR", kind="json_mean"),
    spec("PROTO_AUPR_STD", f"{P5}/loao_prototype_per_action.json", "AUPR", kind="json_std"),
    spec("PROTO_FPR", f"{P5}/loao_prototype_per_action.json", "FPR95", kind="json_mean"),
    spec("PROTO_FPR_STD", f"{P5}/loao_prototype_per_action.json", "FPR95", kind="json_std"),
    spec("LLM_N", f"{RM01}/llm_online_test.json", "n", format_rule="int", protocol="DeepSeek online tasks on structured MI+Timeline"),
    spec("LLM_N_OK", f"{RM01}/llm_online_test.json", "n_ok", format_rule="int"),
    spec("BOX_CONF", f"{RM01}/llm_online_test.json", "records.0.input.motion_intelligence.confidence.value", protocol="RadHAR official test boxing demo"),
    spec("BOX_UNC", f"{RM01}/llm_online_test.json", "records.0.input.motion_intelligence.uncertainty.value"),
    spec("BOX_NWIN", f"{RM01}/llm_online_test.json", "records.0.input.motion_timeline.events.0.n_windows", format_rule="int"),
    spec("BOX_MEAN_V", f"{RM01}/llm_online_test.json", "records.0.input.motion_intelligence.mean_velocity.value"),
    spec("BOX_PEAK_V", f"{RM01}/llm_online_test.json", "records.0.input.motion_intelligence.peak_velocity.value"),
    spec("BOX_ENERGY", f"{RM01}/llm_online_test.json", "records.0.input.motion_intelligence.motion_intensity.value"),
    spec("BOX_EVT_CONF", f"{RM01}/llm_online_test.json", "records.0.input.motion_timeline.events.0.confidence"),
    spec("SMOKE_V01", f"{RM01}/smoke_test_report.md", "", kind="md_pass_count", format_rule="int", protocol="RadarMind v0.1 application smoke"),
    spec("SMOKE_V02", f"{RM02}/smoke_test_report.md", "", kind="md_pass_count", format_rule="int", protocol="RadarMind v0.2 application smoke"),
    spec("CONT_SMOKE", f"{P4}/phase4_smoke.md", "", kind="md_pass_count", format_rule="int", protocol="labeled-interval continuous smoke"),
]


def by_key() -> dict[str, dict[str, Any]]:
    out = {}
    for m in METRICS:
        if m["key"] in out:
            raise ValueError(f"duplicate metric key {m['key']}")
        out[m["key"]] = m
    return out
