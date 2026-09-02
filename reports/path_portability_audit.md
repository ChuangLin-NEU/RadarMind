# Path portability audit

Classes: A = runtime hard-coded path (must be env/relative); B = historical report; C = generated CSV/manifest.
Phase 2 changed Class A scripts to environment variables or repository-relative defaults.
Frozen `answers.json` files may still record the lab path used when the experiment ran. Those files are metric sources and were not rewritten.

| Class | Path | Pattern |
|---|---|---|
| B | `reports/ci4r_split_reference_audit.md` | C:\Users\13919 |
| B | `reports/figure_metric_provenance.md` | C:\Users\13919 |
| B | `reports/path_portability_audit.md` | /home/radar_har |
| B | `reports/public_release_audit.md` | /home/radar_har |
| B | `reports/security_audit.md` | C:\Users\13919, /home/radar_har |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_BEND_D200_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_BEND_D200_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_BEND_D300_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_BEND_D300_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_PICK_D200_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_PICK_D200_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_PICK_D300_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S001_PICK_D300_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_BEND_D200_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_BEND_D200_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_BEND_D300_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_BEND_D300_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_PICK_D200_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_PICK_D200_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_PICK_D300_A000_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/acquisition_toolkit_v01/mock_dataset/metadata/RM_S002_PICK_D300_A045_R001_20260830_120851.json` | C:\Users\13919 |
| B | `results/data_audit/audit_facts.json` | /home/radar_har |
| B | `results/data_audit/dataset_audit_report.md` | /home/radar_har |
| B | `results/phase5_open_set/app_pack.json` | C:\Users\13919 |
| B | `results/phase8_physical_representation/answers.json` | /home/radar_har |
| B | `results/phase8_physical_representation/audit_summary.json` | /home/radar_har |
| B | `results/phase8_physical_representation/missing_data_request.md` | /home/radar_har |
| B | `results/phase8_physical_representation/phase8_physical_representation_report.md` | /home/radar_har |
| B | `results/phase8_physical_representation/phase8b_summary.json` | /home/radar_har |
| B | `results/phase8_physical_representation/preprocessing_chain_77ghz.md` | /home/radar_har |
| B | `results/phase8_physical_representation/raw_access.json` | /home/radar_har |
| B | `results/phase8_physical_representation/remote_zip_probe.json` | /home/radar_har |
| B | `results/phase8_physical_representation/smoke_gate.json` | /home/radar_har |
| B | `results/phase8_physical_representation/smoke_morphology.json` | /home/radar_har |
| B | `results/radarmind_v01/artifact_registry.json` | C:\Users\13919 |
| B | `results/radarmind_v01/radarmind_v01_report.md` | C:\Users\13919 |
| B | `results/radarmind_v02/environment_record.json` | C:\Users\13919 |
| B | `results/runs/bench_radmamba_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/bench_resnet18_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/bench_simple_cnn_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/cross_10GHz_to_77GHz_simple_cnn_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/cross_24GHz_to_77GHz_simple_cnn_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/cross_77GHz_to_24GHz_simple_cnn_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/loso_77GHz_simple_cnn_held0010_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/loso_77GHz_simple_cnn_held0014_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/loso_77GHz_simple_cnn_held0029_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/loso_77GHz_simple_cnn_held0030_seed0/config.json` | C:\Users\13919 |
| B | `results/runs/official_resnet6_seed0/config.json` | C:\Users\13919 |
| B | `results/splits/ci4r_10GHz_last12.json` | C:\Users\13919 |
| B | `results/splits/ci4r_24GHz_last12.json` | C:\Users\13919 |
| B | `results/splits/ci4r_official_last12.json` | C:\Users\13919 |
| B | `results/splits/loso_77GHz/folds.json` | C:\Users\13919 |
| B | `scripts/release/pre_git_audit.py` | C:\Users\13919, /home/radar_har |
| B | `scripts/release/write_phase2_audits.py` | C:\Users\13919, /home/radar_har |
| C | `results/data_audit/dataset_manifest.csv` | /home/radar_har |
| C | `results/data_audit/download_status.csv` | /home/radar_har |
| C | `results/phase10_unified_measurement/representation_manifest.csv` | /home/radar_har |
| C | `results/phase10_velocity_calibration/r2v_manifest.csv` | /home/radar_har |
| C | `results/phase1_pointcloud/failed_samples.csv` | /home/radar_har |
| C | `results/phase8_physical_representation/native_representation_manifest.csv` | /home/radar_har |
| C | `results/phase8_physical_representation/physical_data_inventory.csv` | /home/radar_har |
| C | `results/phase8_physical_representation/r2_manifest.csv` | /home/radar_har |
| C | `results/phase8_physical_representation/raw_processing_status.csv` | /home/radar_har |

Remaining Class A rows: 0
