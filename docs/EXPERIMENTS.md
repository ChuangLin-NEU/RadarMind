# Experiments

Commands below are the actual repository entries. They are not a newly invented CLI.

Metric displays come from `results/final_delivery/metric_snapshot_full.json`.

## E1 Closed-set recognition

| Field | Content |
|---|---|
| Scientific question | How well do the in-domain HAR models perform under each dataset’s own closed-set protocol? |
| Dataset | CI4R 77 GHz 11-class; mRI; RadHAR |
| Protocol | CI4R random-record; mRI SI and random-segment; RadHAR official |
| Entry point | `scripts/ci4r/phase1a_train.py`; `scripts/pointcloud/phase1b_train.py`; `scripts/pointcloud/train_voxel.py` |
| Input | Spectrogram or point-cloud caches; `data_splits/` |
| Output | `results/phase1_ci4r/baselines/*/resnet18/metrics.json`; `results/phase1_pointcloud/baselines/*/metrics.json` |
| Metric | Accuracy, macro-F1 |
| Expected frozen result | RadHAR XYZV Acc/F1 0.949 / 0.949; mRI SI Acc/F1 0.478 / 0.420 |
| Interpretation | In-domain recognition is dataset- and protocol-specific. Groups are not one leaderboard. |
| Limitation | Single-seed freeze. CI4R 11-class is not the 7-class paired split. |

## E2 Subject-independent recognition

| Field | Content |
|---|---|
| Scientific question | Does recognition hold when official subject IDs are disjoint across train/val/test? |
| Dataset | mRI |
| Protocol | Official subject IDs; test subjects 3, 10, 20; T=32, P≤64 |
| Entry point | `scripts/pointcloud/phase1b_train.py` |
| Input | `data_splits/mri_subject_independent.csv` |
| Output | `results/phase1_pointcloud/baselines/mri_si_gru_xyz/metrics.json` |
| Metric | Accuracy 0.478; macro-F1 0.420 |
| Expected frozen result | those two displays |
| Interpretation | This is the only verified subject-independent HAR result in the repository. |
| Limitation | CI4R prefix-group splits are not SI. RadHAR official split is not SI. |

## E3 Cross-radar transfer

| Field | Content |
|---|---|
| Scientific question | How does source-only 7-class HAR transfer across 10 / 24 / 77 GHz? |
| Dataset | CI4R official PNG, 327 agreed triples, 229/49/49 |
| Protocol | source-only; acquisition-timestamp split |
| Entry point | `scripts/cross_radar/train_cross_radar.py`; `run_cross_radar_experiments.py` |
| Input | `results/phase2_cross_radar/cache/` (local); pairing stats |
| Output | `results/phase2_cross_radar/runs/within_*/metrics.json`; `cross_*_to_*/metrics.json` |
| Metric | Accuracy |
| Expected frozen result | within 0.843 / 0.898 / 0.922; 10→24 0.143; 10→77 0.373; 24→10 0.353; 24→77 0.039; 77→10 0.275; 77→24 0.163 |
| Interpretation | Within-radar recognition is strong. Several unseen-radar cells fail. |
| Limitation | Source-only transfer is not UDA and not LORO. |

## E4 LORO / UDA

| Field | Content |
|---|---|
| Scientific question | Do CORAL UDA or leave-one-radar-out training close the unseen-radar gap? |
| Dataset | CI4R paired 7-class |
| Protocol | CORAL = UDA; LORO = domain generalization |
| Entry point | `scripts/cross_radar/train_cross_radar.py` (`coral_uda_*`); `scripts/shared_motion/train_shared_motion.py`; `scripts/phase7/train_m3.py` |
| Input | Same paired split |
| Output | `results/phase2_cross_radar/runs/coral_uda_*/metrics.json`; `results/phase3_shared_motion/runs/*loro*`; `results/phase7_radar_invariance/runs/*loro*` |
| Metric | Accuracy / activity_accuracy |
| Expected frozen result | registry keys `CORAL_*`, `LORO_*`, `M3_LORO_77` |
| Interpretation | These protocols are not interchangeable with source-only transfer or known-radar joint training. |
| Limitation | Negative or limited transfer must stay labeled as such. |

## E5 Shared representation learning

| Field | Content |
|---|---|
| Scientific question | Does a shared encoder improve known-radar activity correspondence? |
| Dataset | CI4R paired 7-class |
| Protocol | known-radar joint training; M0 CE; M1 CE+InfoNCE (λ=0.1 canonical `M1_l01`); M2 physics augmentation; M3 + GRL |
| Entry point | `scripts/shared_motion/train_shared_motion.py`; `scripts/phase7/train_m3.py`; `run_shared_motion_experiments.py` |
| Input | acquisition split 229/49/49 |
| Output | `results/phase3_shared_motion/representation_metrics.csv`; `results/phase7_radar_invariance/` |
| Metric | activity accuracy, retrieval, Radar-ID |
| Expected frozen result | M1 activity 0.932; retrieval 0.935; Radar-ID 1.000 |
| Interpretation | InfoNCE helps semantic correspondence among radars seen in training. |
| Limitation | Radar-ID 1.000: this is not radar invariance. |

## E6 Radar-ID diagnostics

| Field | Content |
|---|---|
| Scientific question | Can a linear probe recover radar identity from a representation? |
| Dataset | CI4R paired maps / embeddings |
| Protocol | 3-radar or 24/77 probes; not a HAR system |
| Entry point | `scripts/shared_motion/evaluate.py`; `scripts/phase7/evaluate.py`; Phase 8–10 `answers.json` writers |
| Input | frozen embeddings or reconstructed maps |
| Output | `representation_radar_id.json`; Phase 8/10 `answers.json` |
| Metric | Radar-ID accuracy |
| Expected frozen result | R2-D 3-radar 0.993; R2-V 24/77 0.939 |
| Interpretation | Sensor identity remains recoverable after several physical steps. |
| Limitation | 3-radar and 24/77 probes are not one trajectory. |

## E7 Physical reconstruction

| Field | Content |
|---|---|
| Scientific question | Do numerical Doppler and unified STFT/MTI change Radar-ID or HAR? |
| Dataset | CI4R raw / official PNG |
| Protocol | R0 official PNG; R1 crop/normalize; R2-D numerical Doppler; R2-V velocity alignment; U1 unified STFT; U2 MTI |
| Entry point | `scripts/phase7/build_r1_cache.py`; `run_r2_benchmark.py`; `scripts/phase10/run_phase10c.py`; `scripts/phase10/run_phase10d.py` |
| Input | `CI4R_RAW_ZIP` when reconstructing; otherwise stored answers |
| Output | `results/phase8_physical_representation/answers.json`; `results/phase10_*/answers.json` |
| Metric | Radar-ID, activity probe, pair cosine. R² when used is reconstruction, not HAR. |
| Expected frozen result | see snapshot keys `RID_R2D_3CLASS`, `RID_R2V_2477`, `U1_*`, `U2_*` |
| Interpretation | Physical re-representation is diagnostic. |
| Limitation | Requires the public raw archive for a from-scratch rebuild. |

## E8 Motion-conditioned difference modeling

| Field | Content |
|---|---|
| Scientific question | Can D(v\|q) or a low-rank 2D Δ explain 24/77 measurement differences? |
| Dataset | CI4R 24/77 paired maps |
| Protocol | B0 mean; B1 ridge D(v\|q); B2 predicted 2D; TEST oracle projection |
| Entry point | `scripts/phase10/run_phase10f.py`; `scripts/phase10/run_phase10g.py` |
| Input | Phase 10 caches (local) |
| Output | `results/phase10_motion_conditioned/answers.json`; `results/phase10_conditional_2d_measurement/answers.json` |
| Metric | R², MAE, correlation; activity probe after applying Δ |
| Expected frozen result | snapshot keys `D2477_B1_R2`, `ORACLE_R2_2D`, `B2_R2_2477` |
| Interpretation | R² is a reconstruction metric. The TEST oracle is a diagnostic upper bound, not a deployable translator. |
| Limitation | Applying Δ as calibration does not establish unseen-radar HAR. |

## E9 Continuous labeled-interval replay

| Field | Content |
|---|---|
| Scientific question | How do window, frame, segment, and boundary scores look on labeled mRI intervals? |
| Dataset | mRI test recordings |
| Protocol | labeled-interval replay; raw / isolated-flip / confidence-aware |
| Entry point | `scripts/continuous/run_all.py`; `run_continuous_demo.py` |
| Input | mRI cache; SI checkpoint (local) |
| Output | `results/phase4_continuous/continuous_runs.csv`; `segment_metrics.csv`; `boundary_metrics.csv` |
| Metric | frame Acc/F1 0.489 / 0.416; window 0.494 / 0.419; segment F1 0.152; boundary 20.6 s |
| Expected frozen result | those displays (`confidence_aware`) |
| Interpretation | This is not full-stream sensing. Unlabeled gaps are not a background class. |
| Limitation | Boundary localization is weak. |

## E10 LOAO open-set rejection

| Field | Content |
|---|---|
| Scientific question | Can a held-out action be rejected rather than forced into a known class? |
| Dataset | mRI |
| Protocol | 13-fold leave-one-action-out; held-out action out of train, val, and threshold fit; MSP / Energy / Prototype |
| Entry point | `scripts/open_set/train_loao.py`; `scripts/open_set/eval_openset.py`; `run_open_set_experiments.py` |
| Input | SI split + per-fold held-out action |
| Output | `results/phase5_open_set/loao_*_per_action.json` |
| Metric | AUROC, AUPR, FPR95 |
| Expected frozen result | Prototype AUROC 0.628 ± 0.125 |
| Interpretation | Open-set rejection, not open-world or open-vocabulary recognition. Output label for reject is `UNKNOWN`. |
| Limitation | Not deployment-grade. |

## E11 Motion Intelligence

| Field | Content |
|---|---|
| Scientific question | Can a typed JSON record carry activity, kinematics, and metadata with explicit provenance? |
| Dataset | any loaded `RadarSample` |
| Protocol | no fallback guessing; UNKNOWN stays UNKNOWN |
| Entry point | `radarmind/intelligence/extract.py`; research app replay pages |
| Input | prediction + optional point cloud + metadata |
| Output | Motion Intelligence JSON |
| Metric | schema completeness and provenance tags, not HAR accuracy |
| Expected frozen result | schema fields as in `docs/MOTION_INTELLIGENCE.md` |
| Interpretation | Structured evidence for the language layer. |
| Limitation | Periodicity and environment are UNKNOWN in v0.1. |

## E12 Language interaction

| Field | Content |
|---|---|
| Scientific question | Can a language model comment on structured Motion Intelligence without seeing raw radar? |
| Dataset | stored MI / timeline records (RadHAR boxing demo in v0.1 online test) |
| Protocol | DeepSeek if a key is present; otherwise template fallback |
| Entry point | `python run_radarmind.py`; `python run_radarmind.py --offline-demo` |
| Input | Motion Intelligence + timeline JSON |
| Output | explanation text; backend `deepseek` or `fallback` |
| Metric | task completion in `results/radarmind_v01/llm_online_test.json` (optional) |
| Expected frozen result | language layer does not change HAR numbers |
| Interpretation | Grounded wording over structured records. |
| Limitation | Not an evidence source. No key required for the rest of RadarMind. |
