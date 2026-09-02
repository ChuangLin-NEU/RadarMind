# Figure rebuild audit

Manuscript audited: `results/final_delivery/RadarMind_Full_Manuscript.md`  
Old generator: `scripts/full_manuscript/figures.py`  
Old files (archived, not reused as new Fig. 1–11): `results/final_delivery/figures/fig1_overview.png` … `fig12_roadmap.png` → `figures/archive_pre_rebuild/`  
Snapshot: `results/final_delivery/metric_snapshot_full.json`

## Manuscript figure inventory (pre-rebuild)

| MS Fig | Caption purpose (latest manuscript) | In-text uses | Table link |
| ------ | ----------------------------------- | ------------ | ---------- |
| Fig. 1 | RadarMind overview (3 layers + current boundary) | Methods 2.1 | — |
| Fig. 2 | Dataset roles; not a unified benchmark | Methods 2.2 | — |
| Fig. 3 | Closed-set + temporal context | Results 3.1; temporal discussed in 3.3 | Table 2 after Fig. 3 |
| Fig. 4 | 3×3 official 7-class train/test matrix | Methods 2.7; Results 3.4 | Table 4 after Fig. 4 |
| Fig. 5 | Radar-ID after reconstruction (3-radar vs 24/77 colors) | Methods 2.7 says “Fig. 5 uses two colors” | — |
| Fig. 6 | Activity vs Radar-ID scatter; also cited in 2.1 as the information-tradeoff diagnostic | Methods 2.1, 2.7 | — |
| Fig. 7 | Shared M0–M2 accuracy / retrieval / Radar-ID | Results 3.5 | Table 5 after Fig. 7 |
| Fig. 8 | Continuous labeled-interval **bars only** | Results 3.7 | Table 7 after Fig. 8 |
| Fig. 9 | Open-set mean AUROC bars | Results 3.8 | Table 8 after Fig. 9 |
| Fig. 10 | LLM pipeline schematic | Results 3.10 | Table 9 after Fig. 10 |
| Fig. 11 | D(v\|q) / oracle R² bars | Results 3.6 (inserted after reconstruction) | Table 6 after Fig. 11 |
| Fig. 12 | Roadmap / module status boxes | Results 3.11; Table 10 cites Fig. 12 | Table 10 |

## Numbering and correspondence failures

1. **Figure order ≠ narrative order.** Shared representation is Results 3.5 but numbered Fig. 7. Physical reconstruction is 3.6 but numbered Fig. 5 / Fig. 6 / Fig. 11.
2. **Fig. 6 is overloaded.** Intro treats Fig. 6 as the information-tradeoff diagnostic; Methods treat Fig. 6 as R2-V vs R0 incomparability; the file is a scatter that also mixes 3-radar, 24/77, M1, and A2 on one axis.
3. **Fig. 8 is bars only.** Manuscript 3.7 discusses labeled-interval replay; the old figure does not show a timeline or unlabeled gaps.
4. **Fig. 9 is mean bars.** Fold-level LOAO files exist (`loao_*_per_action.json`) and are unused in the plot.
5. **Fig. 11 is R² bars only.** Frozen Phase 10G has no npy/npz difference-field maps; panels of measured / predicted / residual fields **cannot** be drawn without fabricating arrays.
6. **Fig. 12 is a leftover main-text number.** New plan collapses status into Fig. 11. Table 10 still cites Fig. 12.
7. **Old Fig. 3 mixes protocols on one leaderboard-like axis** (mRI SI, mRI random, RadHAR XYZV, RadHAR voxel) and omits CI4R 10/24/77 within-radar cells required by the new 3.1 panel.
8. **Old Fig. 5 draws 3-radar and 24/77 Radar-ID as adjacent bars on one axis** (colors differ, but it still reads as one ladder). Forbidden under the new protocol rule.

## Difference-field availability (Fig. 7)

`results/phase10_conditional_2d_measurement/` contains CSV/JSON only (`answers.json`, `oracle_vs_predicted.csv`, `transfer_prediction_2d.csv`, `δ_2d_statistics.csv`, …). **No `.npy` / `.npz` measured or predicted 2D Δ maps.**  
Decision: **do not invent panels a–c.** Rebuild Fig. 7 as a quantitative R² panel only, and state in the caption that TEST oracle is an upper bound, not a deployable translator.

## Decision table

| Current Fig | Manuscript purpose | Existing file | Existing source | Keep / Rebuild / Remove | Reason |
| ----------- | ------------------ | ------------- | --------------- | ----------------------- | ------ |
| Fig. 1 | Three-layer RadarMind framework; radar → recognition → MI → timeline → language; LLM does not see raw radar; unresolved box | `fig1_overview.png` | schematic in `scripts/full_manuscript/figures.py`; no metric scalars required | Rebuild | Old file is a software-layer box list, not the required graphical framework. New file: `fig01_radarmind_framework` |
| Fig. 2 | Dataset–protocol map (CI4R / mRI / RadHAR roles; not a unified benchmark) | `fig2_datasets.png` | snapshot `N_PAIRED_ACQ`, `MRI_N_SUBJECTS`, `MRI_N_ACTIONS`, `RADHAR_N_ACTIONS` | Rebuild | Old file is three isolated boxes. Need a task matrix. New file: `fig02_dataset_protocol_map` |
| Fig. 3 | 3.1 closed-set baselines + temporal context (not architecture ranking) | `fig3_baseline_temporal.png` | `MRI_SI_ACC`, `MRI_RAND_ACC`, `RADHAR_XYZV_ACC`, `RADHAR_VOXEL_ACC`, `MRI_T1_ACC`, `RADHAR_T1_ACC` | Rebuild | Missing CI4R 10/24/77 within; mixes random/voxel into the same board. New file: `fig03_baseline_temporal` |
| Fig. 4 | 3.4 3×3 official 7-class matrix (central negative result) | `fig4_cross_radar.png` | `R0_10_WITHIN` … `R0_77_WITHIN` from Phase 2 `metrics.json` via snapshot | Rebuild | Science is correct; style, naming, and SCI palette must be rebuilt from snapshot (no hard-coded 0.843 / 0.039). New file: `fig04_cross_radar_matrix` |
| Fig. 5 (old) | Reconstruction Radar-ID bars | `fig5_reconstruction.png` | `RID_R0_3CLASS`, `RID_R2D_3CLASS`, `RID_R2D_2477`, `RID_R2V_2477` | Remove as Fig. 5; rebuild into new Fig. 6 | Numbering belongs to shared representation. 3-radar and 24/77 must be split panels, not one axis |
| Fig. 6 (old) | RID vs activity scatter | `fig6_tradeoff.png` | mixed 3-radar / 24/77 / M1 / A2 | Remove as standalone Fig. 6; fold into new Fig. 6 | Mixing probe families on one curve is forbidden. New Fig. 6 may use a scatter **with separate marker families** |
| Fig. 7 (old) | M0–M2 shared representation | `fig7_shared.png` | `M0_ACC/RET`, `M1_ACC/RET/RID`, `M2_ACC/RET` | Rebuild as **new Fig. 5** | Matches 3.5. Include M0–M2 Radar-ID (all 1.000 from snapshot). New file: `fig05_shared_representation` |
| Fig. 8 | Labeled-interval continuous replay | `fig8_continuous.png` | `CONT_FRAME_*`, `CONT_WIN_ACC`, `CONT_SEG_F1` (bars only) | Rebuild | Need GT / raw / smoothed timeline from `phase4_continuous` JSON + summary metrics. New file: `fig08_continuous_inference` |
| Fig. 9 | 13-fold LOAO open-set | `fig9_openset.png` | snapshot means `MSP/ENERGY/PROTO_AUROC` | Rebuild | Use fold-level JSON; show chance 0.5; do not visually inflate Prototype. New file: `fig09_open_set` |
| Fig. 10 | Motion Intelligence → language; LLM isolated from raw radar | `fig10_llm.png` | `BOX_*` snapshot + schematic | Rebuild | Must display stored boxing JSON fields and stored QA, not invented JSON. New file: `fig10_motion_intelligence_llm` |
| Fig. 11 (old) | Measurement-difference R² | `fig11_measurement.png` | `D2477_B1_R2`, `D7724_B1_R2`, `ORACLE_R2_2D`, `B2_R2_2477` | Rebuild as **new Fig. 7** | No frozen difference maps → quantitative only. Caption must mark oracle as TEST upper bound. New file: `fig07_measurement_difference` |
| Fig. 12 | Capability / remaining problems | `fig12_roadmap.png` | `SMOKE_V02` + schematic | Remove as Fig. 12; rebuild as **new Fig. 11** | Not a software dashboard. New file: `fig11_capability_summary` |

## New Fig. 1–11 plan (SCI narrative)

| New Fig | Purpose | Data (registry keys / frozen files) | Protocol note |
| ------- | ------- | ----------------------------------- | ------------- |
| Fig. 1 | Framework | schematic; optional counts from snapshot | LLM does not receive raw radar |
| Fig. 2 | Dataset–task map | `N_PAIRED_ACQ`, `MRI_N_SUBJECTS`, `MRI_N_ACTIONS`, `RADHAR_N_ACTIONS`, `CI4R77_*` | Not a unified benchmark |
| Fig. 3 | 3.1 HAR + temporal | `MRI_SI_ACC`, `RADHAR_XYZV_ACC`, `R0_10/24/77_WITHIN`, `MRI_T1_ACC`, `RADHAR_T1_ACC` | Separated protocol groups; temporal ≠ architecture |
| Fig. 4 | Cross-radar matrix | `R0_*` nine cells | Acquisition-timestamp split; prefix ≠ subject |
| Fig. 5 | Shared representation | `M0/M1/M2_ACC`, `_RET`, `_RID` / `M2_RID_SHARE` | Known-radar only; not LORO |
| Fig. 6 | Radar-ID ladder | 3-radar: `RID_R0_3CLASS`, `RID_R1_3CLASS`, `RID_R2D_3CLASS`; 24/77: `RID_R2D_2477`, `RID_R2V_2477`, `U1_RID`, `U2_RID`, `CAL_*` as available | Two disconnected probe families |
| Fig. 7 | D(v\|q) / 2D R² | `D2477_B1_R2`, `D7724_B1_R2`, `ORACLE_R2_2D`, `B2_R2_2477`, `B2_R2_7724` | R² ≠ HAR; oracle ≠ translator; no fake maps |
| Fig. 8 | Continuous | GT `subject20_0513.json`; predicted events; `CONT_SEG_F1_*`, `CONT_BND_*` | Labeled-interval, not full-stream |
| Fig. 9 | Open-set | `loao_msp/energy/prototype_per_action.json` | LOAO rejection, not open-world |
| Fig. 10 | MI → LLM | `llm_online_test.json` records[0] + QA | Stored JSON only |
| Fig. 11 | Capability / limitation | qualitative + snapshot evidence keys | Implemented stack + unresolved unseen-radar |

## Orphans and missing science

- Old 12 PNGs become orphans after rebuild (archived).
- New plan has no main-text Fig. 12.
- Shared-representation science was present but mis-numbered.
- Continuous timeline and fold-level open-set were missing as figures.
- Difference-field images are missing as **data**, not as a numbering error.

Audit complete. Batch generation may proceed from this table.
