# Paper reproduction audit

Existing generators are kept. Phase 2 does not add a second figure stack.

## Figures (`figures/main/`)

| Figure | File | Generator | Frozen source |
|---|---|---|---|
| Fig. 1 | `fig01_radarmind_framework.svg` | `scripts/paper_figures/figures_v2.py` via `scripts/paper_figures/rebuild.py` | schematic; `scripts/rebuild_paper_figures.py` |
| Fig. 2 | `fig02_dataset_protocol_map.svg` | same | snapshot keys + pairing / summaries |
| Fig. 3 | `fig03_baseline_temporal.svg` | same | Phase 1 metrics in snapshot |
| Fig. 4 | `fig04_cross_radar_matrix.svg` | same | `R0_*` from Phase 2 `metrics.json` |
| Fig. 5 | `fig05_shared_representation.svg` | same | `M0_*` / `M1_*` / `M2_*` |
| Fig. 6 | `fig06_radar_identity_ladder.svg` | same | Phase 7–10 Radar-ID keys |
| Fig. 7 | `fig07_measurement_difference.svg` | same | Phase 10 F/G R² keys |
| Fig. 8 | `fig08_continuous_inference.svg` | same | Phase 4 CSVs |
| Fig. 9 | `fig09_open_set.svg` | same | Phase 5 LOAO JSON |
| Fig. 10 | `fig10_motion_intelligence_llm.svg` | same | `results/radarmind_v01/llm_online_test.json` fields in snapshot |
| Fig. 11 | `fig11_capability_summary.svg` | same | snapshot evidence map |

Entry: `python scripts/rebuild_paper_figures.py`  
Style: `src/paper/visual_style.py`  
Snapshot: `results/final_delivery/metric_snapshot_full.json`

Older helpers still present and not removed: `scripts/paper/build_figures.py`, `scripts/paper_v2/figures.py`.

## Tables (`results/paper_package_v2/tables/`)

| Table | Stem | Builder |
|---|---|---|
| Table 1 dataset | `table1_dataset` | `scripts/paper_v2/tables.py` |
| Table 2 within/cross | `table2_within_cross` | same |
| Table 3 representation | `table3_representation_evolution` | same |
| Table 4 velocity | `table4_velocity_processing` | same |
| Table 5 alignment negative | `table5_alignment_negative` | same |
| Table 6 motion-conditioned | `table6_motion_conditioned` | same |
| Table 7 low-rank 2D | `table7_lowrank_2d` | same |
| Failure ladder | `cross_radar_failure_ladder` | same |

Entry: `python scripts/write_paper_package_v2.py`  
Wrapper added this phase: `python scripts/release/reproduce_from_frozen.py`

Numbers are read from the snapshot / registry. Plotting and table scripts must not hard-code manuscript metrics.
