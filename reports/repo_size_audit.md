# RadarMind Repository Size Audit

**Date:** 2026-09-02  
**Workspace:** `C:\Users\13919\Desktop\RadarMind`  
**Git:** not initialized (sizes are working-tree files, not tracked size)

---

## 1. Working-tree totals

| Location | Size (MB) | Role |
|---|---:|---|
| `results/` | 6087.92 | Frozen metrics + caches + checkpoints + derived HDF5 |
| `data/` | 173.60 | Official CI4R spectrogram PNG copy (1959 files) |
| `third_party/` | 129.16 | AIRHAR (125.49) + CI4R-MULTI3 (3.67) |
| `experiments/` | 114.11 | png_physics_probe side experiment |
| `competitors/` | 14.87 | DAP-Net clone |
| `build_artifacts/` | 9.78 | PPT/docx toolchain |
| `deliverables/` | 8.38 | Review before any delete |
| `Draft-RadarMind.docx` | 7.30 | Draft manuscript |
| `notes/` | 3.59 | Internal notes |
| `figures/` | 3.57 | Paper SVGs + manifest |
| `scripts/` | 2.49 | Scientific + delivery scripts |
| `renders/` | 2.37 | Review |
| `Draft-RadarMind.pdf` | 1.94 | Draft manuscript |
| `demo/` | 1.77 | Legacy Streamlit |
| `reports/` | 1.48 | Audits |
| `data_splits/` | 1.06 | Fixed splits (**keep**) |
| `RadarMind_Presentation.pptx` | 0.71 | Slides |
| `radarmind/` | 0.57 | Runtime package (**keep**) |
| Other small files | <0.3 | README, requirements, runners |

**Approximate working tree:** ~6.6 GB.

---

## 2. Generated-artifact class totals

| Class | Count | Size (MB) | Default git decision |
|---|---:|---:|---|
| `*.pt` / `*.pth` / `*.ckpt` | 120 | 3556.39 | **EXCLUDE** (already in `.gitignore` as `results/**/*.pt`) |
| `*.h5` | 11 | 1375.22 | **EXCLUDE** (already `*.h5`) |
| `*.npy` / `*.npz` | 3679 | 1092.68 | **EXCLUDE** by default; do not blanket-LFS all npy |
| `cache_r2.tar` | 1 | 172.18 | **EXCLUDE** |
| Official PNG under `data/CI4R_MULTI3` | 1959 | ~174 | **EXCLUDE** (dataset copy) |

If these were committed, a GitHub private repo would be multi-gigabyte and slow to clone. Frozen **metrics** needed for paper numbers are kilobytes to a few megabytes.

---

## 3. `results/` breakdown

| Directory | MB | Keep in git? |
|---|---:|---|
| `phase7_radar_invariance` | 1472.41 | metrics only |
| `phase8_physical_representation` | 903.58 | answers/metrics; not `cache_r2.tar` |
| `data` | 750.12 | no (`*.h5`) |
| `phase10_unified_measurement` | 522.31 | reports/answers; not caches |
| `phase3_shared_motion` | 513.45 | metrics only |
| `splits` | 500.55 | no (legacy `*.h5`) |
| `phase2_cross_radar` | 488.71 | metrics + pairing JSON |
| `phase9_radarbridge_v2` | 429.33 | metrics only |
| `phase1_pointcloud` | 123.04 | metrics; not cache/failed_samples |
| `phase10_velocity_calibration` | 115.98 | answers/reports |
| `phase1_ci4r` | 113.12 | metrics |
| `runs` | 43.93 | no (`best.pt`) |
| `phase4_continuous` | 41.47 | yes (small enough if no huge binaries) |
| `data_audit` | 35.46 | review |
| `phase5_open_set` | 18.60 | metrics/eval JSON; not fold `*.pt` |
| `final_delivery` | 5.89 | `metric_snapshot_full.json` yes; Word/PPT audits optional |
| `frozen_2026_08` | 0.04 | **yes** |
| `paper_package_v2` | 0.53 | **yes** |

---

## 4. Top 50 largest files

Sizes rounded to 0.01 MB. Paths relative to workspace root.

| MB | Path | Candidate |
|---:|---|---|
| 172.18 | `results/phase8_physical_representation/cache_r2.tar` | EXCLUDE |
| 125.02 | `results/data/ci4r_77GHz.h5` | EXCLUDE dataset pack |
| 125.02 | `third_party/AIRHAR/datasets/CI4R/ci4r_data.h5` | EXCLUDE |
| 125.02 | `results/data/ci4r_24GHz.h5` | EXCLUDE |
| 125.02 | `results/data/ci4r_10GHz.h5` | EXCLUDE |
| 125.02 | `results/splits/loso_77GHz/held_0029.h5` | EXCLUDE |
| 125.02 | `results/splits/loso_77GHz/held_0030.h5` | EXCLUDE |
| 125.02 | `results/data/cross_10GHz_to_77GHz.h5` | EXCLUDE |
| 125.02 | `results/splits/loso_77GHz/held_0014.h5` | EXCLUDE |
| 125.02 | `results/data/cross_24GHz_to_77GHz.h5` | EXCLUDE |
| 125.02 | `results/splits/loso_77GHz/held_0010.h5` | EXCLUDE |
| 125.02 | `results/data/cross_77GHz_to_24GHz.h5` | EXCLUDE |
| 42.83 | `results/phase9_radarbridge_v2/checkpoints/A2_lam0p1/best.pt` | EXCLUDE / optional private Release |
| 42.83 | `.../A2_lam0p0/best.pt` | same |
| 42.83 | `.../A2_lam0p5/best.pt` | same |
| 42.83 | `.../A2_lam1p0/best.pt` | same |
| 42.83 | `.../A1/best.pt` | same |
| 42.83 | `.../A2/best.pt` | same |
| 42.78 | `.../loro_10_77_to_24/best.pt` | same |
| 42.78 | `.../loro_10_24_to_77/best.pt` | same |
| 42.78 | `.../loro_24_77_to_10/best.pt` | same |
| 42.70 | `results/runs/bench_resnet18_seed0/best.pt` | EXCLUDE |
| 42.70 | many `results/phase7_radar_invariance/runs/*/best.pt` | EXCLUDE |
| 42.70 | `results/phase1_ci4r/baselines/*/resnet18/best.pt` | EXCLUDE |
| 42.69 | `results/phase3_shared_motion/runs/M1_*/best.pt` | EXCLUDE |
| 42.69 | `results/phase9_radarbridge_v2/checkpoints/A0/best.pt` | EXCLUDE |
| 41.50 | `results/phase1_pointcloud/cache/radhar_voxels.npz` | EXCLUDE cache |
| 33.70 | `results/phase1_pointcloud/cache/mri_windows.npz` | EXCLUDE cache |
| 26.19 | `results/phase1_ci4r/cache/ci4r77_airhar_spec_zscore.npz` | EXCLUDE cache |
| 25.92 | `results/phase1_pointcloud/cache/radhar_windows.npz` | EXCLUDE cache |
| 9.42 | `results/phase7_radar_invariance/cache/ci4r_77ghz_r1_png224.npz` | EXCLUDE cache |
| 8.22 | `results/phase2_cross_radar/cache/ci4r_77ghz_png224.npz` | EXCLUDE cache |
| 7.30 | `Draft-RadarMind.docx` | EXCLUDE from first commit |
| 5.03 | `results/phase7_radar_invariance/cache/ci4r_24ghz_r1_png224.npz` | EXCLUDE cache |

Point-cloud checkpoints in the freeze manifest are much smaller (≈0.5 MB for mRI / RadHAR / LOAO folds). Those are still generated weights; default remains: **do not git-add `*.pt`**.

---

## 5. Useful caches (not in Top 50 as a class, but relevant)

| MB | Path | Note |
|---:|---|---|
| 41.50 | `results/phase1_pointcloud/cache/radhar_voxels.npz` | App / voxel training convenience |
| 33.70 | `results/phase1_pointcloud/cache/mri_windows.npz` | App / mRI SI |
| 26.19 | `results/phase1_ci4r/cache/ci4r77_airhar_spec_zscore.npz` | CI4R 11-class |
| 25.92 | `results/phase1_pointcloud/cache/radhar_windows.npz` | RadHAR windows |
| 8.22 | `results/phase2_cross_radar/cache/ci4r_77ghz_png224.npz` | Cross-radar PNG cache |

These are **derived**, not raw IQ. They still reproduce dataset content and should not go into git without a license decision. Prefer documenting how to rebuild them.

---

## 6. Git LFS decision

Do **not** enable blanket LFS for `*.npy` / `*.npz` / `*.pt`.

| Candidate | LFS? | Reason |
|---|---|---|
| `*.pt` / `*.pth` | No for first commit | 3.56 GB. If later shared, use a **private GitHub Release** or documented local path, not LFS on every weight |
| `*.h5` | No | Dataset packs; ignore |
| `cache_r2.tar` | No | 172 MB derived archive |
| Paper SVGs in `figures/main/` | No | Few MB; regular git is enough |
| Split CSVs | No | Text, required |
| `metric_snapshot_full.json` | No | 211 metrics, small |
| Random `*.npy` | No | 3679 files; most are experiment debris |

LFS is **not recommended** until a specific small set of weights is chosen and licensed.

---

## 7. Estimated first-commit size (if ignore rules are followed)

If the first commit includes:

- `radarmind/`, `scripts/` (scientific), `src/paper/`, `data_splits/`
- frozen metrics JSON/CSV/MD (not caches, not `*.pt`, not `*.h5`)
- `figures/main/*.svg`
- docs / README / citation / tests

then a reasonable target is **tens of MB, not GB**.

If `git add .` is used with the **current** `.gitignore` only:

- `*.h5` and `results/**/*.pt` are ignored
- **but** `data/CI4R_MULTI3` PNGs (174 MB), `*.npz` caches (~1 GB), `cache_r2.tar` (172 MB), `third_party/AIRHAR` (minus some logs), and `experiments/` would still be at risk

**Current `.gitignore` is not sufficient** for a clean first commit.

---

## 8. Dataset exclusion list

| Path | Why excluded |
|---|---|
| `data/CI4R_MULTI3/` | Official spectrogram copy (1959 PNG). Public source exists. Secondary redistribution not confirmed |
| `results/data/*.h5` | Packed CI4R spectrograms |
| `third_party/AIRHAR/datasets/CI4R/ci4r_data.h5` | Same content class |
| `results/splits/loso_77GHz/*.h5` | Legacy 125 MB copies |
| Raw IQ (`raw_data.zip` ~93 GB) | **Not present** in this workspace. Must stay that way |

mRI and RadHAR raw recordings were not found as complete public-dataset dumps in this workspace.

---

## 9. Recommendation

1. Expand `.gitignore` before `git init`.
2. Do not use Git LFS in the first commit.
3. Track frozen **metrics**, splits, code, and figure SVGs.
4. Leave checkpoints on this machine; document SHA256 from `results/frozen_2026_08/artifact_manifest.csv`.
5. After `git init`, run `git status` and a size check before the first commit.
