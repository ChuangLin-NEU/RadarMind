# Reproducibility

This repository distinguishes three layers:

1. **Quick verification** — imports, splits, frozen snapshot, schema. No training.
2. **Frozen-result reproduction** — rebuild tables and figures from stored metrics.
3. **Full training reproduction** — rebuild caches and retrain. Costly, GPU-dependent, not bitwise deterministic.

The manuscript numbers are the frozen snapshot, not a later training run.

## Environment

Documented local runtime for RadarMind v0.2:

- Python 3.11
- conda environment name `airhar`
- PyTorch 2.7.1 + a CUDA wheel that matches the local GPU
- `requirements.txt` for scientific libraries
- `requirements-app.txt` for the Streamlit research application

```
conda activate airhar
python -m pip install -r requirements.txt
python -m pip install -r requirements-app.txt
```

Install torch from the CUDA index appropriate to the machine before or after the requirements file. The historical `requirements_lock.txt` is a machine pin and is not the publication dependency list.

GPU training used an NVIDIA RTX 5060 Laptop in the original v0.2 freeze. Other GPUs can train the same configs; expect small numeric drift.

## Datasets

See `docs/DATASETS.md`. Raw archives are local. Rebuild scripts read:

- `CI4R_RAW_ZIP` or `RADARMIND_CI4R_ZIP`
- `RADARMIND_MRI_ROOT`
- `RADARMIND_RADHAR_ROOT`

## Fixed splits

| File | Protocol |
|---|---|
| `data_splits/mri_subject_independent.csv` | mRI subject-independent (test subjects 3, 10, 20) |
| `data_splits/mri_random_segment.csv` | mRI random-segment (not SI) |
| `data_splits/mri_loso_fold01.csv` … `fold20.csv` | mRI LOSO folds |
| `data_splits/radhar_official.csv` | RadHAR official train/test |
| `data_splits/ci4r_random.csv` | CI4R 77 GHz 11-class random-record |
| `data_splits/ci4r_subject_independent.csv` | CI4R prefix-group-disjoint (**not** SI) |
| `data_splits/ci4r_shared_motion_acquisition_split.csv` | 229 / 49 / 49 paired acquisition split |
| `data_splits/ci4r_cross_radar_timestamp.csv` | timestamp pairing index |

Do not recut test sets to raise a number.

## Checkpoints

Weights (`*.pt`) are not in git. See `docs/CHECKPOINTS.md`. Frozen **metrics** do not require loading weights.

## Frozen results

Canonical display values: `results/final_delivery/metric_snapshot_full.json`  
Registry that maps each key to a source file: `scripts/paper_v2/registry.py`  
Freeze ledger: `results/frozen_2026_08/`

```
python scripts/release/validate_metric_registry.py
python scripts/release/quick_verify.py
```

## Quick verification

```
python scripts/release/quick_verify.py
```

Expected: `RADARMIND QUICK VERIFY PASS`. Runtime target is one to two minutes on a local disk with the frozen files present.

## Full evaluation from frozen outputs

```
python scripts/release/reproduce_from_frozen.py
```

This calls the existing paper package and figure rebuilders. It does not train.

Equivalent existing entries:

```
python scripts/write_paper_package_v2.py
python scripts/rebuild_paper_figures.py
python validate_paper_package_v2.py
```

## Training reproduction

Only when a cache or checkpoint is missing. Examples of actual entries:

```
python scripts/ci4r/phase1a_build.py --cache-all
python scripts/ci4r/phase1a_train.py
python scripts/pointcloud/phase1b_build.py --mri-root ... --radhar-root ...
python scripts/pointcloud/phase1b_train.py
python run_cross_radar_experiments.py
python run_shared_motion_experiments.py
python run_open_set_experiments.py
python run_continuous_demo.py
```

Do not overwrite an existing frozen `metrics.json` with a new run.

## Figure and table generation

Tables: `scripts/paper_v2/tables.py` via `scripts/write_paper_package_v2.py`  
Figures: `scripts/paper_figures/rebuild.py` via `scripts/rebuild_paper_figures.py`  
Both read `results/final_delivery/metric_snapshot_full.json` or rebuild that snapshot from the registry. Plotting scripts must not hard-code manuscript numbers.

## Expected results

See `docs/generated/key_results.md` and `tests/expected_manuscript_snapshot.json`.  
Display rounding follows `scripts/paper_v2/snapshot.py` (`f3` → three decimals, `f1` → one decimal).

## Known sources of nondeterminism

- GPU kernels and cuDNN
- DataLoader worker order when training
- CUDA version and driver
- Floating-point reduction order

Do not claim bitwise reproducibility for trained weights. Frozen JSON/CSV metrics are exact on disk. Re-training should be compared with a tolerance appropriate to the protocol (for example 0.01–0.02 accuracy on GPU HAR), not `1e-6`.

## Hardware notes

- Training: GPU recommended
- Snapshot validation, table/figure rebuild, Streamlit: CPU is enough
- Language layer: optional network call; `--offline-demo` uses templates
