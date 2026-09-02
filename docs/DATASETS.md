# Datasets

RadarMind uses three public research corpora. Raw recordings are **not** distributed with this repository. Place local copies outside git and point rebuild scripts at them with environment variables (see `.env.example`).

## CI4R

Multi-frequency human activity radar data at 10 GHz, 24 GHz, and 77 GHz.

### Paired cross-radar subset (7-class)

Used for within-radar and source-only cross-radar evaluation, shared representation, and later physical diagnostics.

| Item | Value |
|---|---|
| Pairing | timestamp-aligned triples across 10 / 24 / 77 GHz |
| Agreed-action triples | 327 |
| Excluded triples | 133, same timestamp but disagreeing action labels across sensors |
| Shared inventory | 7 classes: Away, Bend, Crawl, Kneel, Pick, Sit, Towards |
| Split | acquisition-timestamp: 229 train / 49 validation / 49 test |
| Split file | `data_splits/ci4r_shared_motion_acquisition_split.csv` |
| Pairing index | `data_splits/ci4r_cross_radar_timestamp.csv` |
| Pairing stats | `results/phase2_cross_radar/pairing_stats.json` |
| Representation (R0) | official plotted spectrogram PNG |

This split is **not** subject-independent.

### CI4R 77 GHz 11-class

Used for closed-set spectrogram HAR (ResNet18).

| Protocol | Split file | Meaning |
|---|---|---|
| random-record | `data_splits/ci4r_random.csv` | record-level random split |
| prefix-group-disjoint | `data_splits/ci4r_subject_independent.csv` | filename prefix groups are disjoint across splits |

**PREFIX GROUP ≠ SUBJECT IDENTITY.**  
The second file keeps a legacy name. See `docs/KNOWN_NAMING_ISSUES.md`. Do not report this protocol as subject-independent evaluation.

### Expected local layout (not in git)

Official spectrograms and, if rebuild from IQ is required, the public raw archive. Rebuild entries:

- `scripts/ci4r/phase1a_build.py` (`CI4R_RAW_ZIP` / `RADARMIND_CI4R_ZIP`)
- `scripts/cross_radar/pack_cache.py`

Do not add `data/` or `*.h5` copies to git.

## mRI

Point-cloud human motion corpus with official subject identifiers.

| Item | Value |
|---|---|
| Subjects | 20 |
| Labeled segments | 273 |
| Normalized activities | 13 |
| Window | T = 32, P ≤ 64 |
| Subject-independent split | `data_splits/mri_subject_independent.csv` |
| Test subjects | 3, 10, 20 |
| Also used for | labeled-interval continuous replay; 13-fold LOAO open-set |

This is the only local dataset on which RadarMind reports **subject-independent** recognition.

Rebuild: `scripts/pointcloud/phase1b_build.py` with `RADARMIND_MRI_ROOT`.

## RadHAR

Point-cloud activity corpus.

| Item | Value |
|---|---|
| Actions | 5 |
| Split | official train / test (`data_splits/radhar_official.csv`) |
| Window | T = 60, P ≤ 32 |
| Subject metadata in the local package | not available |

RadHAR results in this repository are **not** subject-independent.

Rebuild: `scripts/pointcloud/phase1b_build.py` with `RADARMIND_RADHAR_ROOT`. Voxel pack: `scripts/pointcloud/pack_voxels.py`.

## Source notes

Download each corpus from its official public source. This repository records loaders, fixed splits, and frozen metrics. It does not re-host the raw archives.
