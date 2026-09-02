# RadarMind Publication Repository Audit

**Date:** 2026-09-02  
**Workspace:** `C:\Users\13919\Desktop\RadarMind`  
**Phase:** Publication Release Audit only. No files deleted. No `git add`. No push.  
**Visibility (user decision):** **PRIVATE**. The GitHub repository must be visible only to the repository owner and GitHub user `ec12312312333`. This is not a public release.

---

## 0. Binding constraints

| Constraint | Status |
|---|---|
| Do not delete code before audit | Observed |
| Do not `git add` / push in this phase | Observed |
| Do not modify frozen scientific results | Observed |
| Do not retrain | Observed |
| Do not invent a second repository | Observed |
| Do not rewrite verified scientific implementations for cosmetics | Observed |
| Repository visibility | **Private**. Owner + `ec12312312333` only |
| Paper authorship vs project contributorship | Keep separate. `ec12312312333` is a project contributor, not a manuscript author unless already listed |

---

## 1. Repository state

| Item | Finding |
|---|---|
| Git repository | **No.** There is no `.git` at the workspace root. |
| Remote | None |
| Default branch | Not created |
| Root LICENSE | Missing |
| Root README | `README.md` is a **RadarBridge** legacy document |
| Canonical app README | `README_RADARMIND.md` (RadarMind v0.2, 2026-08) |
| Installable package | No `pyproject.toml` / `setup.py`. Runtime is repo-root + `sys.path` |
| GitHub CLI (`gh`) | Not installed on this machine |
| Total working tree (approx.) | `results/` 6.09 GB + `data/` 174 MB + `third_party/` 129 MB + other |
| Checkpoints | 120 `*.pt` files, **3.56 GB** |
| HDF5 derived data | 11 `*.h5` files, **1.38 GB** |
| NPZ caches | 3679 `*.npy`/`*.npz` files, **1.09 GB** |

Because there is no Git history, there is also no historical secret-in-commits problem **yet**. The first `git init` must not ingest `.env`, raw data, or checkpoints.

---

## 2. Classification legend

| Class | Meaning |
|---|---|
| **A. PUBLIC CORE** | Required to reproduce paper tables/figures or to run the research application against frozen outputs |
| **B. PUBLIC OPTIONAL** | Research app extras, acquisition toolkit, presentation helpers. Not required for metric reproduction |
| **C. PRIVATE / EXCLUDE** | Raw data, secrets, caches, checkpoints, IDE/build debris, personal paths |
| **D. LEGACY** | RadarBridge-era or superseded paths. Keep on disk until later review; do not treat as the publication entry point |
| **E. UNKNOWN** | Purpose not fully confirmed. **Do not delete** |

“PUBLIC” here means “eligible for the shared scientific repository.” The repository itself remains **private**.

---

## 3. Top-level inventory

| Path | Purpose | Used by manuscript? | Required for reproduction? | Safe to share in private repo? | Generated data? | Secrets? | Class | Action | Reason |
|---|---|---|---|---|---|---|---|---|---|
| `radarmind/` | Runtime package: adapters, engine, models, Motion Intelligence, LLM, Streamlit | Yes (application + MI + language) | Yes (app + inference) | Yes | No | No (reads `.env` only) | A | KEEP | Canonical RadarMind package |
| `scripts/ci4r/` | CI4R 77 GHz cache + ResNet18 training | Yes (E1) | Yes (full training) | Yes | No | No | A | KEEP | Closed-set CI4R 11-class |
| `scripts/pointcloud/` | mRI / RadHAR pack + PointNet-GRU / Voxel CNN-GRU | Yes (E1, E2) | Yes | Yes | No | REVIEW: some hardcoded server paths | A | KEEP | Subject-independent + official RadHAR |
| `scripts/cross_radar/` | Pairing, PNG cache, 3×3 transfer, CORAL | Yes (E3) | Yes | Yes | No | REVIEW: path constants | A | KEEP | Source-only cross-radar |
| `scripts/shared_motion/` | M0 / M1 / M2, InfoNCE, LORO | Yes (E4) | Yes | Yes | No | No | A | KEEP | Shared representation |
| `scripts/continuous/` | Labeled-interval replay | Yes (E8) | Yes | Yes | No | No | A | KEEP | Continuous inference |
| `scripts/open_set/` | 13-fold LOAO, MSP / Energy / Prototype | Yes (E9) | Yes | Yes | No | No | A | KEEP | Open-set rejection |
| `scripts/phase7/` | R1, M3 + GRL, Radar-ID | Yes (E4, E5, E6) | Yes | Yes | No | No | A | KEEP | Invariance / R1 |
| `scripts/phase8/` | R2-D numerical Doppler | Yes (E6) | Yes | REVIEW: `/home/radar_har` in `audit.py` | No | No | A | KEEP | Physical representation |
| `scripts/phase9/` | A0 / A1 / A2 adapters | Yes if Table/Figure retained | Yes for those rows | Yes | No | No | A | KEEP | Radar-specific adapters |
| `scripts/phase10/` | U1/U2, D(v\|q), 2D TEST oracle | Yes (E6, E7) | Yes | Yes | No | No | A | KEEP | Physics / difference models |
| `scripts/models/` | ResNet18, PointNet-GRU, Voxel CNN-GRU | Yes | Yes | Yes | No | No | A | KEEP | Shared model definitions |
| `scripts/freeze/` | Frozen package builder | Supporting | Yes (registry packaging) | Yes | No | No | A | KEEP | Provenance |
| `scripts/paper/` `scripts/paper_v2/` `scripts/paper_figures/` | Table/figure rebuild from frozen files | Yes | Yes (figure/table reproduction) | Yes | No | No | A | KEEP | Must read frozen results, not hardcode numbers |
| `scripts/full_manuscript/` `scripts/final_delivery/` | Manuscript / Word / PPT writers | Supporting | Optional | REVIEW | No | No | B | REVIEW | Delivery helpers, not metric sources |
| `src/paper/` | Figure style / activity colors | Yes (figures) | Yes for figure rebuild | Yes | No | No | A | KEEP | Import root `paper.*` |
| `data_splits/` | Fixed splits | Yes | **Yes** | Yes (no absolute paths found) | No | No | A | KEEP | Canonical splits |
| `run_*.py` | One-shot experiment / app entry points | Yes | Yes | Yes | No | No | A | KEEP | Existing CLI; do not invent a second CLI |
| `smoke_test_all.py` `validate_paper_package_v2.py` | Integrity / number audit | Supporting | Yes | Yes | No | No | A | KEEP | Scientific validation |
| `README_RADARMIND.md` | Current scientific README | Partial | Docs | Yes | No | No | A | KEEP then rewrite as `README.md` | Replace RadarBridge README |
| `README.md` | Legacy RadarBridge setup | No | No | Partial (local paths) | No | No | D | MOVE/ARCHIVE after rewrite | Must not remain the landing page |
| `requirements.txt` `environment.yml` `requirements_lock.txt` | Environment | Supporting | Yes | `environment.yml` has `C:\Users\13919\...` | No | No | A | KEEP after scrub | Do not `pip freeze` again blindly |
| `.gitignore` | Ignore rules | — | Yes | Yes | — | — | A | KEEP and expand | Incomplete for publication |
| `.env.example` | LLM key template | Yes (language layer) | Yes | Yes | No | Placeholder only | A | KEEP | Expand names later |
| `.env` | Live DeepSeek key | No | No | **No** | No | **Yes** | C | EXCLUDE | Never commit. Rotate key |
| `results/phase1_ci4r/` metrics JSON/CSV | Frozen CI4R 11-class | Yes | Yes (metrics) | Metrics yes; `*.pt` no | Yes | No | A / C | KEEP metrics; EXCLUDE `*.pt` | Checkpoints 42.7 MB each |
| `results/phase1_pointcloud/` metrics + cache | mRI / RadHAR | Yes | Metrics yes; NPZ optional | Metrics yes | Yes | Path leakage in `failed_samples.csv` | A / C | KEEP metrics; EXCLUDE cache/failed_samples from git | `failed_samples.csv` has ~23k `/home/radar_har` paths |
| `results/phase2_cross_radar/` | R0 3×3 + CORAL + pairing | Yes | Metrics + `pairing_stats.json` yes | Metrics yes | Yes | No in metrics | A / C | KEEP metrics/pairing; EXCLUDE PNG cache NPZ | Source of 0.843 / 0.898 / 0.922 / 0.039 |
| `results/phase3_shared_motion/` | M0–M2 + LORO | Yes | Metrics yes | Metrics yes | Yes | No | A / C | KEEP metrics; EXCLUDE `*.pt` | M1 acc 0.932, retrieval 0.935 |
| `results/phase4_continuous/` | Continuous replay | Yes | Yes | Yes (small) | Yes | No | A | KEEP metrics/reports | Frame 0.489 / 0.416 |
| `results/phase5_open_set/` | LOAO | Yes | Yes | Yes | Yes | No | A | KEEP metrics; EXCLUDE fold `*.pt` | Prototype AUROC 0.628 ± 0.125 |
| `results/phase7_radar_invariance/` | R1 / M3 / Radar-ID | Yes | Metrics yes | Metrics yes | Yes | No | A / C | KEEP metrics; EXCLUDE caches/`*.pt` | |
| `results/phase8_physical_representation/` | R2-D | Yes | Metrics/answers yes | Manifests leak Windows paths | Yes | No | A / C | KEEP answers/metrics; EXCLUDE `cache_r2.tar` (172 MB) and path-heavy CSVs | |
| `results/phase9_radarbridge_v2/` | A0–A2 | Yes if retained | Metrics yes | Yes | Yes | No | A / C | KEEP metrics; EXCLUDE 42 MB `best.pt` | |
| `results/phase10_*` | Physics chain | Yes | Answers/reports yes | Some manifests leak server paths | Yes | No | A / C | KEEP reports/answers; EXCLUDE large caches | TEST oracle must stay labeled diagnostic |
| `results/frozen_2026_08/` | Freeze ledger | Supporting | Yes | Yes | Yes | No | A | KEEP | Checkpoint SHA256 + findings |
| `results/final_delivery/metric_snapshot_full.json` | 211-key audited snapshot | Yes | Yes | Yes | Yes | No | A | KEEP | Canonical number registry |
| `results/paper_package_v2/` | Generated tables / claims | Yes | Yes (tables) | Yes | Yes | No | A | KEEP | Table reproduction source |
| `results/final_delivery/RadarMind_Full_Manuscript.md` | Generated manuscript | Supporting | No (do not treat as published paper) | REVIEW | Yes | No | B | REVIEW | Useful internally; not a substitute for the journal file |
| `figures/main/fig*.svg` | Rebuilt paper figures | Yes | Yes | Yes | Yes | Manifest has old absolute paths | A | KEEP SVGs; scrub `figure_manifest.json` | Rebuild from frozen metrics |
| `data_splits/ci4r_subject_independent.csv` | CI4R prefix-group split | Yes, but **name is misleading** | Yes | Yes | No | No | A | KEEP file; document / later rename only with import updates | Protocol column is `prefix_tail4_disjoint`. **Not subject identity** |
| `data_splits/mri_subject_independent.csv` | Official mRI SI | Yes | Yes | Yes | No | No | A | KEEP | Test subjects include 3, 10, 20 |
| `data_splits/radhar_official.csv` | Official Train/Test | Yes | Yes | Yes | No | No | A | KEEP | No subject-independent claim |
| `data_splits/ci4r_shared_motion_acquisition_split.csv` | 229 / 49 / 49 | Yes | Yes | Yes | No | No | A | KEEP | Acquisition-timestamp split |
| `data_splits/ci4r_cross_radar_timestamp.csv` | Pairing index | Yes | Yes | Yes | No | No | A | KEEP | |
| `data/` (`CI4R_MULTI3` PNG, 1959 files, 174 MB) | Official spectrogram copies | Input to cache build | For training from scratch | **Do not redistribute by default** | Dataset copy | No | C | EXCLUDE | Public source exists; license for secondary distribution not confirmed here |
| `results/data/*.h5` (125 MB × several) | Packed spectrogram HDF5 | Legacy RadarBridge | No for frozen-metric repro | No (size + derived dataset) | Yes | No | C | EXCLUDE | Already gitignored as `*.h5` |
| `results/splits/loso_77GHz/*.h5` | Legacy LOSO packs | Legacy | No | No | Yes | No | C / D | EXCLUDE | 125 MB copies |
| `third_party/AIRHAR/` | Upstream AIRHAR/RadMamba | Historical Phase 1A | No for current RadarMind tables | Nested `.git`; Apache-2.0; contains 125 MB `ci4r_data.h5` | Yes | No | D / C | EXCLUDE from first commit; attribute if any file is later vendored | Paper models are `scripts/models/resnet18_spec.py`, not AIRHAR training |
| `third_party/CI4R-MULTI3/` | Upstream data-repo scripts | Historical | No | Nested `.git`; **no LICENSE file** | No | No | D | EXCLUDE from first commit | Cite upstream URL instead |
| `competitors/dap_net_official/` | DAP-Net clone | Not in frozen RadarMind tables | No | Nested `.git`; no LICENSE | No | No | D / C | EXCLUDE | Competitor reproduction notes stay local |
| `research/dap_net/` | DAP-Net audit notes | No | No | Third-party email only | No | No | E / B | REVIEW | Not paper-core |
| `demo/` | RadarBridge Explorer Streamlit | No | No | Yes | No | No | D | EXCLUDE from scientific landing | Superseded by `radarmind/app/streamlit_app.py` |
| `radarmind_acquisition/` | Lab acquisition toolkit (MOCK) | No | No | Yes | Mock only | No | B | OPTIONAL later | Not required to reproduce paper numbers |
| `experiments/png_physics_probe/` | Side probe + large CSVs | No | No | Path leakage | Yes | No | C / E | EXCLUDE from git | 114 MB |
| `gpu_jobs/` | Remote GPU launchers | No | No | Contains `C:\Users\13919` | No | No | C | EXCLUDE | Machine-specific |
| `build_artifacts/` | PPT/docx build + node_modules | No | No | No | Yes | No | C | EXCLUDE | 9.8 MB + licenses noise |
| `build_radarmind_presentation.js` | PPT builder | No | No | Yes | No | No | B | REVIEW | Presentation, not science |
| `Draft-RadarMind.docx` `.pdf` | Draft manuscript | Supporting | No | REVIEW (author draft) | Yes | No | B / C | EXCLUDE from git unless user later asks | 7.3 MB / 1.9 MB |
| `RadarMind_Presentation.pptx` | Internal slides | No | No | REVIEW | Yes | No | B / C | EXCLUDE from first commit | 0.71 MB |
| `notes/` | Dev notes, GCS listing | No | No | Some machine notes | No | No | C / D | EXCLUDE | Internal |
| `reports/` | Internal audits including this file | This audit | Audit trail | Yes after scrub | Yes | No | B | KEEP audit reports | Later: `reports/release_audit/` |
| `research/` | DAP-Net notes | No | No | Yes | No | No | E | REVIEW | Do not delete |
| `deliverables/` | Empty on disk (size 8.38 MB reported at listing time; verify before delete) | Unknown | Unknown | Unknown | Unknown | Unknown | E | REVIEW | Do not delete |
| `renders/` | 2.37 MB | Unknown | Unknown | Unknown | Likely generated | Unknown | E | REVIEW | Do not delete |
| `apps/` | Does not exist | — | — | — | — | — | — | Do not invent yet | Current app is `radarmind/app/` + `run_radarmind.py` |
| `configs/` | Does not exist as root tree | — | — | — | — | — | — | Do not invent yet | Configs live beside phase scripts |
| `splits/` | Does not exist; use `data_splits/` | — | — | — | — | — | — | KEEP current name | Renaming breaks loaders |

---

## 4. Manuscript coverage vs repository

Frozen numbers below are read from `results/final_delivery/metric_snapshot_full.json`, which itself points at phase `metrics.json` / CSV files. They are **not** hard-coded in this audit as model outputs.

### A. Dataset / preprocessing

| Dataset | Loader | Split | Preprocess | Raw data in repo? |
|---|---|---|---|---|
| CI4R 77 GHz 11-class | `radarmind/data/ci4r_adapter.py`, `scripts/ci4r/` | `data_splits/ci4r_random.csv`, `ci4r_subject_independent.csv` (**prefix-group, not SI**) | `scripts/ci4r/phase1a_build.py` | Local PNG/H5 copies exist; **do not git-add** |
| CI4R 10/24/77 7-class paired | `radarmind/data/ci4r_band_adapter.py`, `scripts/cross_radar/` | `ci4r_cross_radar_timestamp.csv`, `ci4r_shared_motion_acquisition_split.csv` (229/49/49) | `scripts/cross_radar/pack_cache.py`, `audit_ci4r.py` | Same |
| mRI | `radarmind/data/mri_adapter.py`, `scripts/pointcloud/` | `mri_subject_independent.csv`, LOSO folds | `scripts/pointcloud/phase1b_build.py` | Raw CSV not in this tree |
| RadHAR | `radarmind/data/radhar_adapter.py` | `radhar_official.csv` | `phase1b_build.py`, `pack_voxels.py` | Raw not in this tree |

Pairing facts (from `results/phase2_cross_radar/pairing_stats.json`):

- 327 agreed timestamp-aligned triples
- 133 inconsistent-label triples excluded
- Shared 7-action subset: Away, Bend, Crawl, Kneel, Pick, Sit, Towards

### B. Recognition

| Model | Definition | Training | Frozen metrics present |
|---|---|---|---|
| ResNet18 spectrogram | `scripts/models/resnet18_spec.py` | `scripts/ci4r/phase1a_train.py`, `scripts/cross_radar/train_cross_radar.py` | Yes |
| PointNet-GRU | `scripts/models/point_encoder.py` | `scripts/pointcloud/phase1b_train.py` | Yes |
| Voxel CNN-GRU | `scripts/models/point_encoder.py` | `scripts/pointcloud/train_voxel.py` | Yes |

### C. Subject-independent evaluation

| Protocol | Dataset | File | Verdict |
|---|---|---|---|
| Official subject IDs, test subjects 3 / 10 / 20 | mRI | `data_splits/mri_subject_independent.csv` → Acc **0.478**, Macro-F1 **0.420** | This is the only verified SI claim |
| Prefix-group-disjoint | CI4R 77 GHz | `data_splits/ci4r_subject_independent.csv` protocol=`prefix_tail4_disjoint` | **Not subject-independent.** Filename is a documentation hazard |
| Official Train/Test | RadHAR | Acc **0.949**, F1 **0.949** | No SI claim |

### D. CI4R paired cross-radar (R0 source-only)

Frozen 3×3 (accuracy, from snapshot):

|  | →10 | →24 | →77 |
|---|---|---|---|
| 10 | 0.843 | 0.143 | 0.373 |
| 24 | 0.353 | 0.898 | 0.039 |
| 77 | 0.275 | 0.163 | 0.922 |

Sources: `results/phase2_cross_radar/runs/within_*` and `cross_*_to_*/metrics.json`.

### E. Domain generalization / adaptation

| Method | Code | Frozen | Label |
|---|---|---|---|
| Source-only transfer | `scripts/cross_radar/train_cross_radar.py` | 3×3 above | **source-only** |
| CORAL UDA | same script, `coral_uda_*` | `results/phase2_cross_radar/runs/coral_uda_*` | **UDA**, not DG. Negative result retained |
| LORO | `scripts/shared_motion/`, `scripts/phase7/train_m3.py`, `scripts/phase9/` | `*_loro_*` | **LORO domain generalization** |
| Known-radar joint training | M0–M3 shared runs | `M*_shared` | **not** zero-shot transfer |

### F. Shared representation

| Variant | Implementation | Snapshot keys (examples) |
|---|---|---|
| M0 CE | `scripts/shared_motion/train_shared_motion.py` | M0 activity / Radar-ID / retrieval |
| M1 CE + InfoNCE | same; canonical λ=0.1 is `M1_l01` | M1_ACC 0.932, M1_RET 0.935, M1_RID 1.000 |
| M2 physics-informed aug | same | M2_ACC 0.878 |
| M3 CE + InfoNCE + GRL | `scripts/phase7/train_m3.py` | `results/phase7_radar_invariance/` |
| A0 / A1 / A2 | `scripts/phase9/` | `results/phase9_radarbridge_v2/` |

### G. Physics-informed analysis

Present: R0, R1, R2-D, R2-V, U1, U2, D(v|q), low-rank 2D, B2, TEST oracle (`scripts/phase10/run_phase10g.py`).  
Required documentation later: TEST oracle = **diagnostic upper bound, not a deployable translator**; R² = **reconstruction metric, not HAR accuracy**.

### H. Continuous inference

`scripts/continuous/` + `radarmind/replay.py`. Frozen: frame Acc/F1 **0.489 / 0.416**.  
This is **labeled-interval replay**, not full-stream continuous sensing. No background class should be invented for the release.

### I. Open-set

`scripts/open_set/` 13-fold LOAO. Prototype AUROC **0.628 ± 0.125**.  
This is **open-set rejection**, not open-world / open-vocabulary. Unknown output must remain `UNKNOWN`.

### J. Motion Intelligence

`radarmind/intelligence/schema.py`: activity, confidence, direction, velocities, intensity, periodicity, duration, motion_state, uncertainty, dataset, radar, subject, environment, evidence. Provenance tags: MODEL / PHYSICS_DERIVED / METADATA / UNKNOWN. Unsupported quantities stay UNKNOWN.

### K. Language layer

`radarmind/llm/` (DeepSeek or template fallback). Key is read from environment. LLM must not receive raw IQ / spectrogram / point cloud / hidden features.

### L. Research application

Primary: `python run_radarmind.py` → `radarmind/app/streamlit_app.py` (:8502).  
No “RadarMind Studio” product exists. Do not invent one for the release.  
Legacy: `demo/app.py` (RadarBridge Explorer).

---

## 5. Recommended repository tree (minimal move)

Do **not** mechanically rename `radarmind/`, `scripts/`, or `data_splits/` to match a template. Imports already work from the repo root.

Recommended **first-commit scientific tree** (after later cleanup, not done in this phase):

```
RadarMind/
  README.md                         # rewritten scientific README (private-repo wording)
  LICENSE or LICENSE_SELECTION.md
  CITATION.cff
  CONTRIBUTORS.md                   # Li Luo, Xuehong Sun, @ec12312312333
  CODE_OF_CONDUCT.md
  CONTRIBUTING.md
  SECURITY.md
  requirements.txt
  environment.yml                   # scrub absolute path
  .gitignore                        # expanded
  .env.example
  radarmind/                        # KEEP in place
  scripts/                          # KEEP scientific phase trees
  src/paper/
  data_splits/                      # KEEP name
  results/
    frozen_2026_08/
    final_delivery/metric_snapshot_full.json
    paper_package_v2/               # tables / claims
    phase1_* … phase10_*/           # metrics/json/md only
  figures/main/                     # SVG
  run_*.py
  smoke_test_all.py
  validate_paper_package_v2.py
  docs/                             # to be written in next phase
  reports/                          # this audit set
  tests/                            # existing + later paper-metric tests
```

Explicitly **out of the first git commit**:

- `.env`
- `data/`
- `*.h5` `*.pt` `*.pth` `*.ckpt`
- `results/**/cache/` and `cache_r2.tar`
- `third_party/` nested clones
- `competitors/`
- `build_artifacts/`
- `gpu_jobs/`
- `notes/`
- `Draft-RadarMind.*` `RadarMind_Presentation.pptx`
- `experiments/png_physics_probe/data/`

---

## 6. KEEP / MOVE / EXCLUDE / REVIEW

### KEEP (do not relocate in the next phase unless an import breaks)

- `radarmind/`
- `scripts/{ci4r,pointcloud,cross_radar,shared_motion,continuous,open_set,phase7,phase8,phase9,phase10,models,freeze,paper,paper_v2,paper_figures}`
- `src/paper/`
- `data_splits/`
- `results/frozen_2026_08/`
- `results/final_delivery/metric_snapshot_full.json`
- `results/paper_package_v2/` tables and audits
- Phase `metrics.json` / `eval.json` / `answers.json` / pairing stats / representation CSVs
- `figures/main/fig*.svg`
- Root `run_*.py`, `smoke_test_all.py`, `validate_paper_package_v2.py`
- `.env.example`, current scientific scripts

### MOVE (only later, and only with import/path updates)

| Current | Proposed | Why later, not now |
|---|---|---|
| `README_RADARMIND.md` content | become `README.md` | Landing page must be RadarMind, not RadarBridge |
| `README.md` (RadarBridge) | `docs/legacy/README_RadarBridge.md` or stay untracked | Avoid two competing landing pages |
| New audit reports | `reports/release_audit/` | Optional tidy; not required for science |
| Streamlit app | stay in `radarmind/app/`; optional symlink `apps/research_app/` | Do not break `run_radarmind.py` |

Do **not** rename `data_splits/` → `splits/` in the next phase.

### EXCLUDE (never git-add)

- `.env` (live key)
- `data/CI4R_MULTI3/` (1959 official PNGs)
- All `*.h5` (1.38 GB)
- All `*.pt` / `*.pth` (3.56 GB)
- `results/phase8_physical_representation/cache_r2.tar` (172 MB)
- NPZ caches under `results/*/cache/`
- `third_party/AIRHAR/datasets/CI4R/ci4r_data.h5` and nested `.git`
- `build_artifacts/`, `gpu_jobs/`, `experiments/png_physics_probe/data/`
- Path-dump CSVs such as `results/phase1_pointcloud/failed_samples.csv`
- `__pycache__/`, `.idea/`, `.vscode/`

### REVIEW (do not delete; decide in the organization phase)

| Item | Question |
|---|---|
| `data_splits/ci4r_subject_independent.csv` filename | Document as prefix-group now; rename only if all loaders are updated |
| Checkpoint delivery | Local-only vs GitHub Release (private) vs documented “train to reproduce” |
| NPZ caches (~100 MB useful ones) | App convenience vs license/size. Default: do not commit |
| `radarmind_acquisition/` | Optional module vs omit from v0.1 scientific commit |
| `demo/` | Archive vs omit |
| `third_party/` | Cite + exclude vs sparse attribution copies without nested `.git` |
| `competitors/` + `research/dap_net/` | Keep local only |
| Draft Word/PDF/PPT | Keep local; not part of the code commit |
| `deliverables/`, `renders/` | Inspect before any delete |
| LICENSE | User has not chosen. Prepare `LICENSE_SELECTION.md` (MIT vs Apache-2.0) |
| `ci4r_loso_fold*.csv` | Confirm whether manuscript uses CI4R LOSO (mRI LOSO is used for open-set) |

---

## 7. Blockers before any `git init` + first commit

1. **Secret:** `.env` contains a live DeepSeek API key. It is already listed in `.gitignore`, but `.gitignore` must be verified after `git init` **before** any add. Recommend rotating the key because it exists in plaintext on this disk.
2. **No Git yet:** first commit must be curated. `git add .` would ingest datasets and checkpoints if ignore rules are incomplete.
3. **LICENSE unset.**
4. **RadarBridge README** would become the GitHub landing page if committed as-is.
5. **Absolute paths** in several result CSVs / `environment.yml` / figure manifests.
6. **Misleading split filename** `ci4r_subject_independent.csv`.
7. **Checkpoints are local-only** (already ignored). Reproduction from frozen **metrics** works without them; full training reproduction does not.
8. **`gh` is not installed.** Private GitHub creation cannot happen from this machine until GitHub CLI or equivalent is available.
9. Nested `.git` in `third_party/` and `competitors/` would confuse a parent repository if those trees are added.

None of these require deleting scientific code.

---

## 8. What this phase did not do

- No files deleted
- No code moved
- No scientific numbers edited
- No training
- No `git init`
- No remote
- No collaborator permission change (documentation only)

Next phase (only after you confirm): expand `.gitignore`, write publication docs, add contributor/citation files, prepare a curated first commit, then create a **private** GitHub repository named `RadarMind` and grant `ec12312312333` access.
