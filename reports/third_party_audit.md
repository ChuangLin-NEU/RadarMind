# RadarMind Third-Party Code Audit

**Date:** 2026-09-02  
**Workspace:** `C:\Users\13919\Desktop\RadarMind`  
**Root LICENSE:** missing. User has not chosen MIT vs Apache-2.0.

This audit records what was copied or cloned into the tree. RadarMind must not present third-party code as original.

---

## 1. Summary

| Tree | Origin | License in tree | Nested `.git` | Used by frozen RadarMind tables? | First-commit recommendation |
|---|---|---|---|---|---|
| `third_party/AIRHAR/` | https://github.com/lab-emi/AIRHAR.git | Apache-2.0 (`LICENSE`) | Yes | Historical CI4R path only. Current tables use `scripts/models/resnet18_spec.py` | **EXCLUDE** clone; keep attribution in docs |
| `third_party/CI4R-MULTI3/` | https://github.com/ci4r/CI4R-MULTI3.git | **No LICENSE file** | Yes | Historical download/scripts | **EXCLUDE**; cite dataset URL |
| `competitors/dap_net_official/` | https://github.com/jolin830/DAP-Net.git | **No LICENSE file** | Yes | No | **EXCLUDE** |
| `research/dap_net/` | Project notes about DAP-Net | N/A (notes) | No | No | REVIEW / keep local |
| `scripts/models/resnet18_spec.py` | Thin wrapper over `torchvision.models.resnet18` | PyTorch / BSD via dependency | No | Yes | KEEP; dependency, not vendored source |
| `scripts/models/point_encoder.py` | Project PointNet-style + GRU / Voxel CNN-GRU | No upstream header found | No | Yes | KEEP as project code unless a later review finds a copied file |
| `third_party/AIRHAR/backbones/ResNet.py` | AIRHAR vendored backbone | Apache-2.0 at repo level | — | Not the frozen RadarMind ResNet18 path | Do not re-export as RadarMind original |
| `build_artifacts/ppt/node_modules/` | npm packages | Many third-party licenses | No | No | **EXCLUDE** |
| `radarmind/llm/` | Project code calling DeepSeek HTTP API | N/A | No | Language layer | KEEP; no vendor SDK copy found |

---

## 2. AIRHAR / RadMamba

| Field | Value |
|---|---|
| Path | `third_party/AIRHAR/` |
| Upstream | `https://github.com/lab-emi/AIRHAR.git` |
| Recorded commit (notes) | `d49e2c7ed7365e4d2d1724e5d3e4f2dfffc6f6c0` (`notes/00_clone_and_env.md`) |
| License | Apache License 2.0 |
| Size | 125.49 MB (includes `datasets/CI4R/ci4r_data.h5` 125 MB) |
| Nested git | Yes |

AIRHAR is a legitimate research dependency for the **RadarBridge / Phase 1A** era. The current manuscript recognition path for CI4R spectrograms is the project ResNet18 wrapper, not a claim that RadarMind authored RadMamba.

If any AIRHAR file is later kept in the scientific repo:

- retain copyright and Apache-2.0 license text
- add `THIRD_PARTY_NOTICES`
- do not include the nested `.git` or the 125 MB HDF5

Default: **do not add this clone to the first commit.** Document the URL and the commit used historically.

Apache-2.0 is compatible with a later RadarMind MIT or Apache-2.0 license **if** notices are preserved for any retained files.

---

## 3. CI4R-MULTI3

| Field | Value |
|---|---|
| Path | `third_party/CI4R-MULTI3/` |
| Upstream | CI4R multi-frequency dataset/scripts repository |
| License file | **Absent in this copy** |
| Nested git | Yes |
| Size | 3.67 MB |

The **dataset** (spectrograms / IQ) is a public research dataset. This folder is a clone of helper scripts and notebooks, not RadarMind original work.

Do not upload:

- the clone
- `data/CI4R_MULTI3/` (1959 official PNGs, 174 MB)
- packed `*.h5` copies

Publication docs must point to the official dataset source and expected directory layout. If the upstream license forbids redistribution, raw and packed spectrograms stay local.

---

## 4. DAP-Net

| Field | Value |
|---|---|
| Path | `competitors/dap_net_official/` |
| Upstream | `https://github.com/jolin830/DAP-Net.git` |
| License file | **Absent in this copy** |
| Nested git | Yes |
| Notes | `research/dap_net/*.md`, `PATCHLOG.md` |

This is competitor reproduction material. It is **not** required to regenerate the frozen RadarMind tables. Do not wrap it as RadarMind code. Do not add the clone until a license is confirmed and attribution is written.

Third-party research email in that README is acceptable **inside that upstream project**, not as a RadarMind contact.

---

## 5. Models implemented in this project

### 5.1 ResNet18 spectrogram

`scripts/models/resnet18_spec.py` uses torchvision. That is a normal dependency. Cite PyTorch / torchvision in requirements. Do not copy a full ResNet implementation from AIRHAR into RadarMind files and call it original.

### 5.2 PointNet-GRU / Voxel CNN-GRU

`scripts/models/point_encoder.py` is a small project encoder (PointNet-style per-point MLP + temporal GRU, plus a voxel CNN-GRU). No third-party copyright header was found in that file.

PointNet as a **method** should be cited in the paper/docs (Qi et al.). That is citation, not a license block, unless a specific upstream file was copied. No evidence of a full official PointNet repo vendored under `scripts/`.

### 5.3 CORAL / GRL / InfoNCE

These are standard methods implemented in project training scripts (`scripts/cross_radar/train_cross_radar.py`, `scripts/shared_motion/`, `scripts/phase7/train_m3.py`). Treat as project implementations of published algorithms. Cite the original papers in documentation. No separate vendor LICENSE files were found next to those functions.

---

## 6. Datasets (not code, but license-relevant)

| Dataset | In this workspace | Redistribute in git? |
|---|---|---|
| CI4R official PNG | `data/CI4R_MULTI3/`, several `*.h5` | **No** until redistribution is confirmed |
| CI4R raw IQ | Not present (~93 GB mentioned in notes) | **No** |
| mRI | Window cache NPZ only | **No** raw dump |
| RadHAR | Window / voxel cache NPZ | **No** raw dump |

The repository should document download/source, expected layout, and split commands. That is next-phase `docs/DATASETS.md` work.

---

## 7. npm / presentation toolchain

`build_artifacts/ppt/node_modules/` contains many third-party JavaScript licenses. It is unrelated to paper metric reproduction. **Exclude.**

---

## 8. Nested Git repositories

| Path | Risk |
|---|---|
| `third_party/AIRHAR/.git` | If added under a parent repo, Git records a gitlink or a messy nested history |
| `third_party/CI4R-MULTI3/.git` | Same |
| `competitors/dap_net_official/.git` | Same |

Do not `git add` those directories. If a sparse copy is needed later, copy selected files **without** `.git`.

---

## 9. License choice for RadarMind (not decided)

No root `LICENSE`. This audit does **not** pick one.

| Option | Typical use |
|---|---|
| MIT | Short, permissive, common for research code |
| Apache-2.0 | Permissive + patent clause; closer to AIRHAR if AIRHAR files are retained |

Prepare `LICENSE_SELECTION.md` in the documentation phase. Do not invent a journal DOI. Do not put a license text in this audit as if it were already chosen.

Third-party compatibility:

- torchvision / PyTorch: standard pip dependency
- AIRHAR Apache-2.0: compatible with either choice **if notices kept**
- CI4R-MULTI3 / DAP-Net: unknown until upstream license is recorded — do not vendor

---

## 10. Attribution requirements if those trees stay out of git

README / `docs/DATASETS.md` / `THIRD_PARTY_NOTICES` (later) should name:

1. CI4R / CI4R-MULTI3 dataset authors and official source
2. mRI dataset authors and official source
3. RadHAR dataset authors and official source
4. AIRHAR / RadMamba (historical tooling)
5. Method citations: ResNet, PointNet, CORAL, InfoNCE, GRL, open-set scores

Do not list `ec12312312333` as a dataset author or paper author.

---

## 11. Recommendation

1. First scientific commit: **project code + splits + frozen metrics + figures**, no vendor clones.
2. Add `THIRD_PARTY_NOTICES` when any third-party file is actually tracked.
3. Do not delete the local clones in this phase; they are **EXCLUDE from git**, not “erase from disk.”
4. Resolve RadarMind’s own license with the user before the first tagged release.
