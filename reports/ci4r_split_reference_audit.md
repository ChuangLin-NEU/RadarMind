# CI4R split reference audit

Legacy filename `data_splits/ci4r_subject_independent.csv` is kept.
Protocol in the file is `prefix_tail4_disjoint`. Not subject-independent.

| Path | Line | Usage |
|---|---:|---|
| `CONTRIBUTING.md` | 36 | ``data_splits/ci4r_subject_independent.csv` is a legacy filename. The protocol is prefix-group-disjoint, not subject-independent. See `docs/KNOWN_NAMING_ISSUES.md`.` |
| `README.md` | 78 | ``data_splits/ci4r_subject_independent.csv` is a legacy filename for a prefix-group-disjoint split. See `docs/KNOWN_NAMING_ISSUES.md`.` |
| `docs/DATASETS.md` | 34 | `\| prefix-group-disjoint \| `data_splits/ci4r_subject_independent.csv` \| filename prefix groups are disjoint across splits \|` |
| `docs/KNOWN_NAMING_ISSUES.md` | 5 | `## `data_splits/ci4r_subject_independent.csv`` |
| `docs/KNOWN_NAMING_ISSUES.md` | 9 | `\| Legacy filename \| `data_splits/ci4r_subject_independent.csv` \|` |
| `docs/REPRODUCIBILITY.md` | 48 | `\| `data_splits/ci4r_subject_independent.csv` \| CI4R prefix-group-disjoint (**not** SI) \|` |
| `radarmind/paths.py` | 19 | `CI4R_PREFIX_SPLIT = SPLITS / "ci4r_subject_independent.csv"` |
| `reports/candidate_repository_files.txt` | 14 | `data_splits/ci4r_subject_independent.csv` |
| `reports/ci4r_split_reference_audit.md` | 3 | `Legacy filename `data_splits/ci4r_subject_independent.csv` is kept.` |
| `reports/ci4r_split_reference_audit.md` | 8 | `\| `CONTRIBUTING.md` \| 36 \| ``data_splits/ci4r_subject_independent.csv` is a legacy filename. The protocol is prefix-group-disjoint, not subject-independent. See `docs/KNOWN_NAMING_ISSUES.md`.` \|` |
| `reports/ci4r_split_reference_audit.md` | 9 | `\| `README.md` \| 78 \| ``data_splits/ci4r_subject_independent.csv` is a legacy filename for a prefix-group-disjoint split. See `docs/KNOWN_NAMING_ISSUES.md`.` \|` |
| `reports/ci4r_split_reference_audit.md` | 10 | `\| `docs/DATASETS.md` \| 34 \| `\\| prefix-group-disjoint \\| `data_splits/ci4r_subject_independent.csv` \\| filename prefix groups are disjoint across splits \\|` \|` |
| `reports/ci4r_split_reference_audit.md` | 11 | `\| `docs/KNOWN_NAMING_ISSUES.md` \| 5 \| `## `data_splits/ci4r_subject_independent.csv`` \|` |
| `reports/ci4r_split_reference_audit.md` | 12 | `\| `docs/KNOWN_NAMING_ISSUES.md` \| 9 \| `\\| Legacy filename \\| `data_splits/ci4r_subject_independent.csv` \\|` \|` |
| `reports/ci4r_split_reference_audit.md` | 13 | `\| `docs/REPRODUCIBILITY.md` \| 48 \| `\\| `data_splits/ci4r_subject_independent.csv` \\| CI4R prefix-group-disjoint (**not** SI) \\|` \|` |
| `reports/ci4r_split_reference_audit.md` | 14 | `\| `radarmind/paths.py` \| 19 \| `CI4R_PREFIX_SPLIT = SPLITS / "ci4r_subject_independent.csv"` \|` |
| `reports/ci4r_split_reference_audit.md` | 15 | `\| `reports/public_release_audit.md` \| 104 \| `\\| `data_splits/ci4r_subject_independent.csv` \\| CI4R prefix-group split \\| Yes, but **name is misleading** \\| Yes \\| Yes \\| No \\| No \\| A \\| KEEP file; do` |
| `reports/ci4r_split_reference_audit.md` | 16 | `\| `reports/public_release_audit.md` \| 143 \| `\\| CI4R 77 GHz 11-class \\| `radarmind/data/ci4r_adapter.py`, `scripts/ci4r/` \\| `data_splits/ci4r_random.csv`, `ci4r_subject_independent.csv` (**prefix-gro` |
| `reports/ci4r_split_reference_audit.md` | 17 | `\| `reports/public_release_audit.md` \| 167 \| `\\| Prefix-group-disjoint \\| CI4R 77 GHz \\| `data_splits/ci4r_subject_independent.csv` protocol=`prefix_tail4_disjoint` \\| **Not subject-independent.** File` |
| `reports/ci4r_split_reference_audit.md` | 18 | `\| `reports/public_release_audit.md` \| 329 \| `\\| `data_splits/ci4r_subject_independent.csv` filename \\| Document as prefix-group now; rename only if all loaders are updated \\|` \|` |
| `reports/ci4r_split_reference_audit.md` | 19 | `\| `reports/public_release_audit.md` \| 350 \| `6. **Misleading split filename** `ci4r_subject_independent.csv`.` \|` |
| `reports/ci4r_split_reference_audit.md` | 20 | `\| `results/phase1_ci4r/phase1_ci4r_report.md` \| 112 \| `主结果用 **记号不相交**（`prefix_tail4_disjoint`，文件仍叫 `ci4r_subject_independent.csv`，protocol 字段写的是前者）：` \|` |
| `reports/ci4r_split_reference_audit.md` | 21 | `\| `results/radarmind_v01/artifact_registry.json` \| 18 \| `"prefix_group_disjoint": "C:\\Users\\13919\\Desktop\\RadarBridge\\data_splits\\ci4r_subject_independent.csv"` \|` |
| `reports/ci4r_split_reference_audit.md` | 22 | `\| `scripts/ci4r/phase1a_build.py` \| 477 \| `split_dir / "ci4r_subject_independent.csv",` \|` |
| `reports/ci4r_split_reference_audit.md` | 23 | `\| `scripts/release/quick_verify.py` \| 42 \| `ci4r_legacy = ROOT / "data_splits" / "ci4r_subject_independent.csv"` \|` |
| `reports/ci4r_split_reference_audit.md` | 24 | `\| `scripts/release/write_phase2_audits.py` \| 97 \| `needle = "ci4r_subject_independent"` \|` |
| `reports/ci4r_split_reference_audit.md` | 25 | `\| `scripts/release/write_phase2_audits.py` \| 108 \| `"Legacy filename `data_splits/ci4r_subject_independent.csv` is kept.",` \|` |
| `reports/public_release_audit.md` | 104 | `\| `data_splits/ci4r_subject_independent.csv` \| CI4R prefix-group split \| Yes, but **name is misleading** \| Yes \| Yes \| No \| No \| A \| KEEP file; document / later rename only with import updates \| Proto` |
| `reports/public_release_audit.md` | 143 | `\| CI4R 77 GHz 11-class \| `radarmind/data/ci4r_adapter.py`, `scripts/ci4r/` \| `data_splits/ci4r_random.csv`, `ci4r_subject_independent.csv` (**prefix-group, not SI**) \| `scripts/ci4r/phase1a_build.py` ` |
| `reports/public_release_audit.md` | 167 | `\| Prefix-group-disjoint \| CI4R 77 GHz \| `data_splits/ci4r_subject_independent.csv` protocol=`prefix_tail4_disjoint` \| **Not subject-independent.** Filename is a documentation hazard \|` |
| `reports/public_release_audit.md` | 329 | `\| `data_splits/ci4r_subject_independent.csv` filename \| Document as prefix-group now; rename only if all loaders are updated \|` |
| `reports/public_release_audit.md` | 350 | `6. **Misleading split filename** `ci4r_subject_independent.csv`.` |
| `results/phase1_ci4r/phase1_ci4r_report.md` | 112 | `主结果用 **记号不相交**（`prefix_tail4_disjoint`，文件仍叫 `ci4r_subject_independent.csv`，protocol 字段写的是前者）：` |
| `results/radarmind_v01/artifact_registry.json` | 18 | `"prefix_group_disjoint": "C:\\Users\\13919\\Desktop\\RadarBridge\\data_splits\\ci4r_subject_independent.csv"` |
| `scripts/ci4r/phase1a_build.py` | 477 | `split_dir / "ci4r_subject_independent.csv",` |
| `scripts/release/quick_verify.py` | 42 | `ci4r_legacy = ROOT / "data_splits" / "ci4r_subject_independent.csv"` |
| `scripts/release/write_phase2_audits.py` | 100 | `needle = "ci4r_subject_independent"` |
| `scripts/release/write_phase2_audits.py` | 111 | `"Legacy filename `data_splits/ci4r_subject_independent.csv` is kept.",` |

Decision this phase: KEEP LEGACY PATH; correct semantics in docs.
