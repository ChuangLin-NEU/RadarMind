# Known naming issues

These names remain on disk because loaders already point at them. Semantics in documentation take precedence over the filename.

## `data_splits/ci4r_subject_independent.csv`

| Field | Value |
|---|---|
| Legacy filename | `data_splits/ci4r_subject_independent.csv` |
| Constant in code | `radarmind.paths.CI4R_PREFIX_SPLIT` |
| Actual protocol field | `prefix_tail4_disjoint` |
| Scientific name | prefix-group-disjoint / acquisition-group-disjoint |
| Subject-independent? | **No** |

CI4R filename prefixes (for example the last four digits used as `name_token_tail4`) are acquisition groups. They are not verified subject identities.

The only subject-independent split in this repository is `data_splits/mri_subject_independent.csv`.

Do not rename the CI4R file until every reference listed in `reports/ci4r_split_reference_audit.md` has been updated. Until then, keep the legacy path and use the correct wording in papers, README, and docs.
