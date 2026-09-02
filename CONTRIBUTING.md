# Contributing

RadarMind is a research repository. Changes should preserve the frozen manuscript numbers unless a new experiment is explicitly added.

## Workflow

1. Fork the repository or work on a branch.
2. Keep `.env` and local datasets off the branch.
3. Add or update tests when the change can break metric traceability or documentation wording.
4. Open a pull request that states what was measured and what was not retrained.

## Scientific integrity

Do not hand-edit:

- frozen metrics under `results/phase*/**/metrics.json` and related `answers.json`
- `results/final_delivery/metric_snapshot_full.json`
- official split tables in `data_splits/`

If an experiment is added, record:

- dataset
- split
- config
- seed, if applicable
- checkpoint path (local; weights are not in git)
- metric
- result path

New numbers must go to a new result directory. Do not silently overwrite a manuscript frozen result.

Scientific values in the manuscript must not be edited to match a later run.

## CI4R naming

`data_splits/ci4r_subject_independent.csv` is a legacy filename. The protocol is prefix-group-disjoint, not subject-independent. See `docs/KNOWN_NAMING_ISSUES.md`.

## Language

Do not describe RadarMind as having achieved radar invariance, full-stream continuous recognition, open-world recognition, or CI4R subject-independent evaluation.
