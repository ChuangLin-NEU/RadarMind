# Pre-git audit

Git is not initialized in this phase. This file only records whether a later `git init` is allowed.

Status: **PASS**

| Check | Result | Detail |
|---|---|---|
| .env excluded by gitignore | PASS |  |
| *.pt excluded | PASS |  |
| *.h5 excluded | PASS |  |
| data/ excluded | PASS |  |
| third_party/ excluded | PASS |  |
| competitors/ excluded | PASS |  |
| cache_r2.tar excluded | PASS |  |
| Draft docx excluded | PASS |  |
| .env.example not ignored | PASS |  |
| .env.example has no sk-/token | PASS |  |
| CONTRIBUTORS.md present | PASS |  |
| ec12312312333 in CONTRIBUTORS.md | PASS |  |
| ec12312312333 in README.md | PASS |  |
| CITATION.cff present | PASS |  |
| README.md present | PASS |  |
| LICENSE not claimed | PASS |  |
| CITATION has no invented DOI | PASS |  |
| CITATION authors are paper authors only | PASS |  |
| secret candidate files | PASS | 0 |
| checkpoints in candidate | PASS | 0 |
| h5 in candidate | PASS | 0 |
| raw data/ in candidate | PASS | 0 |
| personal absolute paths in candidate runtime .py | PASS |  |
| candidate size < 1 GB | PASS | 28.16 MB |
| candidate size explained if > 500 MB | PASS | 28.16 MB |
| quick verify | PASS | exit 0 |
| metric verification | PASS | exit 0 |

Candidate files: 1157
Candidate size: 28.16 MB
Largest included file: `results/phase5_open_set/eval_summary.json` (0.98 MB)

If FAIL, do not run git init.
