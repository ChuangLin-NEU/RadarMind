# RadarMind Security and Privacy Audit

**Date:** 2026-09-02  
**Workspace:** `C:\Users\13919\Desktop\RadarMind`  
**Git history:** none (no `.git`). History cleanup (`git filter-repo`) is **not applicable yet**.  
**Push status:** **BLOCKED** until `.env` is confirmed ignored and the first add is curated.

---

## 1. Verdict

| Check | Result |
|---|---|
| Live secret on disk | **Yes** — DeepSeek API key in `.env` |
| Secret in source / results / notebooks | Not found |
| Secret already in Git history | **N/A** (no repository) |
| SSH / PEM / GitHub token / AWS key | Not found |
| Personal phone / home address / private email | Not found |
| Machine and server path leakage | **Yes** — Windows user path and `/home/radar_har` |
| Safe to `git add .` | **No** |
| Safe to push | **No** (no remote; would be blocked anyway) |

A private repository does **not** make a live API key acceptable. `.env` must never be committed, including to a private remote shared with `ec12312312333`.

---

## 2. Secrets

### 2.1 Live credential

| File | Line | Kind | Value in report |
|---|---|---|---|
| `.env` | 1 | DeepSeek API key | `DEEPSEEK_API_KEY=sk-` + 24 hex chars (redacted) |

The application reads this key at runtime (`radarmind/llm/provider.py`, `radarmind/llm/envload.py`) and sends `Authorization: Bearer …`. That pattern is acceptable **if** the key stays in the environment.

`.env` is listed in the current `.gitignore`. After `git init`, confirm with `git check-ignore -v .env` before any add.

**Action:** rotate the DeepSeek key. It exists in plaintext on this disk and was visible to this audit process. After rotation, keep only the new key in a local `.env` that remains untracked.

### 2.2 Safe template

| File | Status |
|---|---|
| `.env.example` | Placeholder `your_key_here`. Safe to keep. Expand later to `RADARMIND_LLM_*` names if desired; do not break the current `DEEPSEEK_*` readers without a code update |

### 2.3 Patterns searched and not found in project files

- OpenAI / Anthropic keys
- `ghp_`, `github_pat_`
- AWS `AKIA…`
- Slack `xox…`
- `BEGIN OPENSSH` / `BEGIN RSA PRIVATE`
- `*.pem`, `*.key`, `id_rsa`
- `credentials.json`
- Shell history files

Existing helper: `scripts/v02/secret_audit.py` (skips `.env` by design). Re-run after the first staged tree.

---

## 3. Privacy

### 3.1 Local Windows identity

Pattern: `C:\Users\13919` / `C:/Users/13919`.

This is a machine account path, not a manuscript author email. It should not be committed.

Examples:

- `environment.yml` — `recorded_from: C:\Users\13919\.conda\envs\airhar\python.exe`
- `figures/figure_manifest.json` and `results/final_delivery/figure_manifest.json`
- `reports/visual_qa_v2.md`, `reports/figure_metric_provenance.md`
- `results/phase8_physical_representation/r2_cache_manifest.csv` (~981 rows)
- `results/acquisition_toolkit_v01/mock_dataset/manifests/dataset_manifest.csv`
- `experiments/png_physics_probe/data/*.csv`
- `results/splits/ci4r_*_last12.csv`
- `gpu_jobs/train_missing_models.sh` / `.ps1`
- `scripts/pointcloud/train_voxel.py` (hardcoded missing-checkpoint path)
- Some v01/v02 smoke reports

### 3.2 Lab / server paths

Pattern: `/home/radar_har/…` — tens of thousands of occurrences, mostly generated CSVs.

Highest counts:

| File | Approx. hits |
|---|---:|
| `results/phase1_pointcloud/failed_samples.csv` | 23606 |
| `results/phase10_unified_measurement/representation_manifest.csv` | 1962 |
| `results/phase8_physical_representation/*.csv` | ~981 each |
| `results/phase10_velocity_calibration/r2v_manifest.csv` | 654 |
| `scripts/phase8/audit.py` | hardcoded `ZIP_01 = "/home/radar_har/ci4r/raw_data.zip"` |

These paths describe a lab account and filesystem layout. For a two-person private repo they are still unnecessary and should not be the published source tree. **Do not delete the scientific metrics** that sit beside them; omit the path-dump CSVs from git or regenerate with relative paths later.

### 3.3 Emails

| Location | Address | Assessment |
|---|---|---|
| `competitors/dap_net_official/README.md` | third-party dataset/author contact | Keep only if that tree is kept; attribution, not RadarMind PII |
| Manuscript authors | Li Luo, Xuehong Sun | May appear in `CITATION.cff` / README. Do not invent extra personal emails |

No owner phone number, street address, or student mailbox was found in project source (excluding `node_modules` documentation).

### 3.4 Subject identifiers

mRI uses `subject1` … official IDs including test subjects 3, 10, 20. These are dataset subject codes, not personal names. Keep in splits.

CI4R filename prefixes (`0000`, `0010`, …) are **acquisition groups**, not verified person identities.

---

## 4. Git history

There is no Git repository.

| Question | Answer |
|---|---|
| Has the DeepSeek key been committed? | No commits exist |
| Is `git filter-repo` required now? | No |
| Risk at `git init` | High if `git add .` is used before ignore rules are expanded |

After the first commit, re-scan `git log` and `git rev-list --all` for `sk-` / `DEEPSEEK_API_KEY=` / `.env` contents.

---

## 5. Required ignore rules (minimum addition)

Current `.gitignore` already has `data/`, `*.h5`, `*.bin`, `results/**/*.pt`, `.env`.

Still missing or incomplete for a safe first commit:

```
.env.*
!.env.example
.venv/
venv/
__pycache__/
.idea/
.vscode/
outputs/
logs/
cache/
caches/
data/raw/
data/cache/
wandb/
runs/
*.log
tmp/
temp/
local/
private/
secrets/
*.npz
*.npy
*.tar
build_artifacts/
gpu_jobs/
experiments/png_physics_probe/data/
third_party/
competitors/
```

Do **not** ignore:

- `data_splits/`
- `results/frozen_2026_08/`
- `results/final_delivery/metric_snapshot_full.json`
- `results/paper_package_v2/` tables
- Phase `metrics.json` / `answers.json` / pairing stats
- `figures/main/*.svg`

`.gitignore` must be expanded **before** `git init` + first add. That is organization-phase work, not done here.

---

## 6. Visibility and access

User instruction: the GitHub repository must be visible **only** to the owner and `ec12312312333`.

| Action | This phase | Later |
|---|---|---|
| Create public repo | Forbidden | Forbidden |
| Create private repo named `RadarMind` | Not yet | After cleanup + first commit |
| List `@ec12312312333` in README / `CONTRIBUTORS.md` | Not yet (docs phase) | Yes |
| Add GitHub collaborator `ec12312312333` | Not yet | Yes, because visibility for that user requires access. This is an explicit user request, not a default |
| Forge commits as that user | Never | Never |

`gh` is not installed on this machine. Private repo creation will need GitHub CLI, the website, or another authenticated client.

---

## 7. Pre-push checklist (for the later release phase)

- [ ] `.env` untracked (`git check-ignore -v .env`)
- [ ] DeepSeek key rotated
- [ ] No `sk-` in tracked files
- [ ] `data/` and `*.h5` untracked
- [ ] `*.pt` untracked
- [ ] Large `*.npz` / `*.tar` untracked
- [ ] No `C:\Users\13919` in tracked files
- [ ] No `/home/radar_har` in tracked files, or only in explicitly reviewed scripts with placeholders
- [ ] Nested `.git` directories not added
- [ ] `git status` reviewed (`reports/git_staging_audit.md` to be written then)

**Current recommendation: do not initialize Git until `.gitignore` is expanded and the keep/exclude list is applied.**
