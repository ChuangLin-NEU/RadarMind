# Dependency audit

No `pip freeze` of the whole machine was taken for publication.

## Core (already in `requirements.txt`)

numpy, scipy, pandas, matplotlib, tqdm, scikit-learn, h5py, einops, opencv-python-headless, tensorboard

Used by training, evaluation, and figure rebuild.

## Training

PyTorch + torchvision. Installed from a CUDA wheel index, not pinned inside `requirements.txt`, because the local freeze used `torch==2.7.1+cu128` on an RTX 5060. Other machines need a wheel that matches their GPU.

## App (`requirements-app.txt`)

streamlit — required for `python run_radarmind.py`. Not listed in the original `requirements.txt`.

## Optional LLM

No extra package. `radarmind/llm/provider.py` uses `urllib` and environment variables.

## Dev (`requirements-dev.txt`)

pytest optional. Manuscript tests use `unittest`.

## Historical

`requirements_lock.txt` is a local pin and contains a conda `file://` URL. It is gitignored and not the publication specification.

`environment.yml` now references `requirements.txt` and no longer records a personal `python.exe` path.

## Action this phase

- Kept `requirements.txt`
- Added `requirements-app.txt` and `requirements-dev.txt`
- Did not add `pyproject.toml`
