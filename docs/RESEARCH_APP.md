# Research application

The paper-facing interactive application is the Streamlit app already used for RadarMind v0.2.

## Start

```
conda activate airhar
python run_radarmind.py
```

http://localhost:8502

Offline language layer (models and frozen files stay real):

```
python run_radarmind.py --offline-demo
```

or `python run_radarmind_offline.py`.

There is no `python -m radarmind.cli` entry in this repository.

## Pages

Implemented in `radarmind/app/streamlit_app.py`:

- home
- discrete replay
- continuous labeled-interval replay
- open-set
- generalization
- ablation
- cross-radar
- radar invariance
- shared motion
- pipeline
- status

## Frozen data requirement

The app reads processed caches and checkpoints from `results/` (see `radarmind/paths.py`). Those caches and `*.pt` files are local. Without them, pages that infer on samples will not load; pages that only read JSON metrics still can.

## Language layer

- Online: `.env` with `DEEPSEEK_API_KEY` or `RADARMIND_LLM_API_KEY`
- Fallback: no key, provider error, or `--offline-demo`

The model receives Motion Intelligence and timeline JSON. It does not receive raw IQ, spectrograms, point clouds, or hidden features.

## Scientific limits in the UI

The UI must not present:

- CI4R prefix-group results as subject-independent
- continuous replay as full-stream sensing
- open-set rejection as open-world recognition
- Radar-ID collapse as solved radar invariance

`demo/app.py` is the older RadarBridge Explorer. It is not the RadarMind research application.
