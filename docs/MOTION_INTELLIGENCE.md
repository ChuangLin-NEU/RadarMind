# Motion Intelligence

Motion Intelligence is a structured record built from a model prediction, point-cloud kinematics when available, and dataset metadata. Implementation: `radarmind/intelligence/schema.py`, `radarmind/intelligence/extract.py`, `radarmind/schema.py`.

## Record

```json
{
  "activity": {"value": "...", "source": "MODEL", "unit": "UNKNOWN", "note": ""},
  "confidence": {"value": 0.0, "source": "MODEL", "unit": "UNKNOWN", "note": ""},
  "direction": {"value": "UNKNOWN", "source": "UNKNOWN", "unit": "UNKNOWN", "note": "..."},
  "mean_velocity": {"value": "...", "source": "PHYSICS_DERIVED", "unit": "...", "note": ""},
  "peak_velocity": {"value": "...", "source": "PHYSICS_DERIVED", "unit": "...", "note": ""},
  "motion_intensity": {"value": "...", "source": "PHYSICS_DERIVED", "unit": "...", "note": ""},
  "periodicity": {"value": "UNKNOWN", "source": "UNKNOWN", "unit": "UNKNOWN", "note": "no period estimator in v0.1"},
  "duration": {"value": "...", "source": "METADATA", "unit": "...", "note": ""},
  "motion_state": {"value": "...", "source": "MODEL", "unit": "UNKNOWN", "note": ""},
  "uncertainty": {"value": "...", "source": "MODEL", "unit": "UNKNOWN", "note": ""},
  "dataset": {"value": "...", "source": "METADATA", "unit": "UNKNOWN", "note": ""},
  "radar": {"value": "...", "source": "METADATA", "unit": "UNKNOWN", "note": ""},
  "subject": {"value": "...", "source": "METADATA", "unit": "UNKNOWN", "note": ""},
  "environment": {"value": "UNKNOWN", "source": "UNKNOWN", "unit": "UNKNOWN", "note": "no validated environment metadata"},
  "evidence": {}
}
```

Each scientific field is a `Tagged` object: `value`, `source`, `unit`, `note`.

## Provenance

| Tag | Meaning |
|---|---|
| `MODEL` | Taken from a stored classifier or open-set score |
| `PHYSICS_DERIVED` | Computed from valid radar points (for example Doppler/velocity statistics) |
| `METADATA` | Copied from dataset or split tables |
| `UNKNOWN` | Not supported by the current estimators or metadata |

`UNKNOWN` is not `0`, not an imputed number, and not a guessed activity name. It must remain unavailable through the semantic layer. The extractor does not invent direction or SI units to fill empty fields.

## Timeline

`radarmind/intelligence/timeline.py` aggregates window records into events. Serialization is JSON. Unsupported quantities stay `UNKNOWN`.

## Language layer

The language model receives this JSON and the timeline, not raw radar. Template fallback is used when no key is configured or the provider fails.

LLM output is not a scientific evidence source. It does not change HAR accuracy and must not be written back as a provenance tag on Motion Intelligence fields.
