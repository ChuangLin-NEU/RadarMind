# mRI signal / label schema

Evidence: Dryad README (inner `dataset_release`) and `competitors/dap_net_official/dataset/makecsv/mRI2csv.py`.

## Hierarchy

```
subject (JSON field `subject`, e.g. subject1)
  └── recording: one raw CSV `raw_data/radar/{subject}_{MMDD}.csv`
        └── action segment: videolabels JSON `labels[name] = [start, end]` inclusive on Frame #
              └── frame: all CSV rows with that Frame #
                    └── points: columns X,Y,Z,Doppler,Intensity
```

One subject file = one recording. The `_MMDD` token is stored as `session` from the filename only.

## How labels attach

JSON integers are **Frame #**, not `Time`.

- `Time` is a wall-clock unix stamp (~1.65e9). It is never in the JSON ranges (e.g. 98–139).
- DAP-Net compares `Frame # == t` for `t` in `[a, b]` inclusive.
- Radar `Frame #` is not gap-free (subject1: 164 gaps, max gap 51). Missing IDs in a JSON interval are unlabeled holes, not extra points.
- `# Obj` matches the number of rows of that Frame # on the files checked.

## Overlap and unlabeled

Adjacent JSON intervals often share the end/start frame (e.g. pose_1 [169,640] and pose_2 [640,1122]). Those frames are marked with `action` joined by `|` in the frame manifest.

Frames before the first interval, after the last, and in interior gaps are `UNLABELED`.

`T pose 2` appears on 13/20 subjects as a short closing T-pose. Training uses canonical name `T pose` (same pose words). The raw JSON key is kept in `action_raw`.

## Rate

`Time` span / unique frames is about **10 Hz** (not 30 Hz). DAP-Net window T=32 is ~3.2 s.
