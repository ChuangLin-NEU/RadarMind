"""Extract labeled-interval timeline slices from frozen Phase 4 JSON. No interpolation."""
from __future__ import annotations

import json
from pathlib import Path


def _skip_ws(text: str, i: int) -> int:
    n = len(text)
    while i < n and text[i] in " \n\r\t,":
        i += 1
    return i


def extract_raw_windows(path: Path, t_max: float) -> list[dict]:
    dec = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    fr = 10.0
    j = text.find('"frame_rate"')
    if j >= 0:
        k = text.find(":", j)
        fr = float(text[k + 1 : text.find(",", k)].strip())
    j = text.find('"windows":')
    if j < 0:
        raise RuntimeError(f"no windows array in {path}")
    i = text.find("[", j)
    i += 1
    out = []
    while True:
        i = _skip_ws(text, i)
        if i >= len(text):
            break
        if text[i] == "]":
            break
        obj, i = dec.raw_decode(text, i)
        if "start_time" in obj:
            t0 = float(obj["start_time"])
            t1 = float(obj.get("end_time", t0))
        else:
            t0 = float(obj["start_frame"]) / fr
            t1 = float(obj["end_frame"]) / fr
        if t0 > t_max:
            break
        out.append(
            {
                "start_time": t0,
                "end_time": t1,
                "activity": obj.get("activity", ""),
                "gt": obj.get("gt", ""),
                "confidence": float(obj.get("confidence", 0.0)),
            }
        )
    return out


def extract_mode_events(path: Path) -> dict[str, list]:
    """Mode-level events arrays (raw, isolated_flip, confidence_aware), in file order."""
    dec = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
    arrays = []
    idx = 0
    while len(arrays) < 3:
        j = text.find('"events":', idx)
        if j < 0:
            break
        k = text.find("[", j)
        arr, end = dec.raw_decode(text, k)
        if (
            isinstance(arr, list)
            and arr
            and isinstance(arr[0], dict)
            and "start_frame" in arr[0]
            and "n_windows" in arr[0]
        ):
            slim = []
            for ev in arr:
                slim.append(
                    {
                        "start_time": float(ev["start_time"]),
                        "end_time": float(ev["end_time"]),
                        "activity": ev.get("activity", ""),
                        "confidence": float(ev.get("confidence", 0.0)),
                    }
                )
            arrays.append(slim)
        idx = end
    if len(arrays) < 3:
        raise RuntimeError(f"expected 3 mode event arrays, got {len(arrays)} in {path}")
    return {"raw": arrays[0], "isolated_flip": arrays[1], "confidence_aware": arrays[2]}


def load_gt_events(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for ev in data["events"]:
        out.append(
            {
                "start_time": float(ev["start_time"]),
                "end_time": float(ev["end_time"]),
                "activity": ev.get("activity", ""),
            }
        )
    return out
