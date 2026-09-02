#!/usr/bin/env python3
"""Build 224x224 z-scored official spectrogram cache + timestamp acquisition split."""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase2_cross_radar"
SEED = 42
CLASS_NAMES = ["Away", "Bend", "Crawl", "Kneel", "Pick", "Sit", "Towards"]


def load_gray224(path: Path) -> tuple[np.ndarray, tuple[int, int]]:
    im = Image.open(path).convert("L")
    native = im.size
    arr = np.asarray(im.resize((224, 224), Image.BICUBIC), dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-6)
    return arr, (native[1], native[0])


def agreed_triples(inv_rows: list[dict]) -> list[str]:
    by = defaultdict(lambda: defaultdict(set))
    files = defaultdict(lambda: defaultdict(list))
    for r in inv_rows:
        if not r["timestamp"]:
            continue
        by[r["timestamp"]][r["radar_frequency"]].add(r["action"])
        files[r["timestamp"]][r["radar_frequency"]].append(r)
    keep = []
    for ts, bands in by.items():
        if set(bands) < {"10GHz", "24GHz", "77GHz"}:
            continue
        acts = set()
        for s in bands.values():
            acts |= s
        if len(acts) == 1 and acts.pop() in CLASS_NAMES:
            keep.append(ts)
    return keep


def split_timestamps(rows: list[dict], timestamps: list[str]) -> dict[str, str]:
    act = {}
    for r in rows:
        if r["timestamp"] in timestamps and r["radar_frequency"] == "77GHz":
            act[r["timestamp"]] = r["action"]
    ts = [t for t in timestamps if t in act]
    rng = np.random.RandomState(SEED)
    by_act: dict[str, list[str]] = {}
    for t in ts:
        by_act.setdefault(act[t], []).append(t)
    tr, va, te = [], [], []
    for a, group in by_act.items():
        rng.shuffle(group)
        n = len(group)
        n_te = max(1, int(round(n * 0.15)))
        n_va = max(1, int(round(n * 0.15)))
        if n_te + n_va >= n:
            n_te = max(1, n // 5)
            n_va = max(1, n // 5)
        te.extend(group[:n_te])
        va.extend(group[n_te : n_te + n_va])
        tr.extend(group[n_te + n_va :])
    out = {}
    for t in tr:
        out[t] = "train"
    for t in va:
        out[t] = "val"
    for t in te:
        out[t] = "test"
    return out


def main() -> None:
    inv = list(csv.DictReader((OUT / "ci4r_radar_inventory.csv").open(encoding="utf-8")))
    keep_ts = agreed_triples(inv)
    assign = split_timestamps(inv, keep_ts)
    split_path = ROOT / "data_splits" / "ci4r_cross_radar_timestamp.csv"
    split_path.parent.mkdir(exist_ok=True)
    with split_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["timestamp", "action", "split", "protocol", "seed"])
        w.writeheader()
        actmap = {}
        for r in inv:
            if r["timestamp"] in assign and r["radar_frequency"] == "77GHz":
                actmap[r["timestamp"]] = r["action"]
        for ts, role in sorted(assign.items()):
            w.writerow({"timestamp": ts, "action": actmap.get(ts, ""), "split": role, "protocol": "acquisition_timestamp", "seed": SEED})

    cache = {"10GHz": [], "24GHz": [], "77GHz": []}
    for r in inv:
        ts = r["timestamp"]
        if ts not in assign:
            continue
        if r["action"] not in CLASS_NAMES:
            continue
        png = ROOT / r["source_file"]
        try:
            x, native = load_gray224(png)
        except Exception as exc:  # noqa: BLE001
            print("[skip]", png, exc, flush=True)
            continue
        cache[r["radar_frequency"]].append(
            {
                "X": x,
                "y": CLASS_NAMES.index(r["action"]),
                "sample_id": f"{r['radar_frequency']}_{ts}_{r['action']}",
                "timestamp": ts,
                "prefix_group": r["prefix_group"],
                "action": r["action"],
                "split": assign[ts],
                "source_file": r["source_file"],
                "native_hw": native,
            }
        )
    dest = OUT / "cache"
    dest.mkdir(exist_ok=True)
    summary = {}
    for freq, items in cache.items():
        if not items:
            continue
        np.savez_compressed(
            dest / f"ci4r_{freq.replace('GHz', '')}ghz_png224.npz",
            X=np.stack([i["X"] for i in items]).astype(np.float32),
            y=np.asarray([i["y"] for i in items], np.int64),
            sample_id=np.array([i["sample_id"] for i in items]),
            timestamp=np.array([i["timestamp"] for i in items]),
            prefix_group=np.array([i["prefix_group"] for i in items]),
            action=np.array([i["action"] for i in items]),
            split=np.array([i["split"] for i in items]),
            source_file=np.array([i["source_file"] for i in items]),
            class_names=np.array(CLASS_NAMES),
        )
        from collections import Counter as C

        roles = C(i["split"] for i in items)
        summary[freq] = {"n": len(items), "split": dict(roles)}
        print("[pack]", freq, len(items), roles, flush=True)
    (dest / "pack_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
