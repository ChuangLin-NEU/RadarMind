#!/usr/bin/env python3
"""CI4R 10/24/77 inventory and pairing from official spectrogram PNGs + documented raw loaders.

Does not unzip raw_data.zip. Prefix tokens are acquisition groups, not subject IDs.
"""
from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SPEC = ROOT / "data" / "CI4R_MULTI3" / "spectrograms" / "11_class_activity_data"
OUT = ROOT / "results" / "phase2_cross_radar"
BAND_DIR = {"10GHz": "Xethru", "24GHz": "24GHz", "77GHz": "77GHz"}
TS_RE = re.compile(r"_(\d{10})")
PREFIX_RE = re.compile(r"^(\d+)_")


def _parse_name(name: str) -> dict:
    stem = Path(name).stem
    m = TS_RE.search(stem)
    p = PREFIX_RE.match(stem)
    return {
        "timestamp": m.group(1) if m else "",
        "prefix_group": p.group(1) if p else "",
        "filename": name,
    }


def scan_pngs() -> list[dict]:
    rows = []
    for freq, folder in BAND_DIR.items():
        base = SPEC / folder
        if not base.is_dir():
            continue
        for png in sorted(base.rglob("*.png")):
            action = png.parent.name
            meta = _parse_name(png.name)
            rows.append(
                {
                    "radar_frequency": freq,
                    "action": action,
                    "prefix_group": meta["prefix_group"],
                    "timestamp": meta["timestamp"],
                    "source_file": str(png.relative_to(ROOT)).replace("\\", "/"),
                    "filename": png.name,
                    "representation": "official_spectrogram_png",
                    "usable": png.is_file() and png.stat().st_size > 0,
                    "bytes": png.stat().st_size if png.is_file() else 0,
                    "native_note": "CI4R-MULTI3 released spectrogram PNG; not raw IQ",
                }
            )
    return rows


def pair_rows(rows: list[dict]) -> tuple[list[dict], dict]:
    by_ts: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["timestamp"] and r["usable"]:
            by_ts[r["timestamp"]][r["radar_frequency"]].append(r)
    manifest = []
    pair_i = 0
    n_exact3 = n_10_77 = n_24_77 = n_10_24 = 0
    for ts in sorted(by_ts):
        bands = by_ts[ts]
        have = set(bands)
        if len(have) < 2:
            continue
        pair_i += 1
        pid = f"pair_{pair_i:05d}"
        if have == {"10GHz", "24GHz", "77GHz"}:
            n_exact3 += 1
            conf, ev = "exact_timestamp", "same unix timestamp in 10/24/77 official PNG filenames"
        elif have == {"10GHz", "77GHz"}:
            n_10_77 += 1
            conf, ev = "exact_timestamp", "same unix timestamp in 10/77 PNG filenames"
        elif have == {"24GHz", "77GHz"}:
            n_24_77 += 1
            conf, ev = "exact_timestamp", "same unix timestamp in 24/77 PNG filenames"
        elif have == {"10GHz", "24GHz"}:
            n_10_24 += 1
            conf, ev = "exact_timestamp", "same unix timestamp in 10/24 PNG filenames"
        else:
            conf, ev = "exact_timestamp", "timestamp overlap"
        actions = {r["action"] for freq in bands for r in bands[freq]}
        action = next(iter(actions)) if len(actions) == 1 else "|".join(sorted(actions))
        if len(actions) > 1:
            conf, ev = "timestamp_action_conflict", "same timestamp, different action folders; not used as an exact pair"
        for freq in sorted(bands):
            for r in bands[freq]:
                manifest.append(
                    {
                        "pair_id": pid,
                        "action": r["action"],
                        "pair_action_label": action,
                        "prefix_group": r["prefix_group"],
                        "timestamp": ts,
                        "radar_frequency": freq,
                        "source_file": r["source_file"],
                        "representation": r["representation"],
                        "usable": r["usable"] and len(actions) == 1,
                        "pairing_confidence": conf,
                        "evidence": ev,
                        "prefix_note": "filename leading digits are acquisition-group tokens, not official subject IDs",
                    }
                )
    # also count pairwise including triples
    for ts, bands in by_ts.items():
        have = set(bands)
        if {"10GHz", "77GHz"} <= have:
            n_10_77 += 0  # already counted exclusive; fix below
    stats = _pair_stats(by_ts, rows)
    stats.update({"n_pair_rows": len(manifest), "n_pair_ids": pair_i})
    return manifest, stats


def _pair_stats(by_ts, rows) -> dict:
    n10 = sum(1 for r in rows if r["radar_frequency"] == "10GHz" and r["usable"])
    n24 = sum(1 for r in rows if r["radar_frequency"] == "24GHz" and r["usable"])
    n77 = sum(1 for r in rows if r["radar_frequency"] == "77GHz" and r["usable"])
    p1077 = p2477 = p1024 = p3 = 0
    act3: Counter = Counter()
    for ts, bands in by_ts.items():
        have = set(bands)
        if {"10GHz", "77GHz"} <= have:
            p1077 += 1
        if {"24GHz", "77GHz"} <= have:
            p2477 += 1
        if {"10GHz", "24GHz"} <= have:
            p1024 += 1
        if {"10GHz", "24GHz", "77GHz"} <= have:
            p3 += 1
            acts = {r["action"] for freq in bands for r in bands[freq]}
            if len(acts) == 1:
                act3[acts.pop()] += 1
            else:
                act3["ACTION_MISMATCH"] += 1
    agreed = {k: v for k, v in act3.items() if k != "ACTION_MISMATCH"}
    return {
        "N_10GHz": n10,
        "N_24GHz": n24,
        "N_77GHz": n77,
        "paired_10_77": p1077,
        "paired_24_77": p2477,
        "paired_10_24": p1024,
        "paired_10_24_77": p3,
        "paired_10_24_77_action_agreed": int(sum(agreed.values())),
        "paired_10_24_77_action_conflict": int(act3.get("ACTION_MISMATCH", 0)),
        "triple_action_counts": dict(act3),
        "common_actions_agreed": sorted(agreed),
        "conflict_note": "same timestamp but different action folders (often 24GHz Toes vs 10/77 Limp); excluded from fair triple subset",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def schema_md(stats: dict) -> str:
    return f"""# CI4R cross-radar schema

依据：`third_party/CI4R-MULTI3/README.md`（三雷达同步采集）、`XeThru_mDopp_bulk.m`、`datToImage_Anchortech.m`、`datToImage_77.m`、`third_party/AIRHAR/datasets/data_prepare_CI4R.py`，以及本地官方谱图 PNG。未全量解压 `raw_data.zip`。文件名前缀数字是 **acquisition group**，不是官方 subject id。

## 1. 可用雷达频率

三个：10 GHz（XeThru UWB）、24 GHz（Ancortek FMCW）、77 GHz（TI IWR1443）。

官方 PNG：`data/CI4R_MULTI3/spectrograms/11_class_activity_data/{{Xethru,24GHz,77GHz}}/`。

## 2. 真实文件格式

| 频率 | raw | 官方 loader | 本阶段训练用 |
|---|---|---|---|
| 10 GHz | `.dat`（XeThru 帧流：uint32 ContentId/Info/Length + 182 float；同目录 616B `.bin` **不是** 77 GHz 8-lane IQ） | `XeThru_mDopp_bulk.m`：拼 IQ → MTI → STFT（PRF=512，window=150） | 官方 released spectrogram PNG |
| 24 GHz | 文本 `.dat`：前 4 个数 fc / Tsweep / NTS / Bw，其后 I/Q | `datToImage_Anchortech.m`：reshape chirp → range FFT → MTI → STFT | 同上 PNG |
| 77 GHz | `*_Raw_0.bin` int16 8-lane DCA1000 | AIRHAR `bin2np` + `RGB_mD`；MATLAB `datToImage_77.m` | 同上 PNG（与 Phase 1A AIRHAR cache 分开，保证三频同一成品） |

统一 tensor 形状 `1×224×224` **不等于** 物理分辨率相同。metadata 保留 `radar_frequency`、`native_*`。

## 3. Preprocessing（官方，不是自造）

- **10 GHz**：XeThru 帧 IQ → 4 阶 highpass MTI → 第 27 距离元 STFT → isodata 去噪 → 20log / −45 dB。
- **24 GHz**：文本 IQ → range FFT 半边 → MTI → 第 19 距离元 STFT → 去噪。
- **77 GHz**：AIRHAR：Range FFT → 门 23:31 求和 → STFT Hann 256 / overlap 200 → 20log clip → 224×224。Phase 1A 已复现。MATLAB 版另有 MTI 和不同 overlap。

本阶段 cross-radar HAR **不重跑 raw**，使用官方已发布谱图，再灰度缩放到 224×224 并逐图 z-score。

## 4. 同步 pairing

官方 README：三雷达 **synched / co-located**。配对证据是文件名中的 10 位 unix timestamp 相同。

当前统计：

```
{json.dumps(stats, indent=2, ensure_ascii=False)}
```

`pair_id` 按 timestamp 分配。若三频 timestamp 对上且动作目录一致，`pairing_confidence=exact_timestamp`。

24 GHz 另有早期补录（README：额外约 180/类）。那些 timestamp 对不上 10/77 的，**不进三频共同子集**，也不假装成同步 pair。

## 5. 泄漏约束

同一 `timestamp` 的 10/24/77 视为一次 acquisition。划分按 timestamp，禁止 77 训练、对应 10 进 val/调参。
"""


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = scan_pngs()
    inv_fields = [
        "radar_frequency",
        "action",
        "prefix_group",
        "timestamp",
        "source_file",
        "filename",
        "representation",
        "usable",
        "bytes",
        "native_note",
    ]
    write_csv(OUT / "ci4r_radar_inventory.csv", rows, inv_fields)
    manifest, stats = pair_rows(rows)
    man_fields = [
        "pair_id",
        "action",
        "pair_action_label",
        "prefix_group",
        "timestamp",
        "radar_frequency",
        "source_file",
        "representation",
        "usable",
        "pairing_confidence",
        "evidence",
        "prefix_note",
    ]
    write_csv(OUT / "ci4r_cross_radar_manifest.csv", manifest, man_fields)
    (OUT / "ci4r_cross_radar_schema.md").write_text(schema_md(stats), encoding="utf-8")
    (OUT / "pairing_stats.json").write_text(json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(stats, indent=2), flush=True)


if __name__ == "__main__":
    main()
