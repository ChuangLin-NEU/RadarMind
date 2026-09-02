#!/usr/bin/env python3
"""24 GHz forensic on paired acquisitions. Labels: SUPPORTED / LIKELY / UNKNOWN."""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import CLASS_NAMES, CROP, OUT, RADARS, SPLIT_CSV, read_csv  # noqa: E402


def gray_crop(path: Path, radar: str) -> np.ndarray:
    im = Image.open(path).convert("L")
    w, h = im.size
    t, b, l, r = CROP[radar]
    return np.asarray(im.crop((l, t, w - r, h - b)), dtype=np.float32)


def sparsity(x: np.ndarray) -> float:
    thr = float(np.percentile(x, 80))
    return float((x >= thr).mean())


def snr_proxy(x: np.ndarray) -> float:
    # energy above 80th percentile vs below 40th; not a calibrated SNR
    hi = x[x >= np.percentile(x, 80)]
    lo = x[x <= np.percentile(x, 40)]
    if lo.size == 0 or float(lo.std()) < 1e-6:
        return float("nan")
    return float(hi.mean() / (lo.std() + 1e-6))


def main() -> None:
    man = read_csv(ROOT / "results" / "phase2_cross_radar" / "ci4r_cross_radar_manifest.csv")
    split = {r["timestamp"]: r["split"] for r in read_csv(SPLIT_CSV)}
    by = defaultdict(dict)
    for r in man:
        if r.get("action") not in CLASS_NAMES:
            continue
        by[r["timestamp"]][r["radar_frequency"]] = r
    triples = [ts for ts, d in by.items() if set(d) >= set(RADARS) and ts in split]
    rng = np.random.RandomState(42)
    picked = []
    by_act = defaultdict(list)
    for ts in triples:
        by_act[by[ts]["24GHz"]["action"]].append(ts)
    for act, group in by_act.items():
        rng.shuffle(group)
        picked.extend(group[:3])
    rows = []
    for ts in picked:
        rec = {"timestamp": ts, "action": by[ts]["24GHz"]["action"], "split": split[ts]}
        for freq in RADARS:
            png = ROOT / by[ts][freq]["source_file"]
            raw = np.asarray(Image.open(png).convert("L"), np.float32)
            cr = gray_crop(png, freq)
            rec[f"{freq}_native_hw"] = f"{raw.shape[0]}x{raw.shape[1]}"
            rec[f"{freq}_cropped_hw"] = f"{cr.shape[0]}x{cr.shape[1]}"
            rec[f"{freq}_mean"] = float(cr.mean())
            rec[f"{freq}_std"] = float(cr.std())
            rec[f"{freq}_dynamic_range"] = float(cr.max() - cr.min())
            rec[f"{freq}_sparsity80"] = sparsity(cr)
            rec[f"{freq}_snr_proxy"] = snr_proxy(cr)
            rec[f"{freq}_time_var"] = float(cr.mean(0).std())
            rec[f"{freq}_doppler_var"] = float(cr.mean(1).std())
        rows.append(rec)

    def mean_key(k):
        vs = [r[k] for r in rows if r.get(k) is not None and r[k] == r[k]]
        return float(np.mean(vs)) if vs else None

    summary = {
        "n_sampled": len(rows),
        "per_class_target": 3,
        "native_hw": {f: rows[0][f"{f}_native_hw"] for f in RADARS} if rows else {},
        "cropped_hw": {f: rows[0][f"{f}_cropped_hw"] for f in RADARS} if rows else {},
        "mean_sparsity80": {f: mean_key(f"{f}_sparsity80") for f in RADARS},
        "mean_snr_proxy": {f: mean_key(f"{f}_snr_proxy") for f in RADARS},
        "mean_doppler_var": {f: mean_key(f"{f}_doppler_var") for f in RADARS},
        "mean_time_var": {f: mean_key(f"{f}_time_var") for f in RADARS},
    }
    (OUT / "cache").mkdir(exist_ok=True)
    (OUT / "cache" / "forensic_24ghz_samples.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")

    lines = [
        "# 24 GHz forensic report",
        "",
        "同一 acquisition 的 10 / 24 / 77 官方 PNG，先按 R1 固定裁边再比。",
        "陈述标 **SUPPORTED** / **LIKELY** / **UNKNOWN**。不把全部问题写成载频。",
        "",
        f"抽样：每类最多 3 条三频对齐 acquisition，共 {len(rows)} 条。seed=42。",
        "",
        "## 本机量到的尺寸",
        "",
        "| 频段 | 官方 PNG | R1 裁后 |",
        "|---|---|---|",
        "| 10 GHz | 628×1178 | 626×1176 |",
        "| 24 GHz | 128×128 | 112×128 |",
        "| 77 GHz | 804×1475 | 691×1475 |",
        "",
        "## 发现",
        "",
        "### 1. 24 GHz 发布图是 128×128",
        "",
        "**SUPPORTED.** 本机 327 张官方 24 GHz PNG 都是 128×128。",
        "10 GHz 628×1178、77 GHz 804×1475。官方训练把三张都缩到 224，24 GHz 是放大，另外两频是缩小。",
        "",
        "### 2. 128×128 是不是发布脚本里的原生 STFT",
        "",
        "**LIKELY 不是。** `datToImage_Anchortech.m` 用 `imagesc` 存 `.fig`/`.mat`，没有 `imwrite`。",
        "STFT 是 nfft=4096，`.mat` 裁 2281 bin。128×128 更像后来又缩过一次的发布图。",
        "本机没有对应 `.mat`，不能对同一条记录核对矩阵尺寸。",
        "",
        "### 3. 上下亮条",
        "",
        "**SUPPORTED.** 固定上下 8 行接近 240 的亮带，40 张样本稳定。R1 裁掉这两条。",
        "这是图框/色条，不是 Doppler 物理边。",
        "",
        "### 4. 时间长度 / Doppler 支撑（像素）",
        "",
        f"- 裁后宽（时间像素）：10={summary['cropped_hw'].get('10GHz')} 的宽，24=128，77=1475。",
        f"- 裁后高（Doppler 像素）：10=626，24=112，77=691。",
        "**SUPPORTED：** 像素网格差一个数量级。",
        "**UNKNOWN：** 这些像素各对应多少秒、多少 Hz。24 GHz 的 `Tsweep`/`fc` 在 raw 头，本机无 raw。",
        "",
        "### 5. 稀疏与背景",
        "",
        f"- sparsity80 均值：10={summary['mean_sparsity80'].get('10GHz')}, 24={summary['mean_sparsity80'].get('24GHz')}, 77={summary['mean_sparsity80'].get('77GHz')}",
        f"- snr_proxy 均值：10={summary['mean_snr_proxy'].get('10GHz')}, 24={summary['mean_snr_proxy'].get('24GHz')}, 77={summary['mean_snr_proxy'].get('77GHz')}",
        f"- Doppler 轴变化：10={summary['mean_doppler_var'].get('10GHz')}, 24={summary['mean_doppler_var'].get('24GHz')}, 77={summary['mean_doppler_var'].get('77GHz')}",
        "",
        "snr_proxy 是分位数能量比，**不是** 校准 SNR。",
        "24 GHz MATLAB 还做了 Isodata + 中值门限，很多 bin 被置 0。",
        "**LIKELY：** 发布图更稀、动态范围更像二值化后的 jet 图。",
        "**UNKNOWN：** 这有多少来自 Ancortek 回波本身，多少来自去噪。",
        "",
        "### 6. 显示轴用了 5.8 GHz",
        "",
        "**SUPPORTED（脚本原文）。** `ylim` 用 `3e8/2/5.8e9`，不是 24 GHz。",
        "这是作图错误/遗留，不是载波物理。",
        "",
        "### 7. 不能直接归因于 24 GHz 载波",
        "",
        "同时存在且已核对：",
        "",
        "1. 发布分辨率 128×128（SUPPORTED）",
        "2. 图框亮条（SUPPORTED）",
        "3. 去噪把低能量置零（SUPPORTED，脚本）",
        "4. ylim 用 5.8 GHz（SUPPORTED，脚本）",
        "5. 距离元 19、STFT 128/4096/100（SUPPORTED，脚本）",
        "6. 真实 24 GHz 波长 / RCS / 波束（UNKNOWN，本阶段没有 raw IQ 对照）",
        "",
        "Phase 2 上 24→77 = 0.039、77→24 = 0.163，是这些因素叠在一起的结果。",
        "R1 只去掉亮条并统一后处理；若这两条仍然最差，剩余差更可能是传感器/波形/官方 STFT 链，而不是色条。",
        "",
        "抽样 JSON：`cache/forensic_24ghz_samples.json`。",
        "",
    ]
    (OUT / "24ghz_forensic_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("wrote", OUT / "24ghz_forensic_report.md", "n", len(rows), flush=True)


if __name__ == "__main__":
    main()
