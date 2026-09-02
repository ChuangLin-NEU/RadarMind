#!/usr/bin/env python3
"""Phase 8 report + answers.json."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase8_physical_representation"


def main() -> None:
    summary = json.loads((OUT / "audit_summary.json").read_text(encoding="utf-8"))
    gate2 = json.loads((OUT / "gate2.json").read_text(encoding="utf-8"))
    per = summary["per_radar"]
    answers = {
        "case": "C",
        "R2_available": "NO",
        "R2_type": "unavailable",
        "physical_representation": "BLOCKED_BY_DATA",
        "Q1": "10 GHz raw 在 raw_data.zip 里有（327/327 .dat），本机没有打开。数值 .mat 不在 zip 里。",
        "Q2": "24 GHz raw 在 zip 里有（327/327 .dat，头含 fc/Tsweep/NTS/Bw），本机没有打开。zip 无 24GHz .mat。",
        "Q3": "77 GHz 327 条都能对到 *_Raw_0.bin 和 activity_spect_matrices .mat；Phase1 cache 也覆盖 327。IQ 未在本机抽出。",
        "learn": "下一步先做 representation（抽出 raw 建 R2），不要再在 PNG 上训 adversarial。",
    }
    (OUT / "answers.json").write_text(json.dumps({**answers, "audit": summary, "gate2": gate2}, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        "# Phase 8 — Physical motion representation reconstruction",
        "",
        "未改 Phase 1–7 / frozen / paper。未训 M1/M2/M3/CORAL。Gate 2 未过，无 HAR 训练。",
        "",
        "## CASE",
        "",
        "**C（对本机建 R2）**：10/24/77 的 raw **都在官方 zip 清单里**，映射已建立；本机没有 `raw_data.zip`，不能打开成员。",
        "不是“raw 不存在”，也不是“文件名对不上”。",
        "",
        "- CI4R raw archive: environment variable `CI4R_RAW_ZIP` or `RADARMIND_CI4R_ZIP`",
        f"- GCS HEAD：{summary.get('gcs', {}).get('content_length')} bytes，reachable={summary.get('gcs', {}).get('reachable')}",
        "",
        "## 覆盖（327 公平 acquisition）",
        "",
        "| 频段 | zip raw | zip 数值 MAT | 本机 PNG | Phase1 77 cache |",
        "|---|---:|---:|---:|---:|",
        f"| 10 | {per['10GHz']['archive_raw']} | {per['10GHz']['archive_num']} | {per['10GHz']['local_png']} | — |",
        f"| 24 | {per['24GHz']['archive_raw']} | {per['24GHz']['archive_num']} | {per['24GHz']['local_png']} | — |",
        f"| 77 | {per['77GHz']['archive_raw']} | {per['77GHz']['archive_num']} | {per['77GHz']['local_png']} | {per['77GHz']['p1_cache']} |",
        "",
        "## 14 问",
        "",
        "1. 10 GHz raw/numerical：**归档 YES / 本机 NO**；zip 无 10 GHz .mat。",
        "2. 24 GHz raw/numerical：**归档 YES / 本机 NO**；zip 无 24 GHz .mat。",
        "3. 77 GHz 327→raw IQ：**归档 YES（映射 327/327）/ 本机未抽出**。Phase1 AIRHAR 224 有，但不是 R2。",
        "4. 10 GHz 物理轴：脚本可给时间与 Doppler **Hz**（PRF=512）；速度 m/s **NO**（UWB 无单一 fc）。本机未重放 STFT。",
        "5. 24 GHz：打开 `.dat` 后可给 Hz（及用头 `fc` 算 v）。ylim 5.8 GHz 只是作图。本机未打开。",
        "6. 77 GHz：脚本可给 Hz；`v=c f_D/(2 fc)` 可用脚本 fc。本机未建网格。",
        "7. R2：**NO**。",
        "8. R2 类型：**unavailable**（不是 time×velocity，也不是已落地的 time×Doppler 矩阵）。",
        "9. fingerprint：R0=1.000，R1=1.000，R2=未测。",
        "10. Radar-ID：R2 未测，谈不上下降。",
        "11. Cross-radar R2：未训。",
        "12. 24 GHz plotting：Phase 7 已见 10→24 因裁色条从 0.143→0.510；其余 24 向仍差。完整归因要等 raw STFT。",
        "13. sensor-domain：R1 后 Radar-ID 仍 1.0、多数 cross 仍差 → 仍强。R2 未检验。",
        "14. 下一步：**representation**。encoder 优化停在 PNG 上已经失败（CORAL/M3）。先抽出 raw 建 R2。",
        "",
        "## Gate 2",
        "",
        "```json",
        json.dumps(gate2, indent=2, ensure_ascii=False),
        "```",
        "",
        "未进入新模型训练。未进入 Phase 9。",
        "",
    ]
    (OUT / "phase8_physical_representation_report.md").write_text("\n".join(lines), encoding="utf-8")
    print("report ok", flush=True)


if __name__ == "__main__":
    main()
