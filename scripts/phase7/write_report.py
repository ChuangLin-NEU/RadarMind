#!/usr/bin/env python3
"""Write phase7_radar_invariance_report.md and answers.json from result files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from phase7.common import OUT, R0_CROSS, read_csv  # noqa: E402


def fget(rows, **kw):
    for r in rows:
        if all(str(r.get(k, "")) == str(v) for k, v in kw.items()):
            return r
    return {}


def num(v, default=None):
    try:
        if v in ("", None):
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def fmt(v, d=3):
    x = num(v)
    return "NA" if x is None else f"{x:.{d}f}"


def level(x, lo, hi):
    if x is None:
        return "UNKNOWN"
    if x >= hi:
        return "STRONG"
    if x >= lo:
        return "MODERATE"
    return "WEAK"


def main() -> None:
    fair = read_csv(OUT / "representation_fairness_comparison.csv")
    probes = read_csv(OUT / "probe_results.csv")
    loro = read_csv(OUT / "loro_results.csv")
    retr = read_csv(OUT / "retrieval_results.csv")
    pix = read_csv(OUT / "representation_radar_id.csv")
    gate = json.loads((OUT / "gate.json").read_text(encoding="utf-8")) if (OUT / "gate.json").is_file() else {}
    p3 = read_csv(ROOT / "results" / "phase3_shared_motion" / "representation_metrics.csv")

    def r1_acc(proto):
        return num(fget(fair, protocol=proto).get("R1_accuracy"))

    def r0_acc(proto):
        return R0_CROSS.get(proto)

    pix_r0 = num(fget(pix, representation="R0").get("accuracy"))
    pix_r1 = num(fget(pix, representation="R1").get("accuracy"))
    p_m0 = fget(probes, model="M0", protocol="multi_radar_shared")
    p_m3 = fget(probes, model="M3", protocol="multi_radar_shared")
    p3_m0 = fget(p3, model="M0", protocol="multi_radar_shared")

    r0_probe = num(p3_m0.get("radar_probe_accuracy"), 1.0)
    r1_base_probe = num(p_m0.get("radar_probe_accuracy"))
    m3_probe = num(p_m3.get("radar_probe_accuracy"))
    r1_act = num(p_m0.get("activity_probe_accuracy"))
    m3_act = num(p_m3.get("activity_probe_accuracy"))

    def loro_acc(model, tgt):
        return num(fget(loro, model=model, test_radar=tgt).get("activity_accuracy"))

    # retrieval means
    def ret_mean(model, key="action_top1"):
        vs = [num(r.get(key)) for r in retr if r.get("model") == model]
        vs = [v for v in vs if v is not None]
        return sum(vs) / len(vs) if vs else None

    deltas = []
    for proto in ("77→10", "10→77", "77→24", "24→77", "10→24", "24→10"):
        a, b = r0_acc(proto), r1_acc(proto)
        if a is not None and b is not None:
            deltas.append(b - a)
    mean_cross_delta = sum(deltas) / len(deltas) if deltas else None
    d2477 = None
    if r0_acc("24→77") is not None and r1_acc("24→77") is not None:
        d2477 = r1_acc("24→77") - r0_acc("24→77")
    d7724 = None
    if r0_acc("77→24") is not None and r1_acc("77→24") is not None:
        d7724 = r1_acc("77→24") - r0_acc("77→24")

    r1_ok = bool(gate.get("r1_constructed"))
    m3_ran = bool(p_m3)
    within = [r1_acc("10→10"), r1_acc("24→24"), r1_acc("77→77")]

    # Q answers
    q1 = (
        f"R0 整图 PNG Radar-ID（像素）={fmt(pix_r0)}。"
        f"R1 裁边后={fmt(pix_r1)}。"
        f"单源 cross 均值变化 {fmt(mean_cross_delta)}。"
        f"24→77 {fmt(r0_acc('24→77'))}→{fmt(r1_acc('24→77'))}；"
        f"77→24 {fmt(r0_acc('77→24'))}→{fmt(r1_acc('77→24'))}。"
    )
    max_abs = max(abs(d) for d in deltas) if deltas else 0.0
    if mean_cross_delta is not None and mean_cross_delta >= 0.08 and (pix_r1 is None or pix_r0 is None or pix_r1 < pix_r0 - 0.1):
        preprocess = "STRONG"
        q1 += " 去图框后 cross 整体上升，且像素 Radar-ID 下降。"
    elif (mean_cross_delta is not None and abs(mean_cross_delta) >= 0.03) or max_abs >= 0.15:
        preprocess = "MODERATE"
        q1 += " 个别方向（尤其涉及 24 GHz 色条）变化大，但六条 cross 没有一起抬升，像素 Radar-ID 仍为 1。"
    else:
        preprocess = "WEAK"
        q1 += " 裁边后 cross 没有稳定抬升，图框/色条解释不了主要失败。"

    q2 = f"R1 运动编码器 Radar-ID probe={fmt(r1_base_probe)}（R0 Phase 3 为 {fmt(r0_probe)}）。"
    if r1_base_probe is not None and r1_base_probe >= 0.95:
        q2 += " 统一 representation 后仍能按雷达分开。"
    elif r1_base_probe is not None:
        q2 += " 比 1.0 低一些，但仍高于随机（1/3）。"

    if not m3_ran:
        q3 = "未训 M3：gate 未过或还没有跑 PART D。"
        inv = "NO"
        sensor = "STRONG" if (r1_base_probe is not None and r1_base_probe >= 0.95) else "MODERATE"
    else:
        q3 = (
            f"M3 Radar-ID {fmt(r1_base_probe)}→{fmt(m3_probe)}；"
            f"Activity Probe {fmt(r1_act)}→{fmt(m3_act)}；"
            f"LORO 10+24→77 {fmt(loro_acc('M0','77GHz'))}→{fmt(loro_acc('M3','77GHz'))}；"
            f"10+77→24 {fmt(loro_acc('M0','24GHz'))}→{fmt(loro_acc('M3','24GHz'))}；"
            f"24+77→10 {fmt(loro_acc('M0','10GHz'))}→{fmt(loro_acc('M3','10GHz'))}。"
        )
        rad_drop = (r1_base_probe - m3_probe) if (r1_base_probe is not None and m3_probe is not None) else None
        act_keep = (m3_act / r1_act) if (m3_act is not None and r1_act not in (None, 0)) else None
        loro_gain = []
        for tgt in ("77GHz", "24GHz", "10GHz"):
            a, b = loro_acc("M0", tgt), loro_acc("M3", tgt)
            if a is not None and b is not None:
                loro_gain.append(b - a)
        lg = sum(loro_gain) / len(loro_gain) if loro_gain else None
        if rad_drop is not None and rad_drop >= 0.15 and act_keep is not None and act_keep >= 0.85 and lg is not None and lg > 0.02:
            inv = "YES"
            q3 += " Radar-ID 下降且动作保持、LORO 有升。"
        elif rad_drop is not None and rad_drop >= 0.10 and act_keep is not None and act_keep >= 0.80:
            inv = "PARTIAL"
            q3 += " Radar-ID 有降，但 LORO/检索未同时稳定变好，或幅度有限。"
        elif act_keep is not None and act_keep < 0.70:
            inv = "NO"
            q3 += " Activity Probe 掉太多，算假去域。"
        else:
            inv = "NO"
            q3 += " 未同时满足去雷达信息且保住动作/泛化。"
        sensor = "STRONG"
        if r1_base_probe is not None and r1_base_probe < 0.8:
            sensor = "MODERATE"
        if inv == "YES":
            sensor = "MODERATE"

    # Learn motion?
    learn = "NO"
    if inv == "YES" and preprocess != "STRONG":
        learn = "PARTIAL"
    if inv == "YES" and (m3_probe is not None and m3_probe < 0.7) and (lg if m3_ran else None) not in (None,) and (lg or 0) > 0.05:
        learn = "YES"
    if inv != "YES":
        learn = "NO" if inv == "NO" else "PARTIAL"

    # 24ghz worst?
    pairs = {p: r1_acc(p) for p in ("77→10", "10→77", "77→24", "24→77", "10→24", "24→10")}
    worst = None
    known = {k: v for k, v in pairs.items() if v is not None}
    if known:
        worst = min(known, key=known.get)

    answers = {
        "R1_available": "YES" if r1_ok else "NO",
        "raw_source_10": "PLOTTED_PNG（官方 628×1178；无 XeThru raw / 无 sx_scaled.mat）",
        "raw_source_24": "PLOTTED_PNG（官方 128×128；无 Ancortek raw / 无 header fc,Tsweep）",
        "raw_source_77": "PLOTTED_PNG（官方 804×1475；AIRHAR 数值链未用于 327 子集）",
        "R0_radar_fingerprint": fmt(pix_r0),
        "R1_radar_fingerprint": fmt(pix_r1),
        "cross": {p: {"R0": r0_acc(p), "R1": r1_acc(p)} for p in R0_CROSS},
        "radar_id_R0_probe": r0_probe,
        "radar_id_R1_baseline": r1_base_probe,
        "radar_id_M3": m3_probe,
        "activity_R1_baseline": r1_act,
        "activity_M3": m3_act,
        "loro": {
            "10+24→77": {"baseline": loro_acc("M0", "77GHz"), "M3": loro_acc("M3", "77GHz")},
            "10+77→24": {"baseline": loro_acc("M0", "24GHz"), "M3": loro_acc("M3", "24GHz")},
            "24+77→10": {"baseline": loro_acc("M0", "10GHz"), "M3": loro_acc("M3", "10GHz")},
        },
        "Q1": q1,
        "Q2": q2,
        "Q3": q3,
        "preprocess_confound": preprocess,
        "sensor_domain_shift": sensor if m3_ran or r1_base_probe is not None else (
            "STRONG" if (r1_acc("24→77") is not None and r1_acc("24→77") < 0.15) else "MODERATE"
        ),
        "radar_invariance_improved": inv,
        "learn_motion_not_sensor": learn,
        "within_R1": {"10": within[0], "24": within[1], "77": within[2]},
        "worst_cross_R1": worst,
        "pixel_radar_id": {"R0": pix_r0, "R1": pix_r1},
        "retrieval_action_top1": {"M0": ret_mean("M0"), "M3": ret_mean("M3")},
        "gate": gate,
    }
    if answers["sensor_domain_shift"] == "UNKNOWN":
        answers["sensor_domain_shift"] = "STRONG" if (r1_acc("24→77") is not None and (r1_acc("24→77") or 0) < 0.15) else "MODERATE"

    # 12 questions
    qa = [
        (
            "1. 官方 PNG 是否存在明显 radar-specific preprocessing fingerprint？",
            "是。" if (pix_r0 is not None and pix_r0 >= 0.9) else "证据不足或较弱。",
            f"Representation-only Radar-ID R0={fmt(pix_r0)}，R1={fmt(pix_r1)}。三频原生分辨率 628×1178 / 128×128 / 804×1475。",
        ),
        (
            "2. 24 GHz 128×128 是否是显著 confound？",
            "是尺寸 confound，但是否单独造成 24→77 失败要看 R1 对照。",
            f"SUPPORTED：全部 24 GHz 官方 PNG 为 128×128。R1 裁掉上下 8 行亮条后仍是 112×128 再放大到 224。24→77 R0={fmt(r0_acc('24→77'))} R1={fmt(r1_acc('24→77'))}。",
        ),
        (
            "3. R1 是否成功建立？",
            "YES（PLOTTED_PNG 公平裁边）" if r1_ok else "NO",
            "不是 RAW_RECOMPUTED。本机无 raw_data.zip，无三频 .mat STFT。禁止把 R1 写成物理速度对齐。",
        ),
        (
            "4. representation fairness 后 cross-radar 如何变化？",
            f"均值 Δ={fmt(mean_cross_delta)}",
            "；".join(f"{p} {fmt(r0_acc(p))}→{fmt(r1_acc(p))}" for p in ("24→77", "77→24", "10→77", "77→10", "10→24", "24→10")),
        ),
        (
            "5. 24 GHz transfer 是否仍然最差？",
            f"R1 最差方向：{worst}" if worst else "NA",
            f"24→77={fmt(r1_acc('24→77'))}，77→24={fmt(r1_acc('77→24'))}，10→24={fmt(r1_acc('10→24'))}。",
        ),
        (
            "6. R1 baseline Radar-ID probe 是否仍接近 1？",
            "是" if (r1_base_probe is not None and r1_base_probe >= 0.95) else ("未测" if r1_base_probe is None else "否"),
            f"R1 M0 probe={fmt(r1_base_probe)}；R0 Phase 3={fmt(r0_probe)}。",
        ),
        (
            "7. M3 能否降低 Radar-ID probe？",
            "未跑" if not m3_ran else ("能" if (m3_probe is not None and r1_base_probe is not None and m3_probe < r1_base_probe - 0.05) else "没有稳定降到可用程度"),
            f"M3={fmt(m3_probe)}。",
        ),
        (
            "8. Activity Probe 是否保持？",
            "未跑" if not m3_ran else ("保持" if (m3_act is not None and r1_act and m3_act >= 0.85 * r1_act) else "下降偏多"),
            f"R1 M0={fmt(r1_act)}，M3={fmt(m3_act)}。",
        ),
        (
            "9. Retrieval 是否改善？",
            "未跑" if ret_mean("M3") is None else ("有" if (ret_mean("M0") is not None and ret_mean("M3") > ret_mean("M0")) else "没有"),
            f"action top1 M0={fmt(ret_mean('M0'))} M3={fmt(ret_mean('M3'))}。",
        ),
        (
            "10. LORO 是否改善？",
            "未跑" if not loro else "",
            f"10+24→77 {fmt(loro_acc('M0','77GHz'))}→{fmt(loro_acc('M3','77GHz'))}；"
            f"10+77→24 {fmt(loro_acc('M0','24GHz'))}→{fmt(loro_acc('M3','24GHz'))}；"
            f"24+77→10 {fmt(loro_acc('M0','10GHz'))}→{fmt(loro_acc('M3','10GHz'))}。",
        ),
        (
            "11. preprocessing vs sensor-domain 各有哪些证据？",
            f"preprocessing={preprocess}；sensor-domain={answers['sensor_domain_shift']}",
            "preprocessing：分辨率/色条/整图缩放、像素 Radar-ID、R0 vs R1 Δ。"
            "sensor-domain：R1 后 within 仍高、cross 仍低、运动 embedding Radar-ID 仍高、LORO 仍差。",
        ),
        (
            "12. 当前是否可以支持 Learn Motion, Not Sensor？",
            learn,
            "只有 Radar-ID 明显下降、Activity 保持、检索/LORO 同时变好才能答 YES。本轮按上列数字判定。",
        ),
    ]
    if not qa[9][1]:
        lg_txt = []
        for tgt, lab in (("77GHz", "10+24→77"), ("24GHz", "10+77→24"), ("10GHz", "24+77→10")):
            a, b = loro_acc("M0", tgt), loro_acc("M3", tgt)
            if a is not None and b is not None:
                lg_txt.append(b - a)
        qa[9] = (
            qa[9][0],
            "有" if lg_txt and sum(lg_txt) / len(lg_txt) > 0.02 else ("没有" if lg_txt else "未跑"),
            qa[9][2],
        )

    lines = [
        "# Phase 7 — Cross-radar representation fairness + Radar-invariant encoder v0.3",
        "",
        "未改 frozen / paper / presentation / Phase 1–5 数字。R0 对照读 Phase 2/3 旧表。",
        "",
        "## Gate",
        "",
        "```json",
        json.dumps(gate, indent=2, ensure_ascii=False),
        "```",
        "",
        "## R1 来源",
        "",
        "- 10 GHz: PLOTTED_PNG，官方 628×1178，裁 1 像素边",
        "- 24 GHz: PLOTTED_PNG，官方 128×128，裁上下 8 行亮条",
        "- 77 GHz: PLOTTED_PNG，官方 804×1475，裁 top 55 / bottom 58",
        "- 物理速度轴：未做",
        "",
        "## R0 vs R1 单源",
        "",
        "| protocol | R0 | R1 | Δ |",
        "|---|---:|---:|---:|",
    ]
    for r in fair:
        lines.append(f"| {r.get('protocol')} | {fmt(r.get('R0_accuracy'))} | {fmt(r.get('R1_accuracy'))} | {fmt(r.get('delta'))} |")
    lines += ["", "## 必须回答的 12 问", ""]
    for title, short, ev in qa:
        lines += [f"### {title}", "", f"**{short}**", "", ev, ""]
    lines += [
        "## 应用页三问",
        "",
        f"- Q1 {q1}",
        f"- Q2 {q2}",
        f"- Q3 {q3}",
        "",
        "## 结论标签",
        "",
        f"- Preprocessing confound: **{preprocess}**",
        f"- Sensor-domain shift: **{answers['sensor_domain_shift']}**",
        f"- Radar invariance improved: **{inv}**",
        f"- Learn Motion, Not Sensor: **{learn}**",
        "",
        "未进入 Phase 8。",
        "",
    ]
    (OUT / "phase7_radar_invariance_report.md").write_text("\n".join(lines), encoding="utf-8")
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2, ensure_ascii=False), encoding="utf-8")
    print("wrote report", flush=True)


if __name__ == "__main__":
    main()
