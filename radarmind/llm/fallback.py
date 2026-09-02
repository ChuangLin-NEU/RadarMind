"""Deterministic text when DEEPSEEK_API_KEY is absent or the API fails."""
from __future__ import annotations


def _val(field) -> object:
    if isinstance(field, dict):
        return field.get("value", "UNKNOWN")
    return field


def _src(field) -> str:
    if isinstance(field, dict):
        return str(field.get("source", "UNKNOWN"))
    return "UNKNOWN"


def _conf(intel: dict | None) -> float:
    if not intel:
        return 0.0
    c = _val(intel.get("confidence"))
    try:
        return float(c)
    except (TypeError, ValueError):
        return 0.0


def _unc(intel: dict | None) -> float:
    if not intel:
        return 1.0
    u = _val(intel.get("uncertainty"))
    try:
        return float(u)
    except (TypeError, ValueError):
        return 1.0


def _hedge(conf: float, unc: float, low: bool = False) -> bool:
    return low or conf < 0.5 or unc >= 0.7


def current_text(intel: dict | None) -> str:
    if not intel:
        return "当前没有 MotionIntelligence，无法说明动作。"
    act = _val(intel.get("activity"))
    conf = _conf(intel)
    unc = _unc(intel)
    if act in (None, "UNKNOWN"):
        return "当前窗口没有模型给出的动作标签。"
    if act == "UNKNOWN_MOTION":
        ev = intel.get("evidence") or {}
        nearest = ev.get("nearest_known_class")
        extra = ""
        if nearest and nearest not in (None, "UNKNOWN"):
            extra = f"该运动与已知 {nearest} 类存在一定相似性，但模型未能可靠归类。"
        return "检测到一段无法可靠归入现有动作类别的运动。" + extra
    if _hedge(conf, unc):
        return f"模型低置信度地判断为「{act}」。当前判断置信度较低（confidence={conf:.3f}，uncertainty={unc:.3f}），不能写成已确认事实。"
    return f"检测到「{act}」。confidence={conf:.3f}。该名称保持数据集原 label，不改写成其他姿势。"


def summary_text(timeline: dict | None) -> str:
    if not timeline or not timeline.get("events"):
        return "当前没有时间线事件，无法做连续行为概括。"
    parts = []
    for e in timeline["events"]:
        act = e.get("activity", "UNKNOWN")
        t0, t1 = e.get("start_time"), e.get("end_time")
        conf = float(e.get("confidence") or 0)
        low = bool(e.get("low_confidence"))
        unit = (timeline or {}).get("smoothing", {}).get("time_unit") or "s"
        span = f"{t0:.2f}–{t1:.2f} {unit}"
        if act == "UNKNOWN_MOTION":
            parts.append(f"{span} 无法归入现有动作类别的运动")
        elif _hedge(conf, 0.0, low):
            parts.append(f"{span} 模型可能检测到「{act}」（置信度较低，{conf:.3f}）")
        else:
            parts.append(f"{span} 「{act}」")
    if len(parts) == 1:
        return "该段时间内：" + parts[0] + "。"
    return "按时间顺序：" + "，随后 ".join(parts) + "。"


def uncertainty_text(intel: dict | None, timeline: dict | None) -> str:
    lines = []
    if intel:
        conf, unc = _conf(intel), _unc(intel)
        if _hedge(conf, unc):
            lines.append(f"当前窗口 confidence={conf:.3f}，uncertainty={unc:.3f}，不宜当作确定结论。")
        for key in ("direction", "mean_velocity", "peak_velocity", "subject", "environment"):
            f = intel.get(key) or {}
            if _val(f) == "UNKNOWN" or _src(f) == "UNKNOWN":
                lines.append(f"字段 {key} 为 UNKNOWN，没有可写的数值。")
    if timeline:
        lows = [e for e in timeline.get("events", []) if e.get("low_confidence") or float(e.get("confidence") or 0) < 0.5]
        if lows:
            acts = "、".join(sorted({e.get("activity", "?") for e in lows}))
            lines.append(f"时间线里有 {len(lows)} 个低置信度事件，涉及：{acts}。")
    if not lines:
        return "当前窗口和时间线没有低于阈值的不确定项。UNKNOWN 字段仍不得补写。"
    return " ".join(lines)


def _events(timeline: dict | None) -> list[dict]:
    return list((timeline or {}).get("events") or [])


def qa_text(question: str, intel: dict | None, timeline: dict | None) -> str:
    q = (question or "").strip()
    ev = _events(timeline)
    stats = ((intel or {}).get("series_stats") or {})
    if not q:
        return " ".join([current_text(intel), summary_text(timeline)])
    if "刚才" in q or "发生了什么" in q:
        return summary_text(timeline)
    if "变化" in q or "转变" in q or "什么时候" in q:
        if len(ev) < 2:
            return "时间线里没有记录到动作类别变化。"
        bits = []
        for i in range(1, len(ev)):
            a, b = ev[i - 1], ev[i]
            bits.append(
                f"{a.get('activity')} → {b.get('activity')}，约 {a.get('end_time')}–{b.get('start_time')}"
            )
        return "检测到的动作变化：" + "；".join(bits) + "。标签保持原名，不做姿势翻译。"
    if "不确定" in q or "uncertainty" in q.lower():
        t = stats.get("min_confidence_time")
        act = stats.get("min_confidence_activity")
        if t is not None and act:
            return f"序列里最低置信度出现在 t={t}，当时模型给出的是「{act}」。这不是 unknown-class 检测。"
        return uncertainty_text(intel, timeline)
    if "最大速度" in q or "速度" in q or "velocity" in q.lower():
        v = stats.get("peak_abs_velocity")
        t = stats.get("peak_velocity_time")
        if v in (None, "UNKNOWN"):
            return "当前 JSON 没有可用的速度峰值。"
        return f"序列摘要里 |velocity| 最大为 {v}，对应时间 {t}。单位未经标定，不能写成米/秒。"
    if "walk" in q.lower() or "是否检测到 walk" in q:
        hits = [e for e in ev if e.get("activity") == "walk"]
        if not hits:
            return "这条时间线的事件列表里没有 walk。"
        return f"时间线里有 {len(hits)} 段 walk，例如 {hits[0].get('start_time')}–{hits[0].get('end_time')}。"
    keys = ("方向", "direction", "人员", "subject", "环境", "environment")
    extra = []
    if any(k in q.lower() or k in q for k in keys):
        extra.append(uncertainty_text(intel, timeline))
        extra.append("问题里问到的物理量，只以 JSON 中非 UNKNOWN 字段为准。")
    return " ".join([current_text(intel), summary_text(timeline)] + extra)


def render(task: str, intel: dict | None, timeline: dict | None, question: str | None = None) -> str:
    if task == "current":
        return current_text(intel)
    if task == "summary":
        return summary_text(timeline)
    if task == "uncertainty":
        return uncertainty_text(intel, timeline)
    if task == "qa":
        return qa_text(question or "", intel, timeline)
    return current_text(intel)
