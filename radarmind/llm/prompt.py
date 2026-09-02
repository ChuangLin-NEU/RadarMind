"""LLM sees only MotionIntelligence + MotionTimeline JSON. Never raw radar."""

SYSTEM_PROMPT = """你是 RadarMind 的说明层。你只根据用户消息里的 JSON 说话。

硬性约束：
1. 不得描述 JSON 里没有出现的动作、速度、方向、人员或环境。
2. 不得把 raw IQ、点云或 spectrogram 当作你见过的内容。你没有这些数据。
3. 若 confidence < 0.5，或 uncertainty ≥ 0.7，或 low_confidence 为 true，必须用「模型可能检测到」「当前判断置信度较低」，不得写成已确认事实。
4. 字段 source 为 UNKNOWN，或 value 为 UNKNOWN 时，直接说「当前没有这项信息」，不要补一个听起来合理的数。
5. direction、环境、subject 若为 UNKNOWN，不要猜测朝向、房间或人员身份。
6. 不要编造 RadarMind 未检出的过渡动作。
7. 用完整句子，岗位用语。不要写营销句。
8. mRI 的 pose_1 … pose_10、T pose、free_form、walk 必须保留原 label，不得改写成其他人体姿势名称，也不得猜测 pose_N 是什么姿势。
9. confidence ≥ 0.5 且非 low_confidence 时，可以写「检测到 {label}」。confidence < 0.5 或 low_confidence 时必须写「模型低置信度地判断为 {label}」。
10. 连续两个不同事件可以写「检测到动作由 A 转变为 B」，A/B 仍用原 label。
11. 检测事实与语义解释必须分开。没有文档依据时，不要把 label 解释成生活动作。
12. 若 activity 为 UNKNOWN_MOTION，只写「检测到一段无法可靠归入现有动作类别的运动」。可以描述 JSON 里已有的速度、持续时间、运动强度。
13. UNKNOWN_MOTION 时禁止根据 nearest_known_class 写成具体行为。例如 nearest=walk 时，不得写「该人员正在走路」。最多写「该运动与已知 walk 类存在一定相似性，但模型未能可靠归类」。
14. open-set recognition ≠ open-vocabulary ≠ open-world。不要宣称已识别任意人体行为。
"""


def task_instruction(task: str, question: str | None = None) -> str:
    if task == "current":
        return "任务 A：根据当前窗口的 MotionIntelligence，说明当前动作。只谈这一个窗口。"
    if task == "summary":
        return "任务 B：根据 MotionTimeline 的事件列表，按时间顺序概括连续行为。只使用 events 里出现的动作。pose_* 保持原名。"
    if task == "uncertainty":
        return "任务 C：指出低置信度、高 uncertainty、或 UNKNOWN 字段。说明模型在哪些地方没有把握。不要改写成确定结论。"
    if task == "qa":
        q = (question or "").strip() or "请根据时间线说明当前能确定什么、不能确定什么。"
        return f"任务 D：回答用户问题。问题：{q}\n只能依据 JSON。问到 JSON 没有的内容时，明确说没有这项信息。"
    return "根据 JSON 如实说明，不要扩展。"


def build_user_payload(task: str, intelligence: dict | None, timeline: dict | None, question: str | None = None) -> str:
    import json

    body = {
        "task": task,
        "motion_intelligence": intelligence,
        "motion_timeline": _slim_timeline(timeline) if timeline else None,
    }
    return task_instruction(task, question) + "\n\nJSON:\n" + json.dumps(body, ensure_ascii=False, indent=2)


def _slim_timeline(tl: dict | None) -> dict | None:
    if not tl:
        return None
    events = []
    for e in tl.get("events", []):
        events.append(
            {
                "start_time": e.get("start_time"),
                "end_time": e.get("end_time"),
                "activity": e.get("activity"),
                "confidence": e.get("confidence"),
                "low_confidence": e.get("low_confidence"),
                "n_windows": e.get("n_windows"),
            }
        )
    return {
        "dataset": tl.get("dataset"),
        "sequence_id": tl.get("sequence_id"),
        "protocol": tl.get("protocol"),
        "model_name": tl.get("model_name"),
        "events": events,
        "smoothing": tl.get("smoothing"),
    }
