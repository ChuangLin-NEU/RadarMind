# Project status

## DONE

- CI4R 77 GHz HAR（random / prefix-group）。原因：metrics 与 checkpoint 可加载。
- mRI / RadHAR point-cloud HAR。原因：training_runs.csv 与 SI/official checkpoint 在。
- mRI cross-subject。原因：官方 subject id + SI split。
- CI4R cross-radar within / zero-shot / CORAL。原因：Phase 2B training_runs.csv。
- Shared Motion v0.1 Task A + probes + LORO。原因：Phase 3 数字冻结。
- Labeled-interval continuous inference。原因：Phase 4 CSV；不是 full-stream。
- Open-set LOAO 13-fold。原因：Phase 5 eval_summary。
- Replay Application + Timeline + DeepSeek/fallback。

## PARTIAL

- sensor-invariant representation。原因：Radar-ID probe=1.0。
- continuous full-stream segmentation。原因：UNLABELED 未做雷达推理。
- open-world understanding。原因：只有 open-set MVP。
- open-vocabulary recognition。原因：未做。

## PENDING

- cross-environment。原因：无环境协议与数据划分。
- MM-Fi / MiliPoint / FAU。原因：本阶段禁止新数据。
- Live Radar。原因：未接硬件。

## BLOCKED_BY_DATA

- true full-stream mRI（含 UNLABELED）。原因：本机无 raw CSV。
- CI4R official cross-subject。原因：无可靠 subject ID。
- RadHAR cross-subject。原因：无 subject metadata。
