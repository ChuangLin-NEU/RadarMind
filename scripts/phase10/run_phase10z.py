#!/usr/bin/env python3
"""Phase 10Z: scientific closure. Read-only on Phase 1–10G. No training."""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "phase10_closure"
FIG = OUT / "figures"
FROZEN = [
    "phase10_motion_factors",
    "phase10_sensor_calibration",
    "phase10_velocity_calibration",
    "phase10_unified_measurement",
    "phase10_hardware_measurement",
    "phase10_motion_conditioned",
    "phase10_conditional_2d_measurement",
]


def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def write_csv(path: Path, fields, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(exist_ok=True)
    freeze = []
    for name in FROZEN:
        p = ROOT / "results" / name
        freeze.append({"directory": f"results/{name}", "exists": p.is_dir(), "status": "READ_ONLY"})
    write_csv(OUT / "freeze_manifest.csv", ["directory", "exists", "status"], freeze)

    write(OUT / "cross_radar_evidence_chain.md", EVIDENCE)
    write_csv(OUT / "hypothesis_ledger.csv", LEDGER_FIELDS, LEDGER)
    write(OUT / "supported_claims.md", CLAIMS)
    write(OUT / "missing_observables.md", MISSING)
    write(OUT / "next_dataset_specification.md", DATASET)
    write(OUT / "acquisition_protocol.md", PROTOCOL)
    write_csv(OUT / "metadata_schema.csv", SCHEMA_FIELDS, SCHEMA)
    write(OUT / "preregistered_splits.md", SPLITS)
    write(OUT / "paper_story_v2.md", PAPER)
    write_csv(OUT / "cross_radar_failure_ladder.csv", LADDER_FIELDS, LADDER)
    make_figure()
    write(OUT / "phase10_closure_report.md", REPORT)
    answers = {
        "empirical_translation": "STOPPED",
        "tune_models_on_ci4r": "NO",
        "radar_invariant_encoder_on_current": "NO",
        "main_bottleneck": "MISSING SPATIAL / ACQUISITION METADATA",
        "next_resource": "DATA",
        "learn_motion_not_sensor": "NO",
        "case": "CLOSURE",
        "har_trained": False,
        "phase11": False,
        "geometry_aware": "BLOCKED BY DATA",
    }
    (OUT / "answers.json").write_text(json.dumps(answers, indent=2), encoding="utf-8")
    print("phase10z done", json.dumps(answers, indent=2), flush=True)


def make_figure() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    stages = ["R0\nPNG", "R1\ncrop", "R2-D\nHz", "R2-V\nm/s", "U1\nSTFT", "U2\nMTI", "M2\nG(v)", "q→D(v)", "2D\nrank-20"]
    rid = [1.000, 1.000, 0.993, 0.939, 0.939, 0.980, 0.949, 1.000, None]
    act = [None, 0.925, 0.878, 0.857, 0.867, 0.867, 0.857, 0.857, 0.735]
    fig, ax = plt.subplots(figsize=(10.4, 4.2))
    x = range(len(stages))
    yr = [v if v is not None else 0 for v in rid]
    ax.bar([i - 0.16 for i in x], yr, 0.32, label="Radar-ID", color="#4C6A92")
    ya = [v if v is not None else 0 for v in act]
    ax.bar([i + 0.16 for i in x], ya, 0.32, label="Activity / probe", color="#C4A35A")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages, fontsize=8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("accuracy")
    ax.axhline(1 / 3, color="0.5", ls="--", lw=0.7)
    ax.legend(loc="lower left")
    ax.set_title("Within-radar works; radar identity survives each alignment step")
    fig.savefig(FIG / "summary_figure.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


EVIDENCE = """# Cross-radar evidence chain

只读 Phase 7–10G 已冻结数字。本文件不改那些目录。
不是 geometry-aware 结论集。经验传感器翻译到此停止。

## A. Official PNG (R0)

Within-radar HAR 高：10/24/77 Accuracy 0.843 / 0.898 / 0.922（Phase 7）。
Cross-radar 塌：例如 24→77 0.039，77→24 0.163。
像素 Radar-ID = 1.000。

## B. R1 plotted-image normalization

裁边 + 统一后处理。preprocessing confound 存在（个别方向数字会动）。
Radar-ID 仍为 1.000。六条 cross 没有一起抬升。
结论：图框/色条不是主因。

## C. R2-D numerical Doppler

去掉绘图，轴为归一化时间 × Doppler Hz。
Radar-ID 0.993。Activity Probe 0.946。
Within：10/24/77 = 0.776 / 1.000 / 0.878。
Cross / LORO 仍差（Phase 8C case D）。
结论：画图伪影去掉以后，雷达身份还在。

## D. R2-V 24/77 velocity alignment

10 GHz：`EFFECTIVE_FC_UNRESOLVED`，不进速度轴。
24/77 Radar-ID：1.000 → 0.939。Activity 0.878 → 0.857。
Pair sep：0.086 → 0.087，基本没变。
Doppler scaling：**WEAK**。Measurement-model shift：**STRONG**。

## E. Unified STFT (U1)

相对 U0，Radar-ID 0.939 → 0.939（Δ=0）。Activity 0.857 → 0.867。
统一窗长/hop 没有消掉指纹。

## F. Unified MTI (U2)

Radar-ID 0.939 → 0.980，更强。Pair sep 下降。
不能把 MTI 差当成主缺口去“对齐”。

## G. Simple sensor calibration (C0–C4 / M1–M3)

Phase 10B：VAL RID gate 全 FAIL。CASE CAL-C。
Phase 10E M2：Radar-ID 0.949，速度边际 0.929，Activity 0.857。Gate FAIL。
一阶/按速度仿射校正不够。

## H. Motion factors (10A)

38 个可解释因子。Activity probe 0.687，Radar-ID 仍 1.000（稳定子集 0.959）。
动作信息在，雷达身份也在。CASE B。

## I. Fixed empirical G_r(v) (10E)

Transfer stability 0.395。mean std D(v) 1.531。动作 η² 0.558。
固定、与动作无关的 G_r(v) 不稳。CASE HW-C。

## J. Motion-conditioned D(v|q) (10F)

24→77 B1：R² 0.744，MAE 0.666，corr 0.868（相对 B0 明显好）。
77→24 B1：R² 0.575，MAE 0.876，corr 0.758。
校正后 Radar-ID 0.939 → 1.000。速度边际仍 1.000。
速度边际可预测，校正解释不了整张谱的传感器差。
真 geometry-aware：**BLOCKED**。

## K. Low-rank 2D Δ(τ,v|q) (10G)

B2 二维 Δ：24→77 R² 0.300；77→24 R² 0.173。
Oracle 天花板：R² 0.418（K=20 TRAIN 解释方差 0.490）。
24→77 预测图会被判成 77，但 Activity 0.898 → 0.735。
Gate FAIL。CASE C2D-C。

## Final

**STOP EMPIRICAL SENSOR TRANSLATION.**
当前 CI4R 327 对不支持再堆经验翻译 / 不变编码器。
下一资源是带几何与采集元数据的新数据，不是新网络。
"""

LEDGER_FIELDS = [
    "hypothesis",
    "experiment",
    "supported",
    "partially_supported",
    "rejected",
    "key_metric",
    "interpretation",
    "next_implication",
]

LEDGER = [
    {
        "hypothesis": "H1 cross-radar failure mainly caused by plotted PNG differences",
        "experiment": "Phase 7 R0 vs R1; Phase 8B R2-D",
        "supported": "no",
        "partially_supported": "yes",
        "rejected": "no",
        "key_metric": "Radar-ID R0=1.000 R1=1.000 R2-D=0.993",
        "interpretation": "图框有 confound，去掉后身份几乎不动",
        "next_implication": "不要把论文失败归因成 PNG",
    },
    {
        "hypothesis": "H2 Doppler-frequency normalization solves most domain shift",
        "experiment": "Phase 8C R2-D CE / LORO",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "R2-D Radar-ID 0.993; LORO near chance on some held-out radars",
        "interpretation": "公共 Doppler 栅格不够",
        "next_implication": "Hz 对齐不是解",
    },
    {
        "hypothesis": "H3 velocity alignment solves 24/77 shift",
        "experiment": "Phase 10C R2-V",
        "supported": "no",
        "partially_supported": "yes",
        "rejected": "no",
        "key_metric": "Radar-ID 1.000→0.939; pair sep 0.086→0.087",
        "interpretation": "载频缩放是弱项",
        "next_implication": "不要再调 common velocity 边界",
    },
    {
        "hypothesis": "H4 STFT mismatch is dominant",
        "experiment": "Phase 10D U1",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "U1 ΔRadar-ID = 0 vs U0",
        "interpretation": "统一窗长/hop 不消指纹",
        "next_implication": "不要再改 STFT 当主杠杆",
    },
    {
        "hypothesis": "H5 MTI mismatch is dominant",
        "experiment": "Phase 10D U2",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "U2 Radar-ID 0.980 > U1 0.939",
        "interpretation": "移植 Butterworth 让身份更清楚",
        "next_implication": "U2 不作正式表示",
    },
    {
        "hypothesis": "H6 simple radar-wise calibration removes fingerprint",
        "experiment": "Phase 10B C0–C4; Phase 10E M1–M3",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "10B gate FAIL; M2 Radar-ID 0.949 vel-RID 0.929",
        "interpretation": "已知雷达一阶校正不够",
        "next_implication": "停止简单 canonicalization",
    },
    {
        "hypothesis": "H7 motion factors are radar-invariant",
        "experiment": "Phase 10A",
        "supported": "no",
        "partially_supported": "yes",
        "rejected": "no",
        "key_metric": "activity 0.687; Radar-ID 1.000 / stable 0.959",
        "interpretation": "有动作信息，不是不变表示",
        "next_implication": "因子可作 q，不能当 invariant encoder",
    },
    {
        "hypothesis": "H8 radar-specific adapters learn shared motion",
        "experiment": "Phase 9 A0–A2 / LORO",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "embedding Radar-ID 1.000; A0>A1>A2; LORO not improved",
        "interpretation": "Adapter 没做成共享运动",
        "next_implication": "当前数据上不要再堆 Adapter",
    },
    {
        "hypothesis": "H9 fixed G_r(v) transfer exists",
        "experiment": "Phase 10E",
        "supported": "no",
        "partially_supported": "no",
        "rejected": "yes",
        "key_metric": "stability 0.395; action η² 0.558",
        "interpretation": "跨动作不稳",
        "next_implication": "禁止写固定硬件传递函数",
    },
    {
        "hypothesis": "H10 motion-conditioned D(v|q) is predictable",
        "experiment": "Phase 10F",
        "supported": "yes",
        "partially_supported": "no",
        "rejected": "no",
        "key_metric": "24→77 B1 R² 0.744; 77→24 R² 0.575",
        "interpretation": "速度边际可预测；校正不消 Radar-ID",
        "next_implication": "可写 motion-dependent response，不能写已校准",
    },
    {
        "hypothesis": "H11 full 2D transformation is low-rank and source-predictable",
        "experiment": "Phase 10G",
        "supported": "no",
        "partially_supported": "yes",
        "rejected": "no",
        "key_metric": "Oracle R² 0.418; B2 0.300/0.173; TRAIN var 0.490 at K=20",
        "interpretation": "有结构，不够低秩也不够可预测",
        "next_implication": "STOP EMPIRICAL SENSOR TRANSLATION",
    },
]

CLAIMS = """# Supported claims

依据 Phase 7–10G。不把未做的写成已做。

## STRONG CLAIM

- Cross-radar HAR 明显弱于 within-radar。Within 约 0.82–1.00；多条 cross 接近或低于有用水平（24→77 R0 0.039）。
- 去掉绘图伪影后，雷达身份仍然容易恢复：R0/R1 = 1.000，R2-D = 0.993。
- 仅做 24/77 速度轴对齐不能消掉域差：Radar-ID 仍 0.939，pair sep 几乎不变。
- 统一 STFT / 简单 MTI / 一阶雷达校正都不够。U1 ΔRID=0；U2 更差；C0–C4 与 M2 gate 未过。
- 当前公开 CI4R 327 对无法验证 geometry-aware 模型：距离、方位、姿态字段不存在。
- 10 GHz 不能可靠地放到与 24/77 相同的速度轴上（有效载频未决）。
- 经验传感器翻译应停止：10G Oracle R² 仅 0.418，Activity 保不住。

## SUPPORTED BUT LIMITED

- 运动状态能解释一部分成对响应差：10F D(v|q) 24→77 R² 0.744。
- 跨雷达测量差里有可结构、随运动变化的成分：10E 动作 η² 0.558；10G B2 好于 B0。
- 低秩二维能抓住一部分未见 acquisition 的差，但不是大部分：Oracle 0.418，K=20 只解释 TRAIN 0.490。
- 载频 Doppler 缩放是弱贡献，不是零。
- Phase 9 Adapter 在已知雷达上也不稳定（A0>A1>A2），不能写成“已知雷达已对齐”。

## NOT SUPPORTED

- Learn Motion, Not Sensor 已经做到。
- 存在 Universal / sensor-invariant radar encoder。
- geometry 造成了当前残差（元数据 UNAVAILABLE，只能列候选隐变量）。
- 10 GHz 可转到可靠公共速度轴。
- 已证明 cross-environment 泛化。
- 已做成可用的 24↔77 传感器翻译或不变表示。
- unseen-radar DG 已被校准或 Adapter 解决（Phase 8C LORO / Phase 9 LORO 未改善）。
"""

MISSING = """# Missing observables

只列当前真正缺、且有科学价值的量。
这些是 **candidate latent variables**。
**不能**写成：它们一定解释了现在的 residual。

| 量 | 状态 | 为何可能有用 |
|---|---|---|
| physical target range | MISSING | 24 无米轴；77 AIRHAR 行号未接到米 |
| subject-to-radar distance | MISSING | 文件名/mapping/split 无逐条距离 |
| azimuth / aspect angle | MISSING | 无测角、无逐条方位 |
| elevation angle | MISSING | 无 |
| body orientation | MISSING | 无姿态字段 |
| radar pose | MISSING | 无雷达位姿 |
| antenna geometry | UNRESOLVED | README 只写 co-located，无基线坐标 |
| channel-level IQ | MISSING | 现用链不是全通道保留后分析 |
| range trajectory | MISSING | 未存逐 chirp 距离轨迹 |
| spatial scattering distribution | MISSING | 无多通道/角度维 |
| subject identity | UNRESOLVED | 文中 6 人，未对上 327 timestamp |
| session identity | MISSING | 无 session 字段 |
| environment identity | MISSING | 单实验室说明，无房间 ID |
| approach/recede as measured geometry | UNRESOLVED | Towards/Away 是动作标签，不是几何量 |

AVAILABLE（与几何无关、已经在用）：
- 雷达种类、动作标签、acquisition timestamp、24/77 的 fc 与公共速度轴、U1 数值谱。

下一轮若要做 hardware-aware 模型，必须先能记录 range / aspect / orientation 或阵列空间量。
现在不能从动作标签代替几何。
"""

DATASET = """# Next dataset specification

目标不是“再多录一些动作”。
目标是：把 sensor / motion / geometry / subject / environment 拆开。

## Minimum viable

雷达：先做 **24 GHz + 77 GHz**。两路 fc 与速度轴已经能恢复。
10 GHz UWB：第二批再加；第一批不要让未决载频拖死速度轴。

同步：两部雷达必须同时看同一个人、同一次动作。禁止分开各做一遍再事后配对。

标识：每条记录统一
`acquisition_id, timestamp, subject_id, action_id, repetition_id, environment_id`。

几何（每条至少）：
subject–radar distance，aspect，body orientation，radar position / height / orientation，动作起止位置。
能记 reference trajectory 就记。

最小 factorial（第一批只控 sensor × subject × geometry × motion）：

- distance：2 m / 3 m / 4 m
- aspect：0° / 45° / 90°
- subject：5–10 人
- action：沿用现有 7 类（Away, Bend, Crawl, Kneel, Pick, Sit, Towards）
- repetition：每条件至少 3 次
- environment：先固定一个房间

environment（房间 / LOS / 遮挡）放到第二阶段，第一轮不要混。

必须能恢复：range [m]、velocity [m/s]、time [s]。
禁止只存 PNG。必须 RAW IQ + metadata。中间量（range-time / range-Doppler / micro-Doppler）可选。
尽量保留全部 RX；有 MIMO 就留虚拟阵。

标定会话与正式动作分开标记：空室、静止反射、站立、简单周期运动。

运动真值：能同步视频就同步；时间戳必须对齐。专业 MoCap 不作第一阶段硬性要求。

模型：数据问题清楚以前，不预定 Transformer / GNN / Mamba / Diffusion。
有几何之后才考虑 M(t, range, velocity, angle) 或 latent motion + sensor operator。
当前 CI4R 不支持。
"""

PROTOCOL = """# Acquisition protocol

## 三个必须能做的实验

当前 CI4R **做不到** 这三组对照。

### E1 Same Motion × Same Geometry × Different Radar

同一人、同一动作、同一距离/方位/朝向，24 与 77 同时采。
用来拆 **sensor effect**。

### E2 Same Motion × Different Geometry × Same Radar

同一雷达、同一人、同一动作，只改 distance / aspect / orientation。
用来拆 **geometry effect**，画单雷达 measurement manifold。

### E3 Same Geometry × Same Action × Different Subject

同一几何、同一动作，换人。
比较 subject variation 与 radar variation 谁大。

## 现场流程（单条）

1. 标定会话（当日一次）：空室、参考反射、站立。标记 `session_type=calibration`。
2. 摆几何：卷尺/地面标记对 distance 与 aspect；记录雷达高度与朝向。
3. 两雷达共触发或硬件同步；记下 `acquisition_id` 与统一时间戳。
4. 被试按动作表做一次；摄像机同时开。
5. 写完 metadata 再收下一条件。缺几何字段的条目标 `unusable`。

## 第一批规模（量级）

7 动作 × 3 距离 × 3 方位 × 8 人 × 3 次 × 2 雷达 ≈ 3000+ 同步条。
可先做 5 人、每条件 2 次作试点，但 grid 不要拆掉。

## 禁止

- 只存频谱图 PNG
- 先采完再按结果改 split
- 第一轮同时换房间、换遮挡、换很多新动作
- 用动作名代替几何量
"""

SCHEMA_FIELDS = ["field", "required", "type", "example", "notes"]
SCHEMA = [
    {"field": "acquisition_id", "required": "yes", "type": "string", "example": "acq_000123", "notes": "两雷达共享"},
    {"field": "timestamp", "required": "yes", "type": "iso8601", "example": "2026-08-30T10:00:00+08:00", "notes": "与视频对齐"},
    {"field": "subject_id", "required": "yes", "type": "string", "example": "S03", "notes": "稳定匿名 ID"},
    {"field": "action_id", "required": "yes", "type": "enum", "example": "Pick", "notes": "7 类之一"},
    {"field": "repetition_id", "required": "yes", "type": "int", "example": "2", "notes": ""},
    {"field": "environment_id", "required": "yes", "type": "string", "example": "roomA", "notes": "第一批固定"},
    {"field": "session_type", "required": "yes", "type": "enum", "example": "activity", "notes": "activity|calibration"},
    {"field": "radar_id", "required": "yes", "type": "enum", "example": "24GHz", "notes": "24GHz|77GHz"},
    {"field": "fc_hz", "required": "yes", "type": "float", "example": "2.475e10", "notes": "写入 header，不事后猜"},
    {"field": "bandwidth_hz", "required": "yes", "type": "float", "example": "1.5e9", "notes": ""},
    {"field": "chirp_duration_s", "required": "yes", "type": "float", "example": "0.001", "notes": ""},
    {"field": "adc_fs_hz", "required": "yes", "type": "float", "example": "", "notes": ""},
    {"field": "n_adc", "required": "yes", "type": "int", "example": "128", "notes": ""},
    {"field": "prf_hz", "required": "yes", "type": "float", "example": "1000", "notes": ""},
    {"field": "n_rx", "required": "yes", "type": "int", "example": "4", "notes": "全通道保存"},
    {"field": "range_fft_def", "required": "yes", "type": "string", "example": "fftshift half-profile", "notes": "能复现米轴"},
    {"field": "distance_m", "required": "yes", "type": "float", "example": "3.0", "notes": "subject–radar"},
    {"field": "aspect_deg", "required": "yes", "type": "float", "example": "45", "notes": ""},
    {"field": "elevation_deg", "required": "no", "type": "float", "example": "0", "notes": "能量到就记"},
    {"field": "body_orientation_deg", "required": "yes", "type": "float", "example": "0", "notes": "相对雷达 LOS"},
    {"field": "radar_x_m", "required": "yes", "type": "float", "example": "0", "notes": "房间坐标"},
    {"field": "radar_y_m", "required": "yes", "type": "float", "example": "0", "notes": ""},
    {"field": "radar_height_m", "required": "yes", "type": "float", "example": "1.2", "notes": ""},
    {"field": "radar_yaw_deg", "required": "yes", "type": "float", "example": "0", "notes": ""},
    {"field": "start_x_m", "required": "yes", "type": "float", "example": "3.0", "notes": "动作起点"},
    {"field": "start_y_m", "required": "yes", "type": "float", "example": "0", "notes": ""},
    {"field": "end_x_m", "required": "yes", "type": "float", "example": "2.0", "notes": ""},
    {"field": "end_y_m", "required": "yes", "type": "float", "example": "0", "notes": ""},
    {"field": "raw_path", "required": "yes", "type": "path", "example": "raw/24/acq_000123.bin", "notes": "IQ"},
    {"field": "video_path", "required": "no", "type": "path", "example": "video/acq_000123.mp4", "notes": "时间戳对齐"},
]

SPLITS = """# Preregistered splits

采集前定死。禁止按 TEST 结果改划分。

| split | 定义 | 用来回答 |
|---|---|---|
| within-subject | 同一 subject 的 rep 划分 train/val/test | 个人复现 |
| cross-subject | 若干 subject 全持出 | 人是否可迁移 |
| cross-geometry | 持出某个 distance 或 aspect | 几何是否可迁移 |
| cross-radar known-sensor | 两雷达都在训练里出现，但是成对/不成对要写明 | 已知雷达校准上限 |
| leave-one-radar-out | 一部雷达的全部 activity 不进训练 | unseen-radar DG |
| cross-environment | 第二阶段才启用；持出 room / clutter | 环境 |

规则：

- 同一 `acquisition_id` 的 24/77 必须落在同一 split。
- calibration session 不进 HAR 的 train/test，只进测量响应估计。
- 主文表格只报预注册 split。探索性切分标 `exploratory`，不进主表。
"""

PAPER = """# Paper story v2

主线不要塞 RadarMind 全部功能。
Continuous / open-set / LLM / 桌面回放：系统扩展，不抢主文。

## 建议题目

Cross-Radar Human Activity Recognition:
When High Within-Radar Accuracy Fails to Generalize

## 结构

1. Benchmark setup — CI4R 327 对，7 动作，10/24/77；acquisition split 229/49/49。
2. Within vs cross-radar gap — within 高，cross 塌。
3. Preprocessing confounds — R0 整图 PNG → R1 裁边；Radar-ID 仍 1。
4. Numerical and velocity-domain reconstruction — R2-D；24/77 R2-V。10 GHz 速度轴未决。
5. Representation fingerprint — 输入 / 边际 / 因子 / embedding 仍能识别雷达。
6. Failure of generic alignment/calibration — 统一 STFT、MTI、C0–C4、M1–M3。
7. Motion-dependent measurement shift — D(v|q) 可预测；二维低秩天花板低。
8. Limits of current public data — 无距离/方位/姿态；不能验证 geometry-aware。
9. Implications — 下一资源是可拆因子的数据，不是更深的 translator。

## 核心表

Cross-Radar Failure Ladder（见 `cross_radar_failure_ladder.csv`）。
让读者看到：这是系统拆解 domain shift，不是试一个模型没过。

## 核心图

From Apparent Generalization to Measurement-Model Limitation
（`figures/summary_figure.png`）

左：within-radar 高。
中：cross-radar 塌。
右：PNG → 数值 → 速度 → 统一 STFT，Radar-ID 仍高。
最后：运动相关测量差 + 缺失空间隐变量。

## 不写进主文的句子

- Universal encoder
- Learn Motion, Not Sensor 已实现
- geometry 已证实是残差原因
- 10 GHz 已速度对齐
"""

LADDER_FIELDS = [
    "stage",
    "representation",
    "radar_id",
    "activity",
    "cross_radar_or_pair",
    "finding",
]

LADDER = [
    {
        "stage": "R0",
        "representation": "official plotted PNG",
        "radar_id": "1.000",
        "activity": "within 0.843/0.898/0.922",
        "cross_radar_or_pair": "24→77 0.039; 77→24 0.163",
        "finding": "within 高，cross 塌，像素能分开雷达",
    },
    {
        "stage": "R1",
        "representation": "cropped plotted PNG",
        "radar_id": "1.000",
        "activity": "probe 0.925; within 0.961/0.857/0.824",
        "cross_radar_or_pair": "cross 未一起抬升",
        "finding": "图框 confound 存在，不是主因",
    },
    {
        "stage": "R2-D",
        "representation": "S(τ, Doppler Hz) numerical",
        "radar_id": "0.993",
        "activity": "within 0.776/1.000/0.878; probe 0.946",
        "cross_radar_or_pair": "LORO 仍差; pair same 0.917 vs diff 0.913",
        "finding": "去掉绘图后身份还在",
    },
    {
        "stage": "R2-V",
        "representation": "24/77 S(τ, v m/s)",
        "radar_id": "0.939",
        "activity": "0.857",
        "cross_radar_or_pair": "pair sep 0.086→0.087",
        "finding": "载频缩放弱；测量差仍强",
    },
    {
        "stage": "U1",
        "representation": "unified STFT R2-V",
        "radar_id": "0.939",
        "activity": "0.867",
        "cross_radar_or_pair": "sep 0.105",
        "finding": "统一 STFT 不降 Radar-ID",
    },
    {
        "stage": "U2",
        "representation": "U1 + transplanted MTI",
        "radar_id": "0.980",
        "activity": "0.867",
        "cross_radar_or_pair": "sep 0.063",
        "finding": "统一 MTI 让身份更强",
    },
    {
        "stage": "M2",
        "representation": "per-velocity affine calibration",
        "radar_id": "0.949",
        "activity": "0.857",
        "cross_radar_or_pair": "vel-RID 0.929; sep 0.072",
        "finding": "一阶 G_r(v) 校正 gate 未过",
    },
    {
        "stage": "motion factors",
        "representation": "38 interpretable factors",
        "radar_id": "1.000 (stable 0.959)",
        "activity": "0.687 (stable 0.490)",
        "cross_radar_or_pair": "paired factor distance 1.18",
        "finding": "有动作信息，不是雷达不变",
    },
    {
        "stage": "D(v|q)",
        "representation": "motion-conditioned velocity transfer",
        "radar_id": "cal 0.939→1.000",
        "activity": "0.867→0.857",
        "cross_radar_or_pair": "24→77 B1 R² 0.744",
        "finding": "边际可预测，校正不消指纹",
    },
    {
        "stage": "2D low-rank",
        "representation": "PCA+Ridge Δ(τ,v|q)",
        "radar_id": "predicted maps flip to target domain",
        "activity": "24→77 predicted 0.735",
        "cross_radar_or_pair": "Oracle R² 0.418; B2 0.300/0.173",
        "finding": "二维翻译天花板低；停止经验翻译",
    },
]

REPORT = """# Phase 10 closure

Phase 10A–10G 只读。本阶段无新训练、无新模型、无 Phase 11。

## 四个判断

A. 现有 CI4R 是否还值得继续大量调模型？ **NO**
B. 是否继续在这 327 对上追求 radar-invariant encoder？ **NO**
C. 下一科研资源： **DATA**
D. 下一批最关键新增变量：subject–radar distance，aspect，body orientation（外加 RAW + 可恢复 range 轴）
E. Learn Motion, Not Sensor： **NO**

## 主瓶颈

MISSING SPATIAL / ACQUISITION METADATA

不是“项目失败”。是：在现有公开成对谱上，经验对齐/校准/低秩翻译已经做到可写进论文的上限。
再堆网络不会把未记录的几何变成可观测量。

## 主文

题目走 within 高、cross 塌、拆解 representation 与测量差。
系统功能（连续、open-set、说明层）不进主线。

## 下一数据

双雷达同步 24/77；最小几何网格；预注册 split；RAW IQ。
有几何之前不写 hardware-aware 深度模型。
"""


if __name__ == "__main__":
    main()
