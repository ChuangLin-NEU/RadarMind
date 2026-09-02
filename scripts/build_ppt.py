"""Build the 15-slide group-meeting deck from notes/ppt_outline.md and results/."""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_AUTO_SHAPE_TYPE
from pptx.enum.text import PP_ALIGN
from pptx.oxml.ns import nsmap
from pptx.oxml import parse_xml
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "results" / "figures" / "ppt"
OUT = ROOT / "notes" / "RadarBridge_group_meeting.pptx"

CLASS_NAMES = [
    "Towards",
    "Away",
    "Pick",
    "Bend",
    "Sit",
    "Kneel",
    "Crawl",
    "Limp",
    "Toes",
    "SStep",
    "Scissor",
]

NAVY = RGBColor(0x1E, 0x3A, 0x4C)
INK = RGBColor(0x2C, 0x2C, 0x2C)
MUTED = RGBColor(0x5A, 0x5A, 0x5A)
PLAN = RGBColor(0x8A, 0x8A, 0x8A)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)
LINE = RGBColor(0xD0, 0xD4, 0xD8)
ALT = RGBColor(0xF3, 0xF5, 0xF6)
TODO_BG = RGBColor(0xEE, 0xEE, 0xEE)
ACCENT = RGBColor(0x3D, 0x5A, 0x6C)
FONT = "微软雅黑"


def _set_run(run, text, size=16, bold=False, color=INK, font=FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = font
    rpr = run._r.get_or_add_rPr()
    # Keep East-Asian font on Windows
    ea = parse_xml(
        f'<a:ea xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{font}"/>'
    )
    rpr.append(ea)


def add_text(box, text, size=16, bold=False, color=INK, align=PP_ALIGN.LEFT):
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    _set_run(run, text, size=size, bold=bold, color=color)
    return tf


def add_lines(box, lines, size=14, color=INK, bold_first=False, spacing=1.05):
    tf = box.text_frame
    tf.clear()
    tf.word_wrap = True
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(4)
        run = p.add_run()
        _set_run(run, line, size=size, bold=(bold_first and i == 0), color=color)
    return tf


def blank_slide(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def title_bar(slide, title, status):
    bar = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(0.82)
    )
    bar.fill.solid()
    bar.fill.fore_color.rgb = NAVY if status == "已完成" else RGBColor(0x4A, 0x4A, 0x4A)
    bar.line.fill.background()
    add_text(bar, title, size=22, bold=True, color=WHITE)
    bar.text_frame.paragraphs[0].alignment = PP_ALIGN.LEFT
    bar.text_frame.margin_left = Inches(0.28)
    bar.text_frame.margin_top = Inches(0.18)

    foot = slide.shapes.add_shape(
        MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(7.22), Inches(13.333), Inches(0.28)
    )
    foot.fill.solid()
    foot.fill.fore_color.rgb = RGBColor(0xF0, 0xF0, 0xF0)
    foot.line.fill.background()
    color = MUTED if status == "已完成" else PLAN
    add_text(foot, f"RadarBridge  ·  {status}", size=11, color=color)
    foot.text_frame.margin_left = Inches(0.28)
    foot.text_frame.margin_top = Inches(0.02)


def notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def point(slide, text, top=0.92):
    box = slide.shapes.add_textbox(Inches(0.28), Inches(top), Inches(12.7), Inches(0.55))
    add_text(box, text, size=15, color=INK)


def table_at(slide, rows, left, top, width, height, col_w=None, header=True):
    n_row, n_col = len(rows), len(rows[0])
    shape = slide.shapes.add_table(n_row, n_col, left, top, width, height)
    tbl = shape.table
    if col_w:
        for i, w in enumerate(col_w):
            tbl.columns[i].width = w
    for r, row in enumerate(rows):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = ""
            p = cell.text_frame.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if c > 0 or (header and r == 0) else PP_ALIGN.LEFT
            run = p.add_run()
            is_head = header and r == 0
            _set_run(run, str(val), size=12 if n_col <= 5 else 11, bold=is_head, color=WHITE if is_head else INK)
            cell.text_frame.word_wrap = True
            fill = NAVY if is_head else (ALT if r % 2 == 0 else WHITE)
            cell.fill.solid()
            cell.fill.fore_color.rgb = fill
            tc = cell._tc
            tcPr = tc.get_or_add_tcPr()
            ln = parse_xml(
                '<a:lnL w="6350" xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
                '<a:solidFill><a:srgbClr val="D0D4D8"/></a:solidFill></a:lnL>'
            )
            # skip extra borders; default is fine
    return shape


def todo_box(slide, text, left, top, width, height):
    box = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, left, top, width, height)
    box.fill.solid()
    box.fill.fore_color.rgb = TODO_BG
    box.line.color.rgb = LINE
    add_text(box, text, size=13, color=MUTED, align=PP_ALIGN.CENTER)
    box.text_frame.word_wrap = True
    box.text_frame.margin_left = Inches(0.12)
    box.text_frame.margin_right = Inches(0.12)
    return box


def flow_boxes(slide, labels, top=1.65, left=0.28, box_w=1.85, gap=0.18):
    h = 0.72
    for i, lab in enumerate(labels):
        x = left + i * (box_w + gap)
        sh = slide.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(x), Inches(top), Inches(box_w), Inches(h))
        sh.fill.solid()
        sh.fill.fore_color.rgb = ACCENT
        sh.line.fill.background()
        add_text(sh, lab, size=12, bold=True, color=WHITE, align=PP_ALIGN.CENTER)
        sh.text_frame.word_wrap = True
        if i < len(labels) - 1:
            arr = slide.shapes.add_shape(
                MSO_AUTO_SHAPE_TYPE.RIGHT_ARROW,
                Inches(x + box_w + 0.02),
                Inches(top + 0.24),
                Inches(gap - 0.04),
                Inches(0.24),
            )
            arr.fill.solid()
            arr.fill.fore_color.rgb = RGBColor(0x9A, 0xA3, 0xAA)
            arr.line.fill.background()


def load_cm(path: Path) -> np.ndarray:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",") if p.strip() != ""]
            if not parts:
                continue
            if not parts[0].lstrip("-").isdigit():
                continue
            rows.append([int(x) for x in parts])
    return np.array(rows, dtype=int)


def save_cm(path: Path, out: Path, title: str) -> None:
    cm = load_cm(path)
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))
    ax.set_xticklabels(CLASS_NAMES, rotation=50, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title, fontsize=11)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)


def make_figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    runs = ROOT / "results" / "runs"
    save_cm(runs / "bench_simple_cnn_seed0" / "cm.csv", FIG / "cm_simple_cnn.png", "simple_cnn  last-12")
    save_cm(runs / "bench_radmamba_seed0" / "cm.csv", FIG / "cm_radmamba.png", "RadMamba  last-12")

    held = ["0010", "0014", "0029", "0030"]
    acc = [0.274, 0.506, 0.323, 0.498]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(held, acc, color="#3D5A6C", width=0.55)
    ax.axhline(0.621, color="#B35A3C", linestyle="--", linewidth=1.4, label="last-12 simple_cnn 0.621")
    ax.axhline(0.400, color="#3D5A6C", linestyle=":", linewidth=1.2, label="LOSO mean 0.400")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("held-out subject")
    ax.legend(fontsize=8, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "loso_bars.png", dpi=150)
    plt.close(fig)

    pairs = ["24→77", "10→77", "77→24"]
    xacc = [0.098, 0.091, 0.250]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    ax.bar(pairs, xacc, color="#3D5A6C", width=0.5)
    ax.axhline(1 / 11, color="#B35A3C", linestyle="--", linewidth=1.4, label="chance 1/11 ≈ 0.091")
    ax.set_ylim(0, 0.7)
    ax.set_ylabel("Accuracy")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "cross_radar_bars.png", dpi=150)
    plt.close(fig)

    bands = ["10 GHz", "24 GHz", "77 GHz"]
    width = [94.6, 24.3, 183.6]
    wstd = [22.3, 3.4, 33.3]
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.bar(bands, width, yerr=wstd, color="#3D5A6C", width=0.5, capsize=4)
    ax.set_ylabel("occupied Doppler bins")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "doppler_width.png", dpi=150)
    plt.close(fig)


def build() -> Path:
    make_figures()
    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    # 1 cover
    s = blank_slide(prs)
    bg = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), Inches(13.333), Inches(7.5))
    bg.fill.solid()
    bg.fill.fore_color.rgb = NAVY
    bg.line.fill.background()
    stripe = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.RECTANGLE, Inches(0), Inches(0), Inches(0.16), Inches(7.5))
    stripe.fill.solid()
    stripe.fill.fore_color.rgb = RGBColor(0xC4, 0xA3, 0x5A)
    stripe.line.fill.background()
    t = s.shapes.add_textbox(Inches(0.7), Inches(2.1), Inches(12), Inches(1.1))
    add_text(t, "RadarBridge", size=40, bold=True, color=WHITE)
    sub = s.shapes.add_textbox(Inches(0.7), Inches(3.25), Inches(12), Inches(0.9))
    add_text(sub, "雷达微多普勒人体行为识别：复现与泛化核验", size=22, color=WHITE)
    sub2 = s.shapes.add_textbox(Inches(0.7), Inches(4.15), Inches(12), Inches(0.6))
    add_text(sub2, "CI4R-MULTI3  +  AIRHAR / RadMamba", size=16, color=RGBColor(0xC8, 0xD0, 0xD4))
    foot = s.shapes.add_textbox(Inches(0.7), Inches(6.4), Inches(12), Inches(0.5))
    add_text(foot, "阶段汇报  ·  已完成：复现与协议核验  ·  尚未提出新模型", size=14, color=RGBColor(0xB0, 0xB8, 0xBC))
    notes(s, "开场先说：这不是刷榜。后面数字都带协议名。有人问提出了什么方法，答：方法还没做，先把泄漏和跨雷达失效说清楚。")

    # 2 background
    s = blank_slide(prs)
    title_bar(s, "雷达回波到微多普勒图", "已完成")
    point(s, "FMCW / UWB 得到距离–慢时间；STFT 得到时间–多普勒图。训练吃的是这张图，不是原始 IQ。")
    flow_boxes(s, ["发射", "人体散射", "回波 IQ", "Range FFT", "STFT", "224×224 谱图"], top=1.6, box_w=1.9, gap=0.22)
    box = s.shapes.add_textbox(Inches(0.28), Inches(2.55), Inches(12.7), Inches(0.7))
    add_text(box, "躯干是较稳的主脊，四肢是两侧周期纹。快时间 = 一个 chirp 内的距离门；慢时间 = chirp 序号。f_D = 2v/λ 见第 12 页。", size=14, color=MUTED)
    cls = s.shapes.add_textbox(Inches(0.28), Inches(3.15), Inches(12.7), Inches(0.32))
    add_text(cls, "11 类：Towards  Away  Pick  Bend  Sit  Kneel  Crawl  Limp  Toes  SStep  Scissor", size=13, bold=True)
    gal77 = ROOT / "results" / "figures" / "micro_doppler_gallery_77GHz.png"
    gal24 = ROOT / "results" / "figures" / "micro_doppler_gallery_24GHz.png"
    if gal77.exists():
        s.shapes.add_picture(str(gal77), Inches(0.28), Inches(3.5), Inches(6.3), Inches(3.35))
        if gal24.exists():
            s.shapes.add_picture(str(gal24), Inches(6.75), Inches(3.5), Inches(6.3), Inches(3.35))
        cap = s.shapes.add_textbox(Inches(0.28), Inches(6.88), Inches(12.7), Inches(0.28))
        add_text(cap, "左 77 GHz、右 24 GHz 各类一张。IQ / Range 不放：raw_data.zip 未下。10 GHz 拼图未生成。", size=11, color=MUTED)
    else:
        todo_box(
            s,
            "TODO  谱图拼图未生成。IQ / Range 图不放：raw_data.zip 未下，不用仿真顶替。",
            Inches(0.28),
            Inches(3.85),
            Inches(12.7),
            Inches(1.55),
        )
    notes(s, "对照 learning_map。有人问为什么不是端到端 IQ：官方入口就是单通道 224×224。")

    # 3 meaning
    s = blank_slide(prs)
    title_bar(s, "为什么还要做雷达行为识别", "已完成")
    point(s, "室内监护、跌倒和异常步态需要无接触观测。雷达不成像。要能用，必须换人、换雷达后仍能认动作。")
    table_at(
        s,
        [
            ["", "同房间摄像头", "可穿戴", "雷达"],
            ["是否成像", "是", "否", "否"],
            ["是否接触人体", "否", "是", "否"],
            ["换人后是否还成立", "常要重训或重标定", "常要重戴、重标定", "待核验"],
            ["换设备后是否还成立", "镜头参数会变", "传感器型号会变", "待核验"],
        ],
        Inches(0.28),
        Inches(1.6),
        Inches(9.4),
        Inches(3.6),
        col_w=[Inches(2.3), Inches(2.4), Inches(2.4), Inches(2.3)],
    )
    side = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(9.9), Inches(1.6), Inches(3.1), Inches(3.6))
    side.fill.solid()
    side.fill.fore_color.rgb = ALT
    side.line.color.rgb = LINE
    add_lines(side, ["当前结论", "跨人、跨频尚未解决", "", "本页不放准确率。", "监护和步态是场景，", "不是已经做成的系统。"], size=14)
    notes(s, "意义落到换人和换雷达会不会掉，不要讲产品或推广。")

    # 4 problem
    s = blank_slide(prs)
    title_bar(s, "同分布准确率高，不能说明学会了动作", "已完成")
    point(s, "四种划分分开看。本页只立问题，数字放第 8–11 页。")
    table_at(
        s,
        [
            ["协议", "训练 / 测试关系", "本仓库状态"],
            ["official last-12", "同雷达、同分布末段切分", "已完成"],
            ["leave-one-subject-out", "同雷达、测试人未进训练", "已完成（仅 simple_cnn，77 GHz）"],
            ["cross-radar", "训练一个载频，测试另一个", "已完成（仅 simple_cnn）"],
            ["unseen radar × unseen subject", "测试时人和雷达都没见过", "计划，无数字"],
        ],
        Inches(0.28),
        Inches(1.6),
        Inches(12.7),
        Inches(3.8),
        col_w=[Inches(3.6), Inches(5.2), Inches(3.9)],
    )
    hint = s.shapes.add_textbox(Inches(0.28), Inches(5.6), Inches(12.7), Inches(1.2))
    add_text(hint, "0.83、0.89 是 last-12，不是跨人、跨雷达。第四行没有结果，不画空柱。", size=14, color=MUTED)
    notes(s, "有人把 0.83、0.89 当成已经能认动作，在这一页拦住：那是 last-12。")

    # 5 data
    s = blank_slide(prs)
    title_bar(s, "CI4R-MULTI3：10 / 24 / 77 GHz", "已完成")
    point(s, "实验室同步采集。本轮输入是官方谱图 PNG，不是从 .bin 现算的 4096 点 STFT。原始 IQ 约 93 GB，未整包下载。")
    table_at(
        s,
        [
            ["载频", "传感器", "公开 PNG 张数"],
            ["10 GHz", "XeThru UWB", "653"],
            ["24 GHz", "Ancortek FMCW", "653"],
            ["77 GHz", "TI IWR1443", "653"],
        ],
        Inches(0.28),
        Inches(1.6),
        Inches(8.4),
        Inches(2.8),
        col_w=[Inches(2.2), Inches(3.4), Inches(2.8)],
    )
    side = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(8.9), Inches(1.6), Inches(4.1), Inches(2.8))
    side.fill.solid()
    side.fill.fore_color.rgb = ALT
    side.line.color.rgb = LINE
    add_lines(side, ["论文名义", "6 人 × 11 类 × 10 次", "", "文件名末 4 位是 subject。"], size=14)
    fn = s.shapes.add_textbox(Inches(0.28), Inches(4.6), Inches(12.7), Inches(2.2))
    add_lines(
        fn,
        [
            "脚注：24 GHz last-12 人员重叠 0030, 0118, 0305；10 GHz last-12 重叠 0030, 0111, 0118。",
            "AIRHAR 的 bin2np() 只处理 77 GHz .bin。",
            "来源：results/analysis/domain_shift_summary.json；ci4r_24GHz_last12.json；ci4r_10GHz_last12.json",
        ],
        size=13,
        color=MUTED,
    )
    notes(s, "跨雷达若仍用 last-12 切测试集，24/10 GHz 也有人员重叠，解释时要说清。")

    # 6 engineering
    s = blank_slide(prs)
    title_bar(s, "对照仓库与最小改动", "已完成")
    point(s, "对照提交 d49e2c7。只改三项，能在本机跑通。未改 RadMamba 结构。")
    table_at(
        s,
        [
            ["文件", "改动"],
            ["data_prepare_CI4R.py", "恢复必填 --path，去掉写死的 /home/yizhuo/..."],
            ["p_scan.py / SSM.py", "zeta 缺失时回退本地 pscan"],
            ["project.py", "去掉 ReduceLROnPlateau(verbose=...)"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(12.7),
        Inches(2.6),
        col_w=[Inches(3.4), Inches(9.3)],
    )
    env = s.shapes.add_textbox(Inches(0.28), Inches(4.35), Inches(12.7), Inches(2.4))
    add_lines(
        env,
        [
            "环境：conda airhar，Python 3.11.15，PyTorch 2.7.1+cu128。本机 RTX 5060（sm_120）。官方 2.5.1+cu126 没有该架构 kernel。",
            "官方没有验证集，按测试 Acc 存 best。复现写明 test-as-val，不要说成正规 val。",
            "来源：notes/CHANGELOG.md，notes/01_audit.md，notes/00_clone_and_env.md",
        ],
        size=14,
        color=INK,
    )
    notes(s, "论文混淆矩阵不在官方代码里，本仓库用 eval_metrics.py 补 Acc / Macro-F1 / 混淆矩阵。")

    # 7 pipeline
    s = blank_slide(prs)
    title_bar(s, "本仓库已跑通的闭环", "已完成")
    point(s, "官方谱图 PNG → 224×224 z-score → h5 → train_local.py → Acc / Macro-F1 / 混淆矩阵。")
    flow_boxes(
        s,
        ["公开桶 PNG", "download", "prepare h5", "train_local", "metrics / cm.csv"],
        top=1.6,
        box_w=2.2,
        gap=0.28,
        left=0.4,
    )
    box = s.shapes.add_textbox(Inches(0.28), Inches(2.55), Inches(12.7), Inches(3.8))
    add_lines(
        box,
        [
            "scripts/download_spectrograms.py  →  scripts/prepare_ci4r_from_png.py",
            "h5：third_party/AIRHAR/datasets/CI4R/ci4r_data.h5，或 results/data/ci4r_24GHz.h5 等",
            "训练：scripts/train_local.py  /  run_unified_bench.py",
            "输出：results/runs/<run>/{metrics.json, history.csv, cm.csv}",
            "",
            "论文写 528 / 132，本机 PNG 路径是 521 / 132，以 results/splits/ci4r_official_last12.json 为准。",
            "官方 spec.json 会覆盖命令行的 backbone 和 epoch。",
        ],
        size=15,
    )
    notes(s, "统一入口是 train_local.py，避免 scripts/models 挡住 AIRHAR 的 models.py。")

    # 8 same-dist
    s = blank_slide(prs)
    title_bar(s, "官方 last-12，77 GHz，同分布", "已完成 · seed 0 · test-as-val")
    point(s, "ResNet6 量级对上论文约 32.3%。RadMamba 0.833，未到论文 91.2%：输入是发布 PNG，train_local 为 30 epoch。不能说复现失败或超越。")
    table_at(
        s,
        [
            ["模型", "Accuracy", "Macro-F1", "参数量", "训练时间 (min)"],
            ["AIRHAR ResNet6", "0.296", "0.192", "24.8k", "0.13"],
            ["simple_cnn", "0.621", "0.595", "24.2k", "0.22"],
            ["RadMamba", "0.833", "0.822", "71.3k", "0.23"],
            ["ResNet18", "0.894", "0.891", "11.2M", "0.37"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(7.5),
        Inches(2.9),
        col_w=[Inches(2.1), Inches(1.3), Inches(1.4), Inches(1.3), Inches(1.4)],
    )
    if (FIG / "cm_simple_cnn.png").exists():
        s.shapes.add_picture(str(FIG / "cm_simple_cnn.png"), Inches(8.0), Inches(1.5), Inches(2.55), Inches(2.25))
        s.shapes.add_picture(str(FIG / "cm_radmamba.png"), Inches(10.6), Inches(1.5), Inches(2.55), Inches(2.25))
    fn = s.shapes.add_textbox(Inches(0.28), Inches(4.6), Inches(12.7), Inches(2.2))
    add_lines(
        fn,
        [
            "脚注：官方 main.py RadMamba 100 epoch，best Acc 82.58%（epoch 17）。该数字在 third_party/AIRHAR/log/，不在 results/tables/。",
            "ResNet18 参数 11.2M，RadMamba 71.3k，不要并排比谁更强。",
            "来源：results/tables/unified_bench.csv，results/tables/all_runs.csv",
        ],
        size=13,
        color=MUTED,
    )
    notes(s, "有人拿 0.894 对论文 91.2% 时，先说输入和划分，再说参数量。")

    # 9 leakage
    s = blank_slide(prs)
    title_bar(s, "官方 last-12 的测试集几乎全是同一个人", "已完成")
    point(s, "77 GHz last-12：521 train / 132 test。测试集人员只有 0030，且 0030 同时在训练集。同分布 Acc 是乐观估计。")
    table_at(
        s,
        [
            ["字段", "内容"],
            ["n_train / n_test", "521 / 132"],
            ["subjects_train", "0000, 0010, 0014, 0029, 0030"],
            ["subjects_test", "0030"],
            ["subjects_in_both", "0030"],
        ],
        Inches(0.28),
        Inches(1.6),
        Inches(8.6),
        Inches(3.2),
        col_w=[Inches(2.8), Inches(5.8)],
    )
    side = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(9.1), Inches(1.6), Inches(3.9), Inches(3.2))
    side.fill.solid()
    side.fill.fore_color.rgb = ALT
    side.line.color.rgb = LINE
    add_lines(side, ["测试文件名示例", "04050030_….png", "", "末四位即 0030。", "", "h5 里没有人员字段，", "必须对文件名核验。"], size=13)
    fn = s.shapes.add_textbox(Inches(0.28), Inches(5.05), Inches(12.7), Inches(1.8))
    add_lines(
        fn,
        [
            "官方按每类末 12 条切。论文写 528 / 132，本机 PNG 少 7 条训练样本，以 json 为准。",
            "这是切分规则的后果，不是说官方造假。",
            "来源：results/splits/ci4r_official_last12.json",
        ],
        size=13,
        color=MUTED,
    )
    notes(s, "不要把泄漏说成官方造假：这是切分规则的后果。")

    # 10 LOSO
    s = blank_slide(prs)
    title_bar(s, "77 GHz，leave-one-subject-out，simple_cnn", "已完成 · 仅 simple_cnn · seed 0")
    point(s, "同一 simple_cnn：last-12 为 0.621，LOSO 均值为 0.400 ± 0.119。认动作的同时也在认人。")
    table_at(
        s,
        [
            ["held-out", "n_train", "n_test", "Accuracy", "Macro-F1"],
            ["0010", "536", "117", "0.274", "0.212"],
            ["0014", "564", "89", "0.506", "0.437"],
            ["0029", "430", "223", "0.323", "0.284"],
            ["0030", "438", "215", "0.498", "0.469"],
            ["mean ± std", "—", "—", "0.400 ± 0.119", "0.351 ± 0.123"],
        ],
        Inches(0.28),
        Inches(1.5),
        Inches(6.9),
        Inches(3.6),
        col_w=[Inches(1.5), Inches(1.2), Inches(1.2), Inches(1.6), Inches(1.4)],
    )
    s.shapes.add_picture(str(FIG / "loso_bars.png"), Inches(7.4), Inches(1.5), Inches(5.6), Inches(3.0))
    fn = s.shapes.add_textbox(Inches(0.28), Inches(5.25), Inches(12.7), Inches(1.7))
    add_lines(
        fn,
        [
            "虚线 0.621 是 last-12，不要和 LOSO 合成总准确率。折间 0.27–0.51，只有四人，先不当成稳定泛化误差。",
            "未做：RadMamba / ResNet18 的 LOSO。",
            "来源：results/tables/cross_subject_77GHz_simple_cnn.json；results/splits/loso_77GHz/folds.json",
        ],
        size=13,
        color=MUTED,
    )
    notes(s, "有人问 RadMamba 跨人会不会好一些：还没跑，不能猜。")

    # 11 cross-radar
    s = blank_slide(prs)
    title_bar(s, "换载频之后，simple_cnn 几乎认不出来", "已完成 · 仅 simple_cnn · seed 0")
    point(s, "11 类随机约 0.091。24→77 为 0.098，10→77 为 0.091。77→24 为 0.250，仍远低于同分布 0.621。")
    table_at(
        s,
        [
            ["训练", "测试", "Accuracy", "Macro-F1"],
            ["24 GHz", "77 GHz", "0.098", "0.024"],
            ["10 GHz", "77 GHz", "0.091", "0.015"],
            ["77 GHz", "24 GHz", "0.250", "0.178"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(6.6),
        Inches(2.6),
        col_w=[Inches(1.6), Inches(1.6), Inches(1.7), Inches(1.7)],
    )
    s.shapes.add_picture(str(FIG / "cross_radar_bars.png"), Inches(7.2), Inches(1.5), Inches(5.8), Inches(3.1))
    fn = s.shapes.add_textbox(Inches(0.28), Inches(4.4), Inches(12.7), Inches(2.4))
    add_lines(
        fn,
        [
            "跨雷达比跨人掉得更狠。77→24 略高于随机，不要说成 77 GHz 更能泛化。",
            "未做：其他 backbone；速度轴对齐后的对照。",
            "来源：results/tables/cross_radar.csv",
        ],
        size=14,
        color=MUTED,
    )
    notes(s, "设备特异性比人员特异性更强，这是后面做速度轴对齐的理由，不是已经做成的方法。")

    # 12 physics
    s = blank_slide(prs)
    title_bar(s, "同一速度，不同载频，谱图尺度对不上", "已完成（统计）· 图待补")
    point(s, "f_D = 2v/λ。77 GHz 波长约 3.9 mm，同样走路速度的频移大约是 10 GHz 的 7.7 倍。这是现象，不是方法。")
    table_at(
        s,
        [
            ["载频", "谱图张数", "平均强度", "多普勒占用宽度（像素）"],
            ["10 GHz", "653", "25.4 ± 3.6", "94.6 ± 22.3"],
            ["24 GHz", "653", "60.6 ± 18.1", "24.3 ± 3.4"],
            ["77 GHz", "653", "43.0 ± 7.3", "183.6 ± 33.3"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(7.3),
        Inches(2.5),
        col_w=[Inches(1.5), Inches(1.5), Inches(1.8), Inches(2.5)],
    )
    s.shapes.add_picture(str(FIG / "doppler_width.png"), Inches(7.8), Inches(1.5), Inches(5.2), Inches(2.85))
    fn = s.shapes.add_textbox(Inches(0.28), Inches(4.25), Inches(12.7), Inches(2.6))
    add_lines(
        fn,
        [
            "宽度是灰度图上超过「列均值 + 0.5 标准差」的行数，不是物理 Hz。方向与公式一致，不能当成已标定的速度轴。",
            "TODO：同动作三频并排谱图（原始 PNG 当前不在 data/）。domain_shift_freq.png 需重跑 analyze_domain_shift.py。",
            "来源：results/analysis/domain_shift_summary.json",
        ],
        size=13,
        color=MUTED,
    )
    notes(s, "尺度对不上时，CNN 容易记雷达而不是动作。")

    # 13 planned method
    s = blank_slide(prs)
    title_bar(s, "计划：把频率轴缩到速度轴再训练", "计划")
    point(s, "按载频把多普勒轴换成 v = f_D · λ / 2，重采样到公共速度栅格，再送给现有网络。现未实现，本页无准确率。")
    flow_boxes(s, ["同一走路速度", "10 / 24 / 77 GHz", "不同多普勒宽度", "v = f·λ/2", "公共速度栅格"], top=1.65, box_w=2.2, gap=0.26, left=0.35)
    warn = s.shapes.add_shape(MSO_AUTO_SHAPE_TYPE.ROUNDED_RECTANGLE, Inches(0.28), Inches(2.7), Inches(12.7), Inches(1.5))
    warn.fill.solid()
    warn.fill.fore_color.rgb = TODO_BG
    warn.line.color.rgb = LINE
    add_text(warn, "未跑通，无准确率。要做有 / 无对齐的消融。不改 RadMamba 内部，只改输入坐标。", size=18, bold=True, color=MUTED, align=PP_ALIGN.CENTER)
    box = s.shapes.add_textbox(Inches(0.28), Inches(4.45), Inches(12.7), Inches(2.3))
    add_lines(
        box,
        [
            "对齐后若跨频仍接近随机，再考虑一致性学习。先不上域对抗。",
            "来源：notes/m8_m9_direction.md（方向说明，无实验结果）",
        ],
        size=15,
    )
    notes(s, "这是物理归一化，不是已经做成的方法。")

    # 14 two weeks
    s = blank_slide(prs)
    title_bar(s, "两周内准备做的对照", "计划")
    point(s, "先补图和统一表，再做速度轴原型和 simple_cnn 跨雷达对照。组会只承诺能做完的对照，不承诺投稿数字。")
    table_at(
        s,
        [
            ["周次", "任务", "状态"],
            ["第 1 周", "三频谱图拼图、域偏移箱线图、混淆矩阵、统一结果表", "计划"],
            ["第 1 周", "速度轴对齐原型；simple_cnn 跨雷达（对齐后）", "计划"],
            ["第 2 周", "ResNet18 / RadMamba 跨雷达；seed 0/1/2", "计划"],
            ["第 2 周", "跨人×跨雷达（能做再做）；feature probe", "计划"],
            ["第 2 周", "更新论文 Results；按本大纲改图", "计划"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(12.7),
        Inches(4.0),
        col_w=[Inches(1.6), Inches(9.0), Inches(2.1)],
    )
    fn = s.shapes.add_textbox(Inches(0.28), Inches(5.75), Inches(12.7), Inches(1.1))
    add_text(fn, "细则见 notes/two_week_plan.md。没做完就留 TODO，不要空填数字。", size=14, color=PLAN)
    notes(s, "详见 two_week_plan.md。")

    # 15 paper
    s = blank_slide(prs)
    title_bar(s, "现在能写什么，还不能写什么", "计划")
    point(s, "工作标题 RadarBridge: Physics-Guided Cross-Radar Generalization… 表示后续方向。方法贡献要等对齐和消融有数。")
    table_at(
        s,
        [
            ["现在能写进稿的", "还不能写"],
            ["可复现闭环与三项补丁", "「本文提出新网络」"],
            ["last-12 人员泄漏", "速度轴对齐已经有效"],
            ["LOSO 0.400 ± 0.119（simple_cnn）", "多 seed 均值"],
            ["跨雷达接近随机（simple_cnn）", "跨人×跨雷达数字"],
            ["占用宽度与 f_D=2v/λ 一致", "保证 SCI 录用"],
        ],
        Inches(0.28),
        Inches(1.55),
        Inches(12.7),
        Inches(3.8),
        col_w=[Inches(6.35), Inches(6.35)],
    )
    fn = s.shapes.add_textbox(Inches(0.28), Inches(5.55), Inches(12.7), Inches(1.3))
    add_lines(
        fn,
        [
            "投稿阶梯：先组会 / 大创中期；有消融再考虑雷达或传感期刊。",
            "若对齐无效，稿子改写成负结果加划分审计，仍然成立。",
        ],
        size=14,
        color=MUTED,
    )
    notes(s, "不要在这一页许诺影响因子。")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(OUT))
    return OUT


if __name__ == "__main__":
    path = build()
    print(path)
