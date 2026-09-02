# -*- coding: utf-8 -*-
"""Build RadarMind formal report deck. Copy is verbatim from the brief."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml import parse_xml
from pptx.util import Emu, Inches, Pt

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "notes"
ASSET = ROOT / "notes" / "RadarMind" / "assets"
# Unicode escapes keep the Chinese filename stable on Windows consoles.
_NAME = "RadarMind_\u6c47\u62a5PPT"
PPTX = OUT_DIR / f"{_NAME}.pptx"
PDF = OUT_DIR / f"{_NAME}.pdf"
PPTX_ROOT = ROOT / f"{_NAME}.pptx"
PDF_ROOT = ROOT / f"{_NAME}.pdf"

BG = RGBColor(0x0B, 0x14, 0x24)
BG2 = RGBColor(0x0E, 0x1A, 0x2E)
CARD = RGBColor(0x15, 0x22, 0x38)
CARD2 = RGBColor(0x1A, 0x2C, 0x48)
BLUE = RGBColor(0x2B, 0x8C, 0xFF)
CYAN = RGBColor(0x3E, 0xC6, 0xE0)
GOLD = RGBColor(0xE8, 0xB8, 0x4A)
RED = RGBColor(0xE0, 0x5A, 0x5A)
GREEN = RGBColor(0x3D, 0xCF, 0x8E)
WHITE = RGBColor(0xE8, 0xEE, 0xF6)
MUTED = RGBColor(0x9A, 0xAD, 0xC2)
LINE = RGBColor(0x2A, 0x3F, 0x5C)
INK = RGBColor(0x0B, 0x14, 0x24)
FONT = "微软雅黑"

W, H = Inches(13.333), Inches(7.5)


def rgb(c: RGBColor) -> str:
    return f"{c[0]:02X}{c[1]:02X}{c[1] and c[1]:02X}"


def _ea(run, name=FONT):
    rpr = run._r.get_or_add_rPr()
    rpr.append(
        parse_xml(
            f'<a:ea xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{name}"/>'
        )
    )


def set_run(run, text, size=16, bold=False, color=WHITE, name=FONT):
    run.text = text
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    _ea(run, name)


def fill(shape, color):
    shape.fill.solid()
    shape.fill.fore_color.rgb = color
    shape.line.fill.background()


def stroke(shape, color, pt=1.0):
    shape.line.color.rgb = color
    shape.line.width = Pt(pt)


def box(slide, l, t, w, h, color=CARD, line=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, l, t, w, h)
    fill(sh, color)
    if line:
        stroke(sh, line, 1.0)
    else:
        sh.line.fill.background()
    sh.adjustments[0] = 0.06
    return sh


def rect(slide, l, t, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, l, t, w, h)
    fill(sh, color)
    return sh


def tb(slide, l, t, w, h, text, size=16, bold=False, color=WHITE, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP):
    sh = slide.shapes.add_textbox(l, t, w, h)
    tf = sh.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        sh.text_frame._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    set_run(run, text, size=size, bold=bold, color=color)
    return sh


def lines(slide, l, t, w, h, items, size=15, color=WHITE, gap=True):
    sh = slide.shapes.add_textbox(l, t, w, h)
    tf = sh.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.space_after = Pt(8 if gap else 4)
        run = p.add_run()
        set_run(run, item, size=size, color=color)
    return sh


def new_prs():
    prs = Presentation()
    prs.slide_width = W
    prs.slide_height = H
    return prs


def blank(prs):
    layout = prs.slide_layouts[6]
    slide = prs.slides.add_slide(layout)
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, H)
    fill(bg, BG)
    return slide


def chrome(slide, title, page, total=30):
    bar = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, W, Inches(0.62))
    fill(bar, BG2)
    accent = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, Inches(0.08), Inches(0.62))
    fill(accent, CYAN)
    tb(slide, Inches(0.28), Inches(0.10), Inches(3.4), Inches(0.22), "RadarMind　电子信息学院科研汇报", 11, False, MUTED)
    tb(slide, Inches(0.28), Inches(0.28), Inches(12.4), Inches(0.32), title, 22, True, WHITE)
    foot = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, Inches(7.18), W, Inches(0.32))
    fill(foot, BG2)
    tb(
        slide,
        Inches(0.28),
        Inches(7.20),
        Inches(11.2),
        Inches(0.26),
        "面向跨雷达、跨人员与跨场景的人体运动雷达智能理解系统",
        10,
        False,
        MUTED,
    )
    tb(slide, Inches(11.9), Inches(7.20), Inches(1.2), Inches(0.26), f"{page} / {total}", 10, False, MUTED, PP_ALIGN.RIGHT)


def render_eq(path: Path, tex: str, fontsize=28, w=7.0, h=1.15):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(w, h), facecolor="#0B1424")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0B1424")
    ax.axis("off")
    ax.text(0.5, 0.5, f"${tex}$", color="#E8EEF6", fontsize=fontsize, ha="center", va="center")
    fig.savefig(path, dpi=180, facecolor=fig.get_facecolor())
    plt.close(fig)


def flow_row(slide, labels, y, x0=0.35, widths=None, colors=None, h=0.58):
    n = len(labels)
    gap = 0.14
    if widths is None:
        usable = 12.6
        aw = (usable - gap * (n - 1)) / n
        widths = [aw] * n
    colors = colors or [CARD2] * n
    x = x0
    boxes = []
    for i, lab in enumerate(labels):
        sh = box(slide, Inches(x), Inches(y), Inches(widths[i]), Inches(h), colors[i], LINE)
        tb(
            slide,
            Inches(x + 0.06),
            Inches(y + 0.10),
            Inches(widths[i] - 0.10),
            Inches(h - 0.16),
            lab,
            12,
            True,
            WHITE,
            PP_ALIGN.CENTER,
            MSO_ANCHOR.MIDDLE,
        )
        boxes.append((x, widths[i]))
        if i < n - 1:
            arr = slide.shapes.add_shape(
                MSO_SHAPE.RIGHT_ARROW,
                Inches(x + widths[i] + 0.02),
                Inches(y + h / 2 - 0.08),
                Inches(0.12),
                Inches(0.16),
            )
            fill(arr, CYAN)
        x += widths[i] + gap
    return boxes


def build():
    ASSET.mkdir(parents=True, exist_ok=True)
    render_eq(ASSET / "eq_fd.png", r"f_D=\dfrac{2v_r f_c}{c}", 36, 6.4, 1.25)
    render_eq(ASSET / "eq_res.png", r"ResNet18=89.4\%", 30, 6.2, 0.9)
    render_eq(ASSET / "eq_loso.png", r"simple\ CNN=0.400\pm0.119", 26, 7.2, 0.85)
    render_eq(ASSET / "eq_xr.png", r"10\rightarrow77\ \mathrm{GHz}=0.091\qquad 24\rightarrow77\ \mathrm{GHz}=0.098", 22, 10.5, 0.85)
    render_eq(ASSET / "eq_chance.png", r"1/11\approx0.091", 26, 4.4, 0.75)
    render_eq(ASSET / "eq_bridge.png", r"Radar_A \leftrightarrow Radar_B", 28, 6.4, 0.85)
    render_eq(ASSET / "eq_goal.png", r"\mathrm{Learn\ Motion,\ Not\ Sensor,\ Subject,\ or\ Environment}", 18, 11.2, 0.7)

    prs = new_prs()
    T = 30

    # 1 cover
    s = blank(prs)
    rect(s, 0, 0, Inches(0.14), H, CYAN)
    for r, col in ((2.1, RGBColor(0x12, 0x3A, 0x5C)), (2.7, RGBColor(0x14, 0x4A, 0x6E)), (3.3, RGBColor(0x16, 0x5A, 0x80))):
        ring = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(10.1), Inches(r), Inches(2.6), Inches(2.6))
        ring.fill.background()
        stroke(ring, col, 1.5)
    sweep = s.shapes.add_shape(MSO_SHAPE.PIE, Inches(10.35), Inches(2.35), Inches(2.1), Inches(2.1))
    fill(sweep, RGBColor(0x14, 0x3A, 0x58))
    stroke(sweep, CYAN, 1.25)
    tb(s, Inches(0.55), Inches(1.55), Inches(9.4), Inches(0.35), "电子信息学院　科研汇报", 14, False, CYAN)
    tb(s, Inches(0.55), Inches(2.05), Inches(9.5), Inches(0.85), "RadarMind", 48, True, WHITE)
    tb(
        s,
        Inches(0.55),
        Inches(2.95),
        Inches(9.6),
        Inches(1.15),
        "面向跨雷达、跨人员与跨场景的人体运动雷达智能理解系统",
        22,
        True,
        WHITE,
    )
    tb(s, Inches(0.55), Inches(4.35), Inches(9.6), Inches(0.45), "副标题：", 14, False, MUTED)
    tb(
        s,
        Inches(0.55),
        Inches(4.75),
        Inches(9.8),
        Inches(0.50),
        "From Radar Signals to Open-World Human Motion Understanding",
        18,
        False,
        GOLD,
    )
    tb(s, Inches(0.55), Inches(6.85), Inches(8.0), Inches(0.30), "1 / 30", 11, False, MUTED)

    # 2 background
    s = blank(prs)
    chrome(s, "项目背景", 2, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.85),
        Inches(8.3),
        Inches(5.9),
        [
            "雷达可以通过发射电磁波并接收人体反射回波，感知人体运动带来的距离、速度、相位和多普勒变化。",
            "人体不同部位在运动中具有不同速度，因此会形成具有动作特征的 micro-Doppler 信号或毫米波点云序列。",
            "本项目希望利用 AI 从雷达信号中理解人体运动，使雷达不只是“检测目标”，而是进一步“理解行为”。",
        ],
        16,
    )
    box(s, Inches(9.0), Inches(1.15), Inches(3.85), Inches(5.4), CARD, LINE)
    tb(s, Inches(9.18), Inches(1.35), Inches(3.5), Inches(0.35), "发射 → 人体反射 → 回波", 13, True, CYAN, PP_ALIGN.CENTER)
    for i, lab in enumerate(["距离", "速度", "相位", "多普勒"]):
        y = 2.05 + i * 0.95
        box(s, Inches(9.35), Inches(y), Inches(3.15), Inches(0.72), CARD2, LINE)
        tb(s, Inches(9.35), Inches(y + 0.18), Inches(3.15), Inches(0.40), lab, 16, True, WHITE, PP_ALIGN.CENTER)

    # 3 why
    s = blank(prs)
    chrome(s, "为什么研究雷达人体运动理解", 3, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.35), "相比摄像头，雷达具有以下优势：", 16, False, WHITE)
    items = [
        "不直接采集人脸和外观，隐私友好；",
        "不依赖光照，适用于夜间和弱光环境；",
        "对速度和微运动敏感；",
        "可用于非接触式监测；",
        "适合居家看护、健康监测、智能安防和车载感知。",
    ]
    for i, it in enumerate(items):
        box(s, Inches(0.40), Inches(1.30 + i * 0.82), Inches(12.5), Inches(0.70), CARD, LINE)
        mark = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(0.62), Inches(1.50 + i * 0.82), Inches(0.28), Inches(0.28))
        fill(mark, CYAN)
        tb(s, Inches(1.10), Inches(1.45 + i * 0.82), Inches(11.5), Inches(0.42), it, 16, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    tb(
        s,
        Inches(0.40),
        Inches(5.50),
        Inches(12.5),
        Inches(1.35),
        "雷达不是替代视觉，而是提供一种重要的非视觉智能感知方式。",
        18,
        True,
        GOLD,
    )

    # 4 signal physics
    s = blank(prs)
    chrome(s, "人的动作改变了什么雷达信号", 4, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.4), Inches(0.32), "人的动作会改变雷达回波的：", 16)
    pairs = [
        ("时延", "对应目标距离；", BLUE),
        ("相位", "对应微小位移；", CYAN),
        ("多普勒频移", "对应运动速度；", GOLD),
        ("幅度", "对应反射强度和姿态变化。", GREEN),
    ]
    for i, (a, b, c) in enumerate(pairs):
        x = 0.40 + (i % 2) * 6.45
        y = 1.28 + (i // 2) * 1.35
        box(s, Inches(x), Inches(y), Inches(6.20), Inches(1.18), CARD, LINE)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(0.10), Inches(1.18))
        fill(bar, c)
        tb(s, Inches(x + 0.28), Inches(y + 0.18), Inches(5.7), Inches(0.36), a, 18, True, WHITE)
        tb(s, Inches(x + 0.28), Inches(y + 0.58), Inches(5.7), Inches(0.40), b, 15, False, MUTED)
    tb(s, Inches(0.40), Inches(4.15), Inches(12.4), Inches(0.32), "其中，人体动作识别最关键的是：", 16)
    box(s, Inches(2.3), Inches(4.58), Inches(8.7), Inches(2.15), CARD, CYAN)
    s.shapes.add_picture(str(ASSET / "eq_fd.png"), Inches(3.3), Inches(4.72), Inches(6.6), Inches(1.15))
    tb(
        s,
        Inches(0.50),
        Inches(6.00),
        Inches(12.3),
        Inches(0.85),
        "即人体各部位运动速度引起的 Doppler / micro-Doppler 特征。",
        15,
        False,
        MUTED,
        PP_ALIGN.CENTER,
    )

    # 5 traditional pipeline
    s = blank(prs)
    chrome(s, "传统 HAR 的基本流程", 5, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.32), "传统雷达人体行为识别通常是：", 16)
    flow_row(
        s,
        ["Radar Signal", "Signal Processing", "Micro-Doppler / Point Cloud", "Deep Learning", "Activity Class"],
        1.35,
        colors=[CARD2, CARD2, RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1E, 0x4A, 0x3A)],
    )
    tb(s, Inches(0.40), Inches(2.25), Inches(12.4), Inches(0.32), "典型模型包括：", 16)
    mods = ["CNN；", "ResNet；", "Transformer；", "Mamba；", "PointNet / PointMLP。"]
    for i, m in enumerate(mods):
        box(s, Inches(0.40 + i * 2.55), Inches(2.70), Inches(2.40), Inches(0.70), CARD, LINE)
        tb(s, Inches(0.40 + i * 2.55), Inches(2.86), Inches(2.40), Inches(0.42), m, 14, True, WHITE, PP_ALIGN.CENTER)

    # 6 table
    s = blank(prs)
    chrome(s, "传统 HAR 的基本流程", 6, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.32), "我们初步复现中：", 16)
    headers = ["Model", "Accuracy"]
    rows = [("simple CNN", "0.621"), ("RadMamba", "0.833"), ("ResNet18", "0.894")]
    box(s, Inches(3.3), Inches(1.50), Inches(6.7), Inches(4.6), CARD, LINE)
    head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.3), Inches(1.50), Inches(6.7), Inches(0.70))
    fill(head, RGBColor(0x1A, 0x3A, 0x58))
    tb(s, Inches(3.5), Inches(1.65), Inches(3.0), Inches(0.42), headers[0], 18, True, CYAN)
    tb(s, Inches(6.7), Inches(1.65), Inches(3.0), Inches(0.42), headers[1], 18, True, CYAN, PP_ALIGN.RIGHT)
    for i, (m, a) in enumerate(rows):
        y = 2.35 + i * 1.10
        if i % 2 == 0:
            stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(3.3), Inches(y), Inches(6.7), Inches(1.10))
            fill(stripe, RGBColor(0x12, 0x1E, 0x32))
        tb(s, Inches(3.5), Inches(y + 0.32), Inches(3.1), Inches(0.45), m, 20, False, WHITE)
        tb(s, Inches(6.7), Inches(y + 0.32), Inches(3.0), Inches(0.45), a, 22, True, GOLD, PP_ALIGN.RIGHT)

    # 7 limits
    s = blank(prs)
    chrome(s, "传统分类模型的局限", 7, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.32), "传统模型通常只能在预设动作类别中选择，例如：", 16)
    for i, lab in enumerate(["walking；", "sitting；", "bending；", "crawling。"]):
        box(s, Inches(0.40 + i * 3.2), Inches(1.30), Inches(3.00), Inches(0.72), CARD, LINE)
        tb(s, Inches(0.40 + i * 3.2), Inches(1.46), Inches(3.00), Inches(0.42), lab, 16, True, WHITE, PP_ALIGN.CENTER)
    lines(
        s,
        Inches(0.40),
        Inches(2.25),
        Inches(12.5),
        Inches(2.2),
        [
            "但真实世界中的人体行为是连续、开放和复杂的。",
            "自动驾驶、智能家居和健康监测场景中，不可能提前穷举所有行为模式。",
            "因此，未来系统不能只做 fixed-class classification，而应走向：",
        ],
        16,
    )
    box(s, Inches(1.6), Inches(5.00), Inches(10.1), Inches(1.55), CARD2, GOLD)
    tb(
        s,
        Inches(1.7),
        Inches(5.45),
        Inches(9.9),
        Inches(0.70),
        "Open-World Human Motion Understanding",
        24,
        True,
        GOLD,
        PP_ALIGN.CENTER,
    )

    # 8 three gaps
    s = blank(prs)
    chrome(s, "当前领域的核心痛点", 8, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.32), "同一个动作会因为以下因素发生变化：", 16)
    cols = [
        (GOLD, "Cross-Subject", "不同人的身高、体型、步态和动作习惯不同。"),
        (GREEN, "Cross-Environment", "房间、距离、角度、遮挡和多径环境不同。"),
        (RED, "Cross-Radar", "雷达频段、波形、带宽、采样方式、分辨率和硬件响应不同。"),
    ]
    for i, (c, t, b) in enumerate(cols):
        x = 0.40 + i * 4.25
        box(s, Inches(x), Inches(1.30), Inches(4.05), Inches(3.55), CARD, LINE)
        top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(1.30), Inches(4.05), Inches(0.12))
        fill(top, c)
        tb(s, Inches(x + 0.18), Inches(1.60), Inches(3.70), Inches(0.70), t, 20, True, c)
        tb(s, Inches(x + 0.18), Inches(2.45), Inches(3.70), Inches(2.10), b, 16, False, WHITE)
    tb(
        s,
        Inches(0.40),
        Inches(5.15),
        Inches(12.5),
        Inches(1.55),
        "因此，模型很容易学到“人、场景或雷达设备”的特征，而不是动作本身。",
        18,
        True,
        WHITE,
    )

    # 9 findings
    s = blank(prs)
    chrome(s, "我们已经发现的问题", 9, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.32), "在 CI4R-MULTI3 数据集中，模型在同分布测试中表现较好：", 16)
    box(s, Inches(0.40), Inches(1.22), Inches(12.5), Inches(1.15), CARD, GREEN)
    s.shapes.add_picture(str(ASSET / "eq_res.png"), Inches(3.5), Inches(1.32), Inches(6.2), Inches(0.90))
    tb(s, Inches(0.40), Inches(2.50), Inches(12.5), Inches(0.30), "但严格测试下性能显著下降：", 16)
    box(s, Inches(0.40), Inches(2.90), Inches(6.15), Inches(2.15), CARD, GOLD)
    tb(s, Inches(0.55), Inches(3.05), Inches(5.8), Inches(0.32), "跨人员 LOSO", 14, True, GOLD)
    s.shapes.add_picture(str(ASSET / "eq_loso.png"), Inches(0.55), Inches(3.40), Inches(5.85), Inches(0.85))
    box(s, Inches(6.75), Inches(2.90), Inches(6.15), Inches(2.15), CARD, RED)
    tb(s, Inches(6.90), Inches(3.05), Inches(5.8), Inches(0.32), "跨雷达", 14, True, RED)
    s.shapes.add_picture(str(ASSET / "eq_xr.png"), Inches(6.85), Inches(3.45), Inches(5.90), Inches(0.80))
    tb(s, Inches(0.40), Inches(5.20), Inches(6.2), Inches(0.30), "接近 11 分类随机猜测：", 16)
    s.shapes.add_picture(str(ASSET / "eq_chance.png"), Inches(6.7), Inches(5.10), Inches(4.2), Inches(0.55))
    tb(s, Inches(0.40), Inches(5.85), Inches(12.5), Inches(0.90), "说明同分布准确率明显高估了真实泛化能力。", 17, True, GOLD)

    # 10 goal
    s = blank(prs)
    chrome(s, "RadarMind 的核心目标", 10, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.90),
        Inches(12.5),
        Inches(2.0),
        [
            "本项目不只是提高某个数据集上的 accuracy，而是希望建立一种更通用的人体运动雷达智能理解框架。",
            "核心目标：",
        ],
        17,
    )
    box(s, Inches(0.70), Inches(3.15), Inches(11.9), Inches(2.15), CARD2, CYAN)
    tb(
        s,
        Inches(0.90),
        Inches(3.55),
        Inches(11.5),
        Inches(0.55),
        "Learn Motion, Not Sensor, Subject, or Environment",
        24,
        True,
        CYAN,
        PP_ALIGN.CENTER,
    )
    tb(
        s,
        Inches(0.90),
        Inches(4.25),
        Inches(11.5),
        Inches(0.75),
        "即让模型尽可能学习人体运动本身，而不是记住某个人、某个场景或某一台雷达。",
        16,
        False,
        WHITE,
        PP_ALIGN.CENTER,
    )

    # 11 bridge to mind
    s = blank(prs)
    chrome(s, "从 RadarBridge 到 RadarMind", 11, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.30), "原始思路 RadarBridge 主要关注：", 16)
    box(s, Inches(2.4), Inches(1.25), Inches(8.5), Inches(1.25), CARD, BLUE)
    s.shapes.add_picture(str(ASSET / "eq_bridge.png"), Inches(3.4), Inches(1.38), Inches(6.5), Inches(0.70))
    tb(s, Inches(0.40), Inches(2.65), Inches(12.4), Inches(0.30), "即跨雷达表征迁移。", 16)
    tb(s, Inches(0.40), Inches(3.10), Inches(12.4), Inches(0.30), "随着目标扩展，项目升级为 RadarMind：", 16)
    box(s, Inches(0.40), Inches(3.50), Inches(12.5), Inches(1.55), CARD, CYAN)
    tb(
        s,
        Inches(0.55),
        Inches(3.75),
        Inches(12.2),
        Inches(1.10),
        "Cross-Radar + Cross-Subject + Cross-Environment + Continuous Motion + LLM Understanding + Application",
        16,
        True,
        WHITE,
        PP_ALIGN.CENTER,
    )
    tb(
        s,
        Inches(0.40),
        Inches(5.30),
        Inches(12.5),
        Inches(1.40),
        "RadarBridge 可作为其中的跨雷达表征桥接模块，而 RadarMind 是完整系统。",
        17,
        True,
        GOLD,
    )

    # 12 data forms
    s = blank(prs)
    chrome(s, "总体技术路线", 12, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.32), "不同雷达数据形态可以不同：", 16)
    forms = ["Raw ADC / IQ；", "Radar Cube；", "Range-Doppler；", "Micro-Doppler；", "Point Cloud。"]
    for i, f in enumerate(forms):
        box(s, Inches(0.40 + i * 2.55), Inches(1.35), Inches(2.40), Inches(1.15), CARD, LINE)
        tb(s, Inches(0.45 + i * 2.55), Inches(1.70), Inches(2.30), Inches(0.50), f, 14, True, WHITE, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(2.80), Inches(12.4), Inches(0.35), "但最终希望映射到统一运动表征：", 16)
    flow_row(
        s,
        ["Sensor Adapter", "Motion Encoder", "LLM", "Application"],
        3.45,
        colors=[CARD2, RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1E, 0x4A, 0x3A)],
        h=0.90,
    )

    # 13 architecture
    s = blank(prs)
    chrome(s, "总体技术路线", 13, T)
    lefts = [
        "Raw IQ ─────────→ Signal Adapter",
        "Radar Cube ─────→ Cube Adapter",
        "Micro-Doppler ─→ Spectrogram Adapter",
        "Point Cloud ────→ Point Adapter",
    ]
    for i, lab in enumerate(lefts):
        box(s, Inches(0.35), Inches(0.90 + i * 1.05), Inches(7.15), Inches(0.90), CARD, LINE)
        tb(s, Inches(0.50), Inches(1.12 + i * 1.05), Inches(6.85), Inches(0.50), lab, 14, True, WHITE, anchor=MSO_ANCHOR.MIDDLE)
        arr = s.shapes.add_shape(MSO_SHAPE.RIGHT_ARROW, Inches(7.60), Inches(1.18 + i * 1.05), Inches(0.42), Inches(0.32))
        fill(arr, CYAN)
    box(s, Inches(8.20), Inches(0.90), Inches(4.70), Inches(5.15), CARD2, CYAN)
    for i, lab in enumerate(["Shared Motion Tokens", "Universal Motion Encoder", "Motion Representation"]):
        box(s, Inches(8.45), Inches(1.35 + i * 1.45), Inches(4.20), Inches(1.10), CARD, LINE)
        tb(s, Inches(8.50), Inches(1.62 + i * 1.45), Inches(4.10), Inches(0.55), lab, 14, True, CYAN, PP_ALIGN.CENTER)
        if i < 2:
            dwn = s.shapes.add_shape(MSO_SHAPE.DOWN_ARROW, Inches(10.20), Inches(2.50 + i * 1.45), Inches(0.28), Inches(0.24))
            fill(dwn, CYAN)

    # 14 three layers of AI
    s = blank(prs)
    chrome(s, "AI 的作用不只是分类", 14, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.4), Inches(0.30), "本项目中 AI 作用分为三层。", 16)
    layers = [
        (BLUE, "第一层：动作识别", "从雷达特征中判断基础动作类别。"),
        (CYAN, "第二层：通用运动表征", "学习对人、场景和雷达变化更鲁棒的 motion representation。"),
        (GOLD, "第三层：语义理解", "结合 LLM，将雷达运动特征转化为自然语言描述、连续行为总结和异常判断。"),
    ]
    for i, (c, t, b) in enumerate(layers):
        box(s, Inches(0.40), Inches(1.28 + i * 1.75), Inches(12.5), Inches(1.58), CARD, LINE)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.40), Inches(1.28 + i * 1.75), Inches(0.12), Inches(1.58))
        fill(bar, c)
        tb(s, Inches(0.75), Inches(1.42 + i * 1.75), Inches(11.9), Inches(0.42), t, 18, True, c)
        tb(s, Inches(0.75), Inches(1.92 + i * 1.75), Inches(11.9), Inches(0.70), b, 16, False, WHITE)

    # 15 sci core
    s = blank(prs)
    chrome(s, "三跨通用：第一阶段科研核心", 15, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.30), "第一篇 SCI 的核心目标可以聚焦在：", 16)
    box(s, Inches(1.1), Inches(1.25), Inches(11.1), Inches(1.35), CARD2, GOLD)
    tb(
        s,
        Inches(1.2),
        Inches(1.62),
        Inches(10.9),
        Inches(0.60),
        "Cross-Radar + Cross-Subject + Cross-Environment HAR",
        20,
        True,
        GOLD,
        PP_ALIGN.CENTER,
    )
    tb(s, Inches(0.40), Inches(2.80), Inches(12.4), Inches(0.30), "具体任务：", 16)
    tasks = [
        "构建多数据集 benchmark；",
        "复现 DAP-Net 等 baseline；",
        "设计更鲁棒的 motion encoder；",
        "评估 unseen subject、unseen environment、unseen radar；",
        "分析模型是否真正减少 radar / subject / environment 信息。",
    ]
    for i, tsk in enumerate(tasks):
        box(s, Inches(0.40), Inches(3.20 + i * 0.70), Inches(12.5), Inches(0.60), CARD, LINE)
        tb(s, Inches(0.65), Inches(3.32 + i * 0.70), Inches(12.0), Inches(0.38), tsk, 15, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # 16 datasets
    s = blank(prs)
    chrome(s, "多数据集计划", 16, T)
    tb(s, Inches(0.40), Inches(0.80), Inches(12.4), Inches(0.30), "计划整合多个公开雷达人体行为识别数据集。", 15)
    table = [
        ("数据集", "主要作用"),
        ("CI4R-MULTI3", "10/24/77 GHz同步雷达，研究跨雷达"),
        ("UniMM-HAR", "异构毫米波 HAR benchmark"),
        ("MM-Fi", "多人员、多动作、多环境"),
        ("mRI", "雷达点云与运动结构"),
        ("RadHAR", "毫米波点云动作识别"),
        ("MiliPoint", "扩大动作类别"),
        ("HIBER", "遮挡与环境变化"),
        ("FAU multistatic", "多站雷达与底层信号"),
    ]
    box(s, Inches(0.55), Inches(1.18), Inches(12.2), Inches(5.70), CARD, LINE)
    for i, (a, b) in enumerate(table):
        y = 1.18 + i * 0.63
        if i == 0:
            head = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(y), Inches(12.2), Inches(0.63))
            fill(head, RGBColor(0x1A, 0x3A, 0x58))
            ca, cb, bold = CYAN, CYAN, True
        else:
            if i % 2 == 0:
                stripe = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.55), Inches(y), Inches(12.2), Inches(0.63))
                fill(stripe, RGBColor(0x12, 0x1E, 0x32))
            ca, cb, bold = WHITE, WHITE, False
        tb(s, Inches(0.75), Inches(y + 0.14), Inches(3.3), Inches(0.38), a, 14, True if i == 0 or True else bold, ca)
        tb(s, Inches(4.20), Inches(y + 0.14), Inches(8.3), Inches(0.38), b, 14, bold, cb)

    # 17 DAP-Net
    s = blank(prs)
    chrome(s, "与 DAP-Net 的关系", 17, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.90),
        Inches(12.5),
        Inches(1.4),
        [
            "DAP-Net 是当前重要 baseline，不是项目终点。",
            "DAP-Net 主要解决：",
        ],
        16,
    )
    box(s, Inches(1.4), Inches(2.35), Inches(10.5), Inches(1.15), CARD, BLUE)
    tb(s, Inches(1.5), Inches(2.65), Inches(10.3), Inches(0.55), "heterogeneous mmWave point-cloud HAR", 18, True, CYAN, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(3.70), Inches(12.5), Inches(0.32), "RadarMind 希望进一步扩展到：", 16)
    box(s, Inches(0.55), Inches(4.12), Inches(12.2), Inches(1.35), CARD, GOLD)
    tb(
        s,
        Inches(0.70),
        Inches(4.45),
        Inches(11.9),
        Inches(0.75),
        "Cross-Radar + Cross-Subject + Cross-Environment + Open-Vocabulary Motion Understanding",
        16,
        True,
        WHITE,
        PP_ALIGN.CENTER,
    )
    tb(s, Inches(0.40), Inches(5.70), Inches(12.5), Inches(1.10), "我们将首先公平复现 DAP-Net，并在统一 benchmark 上比较。", 16, True, GOLD)

    # 18 methods
    s = blank(prs)
    chrome(s, "预期算法思路", 18, T)
    blocks = [
        (
            "Physics-aware Sensor Randomization",
            [
                "模拟不同雷达设备造成的变化：",
                "点云密度变化；",
                "Doppler噪声与分辨率变化；",
                "intensity动态范围变化；",
                "时间采样变化；",
                "空间分辨率变化。",
            ],
        ),
        (
            "Multi-view Motion Representation",
            [
                "同时利用：",
                "三维点云运动；",
                "Doppler速度变化；",
                "time-Doppler动态；",
                "运动统计特征。",
            ],
        ),
        (
            "Domain-Invariant Learning",
            [
                "让模型保留 activity 信息，同时降低 radar、subject、environment 信息。",
            ],
        ),
    ]
    for i, (title, body) in enumerate(blocks):
        x = 0.35 + i * 4.30
        box(s, Inches(x), Inches(0.90), Inches(4.12), Inches(5.85), CARD, LINE)
        top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(0.90), Inches(4.12), Inches(0.12))
        fill(top, [BLUE, CYAN, GOLD][i])
        tb(s, Inches(x + 0.15), Inches(1.15), Inches(3.82), Inches(1.05), title, 15, True, WHITE)
        lines(s, Inches(x + 0.18), Inches(2.30), Inches(3.78), Inches(4.20), body, 14)

    # 19 open vocab
    s = blank(prs)
    chrome(s, "从固定类别到开放动作理解", 19, T)
    tb(s, Inches(0.40), Inches(0.85), Inches(12.4), Inches(0.28), "传统模型输出：", 16)
    box(s, Inches(0.40), Inches(1.20), Inches(12.5), Inches(0.85), CARD, LINE)
    tb(s, Inches(0.55), Inches(1.40), Inches(12.2), Inches(0.48), "Class=Walking", 20, True, MUTED, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(2.20), Inches(12.4), Inches(0.28), "RadarMind 希望进一步输出：", 16)
    box(s, Inches(0.40), Inches(2.55), Inches(12.5), Inches(1.05), CARD, CYAN)
    tb(s, Inches(0.55), Inches(2.75), Inches(12.2), Inches(0.70), "人员向前移动，具有明显周期性步态；随后速度降低并停止。", 16, False, WHITE)
    tb(s, Inches(0.40), Inches(3.75), Inches(12.4), Inches(0.28), "再结合 LLM，形成：", 16)
    box(s, Inches(0.40), Inches(4.10), Inches(12.5), Inches(1.15), CARD, GOLD)
    tb(s, Inches(0.55), Inches(4.30), Inches(12.2), Inches(0.80), "该人员向雷达方向行走，停留后弯腰拾取物体，未检测到异常行为。", 16, False, WHITE)
    tb(s, Inches(0.40), Inches(5.50), Inches(12.5), Inches(1.15), "这使系统从“分类”升级为“理解”。", 18, True, GOLD)

    # 20 LLM position
    s = blank(prs)
    chrome(s, "LLM 的应用位置", 20, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.30), "DeepSeek LLM API 不直接处理原始雷达数据。", 16)
    tb(s, Inches(0.40), Inches(1.15), Inches(12.5), Inches(0.30), "底层雷达模型先输出结构化运动信息：", 16)
    fields = [
        "Direction: toward",
        "Mean velocity: 0.8 m/s",
        "Peak velocity: 2.6 m/s",
        "Periodicity: strong",
        "Motion state: walking → stop → bend",
        "Uncertainty: low",
    ]
    box(s, Inches(0.40), Inches(1.52), Inches(6.35), Inches(5.20), CARD, LINE)
    for i, f in enumerate(fields):
        tb(s, Inches(0.65), Inches(1.70 + i * 0.78), Inches(5.90), Inches(0.60), f, 15, False, CYAN)
    tb(s, Inches(7.00), Inches(1.52), Inches(5.90), Inches(0.32), "LLM 负责：", 16)
    duties = [
        "动作语义解释；",
        "连续行为总结；",
        "异常行为判断；",
        "自然语言问答；",
        "应用层交互。",
    ]
    for i, d in enumerate(duties):
        box(s, Inches(7.00), Inches(1.95 + i * 0.90), Inches(5.85), Inches(0.78), CARD, LINE)
        tb(s, Inches(7.20), Inches(2.12 + i * 0.90), Inches(5.50), Inches(0.45), d, 15, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)

    # 21 continuous
    s = blank(prs)
    chrome(s, "连续动作理解", 21, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.30), "真实行为往往不是单一动作，而是连续动作序列：", 16)
    seq = ["Walk", "Stop", "Bend", "Pick", "Stand", "Walk Away"]
    flow_row(s, seq, 1.25, colors=[CARD2] * 6)
    tb(s, Inches(0.40), Inches(2.10), Inches(12.5), Inches(0.30), "RadarMind 计划进一步实现：", 16)
    for i, lab in enumerate(["动作起止时间检测；", "连续行为切分；", "motion timeline；", "LLM行为总结。"]):
        box(s, Inches(0.40 + i * 3.20), Inches(2.50), Inches(3.05), Inches(0.80), CARD, LINE)
        tb(s, Inches(0.45 + i * 3.20), Inches(2.68), Inches(2.95), Inches(0.48), lab, 14, True, WHITE, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(3.50), Inches(12.5), Inches(0.30), "示例输出：", 16)
    box(s, Inches(0.40), Inches(3.90), Inches(12.5), Inches(2.55), CARD2, GOLD)
    tb(
        s,
        Inches(0.65),
        Inches(4.70),
        Inches(12.0),
        Inches(1.10),
        "人员进入监测区域后向前行走，随后停止并弯腰拾取物体，之后起身离开。",
        18,
        True,
        WHITE,
        PP_ALIGN.CENTER,
    )

    # 22 application + UI sketch
    s = blank(prs)
    chrome(s, "Application 设计", 22, T)
    tb(s, Inches(0.35), Inches(0.78), Inches(4.5), Inches(0.28), "最终计划封装为：", 13)
    tb(s, Inches(0.35), Inches(1.05), Inches(4.5), Inches(0.32), "RadarMind Application", 16, True, CYAN)
    tb(s, Inches(0.35), Inches(1.40), Inches(4.5), Inches(0.26), "软件包含四个模块：", 13)
    mods = [
        ("Radar View", "显示 micro-Doppler、point cloud、Range-Doppler 等雷达特征。"),
        ("Motion Intelligence", "显示当前运动状态、速度、方向、置信度和动作判断。"),
        ("Timeline", "显示连续动作序列。"),
        ("AI Understanding", "通过 DeepSeek 生成自然语言解释和问答。"),
    ]
    for i, (t, b) in enumerate(mods):
        box(s, Inches(0.35), Inches(1.72 + i * 1.22), Inches(4.55), Inches(1.12), CARD, LINE)
        tb(s, Inches(0.50), Inches(1.80 + i * 1.22), Inches(4.25), Inches(0.30), t, 13, True, GOLD)
        tb(s, Inches(0.50), Inches(2.12 + i * 1.22), Inches(4.25), Inches(0.60), b, 12, False, WHITE)

    # UI window — module names only, matching the source copy
    box(s, Inches(5.15), Inches(0.85), Inches(7.75), Inches(5.95), RGBColor(0x10, 0x1C, 0x30), CYAN)
    titlebar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(5.15), Inches(0.85), Inches(7.75), Inches(0.42))
    fill(titlebar, RGBColor(0x1A, 0x3A, 0x58))
    tb(s, Inches(5.30), Inches(0.90), Inches(5.5), Inches(0.32), "RadarMind Application", 12, True, WHITE)
    box(s, Inches(5.30), Inches(1.40), Inches(3.55), Inches(3.15), CARD, LINE)
    tb(s, Inches(5.40), Inches(1.48), Inches(3.35), Inches(0.28), "Radar View", 12, True, CYAN)
    for r, col in ((1.95, RGBColor(0x16, 0x3A, 0x52)), (2.25, RGBColor(0x1A, 0x4A, 0x64)), (2.55, RGBColor(0x22, 0x5A, 0x78))):
        ring = s.shapes.add_shape(MSO_SHAPE.OVAL, Inches(6.15), Inches(r), Inches(1.85), Inches(1.85))
        ring.fill.background()
        stroke(ring, col, 1.15)
    sweep = s.shapes.add_shape(MSO_SHAPE.PIE, Inches(6.40), Inches(2.20), Inches(1.35), Inches(1.35))
    fill(sweep, RGBColor(0x14, 0x3A, 0x58))
    stroke(sweep, CYAN, 1.0)
    box(s, Inches(9.00), Inches(1.40), Inches(3.70), Inches(1.45), CARD, LINE)
    tb(s, Inches(9.10), Inches(1.85), Inches(3.50), Inches(0.55), "Motion Intelligence", 14, True, GOLD, PP_ALIGN.CENTER)
    box(s, Inches(9.00), Inches(2.98), Inches(3.70), Inches(1.57), CARD, LINE)
    tb(s, Inches(9.10), Inches(3.50), Inches(3.50), Inches(0.50), "Timeline", 16, True, GREEN, PP_ALIGN.CENTER)
    box(s, Inches(5.30), Inches(4.70), Inches(7.40), Inches(1.90), CARD, LINE)
    tb(s, Inches(5.40), Inches(5.35), Inches(7.20), Inches(0.55), "AI Understanding", 16, True, GOLD, PP_ALIGN.CENTER)

    # 23 demo goals
    s = blank(prs)
    chrome(s, "软件演示目标", 23, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.28), "近期先实现：", 16)
    box(s, Inches(0.40), Inches(1.15), Inches(6.15), Inches(3.55), CARD, CYAN)
    tb(s, Inches(0.60), Inches(1.30), Inches(5.80), Inches(0.40), "Replay Mode", 20, True, CYAN)
    tb(s, Inches(0.60), Inches(1.75), Inches(5.80), Inches(0.50), "基于公开数据集进行回放式演示。", 15)
    tb(s, Inches(0.60), Inches(2.30), Inches(5.80), Inches(0.30), "展示：", 15)
    lines(
        s,
        Inches(0.60),
        Inches(2.65),
        Inches(5.80),
        Inches(1.90),
        [
            "不同动作对应的雷达图；",
            "同一个动作在不同雷达下的差异；",
            "模型预测结果；",
            "跨雷达/跨人员失败案例；",
            "LLM行为解释。",
        ],
        14,
        WHITE,
        False,
    )
    tb(s, Inches(6.80), Inches(1.15), Inches(6.1), Inches(0.30), "未来扩展：", 16)
    box(s, Inches(6.80), Inches(1.55), Inches(6.10), Inches(3.15), CARD, GOLD)
    tb(s, Inches(7.00), Inches(1.80), Inches(5.75), Inches(0.45), "Live Radar Mode", 20, True, GOLD)
    tb(s, Inches(7.00), Inches(2.45), Inches(5.75), Inches(1.80), "接入真实毫米波雷达，实现实时动作识别与语义理解。", 16)

    # 24 ADAS
    s = blank(prs)
    chrome(s, "与自动驾驶和智能感知的联系", 24, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.85),
        Inches(12.5),
        Inches(1.15),
        [
            "自动驾驶场景中，系统不能仅依赖开发者提前写死有限动作类别。",
            "更理想的系统应能够理解：",
        ],
        16,
    )
    for i, lab in enumerate(
        [
            "行人是否正在进入车道；",
            "骑行者是否减速转向；",
            "人体运动是否存在异常；",
            "目标行为是否具有潜在风险。",
        ]
    ):
        box(s, Inches(0.40 + (i % 2) * 6.45), Inches(2.15 + (i // 2) * 1.05), Inches(6.20), Inches(0.90), CARD, LINE)
        tb(s, Inches(0.55 + (i % 2) * 6.45), Inches(2.38 + (i // 2) * 1.05), Inches(5.95), Inches(0.48), lab, 15)
    tb(s, Inches(0.40), Inches(4.40), Inches(12.5), Inches(0.32), "RadarMind 的目标不是直接替代车载感知系统，而是探索：", 15)
    flow_row(
        s,
        ["Radar", "Motion Understanding", "Semantic Information", "Decision System"],
        4.90,
        colors=[CARD2, RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1A, 0x3A, 0x52), RGBColor(0x1E, 0x4A, 0x3A)],
    )
    tb(s, Inches(0.40), Inches(5.70), Inches(12.5), Inches(1.05), "这一未来智能感知路线。", 16, True, GOLD)

    # 25 curriculum
    s = blank(prs)
    chrome(s, "与电子信息本科培养的关系", 25, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.85),
        Inches(12.5),
        Inches(1.20),
        [
            "该项目非常适合作为电子信息工程本科科研训练。",
            "它贯穿：",
        ],
        16,
    )
    courses = ["雷达感知；", "信号与系统；", "数字信号处理；", "通信原理；", "信息论；", "深度学习；", "大模型；", "软件开发。"]
    for i, c in enumerate(courses):
        x = 0.40 + (i % 4) * 3.20
        y = 2.20 + (i // 4) * 1.35
        box(s, Inches(x), Inches(y), Inches(3.05), Inches(1.15), CARD, LINE)
        tb(s, Inches(x), Inches(y + 0.35), Inches(3.05), Inches(0.45), c, 16, True, WHITE, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(5.15), Inches(12.5), Inches(1.55), "对即将学习数字信号处理和信息论基础的本科生尤其有帮助。", 17, True, GOLD)

    # 26 student value
    s = blank(prs)
    chrome(s, "对本科生能力培养的价值", 26, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.28), "通过该项目，学生可以系统理解：", 16)
    vals = [
        "雷达信号如何产生；",
        "FFT / STFT / Doppler 如何用于动作表征；",
        "为什么同分布 accuracy 不能代表泛化能力；",
        "什么是 domain shift；",
        "如何设计科研实验；",
        "如何阅读和复现前沿工作；",
        "如何从算法走向应用系统。",
    ]
    for i, v in enumerate(vals):
        box(s, Inches(0.40), Inches(1.18 + i * 0.58), Inches(12.5), Inches(0.52), CARD, LINE)
        tb(s, Inches(0.60), Inches(1.26 + i * 0.58), Inches(12.1), Inches(0.36), v, 14, False, WHITE, anchor=MSO_ANCHOR.MIDDLE)
    tb(s, Inches(0.40), Inches(5.30), Inches(12.5), Inches(0.28), "这不是简单训练一个模型，而是完整经历：", 15)
    flow_row(s, ["Signal", "Data", "Model", "Generalization", "Application"], 5.70)

    # 27 phases
    s = blank(prs)
    chrome(s, "分阶段执行计划", 27, T)
    phases = [
        ("Phase I：数据与基线", ["整合公开数据集；", "下载 raw radar / point cloud；", "复现 CI4R 与 DAP-Net；", "建立统一 metadata。"], BLUE),
        ("Phase II：三跨 HAR", ["cross-radar；", "cross-subject；", "cross-environment；", "domain shift 分析；", "形成 SCI 保底成果。"], CYAN),
        ("Phase III：运动表征与LLM", ["motion encoder；", "open-vocabulary motion representation；", "DeepSeek LLM解释层；", "连续动作理解。"], GOLD),
        ("Phase IV：软件系统", ["RadarMind Application；", "replay demo；", "live radar扩展；", "PPT与现场演示。"], GREEN),
    ]
    for i, (t, body, c) in enumerate(phases):
        x = 0.35 + (i % 2) * 6.45
        y = 0.88 + (i // 2) * 3.00
        box(s, Inches(x), Inches(y), Inches(6.25), Inches(2.80), CARD, LINE)
        top = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(x), Inches(y), Inches(6.25), Inches(0.10))
        fill(top, c)
        tb(s, Inches(x + 0.20), Inches(y + 0.22), Inches(5.85), Inches(0.40), t, 16, True, c)
        lines(s, Inches(x + 0.22), Inches(y + 0.70), Inches(5.85), Inches(1.95), body, 13, WHITE, False)

    # 28 outcomes
    s = blank(prs)
    chrome(s, "成果目标", 28, T)
    goals = [
        (GREEN, "保底目标", "完成跨雷达、跨人员、跨场景 HAR，形成一篇普通 SCI 论文。"),
        (CYAN, "中期目标", "构建通用雷达运动表征模型，实现多数据集泛化。"),
        (GOLD, "理想目标", "形成 RadarMind："),
    ]
    for i, (c, t, b) in enumerate(goals):
        box(s, Inches(0.40), Inches(0.90 + i * 1.35), Inches(12.5), Inches(1.20), CARD, LINE)
        bar = s.shapes.add_shape(MSO_SHAPE.RECTANGLE, Inches(0.40), Inches(0.90 + i * 1.35), Inches(0.12), Inches(1.20))
        fill(bar, c)
        tb(s, Inches(0.75), Inches(1.00 + i * 1.35), Inches(12.0), Inches(0.35), t, 16, True, c)
        tb(s, Inches(0.75), Inches(1.40 + i * 1.35), Inches(12.0), Inches(0.50), b, 15, False, WHITE)
    box(s, Inches(0.70), Inches(5.05), Inches(11.9), Inches(1.65), CARD2, GOLD)
    tb(
        s,
        Inches(0.85),
        Inches(5.50),
        Inches(11.6),
        Inches(0.80),
        "Radar Sensing → Motion Representation → LLM Understanding → Interactive Application",
        16,
        True,
        GOLD,
        PP_ALIGN.CENTER,
    )

    # 29 supervisor
    s = blank(prs)
    chrome(s, "与孙老师方向的结合", 29, T)
    lines(
        s,
        Inches(0.40),
        Inches(0.85),
        Inches(12.5),
        Inches(1.55),
        [
            "本项目与雷达信号处理、无线感知、低慢小目标智能探测、多源感知和 AI 应用高度相关。",
            "人体运动 micro-Doppler 来自手臂、腿部和躯干运动；低慢小目标 micro-Doppler 可来自旋翼、机体振动和目标运动。",
            "二者共同关注：",
        ],
        15,
    )
    shared = ["雷达回波；", "Doppler / micro-Doppler；", "弱目标识别；", "跨设备泛化；", "多源感知；", "AI语义判断。"]
    for i, lab in enumerate(shared):
        x = 0.40 + (i % 3) * 4.25
        y = 2.70 + (i // 3) * 1.20
        box(s, Inches(x), Inches(y), Inches(4.05), Inches(1.05), CARD, LINE)
        tb(s, Inches(x), Inches(y + 0.30), Inches(4.05), Inches(0.45), lab, 15, True, WHITE, PP_ALIGN.CENTER)
    tb(s, Inches(0.40), Inches(5.30), Inches(12.5), Inches(1.40), "因此，本项目可作为雷达智能感知方法的可控验证入口。", 17, True, GOLD)

    # 30 summary
    s = blank(prs)
    chrome(s, "总结", 30, T)
    tb(s, Inches(0.40), Inches(0.82), Inches(12.5), Inches(0.55), "RadarMind 的目标是将雷达人体行为识别从固定分类任务推进到通用运动理解系统。", 16)
    tb(s, Inches(0.40), Inches(1.40), Inches(12.5), Inches(0.28), "核心主线：", 15)
    box(s, Inches(0.70), Inches(1.75), Inches(11.9), Inches(1.20), CARD2, CYAN)
    tb(
        s,
        Inches(0.85),
        Inches(2.10),
        Inches(11.6),
        Inches(0.55),
        "Learn Motion, Not Sensor, Subject, or Environment",
        20,
        True,
        CYAN,
        PP_ALIGN.CENTER,
    )
    tb(s, Inches(0.40), Inches(3.15), Inches(6.1), Inches(0.28), "近期目标：", 15)
    box(s, Inches(0.40), Inches(3.48), Inches(6.15), Inches(1.25), CARD, LINE)
    tb(s, Inches(0.55), Inches(3.80), Inches(5.85), Inches(0.70), "跨雷达、跨人员、跨场景 HAR。", 16, False, WHITE)
    tb(s, Inches(6.80), Inches(3.15), Inches(6.1), Inches(0.28), "长期目标：", 15)
    box(s, Inches(6.80), Inches(3.48), Inches(6.10), Inches(1.25), CARD, LINE)
    tb(s, Inches(6.95), Inches(3.75), Inches(5.80), Inches(0.80), "雷达信号到开放式人体运动理解，再到可交互 AI 应用。", 15)
    tb(s, Inches(0.40), Inches(4.90), Inches(12.5), Inches(0.28), "最后一句建议放大：", 14, False, MUTED)
    box(s, Inches(0.70), Inches(5.25), Inches(11.9), Inches(1.45), CARD2, GOLD)
    tb(
        s,
        Inches(0.85),
        Inches(5.65),
        Inches(11.6),
        Inches(0.70),
        "From Radar Signals to Human Motion Understanding",
        22,
        True,
        GOLD,
        PP_ALIGN.CENTER,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("RadarMind_*.pptx"):
        if old.name != PPTX.name:
            try:
                old.unlink()
            except OSError:
                pass
    prs.save(str(PPTX))
    PPTX_ROOT.write_bytes(PPTX.read_bytes())
    print(f"wrote {PPTX}  slides={len(prs.slides)}")
    print(f"copied {PPTX_ROOT}")
    return PPTX


def export_pdf(pptx: Path, pdf: Path) -> Path:
    pdf.parent.mkdir(parents=True, exist_ok=True)
    if pdf.exists():
        pdf.unlink()
    try:
        import win32com.client  # type: ignore
    except ImportError:
        raise SystemExit("need pywin32 to export PDF via PowerPoint")
    app = win32com.client.Dispatch("PowerPoint.Application")
    app.Visible = True
    app.DisplayAlerts = 0
    # 32 = ppSaveAsPDF
    pres = app.Presentations.Open(str(pptx.resolve()), WithWindow=False)
    try:
        pres.SaveAs(str(pdf.resolve()), 32)
    finally:
        pres.Close()
        app.Quit()
    PDF_ROOT.write_bytes(pdf.read_bytes())
    print(f"wrote {pdf}")
    print(f"copied {PDF_ROOT}")
    return pdf


if __name__ == "__main__":
    pptx = build()
    export_pdf(pptx, PDF)
