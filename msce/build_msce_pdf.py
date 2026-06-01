#!/usr/bin/env python3
"""Build MSCE Product Architecture & Mechanism PDF."""

import os
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_JUSTIFY
from reportlab.lib.colors import (
    HexColor, black, white, grey, lightgrey, navy, darkblue, darkred, darkgreen
)
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle,
    KeepTogether, Preformatted, HRFlowable, ListFlowable, ListItem, Frame,
    PageTemplate, BaseDocTemplate, NextPageTemplate, PageTemplate
)
from reportlab.platypus.flowables import Flowable
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

# ── Register Chinese fonts ──
FONT_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'
FONT_BOLD_PATH = '/System/Library/Fonts/STHeiti Medium.ttc'
pdfmetrics.registerFont(TTFont('STHeiti', FONT_PATH, subfontIndex=0))
pdfmetrics.registerFont(TTFont('STHeitiBold', FONT_BOLD_PATH, subfontIndex=0))

# Register a monospace font for code blocks
try:
    # Try Menlo on macOS
    from reportlab.pdfbase.ttfonts import TTFont as TT
    pdfmetrics.registerFont(TT('Menlo', '/System/Library/Fonts/Menlo.ttc', subfontIndex=0))
    CODE_FONT = 'Menlo'
except:
    CODE_FONT = 'Courier'

# ── Colors ──
PRIMARY = HexColor('#1a1a2e')
SECONDARY = HexColor('#16213e')
ACCENT = HexColor('#0f3460')
HIGHLIGHT = HexColor('#e94560')
MUTED = HexColor('#7f8c8d')
BG_LIGHT = HexColor('#f5f6fa')
CODE_BG = HexColor('#2d3436')
TABLE_HEADER = HexColor('#2c3e50')
BORDER_COLOR = HexColor('#bdc3c7')
SUCCESS_GREEN = HexColor('#27ae60')
WARN_ORANGE = HexColor('#e67e22')

WIDTH, HEIGHT = A4

# ── Custom Styles ──
styles = getSampleStyleSheet()

# Base body style
body_style = ParagraphStyle(
    'CNBody', fontName='STHeiti', fontSize=10, leading=16,
    spaceAfter=6, alignment=TA_JUSTIFY, wordSpace=0.5,
)

# Title styles
cover_title_style = ParagraphStyle(
    'CoverTitle', fontName='STHeitiBold', fontSize=32, leading=42,
    alignment=TA_CENTER, textColor=white, spaceAfter=10,
)

cover_subtitle_style = ParagraphStyle(
    'CoverSubtitle', fontName='STHeiti', fontSize=16, leading=24,
    alignment=TA_CENTER, textColor=HexColor('#a0a0c0'), spaceAfter=6,
)

h1_style = ParagraphStyle(
    'CNH1', fontName='STHeitiBold', fontSize=22, leading=30,
    spaceBefore=24, spaceAfter=14, textColor=PRIMARY,
    borderPadding=(0, 0, 2, 0),
)

h2_style = ParagraphStyle(
    'CNH2', fontName='STHeitiBold', fontSize=16, leading=22,
    spaceBefore=18, spaceAfter=10, textColor=SECONDARY,
)

h3_style = ParagraphStyle(
    'CNH3', fontName='STHeitiBold', fontSize=13, leading=18,
    spaceBefore=14, spaceAfter=8, textColor=ACCENT,
)

code_style = ParagraphStyle(
    'CodeBlock', fontName=CODE_FONT, fontSize=7.5, leading=10,
    leftIndent=8, rightIndent=8, spaceAfter=8, spaceBefore=4,
    textColor=HexColor('#dfe6e9'), backColor=CODE_BG,
    borderPadding=8, borderWidth=0.5, borderColor=HexColor('#636e72'),
)

inline_code_style = ParagraphStyle(
    'InlineCode', fontName=CODE_FONT, fontSize=9, textColor=HexColor('#d63031'),
)

table_cell_style = ParagraphStyle(
    'TableCell', fontName='STHeiti', fontSize=8.5, leading=13,
    alignment=TA_LEFT,
)

table_header_style = ParagraphStyle(
    'TableHeader', fontName='STHeitiBold', fontSize=9, leading=13,
    alignment=TA_CENTER, textColor=white,
)

caption_style = ParagraphStyle(
    'Caption', fontName='STHeiti', fontSize=8, leading=12,
    alignment=TA_CENTER, textColor=MUTED, spaceAfter=12,
)

bullet_style = ParagraphStyle(
    'CNBullet', fontName='STHeiti', fontSize=10, leading=16,
    leftIndent=16, spaceAfter=3, bulletIndent=6,
)

footer_style = ParagraphStyle(
    'Footer', fontName='STHeiti', fontSize=8, leading=10,
    alignment=TA_CENTER, textColor=MUTED,
)


# ── Page template with header/footer ──
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        canvas.Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_footer(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_footer(self, num_pages):
        self.saveState()
        self.setFont('STHeiti', 8)
        self.setFillColor(MUTED)
        text = f"MSCE 底层架构与机制 — 技术白皮书 | 第 {self._pageNumber} 页 / 共 {num_pages} 页"
        self.drawCentredString(WIDTH / 2, 18 * mm, text)
        # Top line
        self.setStrokeColor(BORDER_COLOR)
        self.setLineWidth(0.5)
        self.line(20 * mm, 22 * mm, WIDTH - 20 * mm, 22 * mm)
        self.restoreState()


# ── Custom flowables ──
class HRFlowableCustom(Flowable):
    """Horizontal rule."""
    def __init__(self, width=None, color=BORDER_COLOR, thickness=0.5):
        Flowable.__init__(self)
        self.width = width
        self.color = color
        self.thickness = thickness

    def draw(self):
        w = self.width if self.width else self.canv._pagesize[0] - 40 * mm
        self.canv.setStrokeColor(self.color)
        self.canv.setLineWidth(self.thickness)
        self.canv.line(0, 0, w, 0)

    def wrap(self, availWidth, availHeight):
        return (self.width or availWidth, 4)


class BoxFlowable(Flowable):
    """A simple colored box with text."""
    def __init__(self, text, bg_color=BG_LIGHT, border_color=BORDER_COLOR, padding=10, font_size=9):
        Flowable.__init__(self)
        self.text = text
        self.bg_color = bg_color
        self.border_color = border_color
        self.padding = padding
        self.font_size = font_size
        self._content_width = None

    def wrap(self, availWidth, availHeight):
        self._content_width = availWidth - 2 * self.padding - 4
        # Estimate height
        lines = self.text.count('\n') + 1
        self._height = lines * (self.font_size + 4) + 2 * self.padding
        return (availWidth, self._height)

    def draw(self):
        c = self.canv
        c.saveState()
        # Background
        c.setFillColor(self.bg_color)
        c.setStrokeColor(self.border_color)
        c.setLineWidth(0.5)
        c.roundRect(0, 0, self.width, self._height, 4, fill=1, stroke=1)
        # Text
        c.setFillColor(black)
        c.setFont(CODE_FONT, self.font_size)
        lines = self.text.split('\n')
        y = self._height - self.padding - self.font_size
        for line in lines:
            c.drawString(self.padding + 2, y, line)
            y -= (self.font_size + 3)
        c.restoreState()


# ── Helper functions ──
def h1(text):
    return Paragraph(text, h1_style)

def h2(text):
    return Paragraph(text, h2_style)

def h3(text):
    return Paragraph(text, h3_style)

def body(text):
    return Paragraph(text, body_style)

def code(text):
    return Preformatted(text, code_style, maxLineLength=110)

def bullet(text):
    return Paragraph(f"• {text}", bullet_style)

def hr():
    return HRFlowableCustom(width=WIDTH - 40 * mm)

def spacer(h=6):
    return Spacer(1, h)

def make_table(headers, rows, col_widths=None):
    """Create a styled table with headers and rows."""
    header_cells = [Paragraph(h, table_header_style) for h in headers]
    data = [header_cells]
    for row in rows:
        data.append([Paragraph(str(c), table_cell_style) for c in row])

    if col_widths is None:
        available = WIDTH - 40 * mm
        col_widths = [available / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths, repeatRows=1)
    style_cmds = [
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER_COLOR),
        ('TOPPADDING', (0, 0), (-1, -1), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]
    # Alternating row colors
    for i in range(1, len(data)):
        if i % 2 == 0:
            style_cmds.append(('BACKGROUND', (0, i), (-1, i), HexColor('#ecf0f1')))
    t.setStyle(TableStyle(style_cmds))
    return t


def section_page(title, subtitle=None):
    """Create a section divider page."""
    elements = []
    elements.append(Spacer(1, 80 * mm))
    elements.append(Paragraph(title, ParagraphStyle(
        'SectionTitle', fontName='STHeitiBold', fontSize=28, leading=36,
        alignment=TA_CENTER, textColor=PRIMARY, spaceAfter=16,
    )))
    if subtitle:
        elements.append(Paragraph(subtitle, ParagraphStyle(
            'SectionSubtitle', fontName='STHeiti', fontSize=14, leading=20,
            alignment=TA_CENTER, textColor=MUTED,
        )))
    elements.append(PageBreak())
    return elements


# ══════════════════════════════════════════════════════════════════════════════
# DOCUMENT CONTENT
# ══════════════════════════════════════════════════════════════════════════════

def build_document():
    output_path = '/Users/dengxinhang/paper/constraint_residual/msce/MSCE_底层架构与机制.pdf'

    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=25 * mm,
        bottomMargin=25 * mm,
        title='MSCE 底层架构与机制',
        author='MSCE Research Team',
        subject='认知对抗引擎技术白皮书',
    )

    elements = []

    # ═══════════════════════════════════════════════════════════════════
    # COVER PAGE
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 60 * mm))
    elements.append(Paragraph("MSCE", ParagraphStyle(
        'CoverMSCE', fontName='STHeitiBold', fontSize=56, leading=64,
        alignment=TA_CENTER, textColor=PRIMARY, spaceAfter=6,
    )))
    elements.append(Paragraph("底层架构与机制", ParagraphStyle(
        'CoverCN', fontName='STHeitiBold', fontSize=34, leading=44,
        alignment=TA_CENTER, textColor=PRIMARY, spaceAfter=20,
    )))
    elements.append(HRFlowableCustom(width=120 * mm, color=ACCENT, thickness=2))
    elements.append(spacer(14))
    elements.append(Paragraph("认知对抗引擎技术白皮书", ParagraphStyle(
        'CoverSub', fontName='STHeiti', fontSize=18, leading=26,
        alignment=TA_CENTER, textColor=MUTED, spaceAfter=10,
    )))
    elements.append(Paragraph("Multi-model Self-Consistency Engine", ParagraphStyle(
        'CoverEng', fontName=CODE_FONT, fontSize=12, leading=16,
        alignment=TA_CENTER, textColor=HexColor('#636e72'), spaceAfter=8,
    )))
    elements.append(spacer(20))
    elements.append(Paragraph("版本 1.0 | 2026年5月", ParagraphStyle(
        'CoverVer', fontName='STHeiti', fontSize=11, leading=16,
        alignment=TA_CENTER, textColor=MUTED,
    )))
    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # TABLE OF CONTENTS (manual)
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("目录"))
    elements.append(hr())
    elements.append(spacer(8))
    toc_items = [
        ("1", "系统概览", "系统架构、数据流、核心组件"),
        ("2", "生成器层", "六种异构认知策略、提示词模板、设计原理"),
        ("3", "对抗淘汰机制", "锦标赛机制、冲突消解、淘汰规则"),
        ("4", "裁判系统", "投票机制、自适应提前退出、置信度评分"),
        ("5", "不确定性量化", "分歧度计算、置信度熵、推理路径散度"),
        ("6", "约束残差框架", "理论基础、五层规则体系、跨维信号一致性"),
        ("7", "领域自适应", "数学领域优化、裁判提示词注入"),
        ("8", "生产优化", "四生成器模式、超时策略、渐进式深度"),
        ("9", "API设计", "REST端点、请求/响应格式、批量处理"),
        ("10", "基准测试方法", "MMLU流程、评分算法、对比方法"),
        ("附录", "基准测试数据表与对比图", "完整实验数据"),
    ]
    for num, title, desc in toc_items:
        elements.append(Paragraph(
            f'<b>{num}. {title}</b>  <font color="#7f8c8d" size="9">— {desc}</font>',
            ParagraphStyle('TOCItem', fontName='STHeiti', fontSize=11, leading=20,
                          leftIndent=12, spaceAfter=4)
        ))
    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 1. SYSTEM OVERVIEW
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("1. 系统概览"))
    elements.append(hr())

    elements.append(body(
        "MSCE（Multi-model Self-Consistency Engine，多模型自一致性引擎）是一个认知对抗引擎，"
        "通过异构大语言模型的认知对抗与多轮裁判机制，实现对复杂问题的可靠推理与不确定性量化。"
        "其核心理念是：单个模型可能\"自信地错误\"（confidently wrong），"
        "但当多个具有不同认知策略的模型独立推理、互相制衡时，系统不仅能给出更准确的答案，"
        "还能识别自身的知识边界——\"知道什么时候不知道\"（knows when it doesn't know）。"
    ))

    elements.append(spacer(4))
    elements.append(h2("1.1 架构总览"))
    elements.append(body(
        "MSCE 采用三层流水线架构：生成器层（Generator Layer）、裁判层（Judge Layer）、"
        "置信度量化层（Confidence Layer）。数据流为单向管道："
    ))

    # ASCII architecture diagram
    arch_diagram = """    ┌─────────────────────────────────────────────────────────────────────┐
    │                        MSCE 系统架构 (v5.0)                         │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                     │
    │  ┌──────────┐                                                       │
    │  │ Question │                                                       │
    │  │  输入问题 │                                                       │
    │  └────┬─────┘                                                       │
    │       │                                                             │
    │       ▼                                                             │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │              GENERATOR LAYER (生成器层)                      │   │
    │  │  6 异构生成器 | ThreadPoolExecutor | 并发执行                 │   │
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
    │  │  │ deep     │ │ breadth  │ │ counter  │ │ direct   │       │   │
    │  │  │ _first   │ │ _first   │ │ -factual │ │          │       │   │
    │  │  │ GPT-4o   │ │Gemini2.5 │ │DS-Chat   │ │DS-Chat   │       │   │
    │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
    │  │  ┌──────────┐ ┌───────────────┐                             │   │
    │  │  │ science  │ │ constraint    │                             │   │
    │  │  │ _deep    │ │ _propagation  │                             │   │
    │  │  │ o1       │ │ GPT-4o        │                             │   │
    │  │  └──────────┘ └───────────────┘                             │   │
    │  └────────────────────────────┬────────────────────────────────┘   │
    │                               │ 6个候选答案                          │
    │                               ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │              JUDGE LAYER (裁判层)                            │   │
    │  │  ┌──────────────────┐   ┌──────────────────┐                │   │
    │  │  │ Best-of-N Judge  │──▶│ Appeal Mechanism │                │   │
    │  │  │ DeepSeek-Reasoner│   │ 反事实检查被淘汰 │                │   │
    │  │  │ n=3, adaptive    │   │ DeepSeek-Reasoner│                │   │
    │  │  └──────────────────┘   └──────────────────┘                │   │
    │  │  + 相似度强制淘汰 (threshold=0.75)                           │   │
    │  │  + 域自适应裁判提示                                          │   │
    │  └────────────────────────────┬────────────────────────────────┘   │
    │                               │ {eliminated, surviving, top3}        │
    │                               ▼                                     │
    │  ┌─────────────────────────────────────────────────────────────┐   │
    │  │         CONFIDENCE LAYER (置信度量化层)                      │   │
    │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐       │   │
    │  │  │Confidence│ │Disagree- │ │Reasoning │ │Low Conf  │       │   │
    │  │  │  Score   │ │  ment    │ │  Trail   │ │  Flag    │       │   │
    │  │  │ max*agree │ │clusters  │ │per-cand  │ │<0.5 warn │       │   │
    │  │  └──────────┘ └──────────┘ └──────────┘ └──────────┘       │   │
    │  └────────────────────────────┬────────────────────────────────┘   │
    │                               │                                     │
    │                               ▼                                     │
    │  ┌──────────────────────────────────────────────────────────────┐  │
    │  │  OUTPUT: {answer, confidence, disagreement_score,             │  │
    │  │           reasoning_trail, vote_details, elapsed_s}          │  │
    │  └──────────────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────────────┘"""
    elements.append(Preformatted(arch_diagram, ParagraphStyle(
        'ArchDiagram', fontName=CODE_FONT, fontSize=6.5, leading=8,
        leftIndent=4, rightIndent=4, spaceAfter=10, spaceBefore=6,
        textColor=HexColor('#2d3436'), backColor=HexColor('#f5f6fa'),
        borderPadding=6, borderWidth=0.5, borderColor=BORDER_COLOR,
    ), maxLineLength=130))
    elements.append(Paragraph("图 1-1: MSCE 三层流水线架构", caption_style))

    elements.append(h2("1.2 核心设计原则"))
    principles = [
        ("认知多样性", "不同模型 + 不同推理策略 > 同模型多次采样。模型差异优先于提示词差异。"),
        ("疑罪从无", "裁判不确定时不淘汰，宁可保留弱候选也不误杀强候选。误杀不可逆。"),
        ("不确定性即输出", "系统不仅输出答案，还输出置信度和分歧度。低置信度是有效输出，意味着\"需要人工干预\"。"),
        ("约束残差驱动", "从物理常数(L-1)到启发式规则(L3)的五层约束体系，跨维信号一致性作为可靠性判据。"),
    ]
    for title, desc in principles:
        elements.append(Paragraph(f"<b>{title}</b>: {desc}", bullet_style))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 2. GENERATOR LAYER
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("2. 生成器层"))
    elements.append(hr())

    elements.append(body(
        "生成器层是MSCE的认知多样性来源。系统同时运行6个异构生成器，每个生成器使用不同的"
        "推理策略和后端模型，对同一问题进行独立思考。策略多样性保证了系统能从多个认知角度"
        "审视问题，避免单一模型的系统性偏见。"
    ))

    elements.append(h2("2.1 生成器配置矩阵"))

    gen_headers = ["策略ID", "策略名称", "后端模型", "API来源", "超时", "最大Token"]
    gen_rows = [
        ["deep_first", "深度优先推理", "gpt-4o", "mkeai", "60s", "2000"],
        ["breadth_first", "广度优先枚举", "gemini-2.5-pro", "mkeai", "60s", "2000"],
        ["counterfactual", "反事实推理", "deepseek-chat", "deepseek", "60s", "2000"],
        ["direct", "直接推理", "deepseek-chat", "deepseek", "60s", "2000"],
        ["science_deep", "科学深度推理", "o1", "mkeai", "60s", "4000"],
        ["constraint_propagation", "约束传播", "gpt-4o", "mkeai", "60s", "2000"],
    ]
    elements.append(make_table(gen_headers, gen_rows))
    elements.append(Paragraph("表 2-1: 六生成器配置矩阵", caption_style))

    elements.append(h2("2.2 策略详细说明"))

    # Strategy 1: deep_first
    elements.append(h3("2.2.1 深度优先推理 (deep_first)"))
    elements.append(body(
        "深度优先推理者从已知事实出发，逐步推理，每一步只选择最确定的下一步。"
        "不允许跳步或猜测，目标是得到一个逻辑完整的推理链。策略适用于需要长链推导的数学和逻辑问题，"
        "优势在于推理过程可追溯、可验证。模型选择GPT-4o利用其强逻辑一致性。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'DEEP_FIRST_PROMPT = """你是一个"深度优先"推理者。\n\n'
        '从已知事实出发，一步一步推理。每一步只选择最确定的下一步。\n'
        '不要跳步，不要猜测。你的目标是得到一个逻辑完整的推理链，即使它很长。\n'
        '最终输出：清晰的推理步骤 + 最终答案。"""'
    ))

    # Strategy 2: breadth_first
    elements.append(h3("2.2.2 广度优先枚举 (breadth_first)"))
    elements.append(body(
        "广度优先推理者同时考虑所有可能的解题路径，不深入任何一条，列出所有可能性并评估每个"
        "路径的初始可信度。目标是覆盖所有选项而非找到单一答案，为后续淘汰提供全景视图。"
        "模型选择Gemini 2.5 Pro利用其广视野优势。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'BREADTH_FIRST_PROMPT = """你是一个"广度优先"推理者。\n\n'
        '同时考虑所有可能的路径，不要深入任何一条。列出所有可能性，\n'
        '评估每个的初始可信度。你的目标是覆盖所有选项，不是找到答案。\n'
        '最终输出：所有可能答案的列表 + 每个的初步可信度评估。"""'
    ))

    # Strategy 3: counterfactual
    elements.append(h3("2.2.3 反事实推理 (counterfactual)"))
    elements.append(body(
        "反事实推理者假设最常见的答案（直觉答案）是错误的，主动寻找最不可能但逻辑上仍然成立"
        "的答案。目标是打破思维惯性和群体思维，防止全体生成器落入相同陷阱。"
        "模型选择DeepSeek-Chat利用其创造性推理能力。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'COUNTERFACTUAL_PROMPT = """你是一个"反事实"推理者。\n\n'
        '假设最常见的答案（直觉答案）是错误的。\n'
        '找出最不可能但逻辑上仍然成立的答案。\n'
        '你的目标是打破思维惯性。\n'
        '最终输出：反直觉的答案 + 为什么它可能正确，为什么直觉答案可能错误。"""'
    ))

    # Strategy 4: direct
    elements.append(h3("2.2.4 直接推理 (direct)"))
    elements.append(body(
        "直接推理者不使用任何特殊策略，以正常、自然的方式进行推理并回答。该生成器作为"
        "对照组，代表模型的默认行为，为其他策略化的生成器提供基线参考。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'DIRECT_PROMPT = """你是一个直接推理者。直接回答以下问题，\n'
        '给出清晰的推理过程和最终答案。不要用特殊策略，正常思考即可。"""'
    ))

    # Strategy 5: science_deep
    elements.append(h3("2.2.5 科学深度推理 (science_deep)"))
    elements.append(body(
        "科学深度推理者专为物理和科学问题设计。其遵循严格的科学方法：明确引用相关定律"
        "（如牛顿定律、阿基米德原理），在题目给定假设条件下推理，分步计算并标注公式。"
        "模型选择o1利用其强大的链式推理（Chain-of-Thought）能力。"
        "注意：o1不支持独立system prompt，系统提示词被合并到user消息中。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'SCIENCE_DEEP_PROMPT = """你是一个科学深度推理者。对于物理和科学问题：\n'
        '1. 明确引用相关物理定律（牛顿定律、阿基米德原理、瑞利散射等）\n'
        '2. 在题目给定的假设条件下进行推理，不质疑题目设定\n'
        '3. 分步计算，每步标注使用的公式\n'
        '4. 最终给出清晰答案"""'
    ))

    # Strategy 6: constraint_propagation
    elements.append(h3("2.2.6 约束传播推理 (constraint_propagation)"))
    elements.append(body(
        "约束传播推理者基于约束满足问题（CSP）方法论，使用系统化排除法求解逻辑谜题和"
        "约束满足问题。六步流程：建立网格 -> 编码约束 -> 传播 -> 分支 -> 验证 -> 输出。"
        "完成后必须逐条回检所有约束，确保无一违反。与深度优先的链式推理互补，"
        "适用于爱因斯坦谜题、座位安排等需要系统化排除的场景。"
    ))
    elements.append(body("<b>提示词模板:</b>"))
    elements.append(code(
        'CONSTRAINT_PROPAGATION_PROMPT = """你是一个约束传播推理者。\n'
        '对于约束满足和逻辑谜题，使用系统化排除法：\n\n'
        '1. **建立网格**：列出所有变量位置和所有属性域\n'
        '2. **编码约束**：将每个文本约束转化为确定关系和排除关系\n'
        '3. **传播**：当单元格确定时，立即在同行/同列传播排除，迭代至闭合\n'
        '4. **分支**：不确定时假设→传播→矛盾则排除\n'
        '5. **验证**：完成赋值后，逐条回检所有约束，确保无一违反\n'
        '6. **输出**：完整赋值表 + 问题的明确答案\n\n'
        '关键：不跳步，不猜测。完成后必须逐条验证每个约束。"""'
    ))

    elements.append(h2("2.3 并行执行机制"))
    elements.append(body(
        "所有6个生成器通过Python的ThreadPoolExecutor(max_workers=6)并发执行。"
        "每个生成器独立调用其对应的模型API，互不干扰。使用as_completed()收集结果，"
        "一旦某个生成器返回即记录，无需等待全部完成。结果按策略名排序以保证输出一致性。"
        "API调用封装在generate_candidate()函数中，返回统一的{strategy, model, answer, success}结构。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 3. ADVERSARIAL ELIMINATION
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("3. 对抗淘汰机制"))
    elements.append(hr())

    elements.append(body(
        "对抗淘汰（Adversarial Elimination）是MSCE的核心创新。传统多模型投票（如多数投票）"
        "假设每个模型同等可靠，但在现实中，模型可能集体犯错。MSCE采用\"淘汰赛\"范式："
        "不投票选最佳答案，而是通过对抗性检查淘汰有缺陷的答案。保留下来的是经过检验的答案，"
        "而非仅靠票数胜出的答案。"
    ))

    elements.append(h2("3.1 淘汰规则（按优先级）"))
    rules = [
        ("内部一致性 (Internal Consistency)",
         "逻辑自相矛盾的答案直接淘汰。这是最高优先级规则——如果一个答案无法自洽，"
         "无论其他方面多么合理都应被排除。不确定时保留。"),
        ("外部一致性 (External Consistency)",
         "与公认科学/数学/逻辑事实冲突的答案淘汰。重要例外：对于逻辑谜题（如蓝眼睛岛、"
         "囚徒困境），不检查\"与现实一致\"，只检查与题目前提的一致性。对于假设性物理场景，"
         "不质疑题目设定，仅在给定假设下检查物理定律的正确应用。"),
        ("可验证性 (Verifiability)",
         "推理链不可追溯的答案降权（不淘汰）。清晰可追溯的推理链是评判正确性的基础。"
         "不可追溯的答案即使最终结果正确，也无法信任其推理过程。"),
        ("简洁性 (Simplicity)",
         "同等正确的情况下，更简洁的答案胜出（奥卡姆剃刀原则）。"),
    ]
    for title, desc in rules:
        elements.append(Paragraph(f"<b>{title}</b>: {desc}", bullet_style))

    elements.append(h2("3.2 相似度强制淘汰"))
    elements.append(body(
        "如果多个候选生成高度相似的答案（词重叠率 > 75%），系统强制淘汰其中分数较低者，"
        "防止\"全部保留赛\"——所有候选几乎相同导致裁判无法做出有意义区分的场景。"
        "相似度通过答案尾部300字符的Jaccard系数计算（不考虑词频，仅看词集重叠）。"
    ))
    elements.append(body("<b>相似度计算 (Jaccard系数):</b>"))
    elements.append(code(
        'def _simple_similarity(text_a, text_b):\n'
        '    """用词重叠率估算两个答案的相似度（不依赖embedding）"""\n'
        '    if not text_a or not text_b:\n'
        '        return 0.0\n'
        '    def tokens(s):\n'
        '        tail = s[-300:] if len(s) > 300 else s\n'
        '        return set(re.findall(r\'[一-鿿]+|[a-zA-Z]+\', tail.lower()))\n'
        '    a, b = tokens(text_a), tokens(text_b)\n'
        '    if not a or not b:\n'
        '        return 0.0\n'
        '    return len(a & b) / len(a | b)'
    ))

    elements.append(h2("3.3 上诉机制 (Appeal Mechanism)"))
    elements.append(body(
        "为防止裁判误杀有价值的候选，系统提供二次裁决机制。上诉裁判（DeepSeek-Reasoner）"
        "重新审查所有被淘汰的候选，执行反事实检查：如果该候选被保留，会发生什么？"
        "恢复规则包括：淘汰理由不充分则恢复、核心答案可能正确则恢复（降权0.2）、"
        "不确定则恢复（疑罪从无）。只有根本性错误且淘汰理由成立才维持淘汰。"
        "在生产模式（数学领域）中，上诉默认跳过以节省25-30秒延迟。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 4. JUDGE SYSTEM
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("4. 裁判系统"))
    elements.append(hr())

    elements.append(body(
        "裁判系统（Judge System）是MSCE的决策核心。它不寻找\"正确\"答案，而是识别并淘汰"
        "\"有缺陷\"的答案。裁判使用DeepSeek-Reasoner模型，因其在推理评估任务上的卓越表现。"
    ))

    elements.append(h2("4.1 自适应多数投票 (Adaptive Best-of-N)"))
    elements.append(body(
        "裁判采用自适应多数投票机制（默认n=3）。核心优化是\"快速通道\"：如果第一轮投票的最高"
        "置信度 >= 0.9 且至少淘汰了一个候选，则直接返回结果，跳过剩余2轮投票。"
        "该机制在保持高准确率的同时，将平均裁判时间从约18秒减少到约6-8秒（高置信度时）。"
    ))
    elements.append(body("<b>自适应投票流程:</b>"))
    elements.append(code(
        'def _best_of_n_judge(judge_client, judge_model, question, candidate_text, n=3):\n'
        '    """自适应多数投票：第一轮高置信度直接通过，低置信度才跑完整n次"""\n'
        '    # Step 1: 第一轮投票\n'
        '    v1 = _single_judge(judge_client, judge_model, question, candidate_text)\n'
        '    \n'
        '    scores = [s.get("score", 0) for s in v1.get("surviving", [])]\n'
        '    max_s = max(scores) if scores else 0\n'
        '    elim_count = len(v1.get("eliminated", []))\n'
        '    \n'
        '    # Step 2: 快速通道判定\n'
        '    ADAPTIVE_THRESHOLD = 0.9\n'
        '    if max_s >= ADAPTIVE_THRESHOLD and elim_count > 0:\n'
        '        v1["_judge_votes"] = 1\n'
        '        v1["_adaptive"] = f"fast:score={max_s:.2f},elim={elim_count}"\n'
        '        return v1  # 直接返回，跳过剩余投票\n'
        '    \n'
        '    # Step 3: 低置信度 → 完整n次投票，取最高分\n'
        '    best_verdict = v1\n'
        '    for _ in range(n - 1):\n'
        '        v = _single_judge(judge_client, judge_model, question, candidate_text)\n'
        '        if max_score(v) >= best_max_score:\n'
        '            best_verdict = v\n'
        '    return best_verdict'
    ))

    elements.append(h2("4.2 裁判提示词"))
    elements.append(body(
        "裁判系统使用结构化提示词，包含明确的淘汰规则、优先级排序和输出格式约束。"
        "裁判被要求以严格JSON格式输出，无任何解释文本。这确保输出可被程序化解析。"
    ))
    elements.append(body("<b>裁判提示词:</b>"))
    elements.append(code(
        'JUDGE_PROMPT = """你是一个公正的裁决者。检查候选答案，决定哪些应该被淘汰。\n\n'
        '## 淘汰规则（按顺序）：\n'
        '1. **内部一致性**：逻辑自相矛盾 → 淘汰。不确定 → 保留。\n'
        '2. **外部一致性**：与公认科学/数学/逻辑事实冲突 → 淘汰。不确定 → 保留。\n'
        '   **重要**：\n'
        '   - 对于逻辑谜题，不要检查"与现实一致"。只检查与题目前提的一致性。\n'
        '   - 对于假设性物理场景，不要在"场景是否可能发生"上扣分。\n'
        '     应在题目给定的假设条件下，仅检查答案是否正确应用了物理定律。\n'
        '3. **可验证性**：推理链不可追溯 → 降权（不淘汰）。\n'
        '4. **简洁性**：同等正确，更简洁的胜出（奥卡姆剃刀）。\n\n'
        '## 核心原则：疑罪从无。不确定就保留。误杀不可逆。\n\n'
        '## 输出格式（严格JSON，无解释）：\n'
        '{"eliminated":[],"surviving":[{"id":"策略名","score":0.9}],\n'
        ' "top3":[{"rank":1,"id":"策略名","summary":"一句话答案摘要"}]}\n\n'
        '只输出这一行JSON。"""'
    ))

    elements.append(h2("4.3 JSON修复机制"))
    elements.append(body(
        "由于LLM输出的JSON可能被截断或格式不规范，MSCE实现了多层JSON修复机制(_repair_json):"
    ))
    repair_steps = [
        "去除markdown代码块标记(```json, ```)",
        "直接JSON解析尝试",
        "括号补齐：统计缺失的 } 和 ] 数量并追加",
        "逐行截断修复：从末尾逐行删除并重试解析",
        "正则表达式提取：使用正则提取eliminated/surviving/top3关键字段",
    ]
    for i, step in enumerate(repair_steps, 1):
        elements.append(Paragraph(f"{i}. {step}", bullet_style))

    elements.append(h2("4.4 置信度阈值与低置信度标记"))
    elements.append(body(
        "系统设置置信度阈值 CONFIDENCE_THRESHOLD = 0.5。当所有候选被淘汰或最高分低于阈值时，"
        "系统标记 low_confidence = True，并给出具体原因。这实现了\"知道什么时候不知道\"的能力——"
        "当系统无法可靠回答时，诚实地说\"不确定\"而非给出一个可能错误的答案。"
        "这是MSCE相比单模型的关键优势之一：单模型通常对所有问题都自信作答，"
        "而MSCE能识别自身知识边界。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 5. UNCERTAINTY QUANTIFICATION
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("5. 不确定性量化"))
    elements.append(hr())

    elements.append(body(
        "MSCE的不确定性量化模块将\"系统是否确定\"本身作为一个定量输出。"
        "三个指标从不同维度描述系统的不确定性状态。"
    ))

    elements.append(h2("5.1 置信度 (Confidence)"))
    elements.append(body(
        "置信度由两部分组成：裁判最高分（max_score）乘以答案一致性比率（agreement_ratio）。"
        "答案一致性比率计算为与最高分候选答案相似的幸存者占比（相似度 > 0.5）。"
        "该公式确保高置信度需要同时满足两个条件：裁判高度认可且生成器集体一致。"
    ))
    elements.append(code(
        'max_score = max(s.get("score", 0) for s in surviving)\n'
        'agreeing = sum(1 for s in surviving\n'
        '    if similarity(s_answer, top_answer) > 0.5)\n'
        'agreement_ratio = agreeing / len(surviving)\n'
        'confidence = max_score * agreement_ratio  # range: [0, 1]'
    ))

    elements.append(h2("5.2 分歧度 (Disagreement)"))
    elements.append(body(
        "分歧度衡量幸存答案之间的认知差异程度。通过将幸存者按答案相似度聚类"
        "（阈值0.5），计算：(簇数 - 1) / (幸存者数 - 1)。"
        "全部分歧（每个答案各成一个簇）得1.0；完全一致（所有答案在一个簇）得0.0。"
        "高分歧度表明系统内存在不可调和的认知冲突，需要人工干预。"
    ))
    elements.append(code(
        'def compute_disagreement(surviving, id_to_answer, threshold=0.5):\n'
        '    if len(surviving) <= 1:\n'
        '        return 0.0\n'
        '    clusters = _cluster_answers(surviving, id_to_answer, threshold)\n'
        '    return (len(clusters) - 1) / (len(surviving) - 1)'
    ))

    elements.append(h2("5.3 推理路径散度 (Reasoning Trail)"))
    elements.append(body(
        "每个生成器的完整状态被记录在reasoning_trail中：策略名、模型、成功状态、"
        "幸存/淘汰状态、裁判评分、淘汰原因、答案摘要。这提供了完整的可审计记录，"
        "使用户不仅能看最终答案，还能理解系统如何得出结论、哪些认知路径被认可、"
        "哪些被驳回及原因。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 6. CONSTRAINT RESIDUAL FRAMEWORK
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("6. 约束残差框架"))
    elements.append(hr())

    elements.append(body(
        "约束残差框架（Constraint Residual Framework）是MSCE的理论基础。"
        "该框架认为：答案的可靠性不是一个布尔值，而是一个连续量——约束残差（constraint residual）。"
        "当答案在所有约束层上投影一致时，残差为零，答案可靠；当不同约束层给出冲突信号时，"
        "残差增大，答案不可靠。"
    ))

    elements.append(h2("6.1 五层规则体系"))

    rule_headers = ["层级", "名称", "描述", "示例"]
    rule_rows = [
        ["L-1", "物理常数层", "宇宙基本常数：光速、普朗克常数、引力常数等。不可违反。", "c = 3×10^8 m/s"],
        ["L0", "数学定律层", "数学公理和定理：算术、代数、微积分等。不可违反。", "1+1=2, ∫ x dx = x^2/2"],
        ["L1", "逻辑定律层", "形式逻辑规则：同一律、矛盾律、排中律、演绎推理。", "A→B, B→C ⊢ A→C"],
        ["L2", "领域规则层", "特定领域的经验法则和公认知识。", "阿基米德原理、供求定律"],
        ["L3", "启发式规则层", "敏捷经验法则，可能被覆盖但需要强证据。", "奥卡姆剃刀、类比推理"],
    ]
    elements.append(make_table(rule_headers, rule_rows))
    elements.append(Paragraph("表 6-1: 五层约束规则体系", caption_style))

    elements.append(h2("6.2 跨维信号一致性"))
    elements.append(body(
        "核心公式: 约束残差 = Π = Σ∇σ_i，其中σ_i为第i维的约束梯度。"
        "当一个命题在多个独立推理维度上的投影保持自洽时，该命题很可能是真命题（信号）。"
        "当不同维度的投影相互矛盾时，该命题很可能是假命题（噪声）。"
        "六个生成器对应六个独立的认知维度，裁判作为更高维度的投影验证器，"
        "上诉裁判作为正交维度的交叉验证。"
    ))

    elements.append(h2("6.3 稳定性-可见性反比律"))
    elements.append(body(
        "更深层（L-1, L0）的约束更稳定但更难被观察（高度抽象）。更浅层（L2, L3）"
        "的约束更容易被观察但更不稳定（依赖上下文）。MSCE的设计体现了这一原则："
        "裁判优先检查低层约束（内部一致性→外部一致性→可验证性），"
        "仅当低层检查通过后才考虑高层规则（简洁性）。这对应了从稳定约束到不稳定约束的检查顺序。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 7. DOMAIN ADAPTATION
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("7. 领域自适应"))
    elements.append(hr())

    elements.append(body(
        "MSCE支持根据问题领域动态调整裁判行为。当前支持的领域包括math（数学）、"
        "logic（逻辑）、science（科学）和verbal（语言）。领域自适应通过DOMAIN_JUDGE_HINTS机制实现，"
        "在裁判输入前注入领域特定的裁判指令。"
    ))

    elements.append(h2("7.1 数学领域特殊优化"))
    elements.append(body(
        "数学领域有最强的自适应提示。因为数学题答案有唯一正确的数值或表达式，"
        "不同候选不可能同时正确，裁判必须更果断地做出淘汰决策。数学提示词引导裁判："
    ))
    math_tips = [
        "比较各候选的计算过程和最终数值",
        "淘汰推理有缺陷或数值明显错误的候选",
        "至少淘汰1-2个最弱的候选（防止全部保留）",
        "如果多个候选得出相同答案，选择推理最清晰简洁的",
    ]
    for tip in math_tips:
        elements.append(Paragraph(f"• {tip}", bullet_style))

    elements.append(body("<b>数学裁判提示词注入:</b>"))
    elements.append(code(
        'DOMAIN_JUDGE_HINTS = {\n'
        '    "math": (\n'
        '        "\\n## 裁判提示：数学题\\n"\n'
        '        "- 数学题答案有唯一正确的数值或表达式\\n"\n'
        '        "- 不同候选的数值答案不可能同时正确——你必须做出选择\\n"\n'
        '        "- 比较各候选的计算过程和最终数值\\n"\n'
        '        "- 淘汰推理有缺陷或数值明显错误的候选\\n"\n'
        '        "- 至少淘汰1-2个最弱的候选\\n"\n'
        '        "- 如果多个候选得出相同答案，选择推理最清晰简洁的\\n"\n'
        '    ),\n'
        '    "logic": "",\n'
        '    "science": "",\n'
        '    "verbal": "",\n'
        '}'
    ))

    elements.append(h2("7.2 领域-学科映射"))
    elements.append(body(
        "MMLU基准测试中，系统自动将学科映射到MSCE领域："
    ))
    map_headers = ["学科", "MSCE领域"]
    map_rows = [
        ["abstract_algebra, college_mathematics, high_school_mathematics, elementary_mathematics", "math"],
        ["formal_logic, logical_fallacies", "logic"],
        ["college_physics, college_chemistry, college_biology, electrical_engineering", "science"],
        ["其他所有学科", "None (无特殊提示)"],
    ]
    elements.append(make_table(map_headers, map_rows))
    elements.append(Paragraph("表 7-1: MMLU学科-领域映射", caption_style))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 8. PRODUCTION OPTIMIZATION
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("8. 生产优化"))
    elements.append(hr())

    elements.append(body(
        "Product Engine (product_engine.py) 是MSCE的生产优化版本，目标是在保持准确率的前提下"
        "将每题响应时间从45-65秒降低到15-30秒。以下为关键优化策略："
    ))

    elements.append(h2("8.1 引擎版本对比"))
    compare_headers = ["维度", "engine.py (研究版)", "product_engine.py (生产版)"]
    compare_rows = [
        ["生成器数量", "6个（含breadth_first, counterfactual）", "4个（去掉breadth_first, counterfactual）"],
        ["生成超时", "60秒", "45秒"],
        ["生成max_tokens", "2000 (常规) / 4000 (o1)", "1500 (常规) / 2500 (o1)"],
        ["裁判n值", "3", "2"],
        ["裁判max_tokens", "2000", "1000"],
        ["上诉机制", "默认启用 (数学除外)", "数学默认跳过"],
        ["上诉超时", "30秒", "25秒"],
        ["相似度强制淘汰", "0.75", "0.75 (不变)"],
        ["目标延迟", "45-65秒", "15-30秒"],
    ]
    elements.append(make_table(compare_headers, compare_rows))
    elements.append(Paragraph("表 8-1: 研究版 vs 生产版对比", caption_style))

    elements.append(h2("8.2 四生成器选择逻辑"))
    elements.append(body(
        "生产版去掉breadth_first和counterfactual两个生成器的原因：在数学问题上，"
        "广度优先枚举策略不贡献额外的正确性（数学答案唯一），反事实策略虽然能打破思维惯性"
        "但在数学问题上收益有限（正确计算不依赖\"打破惯性\"）。保留的四个生成器覆盖了"
        "数学推理的核心认知维度：链式推理(deep_first)、直接推理(direct)、科学严谨推理"
        "(science_deep)、系统化排除(constraint_propagation)。"
    ))

    elements.append(h2("8.3 渐进式深度策略"))
    elements.append(body(
        "MSCE实现了隐式的渐进式深度：低延迟常规生成器（GPT-4o 2s, DeepSeek-Chat 3s）先返回，"
        "高延迟深度生成器（o1 ~25s）后台持续运行。裁判在收集到足够生成器结果后即可开始工作，"
        "无需等待所有生成器完成。但当前实现中裁判等待所有生成器完成后统一运行，"
        "流式裁判（streaming judge）是未来优化方向。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 9. API DESIGN
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("9. API设计"))
    elements.append(hr())

    elements.append(body(
        "MSCE提供Python原生API（当前版本），计划提供REST API封装。"
        "以下为API接口定义："
    ))

    elements.append(h2("9.1 核心API: run_msce()"))
    elements.append(body("<b>研究版 (engine.py):</b>"))
    elements.append(code(
        'def run_msce(question, config=None, domain=None) -> dict:\n'
        '    """\n'
        '    运行MSCE完整流水线。\n'
        '    Args:\n'
        '        question: 问题文本\n'
        '        config: 生成器/裁判配置字典 (默认: DEFAULT_CONFIG)\n'
        '        domain: 领域标识 ("math"/"logic"/"science"/"verbal")\n'
        '    Returns:\n'
        '        {\n'
        '            "question": str,         # 原始问题\n'
        '            "candidates": list,      # 6个生成器的完整输出\n'
        '            "verdict": {             # 裁判裁决\n'
        '                "eliminated": list,  # 被淘汰的候选\n'
        '                "surviving": list,   # 保留的候选\n'
        '                "top3": list,        # 前三名排名\n'
        '            },\n'
        '            "low_confidence": bool,  # 低置信度警告\n'
        '            "timestamp": float       # Unix时间戳\n'
        '        }\n'
        '    """'
    ))

    elements.append(h2("9.2 产品API: run_msce_product()"))
    elements.append(body("<b>生产版 (product_engine.py):</b>"))
    elements.append(code(
        'def run_msce_product(question, config=None, domain="math",\n'
        '                     skip_appeal=True) -> dict:\n'
        '    """\n'
        '    运行MSCE产品流水线。\n'
        '    Returns:\n'
        '        {\n'
        '            "question": str,           # 原始问题\n'
        '            "candidates": list,        # 生成器输出\n'
        '            "verdict": dict,           # 裁判裁决\n'
        '            "confidence": float,       # 置信度 [0-1]\n'
        '            "disagreement": float,     # 分歧度 [0-1]\n'
        '            "reasoning_trail": list,   # 推理路径记录\n'
        '            "low_confidence": bool,    # 低置信度标记\n'
        '            "elapsed_time": float,     # 总耗时(秒)\n'
        '            "timestamp": float         # Unix时间戳\n'
        '        }\n'
        '    """'
    ))

    elements.append(h2("9.3 评判学生答案API: run_msce_product() (教育版)"))
    elements.append(code(
        'def run_msce_product(problem: str, student_answer: str,\n'
        '                     domain: str = "math") -> dict:\n'
        '    """\n'
        '    评判学生答案。先运行MSCE获取参考答案，再对比学生答案。\n'
        '    Returns:\n'
        '        {\n'
        '            "verdict": "correct|incorrect|uncertain",\n'
        '            "confidence": float,       # 评判置信度\n'
        '            "disagreement": float,     # 模型分歧度\n'
        '            "reasoning": str,          # 评判理由\n'
        '            "details": {\n'
        '                "models_agree": int,   # 达成一致的模型数\n'
        '                "models_total": int,   # 总模型数\n'
        '                "top_answer": str,     # 最佳答案策略\n'
        '                "judge_votes": int,    # 裁判投票次数\n'
        '                "time_ms": int         # 总耗时(毫秒)\n'
        '            }\n'
        '        }\n'
        '    """'
    ))

    elements.append(h2("9.4 计划中的REST端点"))
    rest_headers = ["方法", "端点", "描述", "延迟目标"]
    rest_rows = [
        ["POST", "/judge", "单题评判", "15-30s"],
        ["POST", "/batch_judge", "批量评判（最多50题）", "异步, 每题15-30s"],
        ["GET", "/health", "健康检查 + 模型可用性", "<500ms"],
        ["GET", "/result/{id}", "查询批量评判结果", "<200ms"],
    ]
    elements.append(make_table(rest_headers, rest_rows))
    elements.append(Paragraph("表 9-1: 计划中的REST API端点", caption_style))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # 10. BENCHMARK METHODOLOGY
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("10. 基准测试方法"))
    elements.append(hr())

    elements.append(h2("10.1 MMLU基准测试流程"))
    elements.append(body(
        "MMLU（Massive Multitask Language Understanding）是评估大语言模型能力的标准基准。"
        "MSCE的MMLU测试流程如下："
    ))
    steps = [
        "学科选择: 选取5个代表性学科 — abstract_algebra (数学), college_physics (科学), formal_logic (逻辑), college_chemistry (化学), us_foreign_policy (人文)",
        "题目加载: 使用HuggingFace datasets库加载cais/mmlu测试集，每学科取n题（默认10题/学科，共50题）",
        "选项格式化: 将A/B/C/D选项拼接到问题末尾",
        "MSCE运行: 每题运行完整流水线 (6生成器 + 裁判 + 上诉)，自动映射学科到领域",
        "选项提取: 从最佳答案中提取选项字母 (A/B/C/D)，5层正则匹配 + LLM降级提取",
        "评分: 提取的字母与正确答案对比，正确=1，错误=0",
        "基线对比: 同一题目分别用GPT-4o和DeepSeek-Chat单模型作答并评分",
    ]
    for i, step in enumerate(steps, 1):
        elements.append(Paragraph(f"{i}. {step}", bullet_style))

    elements.append(h2("10.2 选项提取算法"))
    elements.append(body(
        "从生成器自由文本回答中精确提取选项字母是一个非平凡问题。MSCE采用五层正则匹配："
    ))
    extract_steps = [
        "匹配 **X** 格式 (加粗标记)",
        "匹配 \"答案是X\" 或 \"答案:X\" 格式",
        "匹配 \"选X\" 或 \"选择X\" 格式",
        "末尾100字符内匹配独立字母(\\b[ABCD]\\b$)",
        "LLM降级提取: 使用DeepSeek-Chat专门提取选项字母",
    ]
    for i, step in enumerate(extract_steps, 1):
        elements.append(Paragraph(f"{i}. {step}", bullet_style))

    elements.append(h2("10.3 对比方法论"))
    elements.append(body(
        "公平对比原则：MSCE与单模型基线使用完全相同的题目、相同的API提供商、"
        "相同的时间窗口进行测试。MSCE每次运行涉及6+次API调用（6生成器 + 裁判 + 可选上诉），"
        "单模型基线每次运行1次API调用。成本（API费用）未被作为评估维度，"
        "仅关注准确率和置信度校准质量。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # APPENDIX: BENCHMARK DATA
    # ═══════════════════════════════════════════════════════════════════
    elements.append(h1("附录: 基准测试数据"))
    elements.append(hr())

    elements.append(h2("A.1 中文数学20题测试"))
    elements.append(body(
        "测试集：20道中文数学竞赛题（涵盖代数、几何、数论）。"
        "对比模型：MSCE v5（6生成器 + 裁判 + 上诉）、GPT-4o单模型、DeepSeek-Chat单模型。"
    ))

    math_headers = ["模型", "正确数", "正确率", "平均耗时", "低置信度标记"]
    math_rows = [
        ["MSCE v5", "19/20", "95.0%", "~45s/题", "1题 (5%)"],
        ["GPT-4o (单模型)", "14/20", "70.0%", "~3s/题", "N/A (盲目自信)"],
        ["DeepSeek-Chat (单模型)", "18/20", "90.0%", "~4s/题", "N/A (盲目自信)"],
    ]
    elements.append(make_table(math_headers, math_rows))
    elements.append(Paragraph("表 A-1: 中文数学20题基准测试结果", caption_style))

    elements.append(h2("A.2 MMLU 30题试点测试"))
    elements.append(body(
        "测试集：5个MMLU学科，每个6题，共30题。"
    ))

    mmlu_headers = ["模型", "正确数", "正确率"]
    mmlu_rows = [
        ["MSCE v5", "26/30", "86.7%"],
        ["GPT-4o (单模型)", "17/30", "56.7%"],
        ["DeepSeek-Chat (单模型)", "25/30", "83.3%"],
    ]
    elements.append(make_table(mmlu_headers, mmlu_rows))
    elements.append(Paragraph("表 A-2: MMLU 30题试点结果总览", caption_style))

    elements.append(h2("A.3 MMLU按学科细分"))
    subj_headers = ["学科", "题目数", "MSCE正确率", "GPT-4o正确率", "DeepSeek正确率"]
    subj_rows = [
        ["abstract_algebra (抽象代数)", "6", "83.3%", "50.0%", "83.3%"],
        ["college_physics (大学物理)", "6", "100.0%", "66.7%", "83.3%"],
        ["formal_logic (形式逻辑)", "6", "83.3%", "50.0%", "83.3%"],
        ["college_chemistry (大学化学)", "6", "83.3%", "50.0%", "83.3%"],
        ["us_foreign_policy (美国外交)", "6", "83.3%", "66.7%", "83.3%"],
    ]
    elements.append(make_table(subj_headers, subj_rows))
    elements.append(Paragraph("表 A-3: MMLU按学科细分结果", caption_style))

    elements.append(h2("A.4 文本图表: 模型准确率对比"))

    # ASCII bar chart
    ascii_chart = """    MMLU 30题 准确率对比
    ═══════════════════════════════════════════════════════════════

    MSCE v5          ████████████████████████████▌ 86.7%
    DeepSeek-Chat    ███████████████████████████▏  83.3%
    GPT-4o           ██████████████████▊           56.7%

    ═══════════════════════════════════════════════════════════════

    中文数学 20题 准确率对比
    ═══════════════════════════════════════════════════════════════

    MSCE v5          ████████████████████████████████▌ 95.0%
    DeepSeek-Chat    ██████████████████████████████▏   90.0%
    GPT-4o           ███████████████████████▍          70.0%

    ═══════════════════════════════════════════════════════════════

    关键洞察:
    - 单模型存在"自信地错误"问题: 错误答案以高置信度输出
    - MSCE在错误时能标记低置信度: "知道什么时候不知道"
    - MSCE在DeepSeek-Chat已很强的任务上仍有提升 (90%→95%, 83.3%→86.7%)
    - MSCE最大增益出现在GPT-4o弱项上 (70%→95%, 56.7%→86.7%)
    - 认知多样性带来的提升 > 单模型能力上限"""
    elements.append(Preformatted(ascii_chart, ParagraphStyle(
        'AsciiChart', fontName=CODE_FONT, fontSize=8, leading=10.5,
        leftIndent=4, rightIndent=4, spaceAfter=10, spaceBefore=6,
        textColor=HexColor('#2d3436'), backColor=HexColor('#f5f6fa'),
        borderPadding=8, borderWidth=0.5, borderColor=BORDER_COLOR,
    ), maxLineLength=120))
    elements.append(Paragraph("图 A-1: 基准测试结果文本可视化", caption_style))

    elements.append(h2("A.5 置信度校准观察"))
    elements.append(body(
        "以下为MSCE在中文数学测试中的置信度模式观察（n=20）："
    ))
    cal_headers = ["置信度区间", "题目数", "实际正确率", "校准质量"]
    cal_rows = [
        [">= 0.9 (高置信)", "15题 (75%)", "100% (15/15)", "完美校准"],
        ["0.7-0.9 (中置信)", "3题 (15%)", "100% (3/3)", "保守（过度自信更危险）"],
        ["< 0.5 (低置信)", "2题 (10%)", "50% (1/2)", "正确标记不确定性"],
    ]
    elements.append(make_table(cal_headers, cal_rows))
    elements.append(Paragraph("表 A-5: 置信度校准分析", caption_style))
    elements.append(body(
        "MSCE的置信度校准优于单模型的关键在于：当系统不确定时（低置信度），"
        "它诚实地说\"不确定\"而非猜测。单模型没有这一机制，对所有问题都以相同的高置信度输出。"
    ))

    elements.append(PageBreak())

    # ═══════════════════════════════════════════════════════════════════
    # FINAL PAGE
    # ═══════════════════════════════════════════════════════════════════
    elements.append(Spacer(1, 80 * mm))
    elements.append(Paragraph("— 文档结束 —", ParagraphStyle(
        'EndNote', fontName='STHeiti', fontSize=16, leading=22,
        alignment=TA_CENTER, textColor=MUTED, spaceAfter=10,
    )))
    elements.append(spacer(10))
    elements.append(Paragraph(
        "MSCE: Multi-model Self-Consistency Engine<br/>"
        "认知对抗引擎 | 约束残差框架 | 不确定性量化<br/>"
        "版本 1.0 | 2026年5月",
        ParagraphStyle('EndMeta', fontName='STHeiti', fontSize=10, leading=16,
                      alignment=TA_CENTER, textColor=MUTED)
    ))

    # ── Build ──
    doc.build(elements, canvasmaker=NumberedCanvas)
    print(f"PDF created: {output_path}")
    return output_path


if __name__ == '__main__':
    path = build_document()
    print(f"Done: {path}")
