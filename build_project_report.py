from __future__ import annotations

import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from PIL import Image as PILImage, ImageOps
from pypdf import PdfReader
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, StyleSheet1, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Flowable,
    Image,
    KeepTogether,
    ListFlowable,
    ListItem,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from reportlab.graphics.shapes import Circle, Drawing, Ellipse, Line, Polygon, Rect, String


ROOT = Path(__file__).resolve().parent
REPORT_ASSETS = ROOT / "report_assets"
REFERENCE_REPORT = Path("/Users/akshatmalakar/Documents/Report_File_pdf.pdf")
OUTPUT_REPORT = ROOT / "Mutual_Fund_Recommendation_Assistant_Project_Report_Updated.pdf"

sys.path.insert(0, str(ROOT))

from src.corpus_manager import corpus_summary, discover_corpus_assets, load_corpus_sources  # noqa: E402
from src.document_pipeline import load_document_chunks, load_document_pages  # noqa: E402
from src.fund_data import load_fund_schemes, latest_archive_snapshot_path  # noqa: E402
from src.recommendation_engine import InvestorProfile, recommend_schemes, strategy_blueprint  # noqa: E402
from src.vector_store import load_or_build_chunk_vector_index  # noqa: E402


@dataclass(frozen=True)
class ReportStats:
    scheme_count: int
    category_count: int
    sub_category_count: int
    amc_count: int
    real_nav_count: int
    synthetic_nav_count: int
    latest_archive_snapshot: str
    configured_source_count: int
    indexed_asset_count: int
    document_page_count: int
    document_chunk_count: int
    vector_dimension: int
    vector_chunk_count: int
    source_type_counts: dict[str, int]
    trust_tier_counts: dict[str, int]
    top_recommendations: list[tuple[str, str, str, int]]
    allocation_blueprint: list[tuple[str, int, str]]


TITLE = "Mutual Fund Recommendation Assistant"
SUBTITLE = "A Project Work-I Report"
DEGREE = "BACHELOR OF TECHNOLOGY IN COMPUTER SCIENCE & ENGINEERING"
AUTHORS = [
    "Akshat Malakar (EN22CS304007)",
    "Aniket Tripathi (EN22CS304011)",
    "Harsh Nandwal (EN22CS304027)",
]
GUIDE = "Prof. Kumar Gaurav"
DEPARTMENT = "Department of Computer Science & Engineering"
FACULTY = "Faculty of Engineering"
UNIVERSITY = "MEDICAPS UNIVERSITY, INDORE- 453331"
REPORT_PERIOD = "January - April 2026"


def gather_stats() -> ReportStats:
    hidden_outputs: list[tuple[Path, Path]] = []
    for path in sorted(ROOT.glob("Mutual_Fund_Recommendation_Assistant_Project_Report*.pdf")):
        hidden_path = path.with_suffix(path.suffix + ".tmphidden")
        if hidden_path.exists():
            hidden_path.unlink()
        path.rename(hidden_path)
        hidden_outputs.append((path, hidden_path))

    try:
        discover_corpus_assets.cache_clear()
        load_document_pages.cache_clear()
        load_document_chunks.cache_clear()
        load_or_build_chunk_vector_index.cache_clear()

        schemes = load_fund_schemes()
        pages = load_document_pages()
        chunks = load_document_chunks()
        vector_index = load_or_build_chunk_vector_index(chunks)
        sources = load_corpus_sources()
        assets = discover_corpus_assets()
        latest_snapshot = latest_archive_snapshot_path() or "Not available"
        profile = InvestorProfile(
            age=29,
            goal="Wealth Creation",
            risk_appetite="Moderate",
            horizon_years=7,
            investment_mode="SIP",
            amount=12000,
            needs_tax_saving=False,
        )
        recommendations = recommend_schemes(profile, schemes=schemes, top_n=5)
        allocation = strategy_blueprint(profile)
        return ReportStats(
            scheme_count=len(schemes),
            category_count=len({item.category for item in schemes}),
            sub_category_count=len({item.sub_category for item in schemes}),
            amc_count=len({item.amc_name for item in schemes}),
            real_nav_count=sum(
                1
                for item in schemes
                if (item.nav_source or "").startswith("master") or (item.nav_source or "").startswith("archive")
            ),
            synthetic_nav_count=sum(1 for item in schemes if item.nav_is_synthetic),
            latest_archive_snapshot=Path(latest_snapshot).name if latest_snapshot else "Not available",
            configured_source_count=sum(1 for item in sources if item.enabled),
            indexed_asset_count=len(assets),
            document_page_count=len(pages),
            document_chunk_count=len(chunks),
            vector_dimension=vector_index.dimension,
            vector_chunk_count=vector_index.chunk_count,
            source_type_counts=dict(Counter(item.source_type for item in assets)),
            trust_tier_counts=dict(Counter(item.trust_tier for item in assets)),
            top_recommendations=[
                (item.scheme_name, item.sub_category, item.fit_label, item.score) for item in recommendations
            ],
            allocation_blueprint=[(item.label, item.allocation_pct, item.rationale) for item in allocation],
        )
    finally:
        for original_path, hidden_path in hidden_outputs:
            if hidden_path.exists():
                hidden_path.rename(original_path)
        discover_corpus_assets.cache_clear()
        load_document_pages.cache_clear()
        load_document_chunks.cache_clear()
        load_or_build_chunk_vector_index.cache_clear()


def run_command(command: list[str]) -> None:
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)


def prepare_logo() -> Path | None:
    if not REFERENCE_REPORT.exists():
        return None

    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)
    preview_path = REPORT_ASSETS / "reference_cover_preview.png"
    logo_path = REPORT_ASSETS / "medicaps_logo.png"

    if not preview_path.exists():
        run_command(["sips", "-s", "format", "png", str(REFERENCE_REPORT), "--out", str(preview_path)])

    image = PILImage.open(preview_path).convert("RGBA")
    cropped = image.crop((200, 425, 435, 625))
    alpha = cropped.getchannel("A")
    padded = ImageOps.expand(cropped, border=16, fill=(255, 255, 255, 0))
    padded.putalpha(ImageOps.expand(alpha, border=16, fill=255))
    padded.save(logo_path)
    return logo_path


def prepare_screenshot_variants() -> dict[str, Path]:
    variants: dict[str, Path] = {}
    REPORT_ASSETS.mkdir(parents=True, exist_ok=True)

    configs = {
        "ui_fold": ("ui_fold.png", None),
        "shortlist": ("shortlist_full.png", None),
        "strategy": ("strategy_full.png", None),
        "assistant": ("assistant_full.png", None),
    }

    for key, (filename, crop_box) in configs.items():
        source_path = REPORT_ASSETS / filename
        target_path = REPORT_ASSETS / f"{key}_framed.png"
        if not source_path.exists():
            continue
        image = PILImage.open(source_path).convert("RGB")
        if crop_box is not None:
            image = image.crop(crop_box)
        image = image.resize((900, int(900 * image.height / image.width)), PILImage.Resampling.LANCZOS)
        framed = ImageOps.expand(image, border=24, fill="white")
        framed = ImageOps.expand(framed, border=1, fill="#b6bcc7")
        framed.save(target_path)
        variants[key] = target_path
    return variants


def make_styles() -> StyleSheet1:
    styles = getSampleStyleSheet()

    styles.add(
        ParagraphStyle(
            name="CoverTitle",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=20,
            leading=24,
            alignment=TA_CENTER,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CoverBody",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=11.5,
            leading=16,
            alignment=TA_CENTER,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FrontHeading",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=15,
            leading=18,
            alignment=TA_CENTER,
            underline=True,
            spaceAfter=16,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextAcademic",
            parent=styles["Normal"],
            fontName="Times-Roman",
            fontSize=11,
            leading=16,
            alignment=TA_JUSTIFY,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="BodyTextLeft",
            parent=styles["BodyTextAcademic"],
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="ChapterTitle",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=16,
            leading=20,
            alignment=TA_LEFT,
            underline=True,
            spaceBefore=6,
            spaceAfter=12,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SectionTitle",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=12.5,
            leading=16,
            alignment=TA_LEFT,
            spaceBefore=10,
            spaceAfter=8,
        )
    )
    styles.add(
        ParagraphStyle(
            name="SubSectionTitle",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=11.5,
            leading=14,
            alignment=TA_LEFT,
            spaceBefore=8,
            spaceAfter=6,
        )
    )
    styles.add(
        ParagraphStyle(
            name="FigureCaption",
            parent=styles["Normal"],
            fontName="Times-Bold",
            fontSize=10.5,
            leading=13,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=10,
        )
    )
    styles.add(
        ParagraphStyle(
            name="TableCaption",
            parent=styles["FigureCaption"],
            alignment=TA_LEFT,
        )
    )
    styles.add(
        ParagraphStyle(
            name="CodeBlock",
            parent=styles["Normal"],
            fontName="Courier",
            fontSize=8.7,
            leading=11,
            leftIndent=6,
            rightIndent=6,
        )
    )
    return styles


class AcademicDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, pagesize=A4, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        template = PageTemplate(id="all-pages", frames=[frame], onPage=self.draw_page_footer)
        self.addPageTemplates([template])

    def draw_page_footer(self, canvas, doc) -> None:
        page_no = canvas.getPageNumber()
        if page_no == 1:
            return
        canvas.saveState()
        canvas.setFont("Times-Roman", 10)
        canvas.drawCentredString(A4[0] / 2, 0.45 * inch, str(page_no))
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if not isinstance(flowable, Paragraph):
            return
        text = flowable.getPlainText()
        page = self.canv.getPageNumber()
        style_name = flowable.style.name

        if style_name == "ChapterTitle":
            key = f"chapter-{page}-{len(text)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (0, text, page, key))
        elif style_name == "SectionTitle":
            key = f"section-{page}-{len(text)}"
            self.canv.bookmarkPage(key)
            self.notify("TOCEntry", (1, text, page, key))
        elif style_name == "SubSectionTitle":
            self.notify("TOCEntry", (2, text, page))
        elif style_name == "FigureCaption":
            self.notify("FIGURE", (0, text, page))
        elif style_name == "TableCaption":
            self.notify("TABLE", (0, text, page))


def toc_flowable(styles: StyleSheet1, notify_kind: str = "TOCEntry") -> TableOfContents:
    toc = TableOfContents()
    toc._notifyKind = notify_kind
    toc.levelStyles = [
        ParagraphStyle(
            name=f"{notify_kind}-level-0",
            fontName="Times-Bold",
            fontSize=11,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
        ),
        ParagraphStyle(
            name=f"{notify_kind}-level-1",
            fontName="Times-Roman",
            fontSize=10.5,
            leading=13,
            leftIndent=18,
            firstLineIndent=0,
        ),
        ParagraphStyle(
            name=f"{notify_kind}-level-2",
            fontName="Times-Roman",
            fontSize=10,
            leading=12,
            leftIndent=36,
            firstLineIndent=0,
        ),
    ]
    return toc


def make_bullets(items: Iterable[str], styles: StyleSheet1) -> ListFlowable:
    return ListFlowable(
        [ListItem(Paragraph(item, styles["BodyTextLeft"])) for item in items],
        bulletType="bullet",
        start="circle",
        leftIndent=18,
    )


def code_box(text: str, styles: StyleSheet1) -> Table:
    block = Preformatted(text.strip(), styles["CodeBlock"])
    table = Table([[block]], colWidths=[6.55 * inch])
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f4f6f9")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#d7dce5")),
                ("LEFTPADDING", (0, 0), (-1, -1), 10),
                ("RIGHTPADDING", (0, 0), (-1, -1), 10),
                ("TOPPADDING", (0, 0), (-1, -1), 10),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ]
        )
    )
    return table


def wrap_text_for_width(text: str, max_width: float, font_name: str, font_size: float) -> list[str]:
    wrapped: list[str] = []
    for block in str(text).split("\n"):
        words = block.split()
        if not words:
            wrapped.append("")
            continue
        current = words[0]
        for word in words[1:]:
            trial = f"{current} {word}"
            if pdfmetrics.stringWidth(trial, font_name, font_size) <= max_width:
                current = trial
            else:
                wrapped.append(current)
                current = word
        wrapped.append(current)
    return wrapped


def draw_wrapped_text(
    drawing: Drawing,
    text: str,
    x: float,
    top_y: float,
    max_width: float,
    *,
    font_name: str,
    font_size: float,
    leading: float,
    align: str = "left",
) -> float:
    lines = wrap_text_for_width(text, max_width, font_name, font_size)
    current_y = top_y
    for line in lines:
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        if align == "center":
            draw_x = x + max((max_width - line_width) / 2, 0)
        elif align == "right":
            draw_x = x + max_width - line_width
        else:
            draw_x = x
        drawing.add(String(draw_x, current_y, line, fontName=font_name, fontSize=font_size))
        current_y -= leading
    return len(lines) * leading


def add_floating_label(
    drawing: Drawing,
    text: str,
    center_x: float,
    center_y: float,
    *,
    max_width: float = 82,
    font_name: str = "Helvetica",
    font_size: float = 7.2,
    leading: float = 8.6,
) -> None:
    lines = wrap_text_for_width(text, max_width - 6, font_name, font_size)
    box_width = min(
        max_width,
        max(pdfmetrics.stringWidth(line, font_name, font_size) for line in lines) + 8,
    )
    box_height = (len(lines) * leading) + 4
    left = center_x - (box_width / 2)
    bottom = center_y - (box_height / 2)
    drawing.add(Rect(left, bottom, box_width, box_height, strokeColor=None, fillColor=colors.white))
    baseline = bottom + box_height - leading
    for line in lines:
        line_width = pdfmetrics.stringWidth(line, font_name, font_size)
        drawing.add(
            String(
                left + (box_width - line_width) / 2,
                baseline,
                line,
                fontName=font_name,
                fontSize=font_size,
            )
        )
        baseline -= leading


def simple_table(
    rows: list[list[str]],
    col_widths: list[float],
    header_rows: int = 1,
    *,
    header_font_size: float = 10,
    body_font_size: float = 10,
) -> Table:
    header_style = ParagraphStyle(
        name="TableHeader",
        fontName="Times-Bold",
        fontSize=header_font_size,
        leading=header_font_size + 2,
        alignment=TA_LEFT,
    )
    body_style = ParagraphStyle(
        name="TableBody",
        fontName="Times-Roman",
        fontSize=body_font_size,
        leading=body_font_size + 2,
        alignment=TA_LEFT,
        wordWrap="CJK",
    )

    rendered_rows: list[list[Flowable | str]] = []
    for row_index, row in enumerate(rows):
        current_style = header_style if row_index < header_rows else body_style
        rendered_row: list[Flowable | str] = []
        for cell in row:
            if isinstance(cell, Flowable):
                rendered_row.append(cell)
            else:
                rendered_row.append(Paragraph(str(cell).replace("\n", "<br/>"), current_style))
        rendered_rows.append(rendered_row)

    table = Table(rendered_rows, colWidths=col_widths, repeatRows=header_rows)
    table.setStyle(
        TableStyle(
            [
                ("GRID", (0, 0), (-1, -1), 0.6, colors.black),
                ("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#f1f1f1")),
                ("ALIGN", (0, 0), (0, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ("RIGHTPADDING", (0, 0), (-1, -1), 6),
                ("TOPPADDING", (0, 0), (-1, -1), 5),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    return table


def scaled_image(path: Path, target_width: float, h_align: str = "CENTER") -> Image:
    image = Image(str(path))
    ratio = image.imageHeight / float(image.imageWidth)
    image.drawWidth = target_width
    image.drawHeight = target_width * ratio
    image.hAlign = h_align
    return image


def add_box(drawing: Drawing, x: float, y: float, w: float, h: float, title: str, lines: list[str]) -> None:
    drawing.add(Rect(x, y, w, h, strokeColor=colors.black, fillColor=colors.white, strokeWidth=1))
    padding = 8
    title_font_size = 10
    title_leading = 11.2
    body_font_size = 8.1
    body_leading = 9.5
    available_width = w - (2 * padding)

    body_lines: list[str] = []
    for line in lines:
        body_lines.extend(wrap_text_for_width(line, available_width, "Helvetica", body_font_size))

    while (
        (len(body_lines) * body_leading) + title_leading + 10 > h
        and body_font_size > 6.7
    ):
        body_font_size -= 0.25
        body_leading = body_font_size + 1.1
        body_lines = []
        for line in lines:
            body_lines.extend(wrap_text_for_width(line, available_width, "Helvetica", body_font_size))

    title_y = y + h - 16
    draw_wrapped_text(
        drawing,
        title,
        x + padding,
        title_y,
        available_width,
        font_name="Helvetica-Bold",
        font_size=title_font_size,
        leading=title_leading,
    )

    current_y = title_y - 15
    for line in body_lines:
        drawing.add(String(x + padding, current_y, line, fontName="Helvetica", fontSize=body_font_size))
        current_y -= body_leading


def add_centered_box(
    drawing: Drawing,
    x: float,
    y: float,
    w: float,
    h: float,
    text: str,
    *,
    font_size: float = 8.8,
) -> None:
    drawing.add(Rect(x, y, w, h, strokeColor=colors.black, fillColor=colors.white, strokeWidth=1))
    lines = wrap_text_for_width(text, w - 10, "Helvetica", font_size)
    leading = font_size + 1.6
    total_height = len(lines) * leading
    baseline = y + ((h + total_height) / 2) - leading + 2
    for line in lines:
        line_width = pdfmetrics.stringWidth(line, "Helvetica", font_size)
        drawing.add(
            String(
                x + max((w - line_width) / 2, 4),
                baseline,
                line,
                fontName="Helvetica",
                fontSize=font_size,
            )
        )
        baseline -= leading


def add_arrow(
    drawing: Drawing,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    label: str | None = None,
    *,
    label_dx: float = 0,
    label_dy: float = 0,
    label_max_width: float = 82,
) -> None:
    drawing.add(Line(x1, y1, x2, y2, strokeColor=colors.black, strokeWidth=1))
    dx = x2 - x1
    dy = y2 - y1
    if dx == 0 and dy == 0:
        return
    length = (dx**2 + dy**2) ** 0.5
    ux, uy = dx / length, dy / length
    left_x = x2 - (ux * 10) - (uy * 4)
    left_y = y2 - (uy * 10) + (ux * 4)
    right_x = x2 - (ux * 10) + (uy * 4)
    right_y = y2 - (uy * 10) - (ux * 4)
    drawing.add(Polygon([x2, y2, left_x, left_y, right_x, right_y], strokeColor=colors.black, fillColor=colors.black))
    if label:
        add_floating_label(
            drawing,
            label,
            ((x1 + x2) / 2) + label_dx,
            ((y1 + y2) / 2) + label_dy,
            max_width=label_max_width,
        )


def architecture_diagram() -> Drawing:
    drawing = Drawing(470, 300)
    add_box(drawing, 150, 235, 170, 45, "Investor Interface", ["Profile form", "Shortlist, strategy and assistant"])
    add_box(drawing, 140, 170, 190, 45, "Application Layer", ["app.py orchestration", "Session and presentation logic"])
    add_box(
        drawing,
        65,
        100,
        145,
        45,
        "Recommendation Engine",
        ["Scoring and diversification", "Allocation blueprint"],
    )
    add_box(
        drawing,
        255,
        100,
        150,
        45,
        "Grounded Assistant",
        ["Intent routing", "Answer synthesis", "Citation bundle"],
    )
    add_box(drawing, 40, 30, 130, 45, "Scheme Data Layer", ["Raw CSVs", "Canonical dataset"])
    add_box(drawing, 190, 30, 120, 45, "Document Layer", ["PDF / markdown extraction", "Chunk generation"])
    add_box(drawing, 330, 30, 110, 45, "Vector Index", ["Hashed TF-IDF vectors", "Hybrid retrieval"])
    add_arrow(drawing, 235, 235, 235, 215)
    add_arrow(drawing, 190, 170, 140, 145)
    add_arrow(drawing, 280, 170, 330, 145)
    add_arrow(drawing, 105, 100, 105, 75)
    add_arrow(drawing, 250, 100, 250, 75)
    add_arrow(drawing, 350, 100, 385, 75)
    add_arrow(drawing, 170, 52, 190, 52, "context")
    add_arrow(drawing, 310, 52, 330, 52, "vectors")
    return drawing


def dfd_level0_diagram() -> Drawing:
    drawing = Drawing(470, 260)
    drawing.add(Rect(160, 100, 150, 55, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(175, 132, "Mutual Fund Recommendation", fontName="Helvetica-Bold", fontSize=11))
    drawing.add(String(210, 118, "Assistant", fontName="Helvetica-Bold", fontSize=11))
    drawing.add(Rect(40, 110, 95, 40, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(70, 132, "Investor", fontName="Helvetica-Bold", fontSize=11))
    drawing.add(Rect(340, 170, 90, 38, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(350, 190, "Scheme Data", fontName="Helvetica-Bold", fontSize=10))
    drawing.add(Rect(340, 115, 90, 38, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(350, 135, "Knowledge Corpus", fontName="Helvetica-Bold", fontSize=9))
    drawing.add(Rect(340, 60, 90, 38, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(352, 80, "Vector Store", fontName="Helvetica-Bold", fontSize=10))
    add_arrow(drawing, 135, 130, 160, 130, "profile")
    add_arrow(drawing, 310, 130, 340, 189, "shortlist data")
    add_arrow(drawing, 310, 126, 340, 134, "documents")
    add_arrow(drawing, 310, 122, 340, 79, "retrieval")
    add_arrow(drawing, 160, 118, 135, 118, "recommendations")
    return drawing


def dfd_level1_diagram() -> Drawing:
    drawing = Drawing(470, 280)
    add_box(drawing, 30, 170, 90, 42, "Profile Module", ["Validate inputs", "Build investor profile"])
    add_box(drawing, 150, 170, 95, 42, "Scoring Module", ["Risk/goal/horizon", "Top-N shortlist"])
    add_box(drawing, 275, 170, 95, 42, "Strategy Module", ["Allocation mix", "Action checklist"])
    add_box(drawing, 150, 95, 95, 42, "Corpus Module", ["Load pages/chunks", "Source metadata"])
    add_box(drawing, 275, 95, 95, 42, "Retrieval Module", ["Semantic + hybrid search", "Chunk ranking"])
    add_box(drawing, 210, 20, 125, 42, "Answer Module", ["Citations", "Advisor/Auditor/Researcher"])
    drawing.add(Ellipse(5, 176, 18, 18, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(2, 160, "Investor", fontName="Helvetica", fontSize=8))
    drawing.add(Ellipse(420, 122, 28, 18, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(412, 106, "Corpus", fontName="Helvetica", fontSize=8))
    drawing.add(Ellipse(420, 47, 28, 18, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(404, 31, "UI answer", fontName="Helvetica", fontSize=8))
    add_arrow(drawing, 23, 185, 30, 185)
    add_arrow(drawing, 120, 190, 150, 190)
    add_arrow(drawing, 245, 190, 275, 190)
    add_arrow(drawing, 198, 170, 198, 137)
    add_arrow(drawing, 370, 116, 420, 131)
    add_arrow(drawing, 245, 116, 275, 116)
    add_arrow(drawing, 322, 95, 322, 62)
    add_arrow(drawing, 335, 40, 420, 55)
    add_arrow(drawing, 245, 41, 275, 110, "evidence", label_dx=12, label_dy=12, label_max_width=54)
    return drawing


def use_case_diagram() -> Drawing:
    drawing = Drawing(470, 250)
    drawing.add(String(24, 210, "Investor", fontName="Helvetica-Bold", fontSize=11))
    drawing.add(Circle(42, 180, 10, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(Line(42, 170, 42, 145))
    drawing.add(Line(42, 160, 28, 152))
    drawing.add(Line(42, 160, 56, 152))
    drawing.add(Line(42, 145, 30, 130))
    drawing.add(Line(42, 145, 54, 130))
    drawing.add(Rect(110, 35, 320, 185, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(String(188, 205, "Mutual Fund Recommendation Assistant", fontName="Helvetica-Bold", fontSize=11))
    use_cases = [
        ("Set profile", 180),
        ("Generate shortlist", 150),
        ("Review strategy", 120),
        ("Ask grounded question", 90),
        ("Inspect sources", 60),
    ]
    for label, y in use_cases:
        drawing.add(Ellipse(190, y - 10, 165, 26, strokeColor=colors.black, fillColor=colors.white))
        drawing.add(String(235, y + 1, label, fontName="Helvetica", fontSize=10))
        add_arrow(drawing, 52, 175, 190, y + 3)
    return drawing


def activity_diagram() -> Drawing:
    drawing = Drawing(470, 340)
    drawing.add(Circle(235, 300, 10, strokeColor=colors.black, fillColor=colors.black))
    activity_boxes = [
        (155, 255, "Choose preset or enter profile"),
        (155, 210, "Validate age, goal, horizon and amount"),
        (155, 165, "Score schemes and diversify shortlist"),
        (155, 120, "Review strategy and caution notes"),
        (155, 75, "Ask grounded question in selected lens"),
        (155, 30, "Show answer, citations and next actions"),
    ]
    for x, y, text in activity_boxes:
        add_centered_box(drawing, x, y, 160, 28, text, font_size=8.6)
    drawing.add(Circle(235, 8, 10, strokeColor=colors.black, fillColor=colors.white))
    drawing.add(Circle(235, 8, 5, strokeColor=colors.black, fillColor=colors.black))
    for _, y, _ in activity_boxes[:-1]:
        add_arrow(drawing, 235, y, 235, y - 17)
    add_arrow(drawing, 235, 290, 235, 283)
    add_arrow(drawing, 235, 30, 235, 18)
    add_arrow(drawing, 315, 224, 352, 224, "If invalid, show warning", label_dx=40, label_dy=-2, label_max_width=94)
    return drawing


def sequence_diagram() -> Drawing:
    drawing = Drawing(470, 300)
    actors = [
        ("Investor", 36, 68),
        ("Streamlit UI", 126, 78),
        ("Recommendation Engine", 226, 86),
        ("Document Layer", 328, 76),
        ("RAG Engine", 418, 70),
    ]
    lifeline_x: list[float] = []
    for label, x, width in actors:
        add_centered_box(drawing, x, 245, width, 26, label, font_size=7.6)
        center_x = x + (width / 2)
        lifeline_x.append(center_x)
        drawing.add(Line(center_x, 25, center_x, 245, strokeColor=colors.HexColor("#444444"), strokeDashArray=[3, 2]))
    steps = [
        (lifeline_x[0], lifeline_x[1], 210, "submit profile"),
        (lifeline_x[1], lifeline_x[2], 180, "load schemes"),
        (lifeline_x[2], lifeline_x[1], 150, "top picks"),
        (lifeline_x[1], lifeline_x[3], 120, "search docs"),
        (lifeline_x[3], lifeline_x[4], 90, "rank chunks"),
        (lifeline_x[4], lifeline_x[1], 60, "grounded answer"),
        (lifeline_x[1], lifeline_x[0], 30, "results"),
    ]
    for x1, x2, y, label in steps:
        add_arrow(drawing, x1, y, x2, y, label, label_dy=10, label_max_width=74)
    return drawing


def class_diagram() -> Drawing:
    drawing = Drawing(470, 310)
    add_box(drawing, 16, 188, 128, 92, "InvestorProfile", ["+ age", "+ goal", "+ risk_appetite", "+ horizon_years"])
    add_box(drawing, 170, 188, 132, 92, "Recommendation", ["+ scheme_name", "+ fit_label", "+ score", "+ caution"])
    add_box(drawing, 328, 188, 126, 92, "RagAnswer", ["+ summary", "+ citations", "+ confidence_label"])
    add_box(drawing, 16, 58, 128, 92, "DocumentChunk", ["+ chunk_id", "+ page_numbers", "+ source_label"])
    add_box(drawing, 170, 58, 132, 92, "ChunkVectorIndex", ["+ dimension", "+ chunk_count", "+ records"])
    add_box(drawing, 328, 58, 126, 92, "CorpusSource", ["+ source_id", "+ trust_tier", "+ base_url"])
    add_arrow(drawing, 144, 233, 170, 233, "used by", label_dy=14, label_max_width=48)
    add_arrow(drawing, 302, 233, 328, 233, "supports", label_dy=14, label_max_width=52)
    add_arrow(drawing, 80, 150, 80, 188, "chunked", label_dx=-26, label_dy=6, label_max_width=46)
    add_arrow(drawing, 144, 104, 170, 104, "embedded into", label_dy=14, label_max_width=70)
    add_arrow(drawing, 302, 104, 328, 104, "metadata", label_dy=14, label_max_width=52)
    add_arrow(drawing, 236, 150, 236, 188, "retrieval", label_dx=26, label_dy=6, label_max_width=48)
    return drawing


def er_diagram() -> Drawing:
    drawing = Drawing(470, 320)
    add_box(drawing, 16, 224, 136, 60, "RawSchemeDataset", ["scheme_name", "returns, fees, min amounts"])
    add_box(drawing, 168, 224, 144, 60, "MasterSchemeRecord", ["scheme_code", "NAV, AUM, launch date"])
    add_box(drawing, 336, 224, 118, 60, "ArchiveNavRecord", ["daily NAV snapshot"])
    add_box(drawing, 92, 110, 162, 68, "CanonicalFundDataset", ["enriched scheme rows", "nav_source, confidence"])
    add_box(drawing, 282, 110, 170, 68, "DocumentChunk / VectorRecord", ["chunk text", "source metadata", "vector"])
    add_box(drawing, 180, 20, 130, 52, "CorpusSource", ["trust tier, domain"])
    add_arrow(drawing, 84, 224, 158, 178, "join", label_dx=-8, label_dy=12, label_max_width=40)
    add_arrow(drawing, 240, 224, 192, 178, "match", label_dx=8, label_dy=12, label_max_width=42)
    add_arrow(drawing, 392, 224, 228, 178, "backfill", label_dx=18, label_dy=16, label_max_width=52)
    add_arrow(drawing, 254, 144, 282, 144, "scheme context", label_dy=14, label_max_width=74)
    add_arrow(drawing, 245, 72, 245, 110, "governs", label_dx=-26, label_dy=8, label_max_width=48)
    add_arrow(drawing, 310, 72, 366, 110, "source metadata", label_dx=10, label_dy=14, label_max_width=76)
    return drawing


def component_diagram() -> Drawing:
    drawing = Drawing(470, 270)
    add_box(drawing, 30, 185, 110, 45, "Presentation", ["Streamlit app", "Tabs and cards"])
    add_box(drawing, 180, 185, 120, 45, "Core Services", ["recommendation_engine", "rag_engine"])
    add_box(drawing, 340, 185, 100, 45, "Storage", ["CSV and JSON files"])
    add_box(drawing, 30, 105, 110, 45, "Data Pipeline", ["fund_data", "canonical export"])
    add_box(drawing, 180, 105, 120, 45, "Corpus Pipeline", ["document_pipeline", "corpus_manager"])
    add_box(drawing, 340, 105, 100, 45, "Retrieval", ["vector_store", "hybrid search"])
    add_box(drawing, 180, 25, 120, 45, "Explanation Layer", ["answer modes", "citations"])
    add_arrow(drawing, 140, 207, 180, 207)
    add_arrow(drawing, 300, 207, 340, 207)
    add_arrow(drawing, 85, 185, 85, 150)
    add_arrow(drawing, 240, 185, 240, 150)
    add_arrow(drawing, 390, 185, 390, 150)
    add_arrow(drawing, 240, 105, 240, 70)
    add_arrow(drawing, 300, 127, 340, 127)
    add_arrow(drawing, 140, 127, 180, 127)
    return drawing


def state_diagram() -> Drawing:
    drawing = Drawing(470, 190)
    states = [
        ("Idle", 18, 116, 86),
        ("Profile complete", 126, 116, 96),
        ("Shortlist ready", 246, 116, 94),
        ("Strategy reviewed", 362, 116, 94),
        ("Assistant active", 180, 42, 118),
    ]
    for label, x, y, width in states:
        add_centered_box(drawing, x, y, width, 28, label, font_size=8.5)
    add_arrow(drawing, 104, 130, 126, 130, "inputs", label_dy=12, label_max_width=42)
    add_arrow(drawing, 222, 130, 246, 130, "score", label_dy=12, label_max_width=42)
    add_arrow(drawing, 340, 130, 362, 130, "plan", label_dy=12, label_max_width=42)
    add_arrow(drawing, 293, 116, 248, 70, "ask", label_dx=10, label_dy=4, label_max_width=34)
    add_arrow(drawing, 221, 70, 126, 116, "reset / new profile", label_dx=-14, label_dy=14, label_max_width=88)
    return drawing


def figure_block(drawing: Drawing | None, image_path: Path | None, caption: str, styles: StyleSheet1, image_width: float | None = None) -> KeepTogether:
    items = []
    if drawing is not None:
        items.append(drawing)
    elif image_path is not None:
        items.append(scaled_image(image_path, image_width or 4.8 * inch))
    items.append(Paragraph(caption, styles["FigureCaption"]))
    return KeepTogether(items)


def cover_page(story: list, styles: StyleSheet1, logo_path: Path | None) -> None:
    story.append(Spacer(1, 0.55 * inch))
    story.append(Paragraph(TITLE, styles["CoverTitle"]))
    story.append(Spacer(1, 0.32 * inch))
    story.append(Paragraph(SUBTITLE, styles["CoverBody"]))
    story.append(Paragraph("Submitted in partial fulfillment of requirement of the", styles["CoverBody"]))
    story.append(Paragraph("Degree of", styles["CoverBody"]))
    story.append(Paragraph(DEGREE, styles["CoverTitle"]))
    story.append(Spacer(1, 0.08 * inch))
    story.append(Paragraph("BY", styles["CoverBody"]))
    for author in AUTHORS:
        story.append(Paragraph(author, styles["CoverBody"]))
    story.append(Spacer(1, 0.16 * inch))
    story.append(Paragraph("Under the Guidance of", styles["CoverBody"]))
    story.append(Paragraph(f"<b>{GUIDE}</b>", styles["CoverBody"]))
    if logo_path and logo_path.exists():
        story.append(Spacer(1, 0.08 * inch))
        story.append(scaled_image(logo_path, 2.1 * inch))
    story.append(Spacer(1, 0.18 * inch))
    story.append(Paragraph(DEPARTMENT, styles["CoverBody"]))
    story.append(Paragraph(FACULTY, styles["CoverBody"]))
    story.append(Paragraph(UNIVERSITY, styles["CoverBody"]))
    story.append(Spacer(1, 0.14 * inch))
    story.append(Paragraph(REPORT_PERIOD, styles["CoverBody"]))
    story.append(PageBreak())


def front_matter(story: list, styles: StyleSheet1, stats: ReportStats) -> None:
    story.append(Paragraph("Report Approval", styles["FrontHeading"]))
    story.append(
        Paragraph(
            f'The project work "<b>{TITLE}</b>" is hereby approved as a creditable study of an '
            "engineering/computer application subject carried out and presented in a manner "
            "satisfactory to warrant its acceptance as prerequisite for the degree for which it "
            "has been submitted.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            'It is to be understood that by this approval the undersigned do not endorse any statement '
            'made, opinion expressed, or conclusion drawn therein; but approve the "Project Report" '
            "only for the purpose for which it has been submitted.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Spacer(1, 1.2 * inch))
    story.append(Paragraph("Internal Examiner<br/>Name:<br/>Designation<br/>Affiliation", styles["BodyTextLeft"]))
    story.append(Spacer(1, 1.0 * inch))
    story.append(Paragraph("External Examiner<br/>Name:<br/>Designation<br/>Affiliation", styles["BodyTextLeft"]))
    story.append(PageBreak())

    story.append(Paragraph("Declaration", styles["FrontHeading"]))
    story.append(
        Paragraph(
            f'I/We hereby declare that the project entitled "<b>{TITLE}</b>" submitted in partial fulfillment '
            "for the award of the degree of Bachelor of Technology in Computer Science & Engineering is an "
            "authentic record of our own work carried out under the supervision of the guide named in this report. "
            "The work embodied in this report has not been submitted elsewhere for the award of any other degree, "
            "diploma, fellowship, or similar distinction.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Spacer(1, 0.7 * inch))
    for author in AUTHORS:
        story.append(Paragraph(author, styles["BodyTextLeft"]))
    story.append(PageBreak())

    story.append(Paragraph("Certificate", styles["FrontHeading"]))
    story.append(
        Paragraph(
            f"I, {GUIDE}, certify that the project entitled <b>{TITLE}</b> submitted in partial fulfillment "
            "for the award of the degree of Bachelor of Technology by the above-named students has been "
            "carried out under my supervision in the Department of Computer Science & Engineering, Medicaps University.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "To the best of my knowledge, the matter embodied in this report is genuine and has been prepared "
            "according to the expected academic standards for project documentation.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Spacer(1, 1.4 * inch))
    story.append(Paragraph(f"Project Guide<br/>{GUIDE}", styles["BodyTextLeft"]))
    story.append(PageBreak())

    story.append(Paragraph("Acknowledgement", styles["FrontHeading"]))
    story.append(
        Paragraph(
            "We express our sincere gratitude to the university administration, the Department of Computer Science & "
            "Engineering, and our guide for their academic support during the development of this project. The project "
            "benefited from continuous guidance in system design, presentation clarity, and evaluation-oriented thinking.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "We are equally thankful for the availability of the local datasets, documentation, and institutional resources "
            "that enabled us to convert the project from a simple shortlist prototype into a grounded recommendation assistant "
            "with explainable outputs, corpus governance, and modular RAG-ready design.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Abstract", styles["FrontHeading"]))
    story.append(
        Paragraph(
            f"The {TITLE} is an explainable AI project designed to help investors discover suitable mutual fund schemes "
            "based on their goals, risk appetite, investment horizon, and preferred investment mode. The current implementation "
            f"works over {stats.scheme_count} schemes across {stats.sub_category_count} sub-categories and combines rule-based "
            "scheme ranking with document retrieval and grounded answer generation. Rather than behaving like a black-box chatbot, "
            "the system surfaces shortlist reasoning, caution notes, allocation guidance, and citation-backed assistant responses.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "The project integrates a recommendation engine, a preprocessing and NAV-completion pipeline, a document ingestion layer, "
            "a deterministic vector index, and a citation-aware RAG assistant with advisor, auditor, and researcher modes. It is "
            "implemented using Python and Streamlit and is intentionally lightweight enough to run in an offline-friendly academic setup.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "The report documents the problem context, requirements, algorithms, design diagrams, implementation choices, screenshots, "
            "observed outputs, and future expansion areas. The resulting system is positioned as an educational decision-support tool, "
            "not as a replacement for licensed financial advice.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(PageBreak())

    story.append(Paragraph("Keywords", styles["FrontHeading"]))
    story.append(
        Paragraph(
            "Mutual Fund Recommendation, Explainable AI, Streamlit Dashboard, Retrieval-Augmented Generation, "
            "Hashed TF-IDF Embeddings, Vector Retrieval, Corpus Governance, Investor Profiling",
            styles["BodyTextAcademic"],
        )
    )
    story.append(PageBreak())


def toc_and_indexes(story: list, styles: StyleSheet1) -> None:
    story.append(Paragraph("Table of Contents", styles["FrontHeading"]))
    story.append(toc_flowable(styles, "TOCEntry"))
    story.append(PageBreak())

    story.append(Paragraph("List of Figures", styles["FrontHeading"]))
    story.append(toc_flowable(styles, "FIGURE"))
    story.append(PageBreak())

    story.append(Paragraph("List of Tables", styles["FrontHeading"]))
    story.append(toc_flowable(styles, "TABLE"))
    story.append(PageBreak())

    story.append(Paragraph("Abbreviations and Notations", styles["FrontHeading"]))
    abbreviations = [
        "AI - Artificial Intelligence",
        "AUM - Assets Under Management",
        "AMC - Asset Management Company",
        "CSV - Comma Separated Values",
        "DFD - Data Flow Diagram",
        "ELSS - Equity Linked Savings Scheme",
        "LLM - Large Language Model",
        "NAV - Net Asset Value",
        "RAG - Retrieval-Augmented Generation",
        "SIP - Systematic Investment Plan",
        "UI - User Interface",
    ]
    story.append(make_bullets(abbreviations, styles))
    story.append(PageBreak())


def chapter_one(story: list, styles: StyleSheet1, stats: ReportStats) -> None:
    story.append(Paragraph("Chapter 1: Introduction", styles["ChapterTitle"]))
    story.append(Paragraph("1.1 Introduction", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Mutual fund selection is a practical but cognitively difficult problem for everyday investors. "
            "A typical investor must evaluate return history, category risk, expense ratio, fund age, asset "
            "allocation, suitability, and minimum investment thresholds while also keeping personal goals and "
            "risk comfort in view. Many novice investors respond by over-weighting recent returns or by relying "
            "on generic category advice that lacks transparency.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            f"This project addresses that problem through a profile-aware recommendation assistant that works with "
            f"{stats.scheme_count} local scheme records, enriched NAV and AUM data, and a small but structured knowledge corpus. "
            "The goal is not to predict market outcomes, but to make shortlist generation, reasoning, and follow-up exploration "
            "more understandable and auditable.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("1.2 Domain Review and Project Context", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Investor education material from AMFI, SEBI, Investor.gov, and RBI consistently emphasizes diversification, "
            "goal-based allocation, risk disclosure, and document verification before investment decisions. The project aligns "
            "with that educational framing. Instead of suggesting that one fund is universally best, the system narrows options "
            "based on suitability conditions and then supports the user with evidence-led prompts and caution notes.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "The project also reflects a broader pattern in applied AI systems: structured business logic is used for high-confidence "
            "ranking tasks, while language generation is constrained by retrieved evidence to reduce unsupported claims. This makes the "
            "system more appropriate for finance-oriented academic demos than a free-form chatbot.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("1.3 Objectives", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "To capture investor profile inputs such as age, goal, risk appetite, horizon, investment mode, amount, and tax intent.",
                "To generate a shortlist of suitable mutual fund schemes with transparent fit scores and plain-language reasoning.",
                "To enrich scheme-level records with conservative NAV, AUM, and master-data linkage.",
                "To build a searchable knowledge corpus and vector retrieval layer for grounded investor guidance.",
                "To expose the system through an interactive Streamlit interface suitable for academic presentation and demonstration.",
            ],
            styles,
        )
    )

    story.append(Paragraph("1.4 Significance", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The significance of this project lies in its emphasis on explainability and system design discipline. The assistant does "
            "not simply rank schemes; it explains the basis of that ranking, preserves information-quality caveats, differentiates between "
            "real and synthetic NAV coverage, and uses citations when answering follow-up questions. That combination makes the project "
            "educationally stronger and closer to a serious decision-support tool.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("1.5 Research Design and Scope", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The work follows an iterative project design. The first phase establishes the problem statement and requirements. The second "
            "phase builds structured recommendation logic. The third and fourth phases strengthen data quality and coverage. The fifth phase "
            "adds document ingestion, local embeddings, trust-aware retrieval, and grounded answer generation. The current scope remains a "
            "decision-support prototype and intentionally avoids live execution features such as direct brokerage integration or automatic investing.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("1.6 Source of Data", styles["SectionTitle"]))
    story.append(
        Paragraph(
            f"The project uses two principal local scheme datasets, an archive folder of historical NAV snapshots, and a managed corpus of "
            f"{stats.indexed_asset_count} document assets. The raw scheme dataset contains {stats.scheme_count} schemes, while the corpus layer "
            f"currently exposes {stats.document_chunk_count} searchable chunks after extraction and overlap-based chunking.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("1.7 Chapter Scheme", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Chapter 1 introduces the domain and project goals. Chapter 2 formalizes the requirements. Chapter 3 explains the algorithms, "
            "design, and diagrams. Chapter 4 discusses implementation, tools, and testing. Chapter 5 presents module behavior, screenshots, "
            "and discussion. Chapters 6 and 7 conclude the report and outline future scope, followed by an appendix and references.",
            styles["BodyTextAcademic"],
        )
    )


def chapter_two(story: list, styles: StyleSheet1) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 2: Requirement Specifications", styles["ChapterTitle"]))
    story.append(Paragraph("2.1 User Characteristics", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The primary user is a retail investor who needs a guided starting point rather than raw market data. Typical users include beginners "
            "comparing fund categories for the first time, young professionals planning long-horizon SIPs, and academically minded evaluators who "
            "want to inspect how the recommendation logic and assistant answers are grounded.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("2.2 Functional Requirements", styles["SectionTitle"]))
    story.append(Paragraph("Table 2.1 Functional Requirements Specification", styles["TableCaption"]))
    story.append(
        simple_table(
            [
                ["ID", "Requirement"],
                ["FR1", "Capture investor inputs including age, goal, risk appetite, horizon, amount, mode, and tax preference."],
                ["FR2", "Generate a ranked shortlist of suitable mutual fund schemes."],
                ["FR3", "Explain why a fund fits and what the user should double-check before investing."],
                ["FR4", "Provide a strategy blueprint that groups the shortlist into practical allocation roles."],
                ["FR5", "Extract text from local PDF and markdown sources and convert them into searchable chunks."],
                ["FR6", "Build semantic and hybrid retrieval over document chunks for grounded question answering."],
                ["FR7", "Offer advisor, auditor, and researcher answer lenses inside the same UI."],
                ["FR8", "Persist enriched datasets and vector index files for reuse across sessions."],
            ],
            [0.6 * inch, 5.9 * inch],
        )
    )

    story.append(Paragraph("2.3 Non-functional Requirements", styles["SectionTitle"]))
    story.append(Paragraph("Table 2.2 Non-Functional Requirements", styles["TableCaption"]))
    story.append(
        simple_table(
            [
                ["Type", "Expectation"],
                ["Usability", "The UI should remain approachable for non-expert investors and demo audiences."],
                ["Explainability", "Every recommendation and answer should expose reasons, cautions, or source context."],
                ["Determinism", "The current local pipeline should run without depending on live API access."],
                ["Maintainability", "Core logic should remain separated into modules for data, retrieval, recommendation, and UI."],
                ["Auditability", "Generated answers should retain citations and confidence labels."],
                ["Performance", "Typical local interactions should feel responsive on a lightweight academic workstation."],
            ],
            [1.2 * inch, 5.3 * inch],
        )
    )

    story.append(Paragraph("2.4 Dependencies", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "Python 3 runtime environment",
                "Streamlit for the dashboard interface",
                "pypdf for document text extraction",
                "Local file-system access for datasets, snapshots, and vector index storage",
                "A browser for launching the Streamlit application",
            ],
            styles,
        )
    )

    story.append(Paragraph("2.5 Performance Requirements", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Because the local dataset remains moderate in size, the system is designed to keep profile-based shortlist generation and grounded "
            "assistant responses within a practical interactive range. The current corpus is intentionally compact, which helps the assistant "
            "remain responsive while still demonstrating full RAG structure.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("2.6 Hardware Requirements", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "4 GB RAM minimum; 8 GB recommended for smoother local experimentation",
                "Dual-core processor or better",
                "At least 2 GB free disk space for datasets, vector index files, and report artifacts",
                "Standard display resolution for Streamlit interaction",
            ],
            styles,
        )
    )

    story.append(Paragraph("2.7 Software Requirements", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "macOS, Linux, or Windows with Python installed",
                "Streamlit and pypdf as defined in requirements.txt",
                "A modern browser for localhost access",
                "Optional document tooling for future report-export or office workflows",
            ],
            styles,
        )
    )

    story.append(Paragraph("2.8 Constraints and Assumptions", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "The system is educational and must not be positioned as licensed financial advice.",
                "The current prototype does not depend on a live external LLM or live market feed.",
                "Synthetic NAV completion is used for continuity but must remain visibly distinguished from official values.",
                "Recommendations become stronger when the user verifies the latest scheme documents before any real decision.",
            ],
            styles,
        )
    )


def chapter_three(story: list, styles: StyleSheet1, stats: ReportStats) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 3: Design", styles["ChapterTitle"]))
    story.append(Paragraph("3.1 Algorithm Design", styles["SectionTitle"]))

    story.append(Paragraph("3.1.1 Profile-Based Recommendation Algorithm", styles["SubSectionTitle"]))
    story.append(
        code_box(
            """
Algorithm: BuildShortlist
Input: age, goal, risk_appetite, horizon_years, amount, mode, tax_preference
Output: ranked_recommendations

1. Load scheme catalog from local dataset
2. For each scheme in catalog:
      a. compute risk alignment score
      b. compute goal alignment score
      c. compute horizon alignment score
      d. compute affordability and quality score
      e. apply tax adjustment and style penalty
      f. construct why-it-fits and caution text
3. Sort schemes by total score, rating, and long-horizon return signals
4. Diversify the shortlist across sub-categories
5. Return top-N recommendations with explainability fields
            """,
            styles,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3.1.2 Document Ingestion and Chunking Algorithm", styles["SubSectionTitle"]))
    story.append(
        code_box(
            """
Algorithm: BuildCorpusChunks
Input: PDF files, markdown files, HTML/text assets
Output: document_chunks

1. Discover assets from project docs and source registry folders
2. Extract page-level text using pypdf for PDF and direct reads for text assets
3. Normalize whitespace and strip lightweight markup noise
4. Split each page into overlapping word chunks
5. Attach source metadata such as trust tier, source type, title, and page numbers
6. Return chunk catalog for lexical and vector retrieval
            """,
            styles,
        )
    )
    story.append(Spacer(1, 0.1 * inch))

    story.append(Paragraph("3.1.3 Grounded Retrieval and Answer Algorithm", styles["SubSectionTitle"]))
    story.append(
        code_box(
            """
Algorithm: BuildGroundedAnswer
Input: investor_question, investor_profile, shortlist, vector_index, corpus_chunks, assistant_mode
Output: answer_bundle

1. Classify query intent (fit, compare, risk, tax, strategy, or data quality)
2. Select related shortlisted schemes from question overlap
3. Build retrieval query using question + profile + scheme anchors
4. Perform semantic or hybrid retrieval over chunk vectors
5. Build citations and source cards from matched chunks
6. Synthesize answer, key points, watch-outs, follow-up prompts, and confidence label
7. Return answer bundle to the Streamlit UI
            """,
            styles,
        )
    )

    story.append(Paragraph("3.2 System Design", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The project is structured as a modular Streamlit application. User-facing interaction is handled in the UI layer, while "
            "scheme scoring, corpus ingestion, vector retrieval, and answer generation remain separated into dedicated Python modules. "
            "This improves maintainability and also makes the design easier to explain during academic evaluation.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(figure_block(architecture_diagram(), None, "Figure 3.1 System Architecture Diagram", styles))

    story.append(Paragraph("3.2.1 Data Flow Diagram", styles["SubSectionTitle"]))
    story.append(
        Paragraph(
            "The level-0 data flow view treats the application as one coherent decision-support system. The level-1 view expands the same flow "
            "into profile creation, shortlist generation, strategy building, corpus retrieval, and answer composition modules.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(figure_block(dfd_level0_diagram(), None, "Figure 3.2 Data Flow Diagram - Level 0", styles))
    story.append(figure_block(dfd_level1_diagram(), None, "Figure 3.3 Data Flow Diagram - Level 1", styles))

    story.append(Paragraph("3.2.2 Use Case Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(use_case_diagram(), None, "Figure 3.4 Use Case Diagram", styles))

    story.append(Paragraph("3.2.3 Activity Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(activity_diagram(), None, "Figure 3.5 Activity Diagram", styles))

    story.append(PageBreak())
    story.append(Paragraph("3.2.4 Sequence Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(sequence_diagram(), None, "Figure 3.6 Sequence Diagram", styles))

    story.append(PageBreak())
    story.append(Paragraph("3.2.5 Class Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(class_diagram(), None, "Figure 3.7 Class Diagram", styles))

    story.append(PageBreak())
    story.append(Paragraph("3.2.6 Entity Relationship Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(er_diagram(), None, "Figure 3.8 Entity Relationship Diagram", styles))

    story.append(PageBreak())
    story.append(Paragraph("3.2.7 Component Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(component_diagram(), None, "Figure 3.9 Component Diagram", styles))

    story.append(PageBreak())
    story.append(Paragraph("3.2.8 State Transition Diagram", styles["SubSectionTitle"]))
    story.append(figure_block(state_diagram(), None, "Figure 3.10 State Transition Diagram", styles))

    story.append(Paragraph("3.3 Data and Storage Design", styles["SectionTitle"]))
    story.append(
        Paragraph(
            f"The project stores scheme data, enriched canonical outputs, source metadata, and vector embeddings in local file form rather than a "
            f"traditional SQL database. This is a deliberate choice for portability and demo-friendliness. The current prototype uses "
            f"{stats.scheme_count} raw scheme rows, {stats.document_chunk_count} document chunks, and a {stats.vector_dimension}-dimension vector space.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Paragraph("Table 3.1 Key Project Data Assets and Outputs", styles["TableCaption"]))
    story.append(
        simple_table(
            [
                ["Asset", "Purpose", "Current Detail"],
                ["mutual_funds_data2.csv", "Primary recommendation dataset", f"{stats.scheme_count} scheme rows used for shortlist scoring"],
                ["mutual_fund_data1.csv", "AMFI-style master dataset", "NAV, AUM, scheme code, launch date, category context"],
                ["archive/DailyNAV", "Historical NAV snapshots", f"Latest detected snapshot: {stats.latest_archive_snapshot}"],
                ["fund_schemes_canonical.csv", "Enriched canonical export", f"Real NAV coverage: {stats.real_nav_count}; synthetic coverage: {stats.synthetic_nav_count}"],
                ["source_registry.json", "Corpus governance registry", f"{stats.configured_source_count} configured source families"],
                ["document_chunk_index.json", "Vector retrieval export", f"{stats.vector_chunk_count} chunk vectors of dimension {stats.vector_dimension}"],
            ],
            [1.7 * inch, 2.1 * inch, 2.7 * inch],
        )
    )


def chapter_four(story: list, styles: StyleSheet1, stats: ReportStats) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 4: Implementation, Maintenance and Testing", styles["ChapterTitle"]))
    story.append(Paragraph("4.1 Introduction to Languages, IDEs, Tools and Technologies", styles["SectionTitle"]))
    story.append(Paragraph("Table 4.1 Languages, Tools and Technologies", styles["TableCaption"]))
    story.append(
        simple_table(
            [
                ["Technology", "Role in the Project"],
                ["Python", "Primary implementation language for data, scoring, retrieval, and report logic"],
                ["Streamlit", "Interactive dashboard used for profile capture, shortlist display, and assistant interaction"],
                ["pypdf", "PDF text extraction for corpus ingestion"],
                ["Local JSON / CSV storage", "Portable storage for canonical dataset and vector index"],
                ["ReportLab", "Used to generate this project report in PDF form"],
            ],
            [1.7 * inch, 4.8 * inch],
        )
    )

    story.append(Paragraph("4.2 Implementation Stages", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "Foundation stage: establish the problem statement, requirement notes, and starter Streamlit interface.",
                "Recommendation stage: add investor profile handling, scoring rules, and shortlist explainability.",
                "Data stage: normalize scheme names, enrich master-data linkage, and complete NAV coverage with transparent source tags.",
                "Corpus stage: ingest local documents, create chunks, and attach source governance metadata.",
                "RAG stage: add local vector retrieval, answer modes, citations, and grounded response formatting.",
            ],
            styles,
        )
    )

    story.append(Paragraph("4.3 Testing and Validation", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The current project is validated through functional walkthroughs of the UI, recommendation output inspection, data-pipeline sanity checks, "
            "and grounded assistant verification. Since the prototype is intended for project presentation, the emphasis is on correctness of behavior, "
            "traceability of reasoning, and stability of modular interactions rather than on large-scale production benchmarking.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Paragraph("Table 4.2 Validation and Test Cases", styles["TableCaption"]))
    story.append(
        simple_table(
            [
                ["ID", "Scenario", "Expected Result", "Observed Result"],
                ["TC-01", "Incomplete profile input", "System should request remaining fields", "UI displays completion warning until mandatory values are entered"],
                ["TC-02", "Balanced Builder preset", "Shortlist should appear with top-ranked funds", f"Five recommendations are generated; top recommendation is {stats.top_recommendations[0][0]}"],
                ["TC-03", "Strategy tab", "Allocation guidance should be visible", "System shows a three-slice role-based allocation blueprint"],
                ["TC-04", "Assistant question with Advisor lens", "Grounded answer should include support notes and cited sources", "Answer bundle includes intent, confidence, citations, and referenced funds"],
                ["TC-05", "Corpus/vector bootstrap", "Searchable chunk index should be available", f"{stats.document_chunk_count} chunks and {stats.vector_chunk_count} vectors are available locally"],
            ],
            [0.55 * inch, 1.45 * inch, 2.25 * inch, 2.5 * inch],
            header_font_size=9.6,
            body_font_size=9.0,
        )
    )

    story.append(Paragraph("4.4 Maintenance Considerations", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "The codebase separates scheme scoring, corpus management, retrieval, and UI orchestration into modules for easier updates.",
                "The source registry centralizes trust-tier metadata so future corpus expansion remains governed instead of ad hoc.",
                "Canonical dataset export reduces repeated preprocessing work for downstream experiments.",
                "The vector index can be regenerated locally whenever the corpus changes.",
            ],
            styles,
        )
    )

    story.append(Paragraph("4.5 End User Instructions", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "Install the required packages from requirements.txt and run `streamlit run app.py`.",
                "Open the local browser URL and acknowledge the risk notice.",
                "Use a quick preset or fill the profile form manually.",
                "Inspect the shortlist, strategy suggestions, and caution notes.",
                "Open the assistant tab, choose a lens, and ask a grounded follow-up question.",
            ],
            styles,
        )
    )


def chapter_five(story: list, styles: StyleSheet1, stats: ReportStats, screenshots: dict[str, Path]) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 5: Results and Discussion", styles["ChapterTitle"]))
    story.append(Paragraph("5.1 User Interface Representation", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "The final interface organizes the workflow into three major views: shortlist generation, strategy guidance, and a grounded assistant tab. "
            "The left panel captures investor inputs and quick presets, while the right side summarizes the current spotlight and dynamic tab content. "
            "This layout makes the system easy to present in a classroom or viva setting because the profile, results, and follow-up reasoning remain visible within one window.",
            styles["BodyTextAcademic"],
        )
    )

    story.append(Paragraph("5.2 Brief Description of Various Modules", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                "Profile module: builds an InvestorProfile object from validated user inputs.",
                "Recommendation module: scores each scheme and produces a diversified shortlist with explanations.",
                "Strategy module: translates shortlist behavior into role-based allocation suggestions.",
                "Corpus module: extracts and chunks project and reference documents with source metadata.",
                "Assistant module: answers follow-up questions with intent-aware formatting and evidence citations.",
            ],
            styles,
        )
    )

    story.append(Paragraph("5.3 Screenshots", styles["SectionTitle"]))
    if "ui_fold" in screenshots:
        story.append(figure_block(None, screenshots["ui_fold"], "Figure 5.1 Application Home and Profile View", styles, 2.7 * inch))
    if "shortlist" in screenshots:
        story.append(figure_block(None, screenshots["shortlist"], "Figure 5.2 Recommendation Shortlist View", styles, 2.7 * inch))
    if "strategy" in screenshots:
        story.append(figure_block(None, screenshots["strategy"], "Figure 5.3 Strategy Allocation View", styles, 2.7 * inch))
    if "assistant" in screenshots:
        story.append(figure_block(None, screenshots["assistant"], "Figure 5.4 Grounded Assistant Response View", styles, 2.7 * inch))

    story.append(Paragraph("5.4 Backend Representation", styles["SectionTitle"]))
    story.append(
        make_bullets(
            [
                f"Scheme coverage: {stats.scheme_count} schemes across {stats.category_count} categories and {stats.sub_category_count} sub-categories.",
                f"NAV enrichment: {stats.real_nav_count} rows with real master/archive support and {stats.synthetic_nav_count} rows with synthetic continuity support.",
                f"Corpus footprint: {stats.indexed_asset_count} indexed assets expanded into {stats.document_chunk_count} searchable chunks.",
                f"Vector retrieval: {stats.vector_chunk_count} embeddings represented in a {stats.vector_dimension}-dimension deterministic vector space.",
                "Source governance: corpus sources carry trust-tier and source-type metadata for ranking control and explainability.",
            ],
            styles,
        )
    )

    story.append(Paragraph("5.5 Discussion", styles["SectionTitle"]))
    top_lines = [
        f"{index + 1}. {name} ({sub_category}) - {fit_label}, score {score}"
        for index, (name, sub_category, fit_label, score) in enumerate(stats.top_recommendations[:5])
    ]
    story.append(
        Paragraph(
            "The observed outputs show that the system behaves more like a guided decision-support assistant than a simple filter. For the representative "
            "Balanced Builder profile, the shortlist blends moderate-risk and growth-oriented funds while still surfacing caution notes for higher-risk equity "
            "entries. The assistant output is similarly structured: it reports an intent label, confidence cue, support-note count, referenced funds, and source cards.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(make_bullets(top_lines, styles))
    story.append(
        Paragraph(
            "The design still has clear boundaries. The corpus is intentionally modest, the vector model is local and deterministic rather than transformer-based, "
            "and the system is not intended to issue certified investment advice. Even so, the project demonstrates a meaningful integration of structured scoring, "
            "data-quality transparency, and grounded AI assistance within a coherent academic prototype.",
            styles["BodyTextAcademic"],
        )
    )


def chapter_six(story: list, styles: StyleSheet1) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 6: Summary and Conclusion", styles["ChapterTitle"]))
    story.append(
        Paragraph(
            "The Mutual Fund Recommendation Assistant combines investor profiling, explainable shortlist generation, corpus ingestion, local embeddings, and "
            "citation-aware answer synthesis into one academic prototype. The project demonstrates that a finance-oriented assistant can remain lightweight and "
            "offline-friendly while still exposing disciplined system design choices such as modular separation, trust-tagged sources, and explicit caution layers.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(
        Paragraph(
            "In conclusion, the project successfully meets its objective of making mutual fund exploration more transparent and structured for users. It does not "
            "pretend to remove the need for document verification or professional advice, but it meaningfully reduces confusion by translating investor preferences "
            "into grounded, understandable, and auditable outputs.",
            styles["BodyTextAcademic"],
        )
    )


def chapter_seven(story: list, styles: StyleSheet1) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Chapter 7: Future Scope", styles["ChapterTitle"]))
    story.append(
        make_bullets(
            [
                "Integrate a live external LLM while preserving the current prompt-pack and citation discipline.",
                "Expand the corpus with more AMC factsheets, scheme information documents, and official investor-awareness material.",
                "Add portfolio upload and diversification diagnostics for existing holdings.",
                "Benchmark retrieval quality, answer faithfulness, and consistency across predefined investor scenarios.",
                "Introduce richer comparison views, export options, and goal-based planning workflows.",
            ],
            styles,
        )
    )


def appendix_and_references(story: list, styles: StyleSheet1, stats: ReportStats) -> None:
    story.append(PageBreak())
    story.append(Paragraph("Appendix", styles["ChapterTitle"]))
    story.append(Paragraph("Appendix A: Example Balanced Builder Outcome", styles["SectionTitle"]))
    story.append(
        Paragraph(
            "Representative investor profile used for screenshots and result discussion: age 29, goal Wealth Creation, risk appetite Moderate, horizon 7 years, "
            "investment mode SIP, planned amount Rs. 12,000, tax-saving preference No.",
            styles["BodyTextAcademic"],
        )
    )
    story.append(Paragraph("Appendix B: Allocation Blueprint", styles["SectionTitle"]))
    allocation_rows = [["Label", "Allocation", "Rationale"]]
    for label, pct, rationale in stats.allocation_blueprint:
        allocation_rows.append([label, f"{pct}%", rationale])
    story.append(simple_table(allocation_rows, [2.2 * inch, 1.0 * inch, 3.3 * inch]))

    story.append(PageBreak())
    story.append(Paragraph("References", styles["ChapterTitle"]))
    reference_items = [
        "AMFI India. Category framework and investor reference material. https://www.amfiindia.com/",
        "SEBI Investor Education. Investor-awareness and risk-disclosure guidance. https://investor.sebi.gov.in/",
        "Investor.gov. Diversification and asset-allocation education. https://www.investor.gov/",
        "RBI Financial Education. Financial awareness and debt-market context. https://www.rbi.org.in/",
        "Streamlit Documentation. Application framework reference. https://streamlit.io/",
        "pypdf Documentation. PDF parsing and text extraction reference. https://pypdf.readthedocs.io/",
        "Internal project documentation in README.md and docs/ within the MFRA repository.",
    ]
    story.append(make_bullets(reference_items, styles))


def build_report(output_path: Path) -> Path:
    stats = gather_stats()
    logo_path = prepare_logo()
    screenshots = prepare_screenshot_variants()
    styles = make_styles()

    doc = AcademicDocTemplate(
        str(output_path),
        leftMargin=0.72 * inch,
        rightMargin=0.72 * inch,
        topMargin=0.78 * inch,
        bottomMargin=0.75 * inch,
        title=TITLE,
        author=", ".join(AUTHORS),
    )

    story: list = []
    cover_page(story, styles, logo_path)
    front_matter(story, styles, stats)
    toc_and_indexes(story, styles)
    chapter_one(story, styles, stats)
    chapter_two(story, styles)
    chapter_three(story, styles, stats)
    chapter_four(story, styles, stats)
    chapter_five(story, styles, stats, screenshots)
    chapter_six(story, styles)
    chapter_seven(story, styles)
    appendix_and_references(story, styles, stats)

    doc.multiBuild(story)
    return output_path


def main() -> None:
    output = build_report(OUTPUT_REPORT)
    print(output)


if __name__ == "__main__":
    main()
