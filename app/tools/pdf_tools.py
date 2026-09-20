from __future__ import annotations

import os
import re
import tempfile
from datetime import datetime, timezone
from typing import List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fpdf import FPDF
from fpdf.fonts import FontFace

from app.core.utils import strip_gateway_noise

OUTPUT_DIR = os.environ.get("PDF_OUTPUT_DIR", os.path.join(tempfile.gettempdir(), "reports"))

# Apex Dynamics brand palette
BRAND_BLUE = (47, 82, 150)
ROW_STRIPE = (244, 246, 251)
TEXT_DARK = (34, 34, 34)

# fpdf2's built-in base-14 fonts are Latin-1 only; matplotlib bundles the full
# DejaVu family — register those so the PDF is Unicode-safe without any apt
# font packages.
_FONT_DIR = os.path.join(matplotlib.get_data_path(), "fonts", "ttf")
_FONT_REGULAR = os.path.join(_FONT_DIR, "DejaVuSans.ttf")
_FONT_BOLD = os.path.join(_FONT_DIR, "DejaVuSans-Bold.ttf")

# Characters outside DejaVu coverage (emoji, dingbats) are stripped so
# rendering never fails on LLM output.
_UNSAFE_CHARS = re.compile(
    r"[^\x20-\x7E\n\u00A0-\u017F\u2010-\u2027\u2030-\u205E\u20AC\u2122\u2190-\u2193\u2212]"
)
_MD_BOLD = re.compile(r"\*\*(.+?)\*\*")
_MD_HEADING = re.compile(r"^#{1,6}\s*", re.MULTILINE)
_MD_TABLE_LINE = re.compile(r"^\s*\|.*\|\s*$", re.MULTILINE)
_MD_SEPARATOR_ROW = re.compile(r"^\s*\|[\s:\-|]+\|\s*$", re.MULTILINE)
_FORMAT_LABELS = re.compile(r"^\s*(concise_text|markdown_table)\s*:\s*$", re.MULTILINE | re.IGNORECASE)
# LLMs sometimes echo the section titles we add ourselves — drop them so
# headings are not duplicated.
_ECHOED_HEADINGS = re.compile(
    r"^\s*\d*\.?\s*(Executive Summary|Key Findings)\s*$", re.MULTILINE | re.IGNORECASE
)


def _clean(text: str) -> str:
    """Sanitize LLM output for PDF rendering: strip markdown artifacts,
    gateway-injected notice lines, echoed section titles, and characters
    DejaVu cannot display."""
    t = text or ""
    t = _MD_BOLD.sub(r"\1", t)
    t = _MD_HEADING.sub("", t)
    t = _MD_SEPARATOR_ROW.sub("", t)
    t = _MD_TABLE_LINE.sub("", t)
    t = _FORMAT_LABELS.sub("", t)
    t = strip_gateway_noise(t)
    t = _ECHOED_HEADINGS.sub("", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return _UNSAFE_CHARS.sub("", t).strip()


def _ensure_output_dir() -> str:
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


class _ReportPDF(FPDF):
    def footer(self):
        self.set_y(-12)
        self.set_font("DejaVu", size=8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 8, f"Apex Dynamics — Analytics Platform · Page {self.page_no()}/{{nb}}", align="C")


def _new_pdf() -> _ReportPDF:
    pdf = _ReportPDF()
    pdf.alias_nb_pages()
    pdf.set_auto_page_break(auto=True, margin=16)
    pdf.add_page()
    pdf.add_font("DejaVu", "", _FONT_REGULAR)
    pdf.add_font("DejaVu", "B", _FONT_BOLD)
    pdf.set_text_color(*TEXT_DARK)
    return pdf


def _heading(pdf: FPDF, text: str) -> None:
    pdf.set_font("DejaVu", "B", 12)
    pdf.set_text_color(*BRAND_BLUE)
    pdf.multi_cell(0, 8, text)
    pdf.set_text_color(*TEXT_DARK)
    pdf.ln(1)


def _section_break(pdf: FPDF) -> None:
    pdf.ln(2)
    pdf.set_draw_color(210, 214, 222)
    pdf.set_line_width(0.2)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(3)


def render_bar_chart(
    labels: List[str],
    values: List[float],
    title: str = "Chart",
    filename: Optional[str] = None,
) -> str:
    """Render a simple bar chart to a PNG file and return its absolute path."""
    _ensure_output_dir()
    if filename is None:
        fd, filename = tempfile.mkstemp(suffix=".png", dir=OUTPUT_DIR)
        os.close(fd)
    else:
        filename = os.path.join(OUTPUT_DIR, filename)

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.bar(labels, values, color="#4F81BD")
    ax.set_title(title)
    ax.set_ylabel("Value")
    plt.xticks(rotation=20, ha="right", fontsize=8)
    plt.yticks(fontsize=8)
    plt.tight_layout()
    fig.savefig(filename, dpi=150)
    plt.close(fig)
    return filename


def render_pdf(
    title: str,
    body_text: str,
    subtitle: Optional[str] = None,
    data: Optional[List[dict]] = None,
    chart_paths: Optional[List[str]] = None,
    filename: Optional[str] = None,
) -> str:
    """Render an Apex Dynamics branded report: header, executive summary,
    optional data table and charts."""
    _ensure_output_dir()
    pdf = _new_pdf()

    # --- Header ---
    pdf.set_font("DejaVu", "B", 18)
    pdf.set_text_color(*BRAND_BLUE)
    pdf.multi_cell(0, 10, _clean(title))
    pdf.set_draw_color(*BRAND_BLUE)
    pdf.set_line_width(0.6)
    pdf.line(pdf.l_margin, pdf.get_y(), pdf.w - pdf.r_margin, pdf.get_y())
    pdf.ln(2)
    pdf.set_font("DejaVu", size=9)
    pdf.set_text_color(110, 110, 110)
    pdf.multi_cell(
        0, 5, _clean(subtitle or f"Generated {datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M UTC')}")
    )
    pdf.ln(2)

    # --- Executive Summary ---
    _heading(pdf, "Executive Summary")
    pdf.set_font("DejaVu", size=11)
    pdf.multi_cell(0, 6, _clean(body_text))

    # --- Data table ---
    if data:
        _section_break(pdf)
        _heading(pdf, "Data")
        columns = list(data[0].keys())
        # Right-align columns whose values look numeric.
        aligns = []
        for col in columns:
            sample = data[0].get(col)
            aligns.append("R" if isinstance(sample, (int, float)) else "L")
        pdf.set_font("DejaVu", size=10)
        headings_style = FontFace(
            emphasis="BOLD", color=(255, 255, 255), fill_color=BRAND_BLUE
        )
        with pdf.table(
            borders_layout="ALL",
            text_align=aligns,
            line_height=5.5,
            padding=1.5,
            headings_style=headings_style,
            cell_fill_color=ROW_STRIPE,
            cell_fill_mode="ROWS",
        ) as table:
            header = table.row()
            for col in columns:
                header.cell(str(col))
            for row_data in data:
                row = table.row()
                for col in columns:
                    row.cell(str(row_data.get(col, "")))

    # --- Charts ---
    for chart_no, path in enumerate(chart_paths or [], start=1):
        if path and os.path.exists(path):
            _section_break(pdf)
            _heading(pdf, "Visualization" if len(chart_paths or []) == 1 else f"Visualization {chart_no}")
            chart_w = pdf.epw * 0.75
            pdf.image(path, x=(pdf.w - chart_w) / 2, w=chart_w)

    if filename is None:
        fd, filename = tempfile.mkstemp(suffix=".pdf", dir=OUTPUT_DIR)
        os.close(fd)
    else:
        filename = os.path.join(OUTPUT_DIR, filename)

    pdf.output(filename)
    return filename
