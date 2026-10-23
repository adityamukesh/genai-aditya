"""Turn GER-3620P into LangChain Documents that know their section and page.

The PDF has no outline, so the heading hierarchy is recovered from typography:

    level 1   14 pt, teal      "Maintenance Inspections"
    level 2   10 pt            "Combustion Inspection"
              9 pt, teal       "Running Inspections"  (minor heading, kept at depth 2)

Only Inter-Regular counts: the SemiBold 8-10 pt lines are figure titles and
the 25 pt condensed banner is the document title.

Tables (for example Figure 36, the inspection-interval matrix) are detected
with PyMuPDF and linearised row by row so every value keeps its column
header ("7E.03: 12000/450") instead of becoming a soup of numbers.
"""
import re
from dataclasses import dataclass, field

import pymupdf
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, TABLE_CHUNK_SIZE

TEAL = 0x006061
HEADER_Y, FOOTER_Y = 30, 740  # document-id banner at the top, running footer at the bottom
MIN_FONT = 6.8  # anything smaller is figure axis labels and callouts
FIRST_SECTION = "Introduction"  # everything before it is the cover and table of contents
SKIP_SECTIONS = {"List of Figures", "Revision History"}  # captions index / changelog, not O&M content
CAPTION_RE = re.compile(r"^Figure [A-Z]?-?\d+\.")


@dataclass
class Segment:
    """Running text under one heading path; each paragraph remembers its page."""

    crumbs: tuple[str, ...]
    paragraphs: list[tuple[str, int, int]] = field(default_factory=list)  # (text, pdf page, printed page)


MODEL_RE = re.compile(r"\b(\d{1,2}[A-Z]{1,3})\. (\d{2})\b")  # "7F. 03" -> "7F.03"


def _clean(text: str) -> str:
    return MODEL_RE.sub(r"\1.\2", " ".join(text.split()))


def _join_lines(lines: list[str]) -> str:
    out = ""
    for line in lines:
        if out.endswith("-"):  # word hyphenated across a line break
            out += line
        else:
            out = f"{out} {line}" if out else line
    return out


def _heading_level(span: dict) -> int:
    """1 = section, 2 = subsection, 3 = minor heading (nested at depth 2), 0 = body."""
    if not span["font"].startswith("Inter-Regular"):
        return 0
    size, color = span["size"], span["color"]
    if abs(size - 14) < 0.2 and color == TEAL:
        return 1
    if abs(size - 10) < 0.2:
        return 2
    if abs(size - 9) < 0.2 and color == TEAL:
        return 3
    return 0


def _printed_page(page: pymupdf.Page) -> int | None:
    """The number printed in the footer (PDF page 25 is printed page 21)."""
    for x0, y0, x1, y1, word, *_ in page.get_text("words"):
        if y0 > FOOTER_Y and word.isdigit():
            return int(word)
    return None


def _inside(bbox, rect: pymupdf.Rect) -> bool:
    cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
    return rect.x0 - 1 <= cx <= rect.x1 + 1 and rect.y0 - 1 <= cy <= rect.y1 + 1


def _caption_for(table_rect: pymupdf.Rect, captions: list[tuple[float, str]]) -> str:
    """Figure captions sit just below their table; fall back to the nearest one."""
    if not captions:
        return ""
    below = [c for c in captions if c[0] >= table_rect.y1 - 5]
    pool = below or captions
    return min(pool, key=lambda c: abs(c[0] - table_rect.y1))[1]


def _table_rows(page: pymupdf.Page, table) -> list[str]:
    """Linearise a table, labelling each cell with the most specific header above it."""
    grid = [[_clean(page.get_text("text", clip=cell)) if cell else "" for cell in row.cells] for row in table.rows]
    if not grid:
        return []
    keep = [j for j in range(len(grid[0])) if any(row[j] for row in grid)]
    grid = [[row[j] for j in keep] for row in grid]
    grid = [row for row in grid if any(row)]

    headers = [""] * len(keep)
    lines, prev_was_header = [], False
    for row in grid:
        if not any(ch.isdigit() for cell in row for ch in cell):  # titles or an empty data row: no labels
            lines.append(" | ".join(c for c in row if c))
            prev_was_header = False
            continue
        if not row[0] and any(row[1:]):  # header row: blank stub column, labels across
            if not prev_was_header:
                headers = [""] * len(keep)  # a new sub-table starts
            for j, cell in enumerate(row):
                if cell:
                    headers[j] = cell
            prev_was_header = True
            continue
        prev_was_header = False
        cells = [f"{headers[j]}: {c}" if headers[j] else c for j, c in enumerate(row) if c and j > 0]
        lines.append(" | ".join([row[0], *cells]) if row[0] else " | ".join(cells))
    return lines


def _table_documents(page, table, crumbs, pdf_page, printed, caption) -> list[Document]:
    rows = _table_rows(page, table)
    if not rows:
        return []
    # Cite tables by top-level section + figure number: a figure often sits inside
    # an unrelated subsection on the page (Figure 36 lands in "ABC Inspections").
    section = crumbs[0] if crumbs else ""
    head = f"[{section}]\nTable: {caption or 'untitled'}\n"

    # One chunk per sub-table (each starts with the repeated column-title row) so a
    # question about one model family is not drowned out by the others; a sub-table
    # that is still too long is cut at row boundaries.
    blocks: list[list[str]] = []
    for row in rows:
        if not blocks or (row == rows[0] and blocks[-1]):
            blocks.append([])
        blocks[-1].append(row)
    groups = []
    for block in blocks:
        current: list[str] = []
        for row in block:
            if current and len(head) + len("\n".join(current)) + len(row) > TABLE_CHUNK_SIZE:
                groups.append(current)
                current = []
            current.append(row)
        groups.append(current)
    meta = {
        "section": section,
        "figure": caption,
        "kind": "table",
        "page_start": printed,
        "page_end": printed,
        "pdf_page_start": pdf_page,
        "pdf_page_end": pdf_page,
    }
    return [Document(page_content=head + "\n".join(g), metadata=dict(meta)) for g in groups]


def parse_pdf(path) -> tuple[list[Segment], list[Document]]:
    doc = pymupdf.open(path)
    segments: list[Segment] = []
    table_docs: list[Document] = []
    crumbs: list[str] = []
    started = False
    prev_heading_level = 0  # level of the previous line if it was a heading (for 2-line headings)

    def skipping() -> bool:
        return not started or (crumbs and crumbs[0] in SKIP_SECTIONS)

    for pdf_page, page in enumerate(doc, start=1):
        printed = _printed_page(page) or pdf_page
        tables = page.find_tables().tables
        emitted = set()
        captions = [
            (y0, _clean(text))
            for x0, y0, x1, y1, text, *_ in page.get_text("blocks")
            if CAPTION_RE.match(_clean(text))
        ]

        for block in page.get_text("dict")["blocks"]:
            para: list[str] = []

            def flush():
                if para and not skipping():
                    segments[-1].paragraphs.append((_join_lines(para), pdf_page, printed))
                para.clear()

            for line in block.get("lines", []):
                text = _clean("".join(s["text"] for s in line["spans"]))
                x0, y0, x1, y1 = line["bbox"]
                if not text or y1 < HEADER_Y or y0 > FOOTER_Y:
                    continue

                table_idx = next((i for i, t in enumerate(tables) if _inside(line["bbox"], pymupdf.Rect(t.bbox))), None)
                if table_idx is not None:
                    if table_idx not in emitted and not skipping():
                        rect = pymupdf.Rect(tables[table_idx].bbox)
                        table_docs += _table_documents(
                            page, tables[table_idx], crumbs, pdf_page, printed, _caption_for(rect, captions)
                        )
                    emitted.add(table_idx)
                    prev_heading_level = 0
                    continue

                level = _heading_level(line["spans"][0])
                if level:
                    flush()
                    if level == 1 and text == FIRST_SECTION:
                        started = True
                    depth = min(level, 2)
                    if level == prev_heading_level and len(crumbs) == depth:
                        crumbs[-1] = f"{crumbs[-1]} {text}"  # heading wrapped onto a second line
                        segments[-1].crumbs = tuple(crumbs)
                    else:
                        crumbs = crumbs[: depth - 1] + [text]
                        segments.append(Segment(tuple(crumbs)))
                    prev_heading_level = level
                    continue

                prev_heading_level = 0
                if line["spans"][0]["size"] < MIN_FONT:
                    continue
                if text.startswith("•"):
                    flush()
                para.append(text)
            flush()

        # Tables whose text never appeared in the block stream (pure vector drawings).
        for i, table in enumerate(tables):
            if i not in emitted and not skipping():
                rect = pymupdf.Rect(table.bbox)
                table_docs += _table_documents(page, table, crumbs, pdf_page, printed, _caption_for(rect, captions))

    return [s for s in segments if s.paragraphs], table_docs


def _text_documents(segments: list[Segment]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP, add_start_index=True
    )
    docs = []
    for seg in segments:
        text, starts = "", []  # starts: (char offset, pdf page, printed page) of each paragraph
        for para, pdf_page, printed in seg.paragraphs:
            if text:
                text += "\n"
            starts.append((len(text), pdf_page, printed))
            text += para
        section = " > ".join(seg.crumbs)
        for chunk in splitter.create_documents([text]):
            begin = chunk.metadata["start_index"]
            end = begin + len(chunk.page_content)
            covered = [s for s in starts if s[0] < end]
            first = max((s for s in covered if s[0] <= begin), key=lambda s: s[0], default=covered[0])
            last = covered[-1]
            docs.append(
                Document(
                    page_content=f"[{section}]\n{chunk.page_content}",
                    metadata={
                        "section": section,
                        "figure": "",
                        "kind": "text",
                        "page_start": first[2],
                        "page_end": last[2],
                        "pdf_page_start": first[1],
                        "pdf_page_end": last[1],
                    },
                )
            )
    return docs


def load_documents(path) -> list[Document]:
    segments, table_docs = parse_pdf(path)
    return _text_documents(segments) + table_docs
