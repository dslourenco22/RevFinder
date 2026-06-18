"""Local PDF extraction utilities backed by pdfplumber."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
import re
from typing import BinaryIO, Iterable

import pdfplumber


@dataclass(frozen=True)
class TextBlock:
    text: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float


@dataclass(frozen=True)
class PageExtraction:
    page_number: int
    width: float
    height: float
    text: str
    words: tuple[TextBlock, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class TableExtraction:
    page_number: int
    table_index: int
    rows: tuple[tuple[str, ...], ...]


@dataclass(frozen=True)
class PdfExtraction:
    source_name: str
    page_count: int
    full_text: str
    pages: tuple[PageExtraction, ...]
    tables: tuple[TableExtraction, ...]
    metadata: dict[str, str]
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict:
        return asdict(self)


def extract_pdf(source: str | Path | bytes | BinaryIO, source_name: str | None = None) -> PdfExtraction:
    """Extract text, words, and tables from a local PDF source.

    The source may be a filesystem path, raw bytes, or a binary file-like object.
    No OCR or network calls are performed.
    """

    pdf_input, resolved_name = _resolve_source(source, source_name)
    pages: list[PageExtraction] = []
    tables: list[TableExtraction] = []
    warnings: list[str] = []
    text_parts: list[str] = []

    with pdfplumber.open(pdf_input) as pdf:
        metadata = {str(k): str(v) for k, v in (pdf.metadata or {}).items() if v is not None}

        for page_index, page in enumerate(pdf.pages, start=1):
            try:
                page_text = page.extract_text(x_tolerance=1, y_tolerance=3) or ""
            except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
                page_text = ""
                warnings.append(f"Page {page_index}: text extraction failed: {exc}")

            words = _extract_words(page, page_index, warnings)
            extracted_tables = _extract_tables(page, page_index, warnings)
            # The word-grid reconstruction is a costly second word pass and only a
            # fallback for PDFs whose rows render as loose words. Skip it when real
            # tables were found to save time and avoid duplicate rows in the prompt.
            if not extracted_tables:
                extracted_tables = _extract_word_grid_tables(page, page_index, warnings)

            text_parts.append(f"\n--- Page {page_index} ---\n{page_text}".strip())
            pages.append(
                PageExtraction(
                    page_number=page_index,
                    width=float(page.width or 0),
                    height=float(page.height or 0),
                    text=page_text,
                    words=tuple(words),
                )
            )
            tables.extend(
                TableExtraction(
                    page_number=page_index,
                    table_index=table_index,
                    rows=tuple(tuple(_clean_cell(cell) for cell in row) for row in table),
                )
                for table_index, table in enumerate(extracted_tables, start=1)
                if table
            )

    return PdfExtraction(
        source_name=resolved_name,
        page_count=len(pages),
        full_text="\n\n".join(part for part in text_parts if part),
        pages=tuple(pages),
        tables=tuple(tables),
        metadata=metadata,
        warnings=tuple(warnings),
    )


def flatten_tables(tables: Iterable[TableExtraction]) -> list[list[str]]:
    """Return all extracted table rows as plain lists."""

    rows: list[list[str]] = []
    for table in tables:
        rows.extend([list(row) for row in table.rows])
    return rows


# Company title and document metadata live in the top band of an engineering
# sheet. ~1.5 inches at 72 points/inch keeps that header out of body-text parsers
# so they cannot be fallback-scraped as "generic document text".
HEADER_BAND_POINTS = 108.0


def page_body_text(page: PageExtraction, top_margin: float = HEADER_BAND_POINTS) -> str:
    """Reconstruct a page's text with the top header band excluded.

    Uses word coordinates instead of ``page.text`` so the company title and
    document metadata that sit in the top ``top_margin`` points are dropped. This
    prevents header fields from leaking into block-text parsers (e.g. notes).
    """

    body_words = [word for word in page.words if word.top >= top_margin]
    return _textblocks_to_text(body_words)


def _textblocks_to_text(words: Iterable[TextBlock], y_tolerance: float = 3.0) -> str:
    rows: list[list[TextBlock]] = []
    for word in sorted(words, key=lambda item: (item.top, item.x0)):
        midpoint = (word.top + word.bottom) / 2
        for row in rows:
            row_midpoint = sum((item.top + item.bottom) / 2 for item in row) / len(row)
            if abs(midpoint - row_midpoint) <= y_tolerance:
                row.append(word)
                break
        else:
            rows.append([word])

    lines: list[str] = []
    for row in rows:
        row.sort(key=lambda item: item.x0)
        line = " ".join(word.text for word in row).strip()
        if line:
            lines.append(line)
    return "\n".join(lines)


def _resolve_source(source: str | Path | bytes | BinaryIO, source_name: str | None) -> tuple[str | BytesIO | BinaryIO, str]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        return str(path), source_name or path.name
    if isinstance(source, bytes):
        return BytesIO(source), source_name or "uploaded.pdf"
    return source, source_name or getattr(source, "name", "uploaded.pdf")


def _extract_words(page: pdfplumber.page.Page, page_number: int, warnings: list[str]) -> list[TextBlock]:
    try:
        raw_words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=True,
            extra_attrs=[],
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        warnings.append(f"Page {page_number}: word extraction failed: {exc}")
        return []

    return [
        TextBlock(
            text=str(word.get("text", "")).strip(),
            page=page_number,
            x0=float(word.get("x0") or 0),
            top=float(word.get("top") or 0),
            x1=float(word.get("x1") or 0),
            bottom=float(word.get("bottom") or 0),
        )
        for word in raw_words
        if str(word.get("text", "")).strip()
    ]


def _extract_tables(page: pdfplumber.page.Page, page_number: int, warnings: list[str]) -> list[list[list[str | None]]]:
    settings = {
        "vertical_strategy": "lines",
        "horizontal_strategy": "lines",
        "snap_tolerance": 3,
        "join_tolerance": 3,
        "edge_min_length": 3,
        "min_words_vertical": 2,
        "min_words_horizontal": 1,
        "text_tolerance": 3,
    }
    try:
        tables = page.extract_tables(table_settings=settings)
    except Exception as exc:
        warnings.append(f"Page {page_number}: line-table extraction failed: {exc}")
        tables = []

    if tables:
        return tables

    try:
        return page.extract_tables(
            table_settings={
                **settings,
                "vertical_strategy": "text",
                "horizontal_strategy": "text",
            }
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        warnings.append(f"Page {page_number}: text-table extraction failed: {exc}")
        return []


def _extract_word_grid_tables(page: pdfplumber.page.Page, page_number: int, warnings: list[str]) -> list[list[list[str]]]:
    """Reconstruct a simple table from word coordinates.

    This is a fallback for PDFs where wrapped description text is rendered as
    loose words instead of true table cells. It infers column bands from a
    header row, assigns words by x-position, then folds continuation lines back
    into the previous row's description cell.
    """

    try:
        raw_words = page.extract_words(
            keep_blank_chars=False,
            use_text_flow=False,
            extra_attrs=[],
            x_tolerance=1,
            y_tolerance=3,
        )
    except Exception as exc:  # pragma: no cover - defensive for malformed PDFs
        warnings.append(f"Page {page_number}: word-grid extraction failed: {exc}")
        return []

    words = [
        {
            "text": str(word.get("text", "")).strip(),
            "x0": float(word.get("x0") or 0),
            "x1": float(word.get("x1") or 0),
            "top": float(word.get("top") or 0),
            "bottom": float(word.get("bottom") or 0),
        }
        for word in raw_words
        if str(word.get("text", "")).strip()
    ]
    if not words:
        return []

    lines = _cluster_words_into_lines(words)
    header_index = _find_grid_header_index(lines)
    if header_index is None:
        return []

    header_words = lines[header_index]
    headers = _infer_grid_headers(header_words)
    if len(headers) < 3:
        return []

    boundaries = _column_boundaries(headers, float(page.width or 0))
    rows: list[list[str]] = [[header["label"] for header in headers]]

    for line_words in lines[header_index + 1 :]:
        row = ["" for _ in headers]
        for word in line_words:
            col_index = _column_index_for_word(word, boundaries)
            row[col_index] = f"{row[col_index]} {word['text']}".strip()

        if not any(row):
            continue
        if _is_continuation_row(row, headers) and len(rows) > 1:
            description_index = _header_index(headers, "description")
            target_index = description_index if description_index is not None else min(2, len(headers) - 1)
            continuation = " ".join(cell for cell in row if cell)
            rows[-1][target_index] = f"{rows[-1][target_index]} {continuation}".strip()
            continue
        rows.append(row)

    if len(rows) <= 1:
        return []
    return [rows]


def _cluster_words_into_lines(words: list[dict[str, float | str]], y_tolerance: float = 4.0) -> list[list[dict[str, float | str]]]:
    lines: list[list[dict[str, float | str]]] = []
    for word in sorted(words, key=lambda item: (float(item["top"]), float(item["x0"]))):
        midpoint = (float(word["top"]) + float(word["bottom"])) / 2
        for line in lines:
            line_midpoint = sum((float(item["top"]) + float(item["bottom"])) / 2 for item in line) / len(line)
            if abs(midpoint - line_midpoint) <= y_tolerance:
                line.append(word)
                break
        else:
            lines.append([word])

    for line in lines:
        line.sort(key=lambda item: float(item["x0"]))
    return lines


def _find_grid_header_index(lines: list[list[dict[str, float | str]]]) -> int | None:
    for index, line in enumerate(lines[:30]):
        text = " ".join(str(word["text"]) for word in line).lower()
        score = sum(
            token in text
            for token in ("item", "line", "part", "description", "qty", "quantity", "rev", "vendor", "uom")
        )
        if score >= 3:
            return index
    return None


def _infer_grid_headers(header_words: list[dict[str, float | str]]) -> list[dict[str, float | str]]:
    tokens = [(str(word["text"]), float(word["x0"]), float(word["x1"])) for word in header_words]
    headers: list[dict[str, float | str]] = []
    index = 0
    while index < len(tokens):
        text, x0, x1 = tokens[index]
        label = _canonical_header_label(text)

        if index + 1 < len(tokens):
            next_text, _, next_x1 = tokens[index + 1]
            combined = _canonical_header_label(f"{text} {next_text}")
            if combined != _canonical_header_label(text):
                label = combined
                x1 = next_x1
                index += 1

        if label:
            headers.append({"label": label, "x0": x0, "x1": x1})
        index += 1

    deduped: list[dict[str, float | str]] = []
    seen: set[str] = set()
    for header in headers:
        label = str(header["label"])
        if label in seen:
            continue
        seen.add(label)
        deduped.append(header)
    return deduped


def _canonical_header_label(text: str) -> str:
    label = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    if label in {"item", "item no", "line", "line no", "no"}:
        return "item_id"
    if label in {"part", "part no", "part number", "pn", "p n"}:
        return "part_number"
    if label in {"description", "desc", "item description"}:
        return "description"
    if label in {"qty", "quantity"}:
        return "quantity"
    if label in {"uom", "u m", "unit", "units"}:
        return "unit"
    if label in {"rev", "revision"}:
        return "revision"
    if label in {"manufacturer", "mfg"}:
        return "manufacturer"
    if label in {"vendor", "supplier"}:
        return "vendor"
    if label in {"price", "unit price", "cost", "unit cost"}:
        return "price"
    return ""


def _column_boundaries(headers: list[dict[str, float | str]], page_width: float) -> list[float]:
    starts = [float(header["x0"]) for header in headers]
    boundaries = [max(0.0, starts[0] - 6.0)]
    for left, right in zip(starts, starts[1:]):
        boundaries.append((left + right) / 2)
    boundaries.append(max(page_width, float(headers[-1]["x1"]) + 6.0))
    return boundaries


def _column_index_for_word(word: dict[str, float | str], boundaries: list[float]) -> int:
    midpoint = (float(word["x0"]) + float(word["x1"])) / 2
    for index in range(len(boundaries) - 1):
        if boundaries[index] <= midpoint < boundaries[index + 1]:
            return index
    return max(0, len(boundaries) - 2)


def _is_continuation_row(row: list[str], headers: list[dict[str, float | str]]) -> bool:
    item_index = _header_index(headers, "item_id")
    part_index = _header_index(headers, "part_number")
    quantity_index = _header_index(headers, "quantity")
    has_key = any(row[index] for index in (item_index, part_index, quantity_index) if index is not None)
    return not has_key


def _header_index(headers: list[dict[str, float | str]], label: str) -> int | None:
    for index, header in enumerate(headers):
        if header["label"] == label:
            return index
    return None


def _clean_cell(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())
