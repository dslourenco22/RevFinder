"""Local PDF extraction utilities backed by pdfplumber."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from io import BytesIO
from pathlib import Path
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
            page_count=len(pdf.pages),
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


def _clean_cell(value: object) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())
