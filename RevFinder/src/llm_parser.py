"""Structured engineering document parsing through local Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field, replace
from io import BytesIO
from typing import Any
from urllib.parse import urlparse

import requests

from .extractor import (
    HEADER_BAND_POINTS,
    PdfExtraction,
    flatten_tables,
    page_body_text,
)
from .normalize import (
    PART_LIKE_PATTERN,
    extract_part_key,
    normalize_identifier,
    normalize_price,
    part_match_token,
)


# llama3.2 is the default local model. Stronger models (listed below) generally
# map arbitrary layouts more reliably and can be selected in the UI / passed in.
DEFAULT_MODEL = "llama3.2"
SUGGESTED_MODELS = ["llama3.2", "llama3.1:8b", "qwen2.5:7b", "llama3.2:1b"]
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DESCRIPTOR_TAIL_WORDS = {
    "red",
    "green",
    "blue",
    "black",
    "white",
    "yellow",
    "coil",
    "curve",
    "output",
    "input",
    "momentary",
    "maintained",
    "normally open",
    "normally closed",
}
VALID_UNITS = {
    "ea",
    "each",
    "pc",
    "pcs",
    "piece",
    "pieces",
    "ft",
    "in",
    "mm",
    "m",
    "set",
    "lot",
    "box",
    "roll",
}

LINE_ITEM_FIELDS = [
    "item_id",
    "part_number",
    "mpn",
    "description",
    "quantity",
    "unit",
    "revision",
    "manufacturer",
    "vendor",
    "unit_price",
    "total_price",
    "contextual_notes",
    "source_page",
    "raw_text",
    "confidence",
]


@dataclass
class ParsedDocument:
    source_name: str
    document_type: str = "unknown"
    header: dict[str, Any] = field(default_factory=dict)
    line_items: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    parser: str = "unknown"
    warnings: list[str] = field(default_factory=list)
    # Fields this document could actually provide (None = all fields available, as
    # for PDFs). Spreadsheets set it to their real columns so a column present in
    # one file but absent in the other is not compared as a content change.
    available_fields: list[str] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LocalOllamaParser:
    """Parser that only communicates with a local Ollama endpoint."""

    def __init__(
        self,
        base_url: str = DEFAULT_OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        timeout_seconds: int = 180,
        max_prompt_chars: int = 52_000,
        header_band_points: float = HEADER_BAND_POINTS,
    ) -> None:
        self.base_url = _validate_local_url(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_prompt_chars = max_prompt_chars
        self.header_band_points = header_band_points

    def parse(self, extraction: PdfExtraction) -> ParsedDocument:
        extraction = _stitch_extraction(extraction)

        # Deterministic parsing (block/table/card/line scanners) is reliable and
        # CONSISTENT for structured PO/BoM manifests — both documents are parsed
        # identically, so values, formatting, and column order match. A small local
        # LLM (e.g. llama3.2) tends to corrupt these (swapped prices, dropped
        # quantities, inconsistent formatting), producing false-positive deltas.
        # So we prefer the deterministic result whenever it recovers line items and
        # fall back to the LLM only for layouts it cannot segment.
        deterministic = fallback_parse(extraction, self.header_band_points)
        if deterministic.line_items:
            return deterministic

        payload = {
            "model": self.model,
            "prompt": self._build_prompt(extraction),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 16384,
            },
        }

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            parsed = _decode_ollama_response(response.json(), extraction.source_name)
            parsed.parser = f"ollama:{self.model}"
            # Notes are derived deterministically (coordinate-bounded + keyword
            # anchored) so the model cannot fallback-scrape header metadata.
            parsed.notes = extract_document_notes(extraction, self.header_band_points)
            parsed.line_items = backfill_from_deterministic(parsed.line_items, extraction)
            parsed.line_items = attach_contextual_notes(parsed.line_items, extraction.full_text)
            parsed.warnings.extend(extraction.warnings)
            return _normalize_parsed_document(parsed)
        except Exception as exc:
            deterministic.warnings.append(f"Ollama parse unavailable; used deterministic parser: {exc}")
            return deterministic

    def healthcheck(self) -> tuple[bool, str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            response.raise_for_status()
            models = [item.get("name", "") for item in response.json().get("models", [])]
            if self.model in models:
                return True, f"{self.model} is available"
            if any(model.startswith(self.model) for model in models):
                return True, f"{self.model} compatible model found"
            return False, f"Ollama is reachable, but {self.model} was not listed"
        except Exception as exc:
            return False, str(exc)

    def _build_prompt(self, extraction: PdfExtraction) -> str:
        tables = _tables_as_prompt_text(extraction)
        source_text = extraction.full_text[: self.max_prompt_chars]
        return f"""
You are parsing a confidential engineering document inside a fully local system.
Return only valid JSON matching this schema:
{{
  "document_type": "BOM | PO | ECO | unknown",
  "header": {{
    "document_number": "",
    "revision": "",
    "date": "",
    "project": "",
    "customer": "",
    "vendor": "",
    "title": ""
  }},
  "line_items": [
    {{
      "item_id": "",
      "part_number": "",
      "mpn": "",
      "description": "",
      "quantity": "",
      "unit": "",
      "revision": "",
      "manufacturer": "",
      "vendor": "",
      "unit_price": "",
      "total_price": "",
      "contextual_notes": "",
      "source_page": "",
      "raw_text": "",
      "confidence": 0.0
    }}
  ],
  "notes": []
}}

Rules:
- Extract every BOM, PO, or ECO line item that is present.
- Do not invent values. Use empty strings for unknown fields.
- Documents arrive in MANY layouts: column tables, labeled cards, or cells that stack several fields vertically. Map each line item to the schema by MEANING, not by visual position, and produce identical structured output regardless of layout.
- part_number is the Internal Part Number (IPN) and is the primary key for a line. mpn is the Manufacturer Part Number and manufacturer is the maker. Keep them in separate fields. Never emit a line that only has an mpn/manufacturer; attach that data to the line that owns the IPN.
- Never include a field label (e.g. "IPN:", "MFG:", "MPN:", "Part Reference:") inside a value. If one cell stacks several labeled fields, split them into the matching schema fields.
- A single part number may be split across lines (e.g. "105-ENC-" then "AL-01"); join it into one complete part_number ("105-ENC-AL-01"). Preserve identifiers, decimals, dash suffixes, and revision letters exactly; keep full multi-hyphen keys intact (e.g. "710-HARN-TLM-ALPH-TRK-01A_REV.3") and never truncate at a hyphen.
- unit_price is the per-unit scalar ("UNIT PRICE", "Price per Unit"). total_price is the aggregate line value ("EXT PRICE", "Total Line Val", quantity x unit price). If a price cell stacks two numbers (unit over extended/total), the per-unit value is unit_price and the aggregate/larger value is total_price. Never copy a line total into unit_price.
- contextual_notes holds loose or bracketed engineering remarks tied to a line (e.g. "[QTY DECREASED BY 1 PER ECO]", "[CRITICAL NOTE: ...]"). Attach such a remark to the line item it describes; do not invent one.
- Use source_page when it can be inferred.
- Confidence must be a number from 0 to 1.
- Only put text that appears under an explicit heading such as "Revision Change Engineering Notes", "Engineering Change Notes", or "Revision History" in notes. Never put the company name, page header, or document metadata fields in notes.
- If a description wraps to multiple visual lines, merge the wrapped text back into description. Do not put words like color names, Coil, Curve, Input, or Output in vendor/unit unless the source explicitly labels them there.

Source file: {extraction.source_name}

Extracted tables:
{tables}

Extracted page text:
{source_text}
""".strip()


def fallback_parse(
    extraction: PdfExtraction, header_band_points: float = HEADER_BAND_POINTS
) -> ParsedDocument:
    """Best-effort deterministic parser for table-shaped engineering documents."""

    extraction = _stitch_extraction(extraction)

    # Primary path: parse the clean text stream into line-item blocks. This is
    # robust to the fragmented/garbage tables pdfplumber often returns for ruled
    # layouts, and works on both tabular and card layouts.
    items = _block_scan_parse(extraction)

    # Fallbacks for layouts the block scanner cannot delimit.
    if not items:
        items = _table_scan_parse(extraction)
    if not items:
        items = _card_block_parse(extraction)
    if not items:
        items = _line_scan_parse(extraction)

    items = attach_contextual_notes(items, extraction.full_text)

    return _normalize_parsed_document(
        ParsedDocument(
            source_name=extraction.source_name,
            document_type=_guess_document_type(extraction.full_text),
            header=_extract_header_hints(extraction.full_text),
            line_items=items,
            notes=extract_document_notes(extraction, header_band_points),
            parser="deterministic",
            warnings=list(extraction.warnings),
        )
    )


# --- Block-based deterministic parser (operates on the clean text stream) -----
_ITEM_START = re.compile(r"(?i)^\s*(?:item|line|ln|no\.?)?\s*#?\s*(\d{1,4})\b")
_MONEY = re.compile(r"\$\s*[0-9][0-9,]*\.\d{2}")
_QTY_LABEL = re.compile(r"(?i)\b(?:qty|quantity)(?:\s*ordered)?\s*[:#]?\s*(\d+)\b")
_QTY_BEFORE_PRICE = re.compile(r"(?i)(\d+(?:\.\d+)?)\s*(?:ea|pcs?|units?|each|set|lot|box|roll)?\s*\$")
_MPN_LABEL = re.compile(
    r"(?i)\b(?:mpn|mfr?\s*part(?:\s*(?:number|no))?|manufacturer\s*part(?:\s*(?:number|no))?)\s*[:#]\s*"
    r"([A-Za-z0-9][A-Za-z0-9._/\-]*)"
)
# Capture the manufacturer as a run of Capitalized/UPPER tokens so wrapped
# lowercase description prose does not leak into the value.
_MFG_LABEL = re.compile(
    r"(?i:\b(?:mfg|mfr|manufacturer|maker)\s*[:#]\s*)([A-Z][A-Za-z0-9.&\-]*(?:\s+[A-Z][A-Za-z0-9.&\-]*){0,3})"
)
_FIELD_LABEL = re.compile(
    r"(?i)\b(?:ipn|part\s*(?:reference|number|no|ref)|description|manufacturer|mpn|qty|quantity)\b\s*[:#]"
)


def _block_scan_parse(extraction: PdfExtraction) -> list[dict[str, Any]]:
    """Parse the text stream into line-item blocks delimited by a leading index.

    Each item begins on a line that opens with a line number ("001", "Item #1",
    "LN 1", ...). Within the block, prices are the $-amounts (unit then extended),
    quantity is a labeled or pre-price integer, and the part number is the first
    delimiter-bearing token. Works for tabular and card layouts alike, and does not
    depend on pdfplumber's (often fragmented) table extraction.
    """

    blocks: list[list[str]] = []
    current: list[str] | None = None
    for raw_line in extraction.full_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--- Page"):
            continue
        if _ITEM_START.match(line):
            if current is not None:
                blocks.append(current)
            current = [line]
        elif current is not None:
            current.append(line)
    if current is not None:
        blocks.append(current)

    items: list[dict[str, Any]] = []
    for block_lines in blocks:
        item = _parse_item_block(" ".join(block_lines))
        if item is not None:
            items.append(item)
    return items


def _parse_item_block(block: str) -> dict[str, Any] | None:
    part = _first_part_like(block)
    if not part:
        return None

    prices = [re.sub(r"\s+", "", token) for token in _MONEY.findall(block)]
    quantity = ""
    qty_match = _QTY_LABEL.search(block) or _QTY_BEFORE_PRICE.search(block)
    if qty_match:
        quantity = qty_match.group(1)

    # Reject non-item blocks (page headers, addresses): a real line item has a
    # price, a quantity, or an explicit BoM field label.
    if not (prices or quantity or _FIELD_LABEL.search(block)):
        return None

    mpn_match = _MPN_LABEL.search(block)
    mfg_match = _MFG_LABEL.search(block)
    start_match = _ITEM_START.match(block)
    mpn = mpn_match.group(1).strip(" -|") if mpn_match else ""
    manufacturer = mfg_match.group(1).strip(" -|") if mfg_match else ""

    item = _blank_item()
    item.update(
        {
            "item_id": start_match.group(1) if start_match else "",
            "part_number": part,
            "mpn": mpn,
            "manufacturer": manufacturer,
            "description": _clean_block_description(block, (part, mpn, manufacturer, quantity)),
            "quantity": quantity,
            "unit_price": prices[0] if prices else "",
            "total_price": prices[1] if len(prices) > 1 else "",
            "raw_text": block[:300],
            "confidence": 0.6,
        }
    )
    return item


def _clean_block_description(block: str, structured_values: tuple[str, ...]) -> str:
    """Isolate the human description from a block, dropping labels, prices, the
    leading index, and the already-extracted structured values (part/mpn/mfg/qty)."""

    text = _MONEY.sub(" ", block)
    text = re.sub(
        r"(?i)\b(?:ipn|mpn|mfg|mfr|manufacturer(?:\s*part(?:\s*(?:number|no))?)?|"
        r"mfr?\s*part(?:\s*(?:number|no))?|part\s*(?:reference|number|no|ref)|"
        r"description|qty|quantity(?:\s*ordered)?|price(?:\s*per\s*unit)?|"
        r"total(?:\s*line\s*val(?:ue)?)?|unit)\b\s*[:#]?",
        " ",
        text,
    )
    text = _ITEM_START.sub(" ", text)
    text = re.sub(r"[•|]", " ", text)
    for value in structured_values:
        if value and len(value) >= 2:
            text = re.sub(re.escape(value), " ", text)
    return " ".join(text.split())[:240]


def parse_tabular(data: bytes, source_name: str) -> ParsedDocument:
    """Parse a CSV or Excel manifest into line items.

    Real BoM/cost sheets carry title and metadata rows above the column header, so
    the header row is *detected* (not assumed to be row 0), its labels are mapped to
    schema fields, and each subsequent row becomes a line item.
    """

    rows = _read_tabular_rows(data, source_name)
    items: list[dict[str, Any]] = []
    if rows:
        header_index, header = _find_header_row(rows)
        if header_index is not None:
            column_map = _map_columns(header)
            data_rows = [[str(cell or "").strip() for cell in raw] for raw in rows[header_index + 1 :]]
            if "part_number" in column_map:
                # BoM/ERP exports place one component across several rows (a
                # description continuation, Vends:/Mfgrs: sub-lines, a cost row).
                # Aggregate each component's rows into a single line item.
                items = _aggregate_tabular_records(data_rows, column_map)
            else:
                for cells in data_rows:
                    if not any(cells):
                        continue
                    item = _row_to_item(cells, column_map)
                    item = _repair_shifted_descriptor_tail(item)
                    item["raw_text"] = " | ".join(cell for cell in cells if cell)
                    item["confidence"] = 0.9
                    if item.get("item_id") or item.get("description"):
                        items.append(item)

    document = _normalize_parsed_document(
        ParsedDocument(
            source_name=source_name,
            document_type=_guess_document_type(source_name),
            line_items=items,
            parser="tabular",
        )
    )
    # Only the columns this sheet actually provided are comparable.
    document.available_fields = [
        field_name
        for field_name in LINE_ITEM_FIELDS
        if any(_clean_value(item.get(field_name)) for item in document.line_items)
    ]
    return document


_SUBLABEL_VENDOR = re.compile(r"(?i)\bvend")
_SUBLABEL_MFG = re.compile(r"(?i)\b(?:mfg|mfr|manufactur)")


def _looks_like_part_value(value: str) -> bool:
    """A component/part cell: has a letter and isn't a bare number (a cost cell)."""

    value = value.strip()
    return bool(value) and bool(re.search(r"[A-Za-z]", value)) and not re.fullmatch(r"[\d.,\s%$]+", value)


def _aggregate_tabular_records(data_rows: list[list[str]], column_map: dict[str, int]) -> list[dict[str, Any]]:
    part_col = column_map["part_number"]
    desc_col = column_map.get("description")

    items: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for cells in data_rows:
        if not any(cells):
            continue
        part = cells[part_col].strip() if part_col < len(cells) else ""

        if _looks_like_part_value(part):
            if current is not None:
                items.append(current)
            current = _blank_item()
            current["part_number"] = part
            current["confidence"] = 0.9
            for field_name, index in column_map.items():
                if field_name == "part_number":
                    continue
                if index < len(cells) and cells[index].strip():
                    current[field_name] = cells[index].strip()
            current["raw_text"] = " | ".join(cell for cell in cells if cell.strip())[:300]
        elif current is not None:
            joined = " ".join(cell for cell in cells if cell.strip())
            if _SUBLABEL_VENDOR.search(joined) and not _clean_value(current.get("vendor")):
                current["vendor"] = _value_after_label(cells, _SUBLABEL_VENDOR)
            elif _SUBLABEL_MFG.search(joined) and not _clean_value(current.get("manufacturer")):
                current["manufacturer"] = _value_after_label(cells, _SUBLABEL_MFG)
            elif (
                desc_col is not None
                and desc_col < len(cells)
                and re.search(r"[A-Za-z]", cells[desc_col])  # skip bare cost/number rows
            ):
                current["description"] = f"{_clean_value(current.get('description'))} {cells[desc_col].strip()}".strip()

    if current is not None:
        items.append(current)
    return items


def _value_after_label(cells: list[str], label_pattern: re.Pattern) -> str:
    for index, cell in enumerate(cells):
        if label_pattern.search(cell):
            for following in cells[index + 1 :]:
                if following.strip():
                    return following.strip()
    return ""


def _read_tabular_rows(data: bytes, source_name: str) -> list[tuple[str, ...]]:
    """Read a CSV/Excel file into rows of strings, tolerant of ragged rows and
    title/metadata lines above the header (rows may have different column counts)."""

    name = source_name.lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        import pandas as pd  # local import: only needed for spreadsheet inputs

        try:
            frame = pd.read_excel(BytesIO(data), dtype=str, header=None)
        except Exception:
            return []
        return [
            tuple("" if _is_missing(value) else str(value).strip() for value in row.tolist())
            for _, row in frame.iterrows()
        ]

    # CSV: the stdlib reader handles ragged rows (a 1-column title line above a
    # 4-column header) that pandas' tokenizer rejects.
    import csv

    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")
    return [tuple(cell.strip() for cell in row) for row in csv.reader(text.splitlines())]


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        import math

        return isinstance(value, float) and math.isnan(value)
    except Exception:
        return False


def _table_scan_parse(extraction: PdfExtraction) -> list[dict[str, Any]]:
    """Legacy column-mapped table parser (fallback for genuinely clean tables)."""

    items: list[dict[str, Any]] = []
    for table in extraction.tables:
        if not table.rows:
            continue
        rows = _coalesce_wrapped_rows(table.rows)
        header_index, header = _find_header_row(rows)
        if header_index is None:
            continue
        column_map = _map_columns(header)
        if not column_map:
            continue
        for raw_row in rows[header_index + 1 :]:
            row = [str(cell or "").strip() for cell in raw_row]
            if not any(row):
                continue
            item = _row_to_item(row, column_map)
            item = _repair_shifted_descriptor_tail(item)
            item["source_page"] = table.page_number
            item["raw_text"] = " | ".join(cell for cell in row if cell)
            item["confidence"] = 0.55
            if item.get("part_number") or item.get("item_id") or item.get("description"):
                items.append(item)
    return items


def _validate_local_url(base_url: str) -> str:
    parsed = urlparse(base_url)
    allowed_hosts = {"localhost", "127.0.0.1", "::1"}
    if parsed.scheme not in {"http", "https"} or parsed.hostname not in allowed_hosts:
        raise ValueError("Ollama base URL must point to localhost or 127.0.0.1")
    return base_url


def _decode_ollama_response(payload: dict[str, Any], source_name: str) -> ParsedDocument:
    response_text = str(payload.get("response", "")).strip()
    if not response_text:
        raise ValueError("empty Ollama response")

    try:
        decoded = json.loads(response_text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", response_text, flags=re.DOTALL)
        if not match:
            raise
        decoded = json.loads(match.group(0))

    if not isinstance(decoded, dict):
        raise ValueError("Ollama response was not a JSON object")

    return ParsedDocument(
        source_name=source_name,
        document_type=str(decoded.get("document_type") or "unknown"),
        header=decoded.get("header") if isinstance(decoded.get("header"), dict) else {},
        line_items=decoded.get("line_items") if isinstance(decoded.get("line_items"), list) else [],
        notes=decoded.get("notes") if isinstance(decoded.get("notes"), list) else [],
    )


def _normalize_parsed_document(document: ParsedDocument) -> ParsedDocument:
    normalized_items: list[dict[str, Any]] = []
    for item in document.line_items:
        if not isinstance(item, dict):
            continue
        item = _repair_shifted_descriptor_tail(item)
        normalized = {field: _clean_value(item.get(field, "")) for field in LINE_ITEM_FIELDS}
        normalized["confidence"] = _coerce_confidence(item.get("confidence"))
        if any(normalized.get(field) for field in ("item_id", "part_number", "description", "raw_text")):
            normalized_items.append(normalized)

    document.line_items = normalized_items
    document.document_type = document.document_type or "unknown"
    document.header = {str(k): _clean_value(v) for k, v in document.header.items()}
    document.notes = [cleaned for note in document.notes if (cleaned := _clean_note(note))]
    return document


def _tables_as_prompt_text(extraction: PdfExtraction, max_rows: int = 650) -> str:
    lines: list[str] = []
    row_count = 0
    for table in extraction.tables:
        # Skip fragmented/garbage tables (no detectable header) so they do not
        # confuse the model; the clean page text already carries the data.
        if _find_header_row(table.rows)[0] is None:
            continue
        lines.append(f"[page {table.page_number}, table {table.table_index}]")
        for row in table.rows:
            lines.append(" | ".join(row))
            row_count += 1
            if row_count >= max_rows:
                lines.append("[table rows truncated]")
                return "\n".join(lines)
    return "\n".join(lines) or "(no usable tables extracted)"


def _find_header_row(rows: tuple[tuple[str, ...], ...]) -> tuple[int | None, tuple[str, ...]]:
    for index, row in enumerate(rows[:8]):
        normalized = " ".join(row).lower()
        score = sum(
            token in normalized
            for token in ("part", "item", "qty", "quantity", "description", "rev", "price", "vendor")
        )
        if score >= 2:
            return index, row
    return None, ()


def _coalesce_wrapped_rows(rows: tuple[tuple[str, ...], ...]) -> tuple[tuple[str, ...], ...]:
    """Merge continuation rows created by wrapped table-cell text."""

    if not rows:
        return rows

    header_index, header = _find_header_row(rows)
    if header_index is None:
        return rows
    column_map = _map_columns(header)
    description_index = column_map.get("description", min(2, len(header) - 1))
    key_indexes = [
        index
        for field in ("item_id", "part_number", "quantity")
        if (index := column_map.get(field)) is not None
    ]

    merged: list[list[str]] = [list(row) for row in rows[: header_index + 1]]
    for raw_row in rows[header_index + 1 :]:
        row = [str(cell or "").strip() for cell in raw_row]
        if not any(row):
            continue
        has_key = any(index < len(row) and row[index] for index in key_indexes)
        if not has_key and len(merged) > header_index + 1:
            continuation = " ".join(cell for cell in row if cell)
            while len(merged[-1]) <= description_index:
                merged[-1].append("")
            merged[-1][description_index] = f"{merged[-1][description_index]} {continuation}".strip()
            continue
        merged.append(row)

    return tuple(tuple(row) for row in merged)


def _map_columns(header: tuple[str, ...]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for index, column in enumerate(header):
        label = re.sub(r"[^a-z0-9]+", " ", column.lower()).strip()
        # Order matters: more specific labels (mpn, line total) are matched before
        # the generic "part"/"price" catch-alls they would otherwise collide with.
        if "manufacturer part" in label or "mfg part" in label or label in {"mpn", "mfg pn", "mfr pn", "mfg p n"}:
            mapping["mpn"] = index
        elif "manufacturer" in label or label in {"mfg", "mfr"}:
            mapping["manufacturer"] = index
        elif label in {"item", "item no", "line", "line no", "ln", "no", "#"}:
            mapping["item_id"] = index
        elif "part" in label or "component" in label or label in {"pn", "p n", "ipn", "material", "component number"}:
            mapping["part_number"] = index
        elif "desc" in label or "name" in label:
            mapping["description"] = index
        elif "qty" in label or "quantity" in label:
            mapping["quantity"] = index
        elif label in {"uom", "unit", "units", "um", "u m", "u om"}:
            mapping["unit"] = index
        elif label in {"rev", "revision", "release"}:
            mapping["revision"] = index
        elif "ext" in label or "line total" in label or "total line" in label or "total price" in label or label in {"total", "amount", "extended"}:
            mapping["total_price"] = index
        elif "unit price" in label or "price per unit" in label or "unit cost" in label or label in {"price", "cost"}:
            mapping["unit_price"] = index
        elif "vendor" in label or "supplier" in label:
            mapping["vendor"] = index
    return mapping


def _row_to_item(row: list[str], column_map: dict[str, int]) -> dict[str, Any]:
    item = {field: "" for field in LINE_ITEM_FIELDS}
    for field, index in column_map.items():
        if index < len(row):
            item[field] = row[index]
    return item


def _repair_shifted_descriptor_tail(item: dict[str, Any]) -> dict[str, Any]:
    """Move obvious wrapped-description tails out of vendor/unit fields."""

    repaired = dict(item)
    description = _clean_value(repaired.get("description"))
    for field in ("vendor", "unit"):
        value = _clean_value(repaired.get(field))
        if not value:
            continue
        if field == "unit" and _normalize_token(value) in VALID_UNITS:
            continue
        if field == "vendor" and not _looks_like_description_tail(value):
            continue
        if field == "unit" and not _looks_like_description_tail(value):
            continue
        if value.casefold() not in description.casefold():
            description = f"{description} {value}".strip()
        repaired[field] = ""

    repaired["description"] = description
    return repaired


def _looks_like_description_tail(value: str) -> bool:
    normalized = _normalize_token(value)
    if normalized in DESCRIPTOR_TAIL_WORDS:
        return True
    return bool(
        re.search(
            r"\b(?:\d+\s*(?:v|vac|vdc|a|amp|amps)|c\s*curve|coil|output|input|red|green|blue|black|white|yellow)\b",
            normalized,
        )
    )


def _line_scan_parse(extraction: PdfExtraction) -> list[dict[str, Any]]:
    rows = flatten_tables(extraction.tables)
    candidates = [" | ".join(row) for row in rows if len([cell for cell in row if cell]) >= 3]
    if not candidates:
        candidates = [
            line
            for line in extraction.full_text.splitlines()
            if PART_LIKE_PATTERN.search(line) and not _is_bracket_note(line)
        ]

    items: list[dict[str, Any]] = []
    for index, line in enumerate(candidates, start=1):
        qty_match = re.search(r"\b(?:qty|quantity)?\s*[:#]?\s*(\d+(?:\.\d+)?)\b", line, flags=re.IGNORECASE)
        item = _blank_item()
        item.update(
            {
                "item_id": str(index),
                "part_number": extract_part_key(_first_part_like(line)),
                "description": line[:240],
                "quantity": qty_match.group(1) if qty_match else "",
                "raw_text": line,
                "confidence": 0.35,
            }
        )
        items.append(item)
    return items


# Vertical "card" layouts present each field as a "Label: value" line instead of a
# table row. Map those labels onto the canonical line-item fields.
_CARD_LABELS = {
    "part": "part_number",
    "part no": "part_number",
    "part number": "part_number",
    "part ref": "part_number",
    "part reference": "part_number",
    "ipn": "part_number",
    "internal part": "part_number",
    "internal part number": "part_number",
    "mpn": "mpn",
    "mfg part": "mpn",
    "mfg pn": "mpn",
    "manufacturer part": "mpn",
    "manufacturer part number": "mpn",
    "mfg": "manufacturer",
    "mfr": "manufacturer",
    "manufacturer": "manufacturer",
    "maker": "manufacturer",
    "desc": "description",
    "description": "description",
    "name": "description",
    "qty": "quantity",
    "quantity": "quantity",
    "uom": "unit",
    "unit": "unit",
    "units": "unit",
    "rev": "revision",
    "revision": "revision",
    "price": "unit_price",
    "unit price": "unit_price",
    "price per unit": "unit_price",
    "cost": "unit_price",
    "unit cost": "unit_price",
    "ext price": "total_price",
    "extended price": "total_price",
    "extended": "total_price",
    "line total": "total_price",
    "total line val": "total_price",
    "total price": "total_price",
    "total": "total_price",
    "vendor": "vendor",
    "supplier": "vendor",
}
_CARD_LINE = re.compile(r"^([A-Za-z][A-Za-z /#._-]{0,28}?)\s*[:#]\s*(.+)$")

# Recognize labelled unit/extended prices, even when both appear on one line such
# as "UNIT PRICE: $85.50 | EXT PRICE: $427.50".
_UNIT_PRICE_RE = re.compile(
    r"(?i)(?:unit\s*price|price\s*per\s*unit|price\s*/\s*unit|unit\s*cost)\s*[:#]?\s*(\$?\s*[0-9][0-9,]*(?:\.\d+)?)"
)
_TOTAL_PRICE_RE = re.compile(
    r"(?i)(?:ext(?:ended)?\.?\s*price|total\s*line\s*val(?:ue)?|line\s*total|total\s*price)\s*[:#]?\s*(\$?\s*[0-9][0-9,]*(?:\.\d+)?)"
)


def parse_price_fields(text: Any) -> dict[str, float | None]:
    """Split a line into isolated unit_price and total_price floats.

    Guarantees zero cross-contamination: the per-unit scalar and the aggregate
    line value are matched by their own labels and never share a slot.
    """

    source = str(text or "")
    unit_match = _UNIT_PRICE_RE.search(source)
    total_match = _TOTAL_PRICE_RE.search(source)
    return {
        "unit_price": normalize_price(unit_match.group(1)) if unit_match else None,
        "total_price": normalize_price(total_match.group(1)) if total_match else None,
    }


def _has_price_pair(line: str) -> bool:
    lowered = line.lower()
    return "price" in lowered and ("unit" in lowered or "per unit" in lowered) and ("ext" in lowered or "total" in lowered or "line" in lowered)


def _card_block_parse(extraction: PdfExtraction) -> list[dict[str, Any]]:
    """Reconstruct line items from vertically stacked "Label: value" cards.

    A blank line or a repeated part-number label starts a new card, so the same
    identity properties are recovered whether the source renders horizontally as a
    table or vertically as card blocks.
    """

    items: list[dict[str, Any]] = []
    current: dict[str, str] = {}

    def flush() -> None:
        nonlocal current
        if current.get("part_number") or current.get("description"):
            item = _blank_item()
            item.update(current)
            item["raw_text"] = " | ".join(f"{key}: {value}" for key, value in current.items())
            item["confidence"] = 0.45
            items.append(_repair_shifted_descriptor_tail(item))
        current = {}

    for raw_line in extraction.full_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--- Page"):
            flush()
            continue
        if _has_price_pair(line):
            prices = parse_price_fields(line)
            if prices["unit_price"] is not None:
                current["unit_price"] = f"{prices['unit_price']:.2f}"
            if prices["total_price"] is not None:
                current["total_price"] = f"{prices['total_price']:.2f}"
            continue
        match = _CARD_LINE.match(line)
        if not match:
            continue
        field = _CARD_LABELS.get(_normalize_token(match.group(1)))
        if not field:
            continue
        if field == "part_number" and current.get("part_number"):
            flush()
        current[field] = match.group(2).strip()

    flush()
    return items


def _first_part_like(line: str) -> str:
    match = PART_LIKE_PATTERN.search(line)
    return match.group(0) if match else ""


def _blank_item() -> dict[str, Any]:
    item = {field: "" for field in LINE_ITEM_FIELDS}
    item["confidence"] = 0.0
    return item


# A part-number fragment left dangling by a line wrap: ends with a hyphen and the
# stem contains a digit or an internal delimiter (so plain words like "high-" are
# left alone).
_TRAILING_FRAGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*-$")
_LEADING_TOKEN = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(.*)$")


def _ends_with_part_fragment(line: str) -> bool:
    stripped = line.rstrip()
    if not stripped.endswith("-"):
        return False
    tokens = stripped.split()
    if not tokens or not _TRAILING_FRAGMENT.fullmatch(tokens[-1]):
        return False
    stem = tokens[-1][:-1]
    return "-" in stem or any(char.isdigit() for char in stem)


def stitch_hyphenated_lines(text: str) -> str:
    """Reconstruct part numbers split across a line wrap.

    When a line ends with a dangling part-number fragment (e.g. "105-ENC-"), the
    leading token of the next line ("AL-01") is stitched on to rebuild the whole
    key ("105-ENC-AL-01"). The remainder of the lower line is preserved; pure
    line-wrap whitespace/noise is stripped.
    """

    raw = text.split("\n")
    result: list[str] = []
    index = 0
    while index < len(raw):
        line = raw[index]
        if index + 1 < len(raw) and _ends_with_part_fragment(line):
            match = _LEADING_TOKEN.match(raw[index + 1])
            if match:
                result.append(line + match.group(1))
                remainder = match.group(2).strip()
                if remainder:
                    raw[index + 1] = remainder
                    index += 1
                else:
                    index += 2
                continue
        result.append(line)
        index += 1
    return "\n".join(result)


def _stitch_extraction(extraction: PdfExtraction) -> PdfExtraction:
    stitched = stitch_hyphenated_lines(extraction.full_text)
    if stitched == extraction.full_text:
        return extraction
    return replace(extraction, full_text=stitched)


# Fields the deterministic table parse can reliably backfill into LLM output.
_BACKFILL_FIELDS = (
    "unit_price",
    "total_price",
    "quantity",
    "unit",
    "revision",
    "mpn",
    "manufacturer",
    "vendor",
    "item_id",
    "description",
)


def backfill_from_deterministic(items: list[dict[str, Any]], extraction: PdfExtraction) -> list[dict[str, Any]]:
    """Fill fields the model dropped from the deterministic table extraction.

    pdfplumber reads structured tables (e.g. a stacked "UNIT / EXT PRICE" column)
    far more reliably than a small LLM. For each parsed item, any empty field is
    backfilled from the deterministically parsed row with the same part-number
    token. Universal: it matches by content, not by template, and only fills gaps.
    """

    try:
        deterministic = fallback_parse(extraction).line_items
    except Exception:
        return items
    source_by_key: dict[str, dict[str, Any]] = {}
    for row in deterministic:
        key = normalize_identifier(part_match_token(row.get("part_number")))
        if key:
            source_by_key.setdefault(key, row)
    if not source_by_key:
        return items

    for item in items:
        if not isinstance(item, dict):
            continue
        source = source_by_key.get(normalize_identifier(part_match_token(item.get("part_number"))))
        if not source:
            continue
        for field_name in _BACKFILL_FIELDS:
            if not _clean_value(item.get(field_name)) and _clean_value(source.get(field_name)):
                item[field_name] = source.get(field_name)
    return items


# Loose / bracketed engineering remarks (e.g. "[QTY DECREASED BY 1 PER ECO]" or
# "[CRITICAL NOTE: ...]") that fall outside formal columns.
_BRACKET_NOTE = re.compile(r"\[[^\]]+\]")
_IPN_LABEL = re.compile(r"(?i)\bIPN\b\s*[:#]?\s*([A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)+)")
# A leading part token, tolerating a leading line-number such as "002 ".
_LEADING_PART = re.compile(r"\s*(?:\d+\s+)?([A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)+)")


def _is_bracket_note(line: str) -> bool:
    return line.strip().startswith("[")


def _row_parent_ipn(line: str) -> str:
    """Identify the parent IPN that a row introduces.

    Prefers an explicit ``IPN:`` label, otherwise the first part-number token at
    the start of the row (tolerating a leading line-number like "002"). This stops
    the active parent from going stale on real layouts, which is what let a note
    leak from one part onto the next.
    """

    labelled = _IPN_LABEL.search(line)
    if labelled:
        return labelled.group(1)
    leading = _LEADING_PART.match(line)
    return leading.group(1) if leading else ""


def cluster_contextual_notes(full_text: str) -> dict[str, list[str]]:
    """Bind loose / bracketed remarks to their owning parent IPN.

    Walks the raw text stream; the active parent updates whenever a line actually
    introduces an IPN. A bracket-only remark never becomes a parent, and a remark
    sharing a line with an IPN binds to that same-line IPN. Returns
    ``{normalized_ipn: [notes...]}``; remarks before any item key to "" and are
    dropped during attachment.
    """

    clusters: dict[str, list[str]] = {}
    current_parent = ""
    pending: list[str] = []  # notes seen before any parent -> bind to the first one
    for raw_line in full_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("--- Page"):
            continue
        if not _is_bracket_note(line):
            ipn = _row_parent_ipn(line)
            if ipn:
                current_parent = normalize_identifier(ipn)
                if pending:
                    clusters.setdefault(current_parent, []).extend(pending)
                    pending = []
        for note in _BRACKET_NOTE.findall(line):
            if current_parent:
                clusters.setdefault(current_parent, []).append(note.strip())
            else:
                pending.append(note.strip())
    return clusters


def _merge_notes(item: dict[str, Any], notes: list[str]) -> None:
    """Append notes to an item's contextual_notes, preserving spacing and de-duping."""

    combined = _clean_value(item.get("contextual_notes"))
    for note in notes:
        note = note.strip()
        if note and note not in combined:
            combined = f"{combined} {note}".strip()
    item["contextual_notes"] = combined


def _split_inline_notes(item: dict[str, Any]) -> None:
    """Decouple bracketed remarks from this row's description into contextual_notes.

    An inline note must never remain fused to the generic description (it would
    distort both the text diff and the bounding-box geometry).
    """

    description = _clean_value(item.get("description"))
    brackets = _BRACKET_NOTE.findall(description)
    if not brackets:
        return
    item["description"] = _clean_value(_BRACKET_NOTE.sub(" ", description))
    _merge_notes(item, brackets)


def attach_contextual_notes(items: list[dict[str, Any]], full_text: str) -> list[dict[str, Any]]:
    """Decouple inline notes and bind clustered remarks to each item's parent IPN."""

    clusters = cluster_contextual_notes(full_text)
    for item in items:
        if not isinstance(item, dict):
            continue
        # Per-item: pull this row's own inline brackets out of description first,
        # then add notes clustered to this exact IPN from the raw text stream.
        _split_inline_notes(item)
        notes = clusters.get(normalize_identifier(item.get("part_number")))
        if notes:
            _merge_notes(item, notes)
    return items


def _guess_document_type(text: str) -> str:
    lowered = text.lower()
    if "bill of materials" in lowered or re.search(r"\bbom\b", lowered):
        return "BOM"
    if "purchase order" in lowered or re.search(r"\bpo\b", lowered):
        return "PO"
    if "engineering change" in lowered or re.search(r"\beco\b", lowered):
        return "ECO"
    return "unknown"


def _extract_header_hints(text: str) -> dict[str, str]:
    """Extract only cleanly-labeled document metadata.

    Conservative on purpose: loosely-matched fields (a bare 'Rev' inside a
    description, a title scraped from prose) produced garbage document-level
    changes, so only an explicitly labeled document number and date are captured.
    """

    hints: dict[str, str] = {}
    patterns = {
        "document_number": r"(?i:\b(?:purchase\s*order|p\.?\s*o\.?|document|doc|bom|eco)\s*(?:number|no\.?|#)?\s*[:\-]\s*)([A-Z0-9][A-Z0-9._\-]+)",
        "date": r"(?i:\bdate\s*[:\-]\s*)([A-Za-z]{3,}\.?\s+\d{1,2},?\s+\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if match:
            hints[key] = _clean_value(match.group(1))
    return hints


# A notes block is only recognised when it is introduced by an explicit heading.
# Without an anchor we return nothing rather than fallback-scraping page header
# metadata as "generic document text".
_NOTE_HEADING = re.compile(
    r"(?i)\b(?:revision\s+change\s+engineering\s+notes"
    r"|engineering\s+change\s+notes"
    r"|revision\s+history"
    r"|change\s+(?:summary|log|notes))\b\s*[:\-]?"
)
# Lines that mark the end of a notes block (footer, signatures, header fields
# that may sit below the notes box, or page chrome).
_NOTE_STOP = re.compile(
    r"(?i)^(?:approved\s+by|prepared\s+by|checked\s+by|reviewed\s+by|signature"
    r"|page\s+\d+\s+of\s+\d+|printed\s+(?:on|by)|confidential"
    r"|document\s+type\s*:|system\s+id\s*:|project\s+name\s*:|status\s*:)"
)


def extract_document_notes(
    extraction: PdfExtraction, header_band_points: float = HEADER_BAND_POINTS
) -> list[str]:
    """Extract engineering/revision note blocks deterministically.

    Coordinate bounding excludes the top ``header_band_points`` of each page, and a
    keyword anchor (:data:`_NOTE_HEADING`) is required before any text is captured.
    If no page contains the anchor, an empty list is returned instead of unmapped
    text.
    """

    notes: list[str] = []
    for page in extraction.pages:
        body = page_body_text(page, top_margin=header_band_points)
        if not body:
            continue
        match = _NOTE_HEADING.search(body)
        if not match:
            continue
        block = _capture_note_block(body[match.start() :])
        if block and block not in notes:
            notes.append(block)
    return notes


def _capture_note_block(segment: str, max_lines: int = 40) -> str:
    collected: list[str] = []
    for raw_line in segment.splitlines():
        line = _clean_value(raw_line)
        if not line:
            continue
        if collected and _NOTE_STOP.search(line):
            break
        collected.append(line)
        if len(collected) >= max_lines:
            break
    return "\n".join(collected)


def _clean_value(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())


def _clean_note(value: Any) -> str:
    """Clean a note while preserving line breaks for display and diffing."""

    text = str(value or "").replace("\x00", "")
    lines = [" ".join(line.split()) for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


def _normalize_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", _clean_value(value).casefold()).strip()


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
