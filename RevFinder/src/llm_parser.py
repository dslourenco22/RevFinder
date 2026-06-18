"""Structured engineering document parsing through local Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from .extractor import HEADER_BAND_POINTS, PdfExtraction, flatten_tables, page_body_text


DEFAULT_MODEL = "llama3.2"
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
    "description",
    "quantity",
    "unit",
    "revision",
    "manufacturer",
    "vendor",
    "price",
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
        payload = {
            "model": self.model,
            "prompt": self._build_prompt(extraction),
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0,
                "top_p": 0.1,
                "num_ctx": 8192,
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
            parsed.warnings.extend(extraction.warnings)
            return _normalize_parsed_document(parsed)
        except Exception as exc:
            fallback = fallback_parse(extraction, self.header_band_points)
            fallback.warnings.append(f"Ollama parse unavailable; used deterministic fallback: {exc}")
            return fallback

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
      "description": "",
      "quantity": "",
      "unit": "",
      "revision": "",
      "manufacturer": "",
      "vendor": "",
      "price": "",
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
- Preserve engineering identifiers, decimals, dash suffixes, and revision letters exactly.
- Use source_page when it can be inferred.
- Confidence must be a number from 0 to 1.
- Only put text that appears under an explicit heading such as "Revision Change Engineering Notes", "Engineering Change Notes", or "Revision History" in notes. Never put the company name, page header, or document metadata fields in notes.
- If a table description wraps to multiple visual lines, merge the wrapped text back into description. Do not put words like color names, Coil, Curve, Input, or Output in vendor/unit unless the source explicitly labels them there.

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

    if not items:
        items = _line_scan_parse(extraction)

    return _normalize_parsed_document(
        ParsedDocument(
            source_name=extraction.source_name,
            document_type=_guess_document_type(extraction.full_text),
            header=_extract_header_hints(extraction.full_text),
            line_items=items,
            notes=extract_document_notes(extraction, header_band_points),
            parser="deterministic-fallback",
            warnings=list(extraction.warnings),
        )
    )


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
        lines.append(f"[page {table.page_number}, table {table.table_index}]")
        for row in table.rows:
            lines.append(" | ".join(row))
            row_count += 1
            if row_count >= max_rows:
                lines.append("[table rows truncated]")
                return "\n".join(lines)
    return "\n".join(lines) or "(no tables extracted)"


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
        if label in {"item", "item no", "line", "line no", "no", "#"}:
            mapping["item_id"] = index
        elif "part" in label or label in {"pn", "p n"}:
            mapping["part_number"] = index
        elif "desc" in label or "name" in label:
            mapping["description"] = index
        elif "qty" in label or "quantity" in label:
            mapping["quantity"] = index
        elif label in {"uom", "unit", "units"}:
            mapping["unit"] = index
        elif label in {"rev", "revision", "release"}:
            mapping["revision"] = index
        elif "manufacturer" in label or label == "mfg":
            mapping["manufacturer"] = index
        elif "vendor" in label or "supplier" in label:
            mapping["vendor"] = index
        elif "price" in label or "cost" in label:
            mapping["price"] = index
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
            if re.search(r"\b[A-Z0-9]{2,}[-_][A-Z0-9][A-Z0-9._-]*\b", line, flags=re.IGNORECASE)
        ]

    items: list[dict[str, Any]] = []
    for index, line in enumerate(candidates, start=1):
        part_match = re.search(r"\b[A-Z0-9]{2,}[-_][A-Z0-9][A-Z0-9._-]*\b", line, flags=re.IGNORECASE)
        qty_match = re.search(r"\b(?:qty|quantity)?\s*[:#]?\s*(\d+(?:\.\d+)?)\b", line, flags=re.IGNORECASE)
        items.append(
            {
                "item_id": str(index),
                "part_number": part_match.group(0) if part_match else "",
                "description": line[:240],
                "quantity": qty_match.group(1) if qty_match else "",
                "unit": "",
                "revision": "",
                "manufacturer": "",
                "vendor": "",
                "price": "",
                "source_page": "",
                "raw_text": line,
                "confidence": 0.35,
            }
        )
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
    hints: dict[str, str] = {}
    patterns = {
        "document_number": r"(?:document|doc|po|bom|eco)\s*(?:number|no\.?|#)\s*[:\-]?\s*([A-Z0-9._\-]+)",
        "revision": r"(?:document\s*)?(?:revision|rev)\s*[:#\-]?\s*([A-Z0-9._\-]+)",
        "release_date": r"(?:release|released|effective)?\s*date\s*[:\-]?\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})",
        "date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2}|[A-Z][a-z]+\s+\d{1,2},?\s+\d{4})\b",
        "title": r"(?:title|description)\s*[:\-]\s*(.{4,120})",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hints[key] = match.group(1)
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
