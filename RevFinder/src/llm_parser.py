"""Structured engineering document parsing through local Ollama."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any
from urllib.parse import urlparse

import requests

from .extractor import PdfExtraction, flatten_tables


DEFAULT_MODEL = "llama3.2"
DEFAULT_OLLAMA_URL = "http://localhost:11434"

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
    ) -> None:
        self.base_url = _validate_local_url(base_url).rstrip("/")
        self.model = model
        self.timeout_seconds = timeout_seconds
        self.max_prompt_chars = max_prompt_chars

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
            parsed.warnings.extend(extraction.warnings)
            return _normalize_parsed_document(parsed)
        except Exception as exc:
            fallback = fallback_parse(extraction)
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

Source file: {extraction.source_name}

Extracted tables:
{tables}

Extracted page text:
{source_text}
""".strip()


def fallback_parse(extraction: PdfExtraction) -> ParsedDocument:
    """Best-effort deterministic parser for table-shaped engineering documents."""

    items: list[dict[str, Any]] = []
    for table in extraction.tables:
        if not table.rows:
            continue
        header_index, header = _find_header_row(table.rows)
        if header_index is None:
            continue

        column_map = _map_columns(header)
        if not column_map:
            continue

        for raw_row in table.rows[header_index + 1 :]:
            row = [str(cell or "").strip() for cell in raw_row]
            if not any(row):
                continue
            item = _row_to_item(row, column_map)
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
        normalized = {field: _clean_value(item.get(field, "")) for field in LINE_ITEM_FIELDS}
        normalized["confidence"] = _coerce_confidence(item.get("confidence"))
        if any(normalized.get(field) for field in ("item_id", "part_number", "description", "raw_text")):
            normalized_items.append(normalized)

    document.line_items = normalized_items
    document.document_type = document.document_type or "unknown"
    document.header = {str(k): _clean_value(v) for k, v in document.header.items()}
    document.notes = [_clean_value(note) for note in document.notes if _clean_value(note)]
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
        "revision": r"(?:revision|rev)\s*[:\-]?\s*([A-Z0-9._\-]+)",
        "date": r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{4}-\d{2}-\d{2})\b",
    }
    for key, pattern in patterns.items():
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            hints[key] = match.group(1)
    return hints


def _clean_value(value: Any) -> str:
    return " ".join(str(value or "").replace("\x00", "").split())


def _coerce_confidence(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0
