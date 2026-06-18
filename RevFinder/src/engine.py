"""Deterministic revision comparison engine."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any

import numpy as np
import pandas as pd

from .llm_parser import LINE_ITEM_FIELDS, ParsedDocument


COMPARE_FIELDS = [
    "part_number",
    "description",
    "quantity",
    "unit",
    "revision",
    "manufacturer",
    "vendor",
    "price",
]
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
VALID_UNITS = {"ea", "each", "pc", "pcs", "piece", "pieces", "ft", "in", "mm", "m", "set", "lot", "box", "roll"}


@dataclass
class FieldChange:
    field: str
    old_value: str
    new_value: str


@dataclass
class RowChange:
    status: str
    match_key: str
    old_item: dict[str, Any] = field(default_factory=dict)
    new_item: dict[str, Any] = field(default_factory=dict)
    field_changes: list[FieldChange] = field(default_factory=list)
    similarity: float = 0.0


@dataclass
class DiffResult:
    old_document: ParsedDocument
    new_document: ParsedDocument
    changes: list[RowChange]
    old_items: pd.DataFrame
    new_items: pd.DataFrame
    comparison: pd.DataFrame
    document_changes: pd.DataFrame
    summary: dict[str, int]


def compare_documents(old_document: ParsedDocument, new_document: ParsedDocument) -> DiffResult:
    old_df = _items_to_dataframe(old_document)
    new_df = _items_to_dataframe(new_document)
    old_indexed = _index_by_match_key(old_df)
    new_indexed = _index_by_match_key(new_df)

    changes: list[RowChange] = []
    all_keys = sorted(set(old_indexed) | set(new_indexed))

    for key in all_keys:
        old_item = old_indexed.get(key)
        new_item = new_indexed.get(key)

        if old_item is None and new_item is not None:
            changes.append(RowChange(status="added", match_key=key, new_item=new_item))
            continue
        if new_item is None and old_item is not None:
            changes.append(RowChange(status="removed", match_key=key, old_item=old_item))
            continue
        if old_item is None or new_item is None:
            continue

        field_changes = [
            FieldChange(field=field, old_value=_display(old_item.get(field)), new_value=_display(new_item.get(field)))
            for field in COMPARE_FIELDS
            if not _values_equal(old_item.get(field), new_item.get(field), numeric=field in {"quantity", "price"})
        ]
        changes.append(
            RowChange(
                status="modified" if field_changes else "unchanged",
                match_key=key,
                old_item=old_item,
                new_item=new_item,
                field_changes=field_changes,
                similarity=_row_similarity(old_item, new_item),
            )
        )

    summary = {
        "old_items": int(len(old_df)),
        "new_items": int(len(new_df)),
        "added": sum(change.status == "added" for change in changes),
        "removed": sum(change.status == "removed" for change in changes),
        "modified": sum(change.status == "modified" for change in changes),
        "unchanged": sum(change.status == "unchanged" for change in changes),
    }
    document_changes = _document_changes_dataframe(old_document, new_document)
    summary["document_changes"] = int(len(document_changes))

    return DiffResult(
        old_document=old_document,
        new_document=new_document,
        changes=changes,
        old_items=old_df,
        new_items=new_df,
        comparison=_comparison_dataframe(changes),
        document_changes=document_changes,
        summary=summary,
    )


def filter_changes(diff: DiffResult, status: str) -> pd.DataFrame:
    if diff.comparison.empty or "status" not in diff.comparison.columns:
        return pd.DataFrame(columns=_comparison_columns())
    return diff.comparison[diff.comparison["status"] == status].copy()


def discrepancy_log(diff: DiffResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for change in diff.changes:
        if change.status == "added":
            rows.append(
                {
                    "severity": "review",
                    "status": "added",
                    "match_key": change.match_key,
                    "field": "",
                    "old_value": "",
                    "new_value": _item_label(change.new_item),
                    "message": "Line item appears only in the new revision.",
                }
            )
        elif change.status == "removed":
            rows.append(
                {
                    "severity": "review",
                    "status": "removed",
                    "match_key": change.match_key,
                    "field": "",
                    "old_value": _item_label(change.old_item),
                    "new_value": "",
                    "message": "Line item appears only in the old revision.",
                }
            )
        elif change.status == "modified":
            for field_change in change.field_changes:
                rows.append(
                    {
                        "severity": _severity_for_field(field_change.field),
                        "status": "modified",
                        "match_key": change.match_key,
                        "field": field_change.field,
                        "old_value": field_change.old_value,
                        "new_value": field_change.new_value,
                        "message": f"{field_change.field} changed.",
                    }
                )
    for _, change in diff.document_changes.iterrows():
        rows.append(
            {
                "severity": "high" if change["field"] in {"revision", "release_date", "notes"} else "medium",
                "status": "document_changed",
                "match_key": "document",
                "field": change["field"],
                "old_value": change["old_value"],
                "new_value": change["new_value"],
                "message": f"Document-level {change['field']} changed.",
            }
        )
    return pd.DataFrame(rows)


def _items_to_dataframe(document: ParsedDocument) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(document.line_items, start=1):
        item = _repair_shifted_descriptor_tail(item)
        row = {field: _display(item.get(field)) for field in LINE_ITEM_FIELDS}
        row["_row_number"] = index
        row["_source_name"] = document.source_name
        rows.append(row)
    columns = [*LINE_ITEM_FIELDS, "_row_number", "_source_name"]
    return pd.DataFrame(rows, columns=columns).fillna("")


def _index_by_match_key(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    counters: dict[str, int] = {}
    indexed: dict[str, dict[str, Any]] = {}
    for _, row in df.iterrows():
        item = row.to_dict()
        base_key = _base_match_key(item)
        counters[base_key] = counters.get(base_key, 0) + 1
        key = base_key if counters[base_key] == 1 else f"{base_key}__{counters[base_key]}"
        item["_match_key"] = key
        indexed[key] = item
    return indexed


def _base_match_key(item: dict[str, Any]) -> str:
    for field in ("part_number", "item_id"):
        normalized = _normalize_identifier(item.get(field))
        if normalized:
            return f"{field}:{normalized}"

    raw = " ".join(_display(item.get(field)) for field in ("description", "quantity", "revision"))
    digest = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]
    return f"row:{digest}"


def _comparison_dataframe(changes: list[RowChange]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for change in changes:
        old_item = change.old_item or {}
        new_item = change.new_item or {}
        rows.append(
            {
                "status": change.status,
                "match_key": change.match_key,
                "part_number_old": _display(old_item.get("part_number")),
                "part_number_new": _display(new_item.get("part_number")),
                "description_old": _display(old_item.get("description")),
                "description_new": _display(new_item.get("description")),
                "quantity_old": _display(old_item.get("quantity")),
                "quantity_new": _display(new_item.get("quantity")),
                "revision_old": _display(old_item.get("revision")),
                "revision_new": _display(new_item.get("revision")),
                "changed_fields": ", ".join(field_change.field for field_change in change.field_changes),
                "similarity": round(float(change.similarity), 4),
            }
        )
    return pd.DataFrame(rows, columns=_comparison_columns())


def _comparison_columns() -> list[str]:
    return [
        "status",
        "match_key",
        "part_number_old",
        "part_number_new",
        "description_old",
        "description_new",
        "quantity_old",
        "quantity_new",
        "revision_old",
        "revision_new",
        "changed_fields",
        "similarity",
    ]


def _document_changes_dataframe(old_document: ParsedDocument, new_document: ParsedDocument) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    header_keys = sorted(set(old_document.header) | set(new_document.header))
    for key in header_keys:
        old_value = _display(old_document.header.get(key))
        new_value = _display(new_document.header.get(key))
        if _normalize_text(old_value) != _normalize_text(new_value):
            rows.append(
                {
                    "section": "header",
                    "field": key,
                    "old_value": old_value,
                    "new_value": new_value,
                }
            )

    old_notes = _normalize_notes(old_document.notes)
    new_notes = _normalize_notes(new_document.notes)
    if old_notes != new_notes:
        rows.append(
            {
                "section": "notes",
                "field": "notes",
                "old_value": "\n".join(old_document.notes),
                "new_value": "\n".join(new_document.notes),
            }
        )

    return pd.DataFrame(rows, columns=["section", "field", "old_value", "new_value"])


def _values_equal(old_value: Any, new_value: Any, numeric: bool = False) -> bool:
    if numeric:
        old_number = _to_number(old_value)
        new_number = _to_number(new_value)
        if old_number is not None and new_number is not None:
            return math.isclose(old_number, new_number, rel_tol=1e-9, abs_tol=1e-9)
    return _normalize_text(old_value) == _normalize_text(new_value)


def _row_similarity(old_item: dict[str, Any], new_item: dict[str, Any]) -> float:
    old_text = " ".join(_normalize_text(old_item.get(field)) for field in COMPARE_FIELDS)
    new_text = " ".join(_normalize_text(new_item.get(field)) for field in COMPARE_FIELDS)
    if not old_text and not new_text:
        return 1.0
    return float(SequenceMatcher(None, old_text, new_text).ratio())


def _to_number(value: Any) -> float | None:
    text = _display(value)
    if not text:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", ".", "-."}:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def _normalize_identifier(value: Any) -> str:
    return re.sub(r"\s+", "", _display(value)).upper()


def _normalize_text(value: Any) -> str:
    return " ".join(_display(value).casefold().split())


def _normalize_notes(notes: list[str]) -> str:
    return "\n".join(_normalize_text(note) for note in notes if _normalize_text(note))


def _repair_shifted_descriptor_tail(item: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(item)
    description = _display(repaired.get("description"))
    for field in ("vendor", "unit"):
        value = _display(repaired.get(field))
        if not value:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        if field == "unit" and normalized in VALID_UNITS:
            continue
        if normalized not in DESCRIPTOR_TAIL_WORDS and not re.search(
            r"\b(?:\d+\s*(?:v|vac|vdc|a|amp|amps)|c\s*curve|coil|output|input|red|green|blue|black|white|yellow)\b",
            normalized,
        ):
            continue
        if value.casefold() not in description.casefold():
            description = f"{description} {value}".strip()
        repaired[field] = ""
    repaired["description"] = description
    return repaired


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and np.isnan(value):
        return ""
    return " ".join(str(value).replace("\x00", "").split())


def _item_label(item: dict[str, Any]) -> str:
    return _display(item.get("part_number")) or _display(item.get("description")) or _display(item.get("item_id"))


def _severity_for_field(field: str) -> str:
    if field in {"quantity", "revision", "part_number", "price"}:
        return "high"
    if field in {"manufacturer", "vendor"}:
        return "medium"
    return "low"
