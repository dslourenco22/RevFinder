"""Deterministic revision comparison engine.

Reconciliation is a relational full outer join over part-number (IPN) keyed hash
maps of strongly-typed :class:`~src.models.LineItem` objects (see ``models.py``).
Baseline and revised maps are built independently, so historical values are
always retrieved by key — there is no positional indexing and no shared state.
"""

from __future__ import annotations

import hashlib
import re
import sys
from dataclasses import dataclass, field
from typing import Any, TextIO

import numpy as np
import pandas as pd

from .llm_parser import LINE_ITEM_FIELDS, ParsedDocument
from .models import (
    COMPARE_FIELDS,
    Delta,
    DocumentMetadata,
    LineItem,
    build_item_map,
    field_severity,
    format_log_prefix,
    format_modification,
    full_outer_join,
)
from .normalize import (
    collapse_kerning,
    implied_unit_price,
    normalize_identifier,
    normalize_item_label,
    normalize_price,
    normalize_quantity,
    part_match_token,
)


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

# Map relational join verbs to the report's status vocabulary.
_STATUS_FROM_DELTA = {"ADDED": "added", "DELETED": "removed", "MODIFIED": "modified", "UNCHANGED": "unchanged"}


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
    warnings: list[str] = field(default_factory=list)


def compare_documents(old_document: ParsedDocument, new_document: ParsedDocument) -> DiffResult:
    old_df = _items_to_dataframe(old_document)
    new_df = _items_to_dataframe(new_document)

    # Two isolated, IPN-keyed hash maps -> relational full outer join.
    baseline_map = _build_item_map(old_df)
    revised_map = _build_item_map(new_df)
    # Only compare fields both documents can provide; a column that exists in one
    # file but not the other is a schema difference, not a content change. PDFs
    # advertise all fields (available_fields=None), so they are unrestricted.
    shared = _available_fields(old_document) & _available_fields(new_document)
    compare_fields = [field for field in COMPARE_FIELDS if field in shared]
    changes = [
        _row_change_from_delta(delta)
        for delta in full_outer_join(baseline_map, revised_map, compare_fields=compare_fields)
    ]

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
        warnings=[],
    )


def _available_fields(document: ParsedDocument) -> set[str]:
    """Comparable fields a document can provide (None advertises all — e.g. PDFs)."""

    available = getattr(document, "available_fields", None)
    if available is None:
        return set(COMPARE_FIELDS)
    return set(available) & set(COMPARE_FIELDS)


def _row_change_from_delta(delta: Delta) -> RowChange:
    return RowChange(
        status=_STATUS_FROM_DELTA[delta.change_type],
        match_key=delta.key,
        old_item=dict(delta.old.raw) if delta.old is not None else {},
        new_item=dict(delta.new.raw) if delta.new is not None else {},
        field_changes=[FieldChange(fc.field, fc.old_value, fc.new_value) for fc in delta.field_changes],
        similarity=delta.similarity,
    )


def document_metadata(document: ParsedDocument) -> DocumentMetadata:
    """Expose the typed document-level metadata schema for a parsed document."""

    return DocumentMetadata.from_header(document.header)


def filter_changes(diff: DiffResult, status: str) -> pd.DataFrame:
    if diff.comparison.empty or "status" not in diff.comparison.columns:
        return pd.DataFrame(columns=_comparison_columns())
    return diff.comparison[diff.comparison["status"] == status].copy()


def discrepancy_log(diff: DiffResult) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for change in diff.changes:
        if change.status == "added":
            label = _item_label(change.new_item)
            rows.append(
                {
                    "severity": "review",
                    "status": "added",
                    "match_key": change.match_key,
                    "field": "item",
                    "old_value": "",
                    "new_value": label,
                    "message": format_log_prefix("review", "added", label, "item") + format_modification(None, label),
                }
            )
        elif change.status == "removed":
            label = _item_label(change.old_item)
            rows.append(
                {
                    "severity": "review",
                    "status": "removed",
                    "match_key": change.match_key,
                    "field": "item",
                    "old_value": label,
                    "new_value": "",
                    "message": format_log_prefix("review", "removed", label, "item") + format_modification(label, None),
                }
            )
        elif change.status == "modified":
            part = _item_label(change.new_item) or _item_label(change.old_item)
            for field_change in change.field_changes:
                severity = field_severity(field_change.field)
                rows.append(
                    {
                        "severity": severity,
                        "status": "modified",
                        "match_key": change.match_key,
                        "field": field_change.field,
                        "old_value": field_change.old_value,
                        "new_value": field_change.new_value,
                        "message": format_log_prefix(severity, "modified", part, field_change.field)
                        + format_modification(field_change.old_value, field_change.new_value),
                    }
                )
    for _, change in diff.document_changes.iterrows():
        severity = "high" if change["field"] in {"revision", "release_date", "notes"} else "medium"
        rows.append(
            {
                "severity": severity,
                "status": "document_changed",
                "match_key": "document",
                "field": change["field"],
                "old_value": change["old_value"],
                "new_value": change["new_value"],
                "message": format_log_prefix(severity, "document_changed", "document", change["field"])
                + format_modification(change["old_value"], change["new_value"]),
            }
        )
    return pd.DataFrame(rows)


def print_discrepancy_log(diff: DiffResult, stream: TextIO | None = None) -> None:
    """Emit the pipe-delimited discrepancy log to a stream (default stdout).

    This is the single sanctioned log surface; no legacy squashed format is used.
    """

    stream = stream or sys.stdout
    log = discrepancy_log(diff)
    for _, row in log.iterrows():
        print(row["message"], file=stream)


def _items_to_dataframe(document: ParsedDocument) -> pd.DataFrame:
    rows = []
    for index, item in enumerate(document.line_items, start=1):
        item = _repair_shifted_descriptor_tail(item)
        item = _normalize_money_fields(item)
        item = _fill_implied_unit_price(item)
        row = {field_name: _display(item.get(field_name)) for field_name in LINE_ITEM_FIELDS}
        # Consistent, human display (ERP exports pad with trailing zeros).
        row["quantity"] = _display_quantity(row["quantity"])
        row["unit_price"] = _display_price(row["unit_price"])
        row["total_price"] = _display_price(row["total_price"])
        row["_row_number"] = index
        row["_source_name"] = document.source_name
        rows.append(row)
    columns = [*LINE_ITEM_FIELDS, "_row_number", "_source_name"]
    return pd.DataFrame(rows, columns=columns).fillna("")


def _build_item_map(df: pd.DataFrame) -> dict[str, LineItem]:
    items = [LineItem.from_raw(row.to_dict(), index) for index, (_, row) in enumerate(df.iterrows(), start=1)]
    return build_item_map(items, key_func=lambda line_item: _base_match_key(line_item.raw))


def _base_match_key(item: dict[str, Any]) -> str:
    # The Internal Part Number (IPN) is the absolute primary key. Manufacturer
    # fields (mpn/manufacturer) are nested properties and never form their own
    # key, so an MPN/MFG-only line cannot become a false add/remove.
    part_key = normalize_identifier(part_match_token(item.get("part_number")))
    if part_key:
        return f"part_number:{part_key}"

    # Fall back to the line label only when no IPN exists, normalizing padding and
    # prefixes so "LN: 001", "Item #1", and "LN: 1" collapse to one key.
    item_label = normalize_item_label(item.get("item_id"))
    if item_label:
        return f"item_id:{item_label}"

    # Unpriced / IPN-less items (drawings, notes, reference material) are anchored
    # by their description; identical descriptions are disambiguated by structural
    # position via the per-key counter in build_item_map. This keeps quantity and
    # price fluctuations visible as field changes instead of breaking the match.
    description = _normalize_text(item.get("description"))
    if description:
        digest = hashlib.sha1(description.encode("utf-8")).hexdigest()[:12]
        return f"desc:{digest}"

    raw = " ".join(_display(item.get(field_name)) for field_name in ("quantity", "revision"))
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
        if _normalize_metadata(old_value) != _normalize_metadata(new_value):
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


# Quantitative directives embedded in contextual notes, e.g. "QTY DECREASED BY 1".
_QTY_DIRECTIVE = re.compile(r"(?i)q(?:ty|uantity)\s+(decreased|reduced|increased|raised)\s+by\s+(\d+)")


def _qty_directive(note: Any) -> tuple[str, int] | None:
    match = _QTY_DIRECTIVE.search(_display(note))
    if not match:
        return None
    direction = "down" if match.group(1).lower() in {"decreased", "reduced"} else "up"
    return direction, int(match.group(2))


def _qty_change_matches(directive: tuple[str, int], old_item: dict[str, Any], new_item: dict[str, Any]) -> bool:
    direction, amount = directive
    old_qty = normalize_quantity(old_item.get("quantity"))
    new_qty = normalize_quantity(new_item.get("quantity"))
    if old_qty is None or new_qty is None:
        return False
    if direction == "down":
        return old_qty - new_qty == amount
    return new_qty - old_qty == amount


def note_quantity_consistent(note: Any, old_quantity: Any, new_quantity: Any) -> bool | None:
    """Tie-breaker: does a note's quantitative directive agree with the qty change?

    Returns True/False when the note carries an explicit directive (e.g. "QTY
    DECREASED BY 1"), or None when there is no quantitative directive. This is a
    *tie-breaker* signal for ambiguous (equidistant) note assignment only; it never
    vetoes a note that proximity has already bound to its parent.
    """

    directive = _qty_directive(note)
    if directive is None:
        return None
    return _qty_change_matches(directive, {"quantity": old_quantity}, {"quantity": new_quantity})


# Universal money parsing (not template-specific): a currency-tagged token, and a
# bare decimal money token like "325.00".
_CURRENCY_TOKEN = re.compile(r"\$\s*[0-9][0-9,]*(?:\.\d+)?")
_DECIMAL_MONEY = re.compile(r"\b[0-9][0-9,]*\.\d{2}\b")


def _money_tokens(text: str) -> list[str]:
    return _CURRENCY_TOKEN.findall(text) or _DECIMAL_MONEY.findall(text)


def _normalize_money_fields(item: dict[str, Any]) -> dict[str, Any]:
    """Recover prices stacked in one field or dropped from a row's price columns.

    Universal money parsing, not a template rule: it only acts on currency/decimal
    money patterns, so it is a no-op on genuinely unpriced documents.
      1. A unit_price holding two stacked values (unit over extended) -> unit + total.
      2. Both price fields empty -> harvest currency-tagged amounts from raw_text
         (smallest count: one -> unit; two+ -> unit + extended).
    """

    repaired = dict(item)
    unit = _display(repaired.get("unit_price"))
    total = _display(repaired.get("total_price"))

    if unit and not total:
        tokens = _money_tokens(unit)
        if len(tokens) >= 2:
            repaired["unit_price"] = tokens[0].strip()
            repaired["total_price"] = tokens[1].strip()
            return repaired

    # A combined "UNIT / EXT PRICE" column often lands both values in total_price.
    if total and not unit:
        tokens = _money_tokens(total)
        if len(tokens) >= 2:
            repaired["unit_price"] = tokens[0].strip()
            repaired["total_price"] = tokens[1].strip()
            return repaired

    if not unit and not total:
        tokens = [token.strip() for token in _CURRENCY_TOKEN.findall(_display(repaired.get("raw_text")))]
        if len(tokens) >= 2:
            repaired["unit_price"] = tokens[0]
            repaired["total_price"] = tokens[-1]
        elif len(tokens) == 1:
            repaired["unit_price"] = tokens[0]

    return repaired


def _display_quantity(value: str) -> str:
    """Trim ERP-padded quantities for display: '10.00000000' -> '10', '2.50' -> '2.5'."""

    text = _display(value)
    if not text or re.search(r"[A-Za-z]", text):  # keep unit-bearing quantities like "5 PC"
        return text
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    try:
        number = float(cleaned)
    except ValueError:
        return text
    return str(int(number)) if number == int(number) else f"{number:g}"


def _display_price(value: str) -> str:
    """Normalize price display to a consistent currency format: '342.000000' -> '$342.00'."""

    text = _display(value)
    if not text:
        return ""
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    try:
        number = float(cleaned)
    except ValueError:
        return text
    return f"${number:,.2f}"


def _fill_implied_unit_price(item: dict[str, Any]) -> dict[str, Any]:
    """Backfill a dropped unit price from total / quantity to avoid false deltas."""

    if normalize_price(item.get("unit_price")) is not None:
        return item
    implied = implied_unit_price(item.get("total_price"), item.get("quantity"))
    if implied is None:
        return item
    repaired = dict(item)
    repaired["unit_price"] = f"{implied:.2f}"
    return repaired


def _normalize_text(value: Any) -> str:
    return " ".join(_display(value).casefold().split())


def _normalize_metadata(value: Any) -> str:
    """Normalize a document-level field, repairing kerning fragmentation first.

    Collapsing split-apart tokens (e.g. "SYSTEMS C OR P" -> "SYSTEMS CORP") keeps
    PDF layout noise from registering as a false metadata change.
    """

    return " ".join(collapse_kerning(_display(value)).casefold().split())


def _normalize_notes(notes: list[str]) -> str:
    return "\n".join(_normalize_text(note) for note in notes if _normalize_text(note))


def _repair_shifted_descriptor_tail(item: dict[str, Any]) -> dict[str, Any]:
    repaired = dict(item)
    description = _display(repaired.get("description"))
    for field_name in ("vendor", "unit"):
        value = _display(repaired.get(field_name))
        if not value:
            continue
        normalized = re.sub(r"[^a-z0-9]+", " ", value.casefold()).strip()
        if field_name == "unit" and normalized in VALID_UNITS:
            continue
        if normalized not in DESCRIPTOR_TAIL_WORDS and not re.search(
            r"\b(?:\d+\s*(?:v|vac|vdc|a|amp|amps)|c\s*curve|coil|output|input|red|green|blue|black|white|yellow)\b",
            normalized,
        ):
            continue
        if value.casefold() not in description.casefold():
            description = f"{description} {value}".strip()
        repaired[field_name] = ""
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
