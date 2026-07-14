"""Strongly-typed, immutable data models and relational reconciliation.

This module is the schema + join layer for the comparison pipeline. It defines
typed, immutable representations of a BoM line item and document metadata, and a
pure full-outer-join over part-number (IPN) keyed hash maps. It holds no global
state: every builder returns fresh instances, and baseline / revised maps live in
independent dictionaries so processing one can never mutate the other.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping

from .normalize import (
    canonical_manufacturer,
    normalize_identifier,
    normalize_price,
    normalize_quantity,
    part_match_token,
)


# Fields compared during reconciliation, with their comparison semantics.
COMPARE_FIELDS = [
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
]
QUANTITY_FIELDS = {"quantity"}
PRICE_FIELDS = {"unit_price", "total_price"}
MANUFACTURER_FIELDS = {"manufacturer"}
_HIGH_SEVERITY = {"quantity", "revision", "part_number", "unit_price", "total_price"}
_MEDIUM_SEVERITY = {"manufacturer", "vendor", "mpn"}


def _str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return " ".join(str(value).replace("\x00", "").split())


def _notes_tuple(value: Any) -> tuple[str, ...]:
    if isinstance(value, (list, tuple)):
        return tuple(_str(item) for item in value if _str(item))
    text = _str(value)
    return (text,) if text else ()


def field_kind(name: str) -> str:
    if name in QUANTITY_FIELDS:
        return "quantity"
    if name in PRICE_FIELDS:
        return "price"
    if name in MANUFACTURER_FIELDS:
        return "manufacturer"
    return "text"


def field_severity(name: str) -> str:
    if name in _HIGH_SEVERITY:
        return "high"
    if name in _MEDIUM_SEVERITY:
        return "medium"
    return "low"


def _normalize_text(value: Any) -> str:
    return " ".join(_str(value).casefold().split())


def values_equal(old_value: Any, new_value: Any, kind: str = "text") -> bool:
    """Typed equality: numeric for quantities/prices, alias-aware for makers."""

    if kind == "quantity":
        old_number = normalize_quantity(old_value)
        new_number = normalize_quantity(new_value)
        if old_number is not None and new_number is not None:
            return old_number == new_number
    elif kind == "price":
        old_number = normalize_price(old_value)
        new_number = normalize_price(new_value)
        if old_number is not None and new_number is not None:
            return math.isclose(old_number, new_number, rel_tol=1e-9, abs_tol=1e-9)
    elif kind == "manufacturer":
        return canonical_manufacturer(old_value) == canonical_manufacturer(new_value)
    return _normalize_text(old_value) == _normalize_text(new_value)


def text_similarity(old_value: Any, new_value: Any) -> float:
    """SequenceMatcher ratio over normalized text (for low-severity deltas)."""

    old_text = _normalize_text(old_value)
    new_text = _normalize_text(new_value)
    if not old_text and not new_text:
        return 1.0
    return float(SequenceMatcher(None, old_text, new_text).ratio())


@dataclass(frozen=True)
class LineItem:
    """An immutable, typed BoM line item.

    Typed fields drive comparison; ``raw`` keeps the exact extracted strings so
    logs preserve the original tokens (never coerced to empty).
    """

    part_number: str
    quantity: int | None
    unit_price: float | None
    total_price: float | None
    mpn: str
    vendor: str
    description: str
    contextual_notes: tuple[str, ...]
    source_index: int
    raw: Mapping[str, Any]

    @classmethod
    def from_raw(cls, data: Mapping[str, Any], source_index: int = 0) -> "LineItem":
        return cls(
            part_number=_str(data.get("part_number")),
            quantity=normalize_quantity(data.get("quantity")),
            unit_price=normalize_price(data.get("unit_price")),
            total_price=normalize_price(data.get("total_price")),
            mpn=_str(data.get("mpn")),
            vendor=_str(data.get("vendor")),
            description=_str(data.get("description")),
            contextual_notes=_notes_tuple(data.get("contextual_notes")),
            source_index=int(source_index),
            raw=dict(data),
        )

    def value(self, name: str) -> str:
        """Raw display value for a field (original extracted token)."""

        return _str(self.raw.get(name))


@dataclass(frozen=True)
class DocumentMetadata:
    """Immutable document-level metadata schema."""

    revision: str = ""
    date: str = ""
    po_number: str = ""
    vendor: str = ""

    @classmethod
    def from_header(cls, header: Mapping[str, Any] | None) -> "DocumentMetadata":
        header = header or {}

        def pick(*keys: str) -> str:
            for key in keys:
                value = _str(header.get(key))
                if value:
                    return value
            return ""

        return cls(
            revision=pick("revision"),
            date=pick("date", "release_date"),
            po_number=pick("document_number", "po_number", "po"),
            vendor=pick("vendor", "supplier", "customer"),
        )


@dataclass(frozen=True)
class FieldDelta:
    field: str
    old_value: str
    new_value: str


@dataclass(frozen=True)
class Delta:
    key: str
    change_type: str  # ADDED | DELETED | MODIFIED | UNCHANGED
    part_number: str
    field_changes: tuple[FieldDelta, ...] = ()
    old: LineItem | None = None
    new: LineItem | None = None
    similarity: float = 1.0


def build_item_map(
    items: list[Any],
    key_func: Callable[[LineItem], str] | None = None,
) -> dict[str, LineItem]:
    """Stateless: build an IPN-keyed hash map of LineItems.

    Duplicate keys are disambiguated with a positional suffix so repeated part
    numbers remain individually addressable. Accepts raw dicts or LineItems.
    """

    counters: dict[str, int] = {}
    result: dict[str, LineItem] = {}
    for index, item in enumerate(items, start=1):
        line_item = item if isinstance(item, LineItem) else LineItem.from_raw(item, index)
        base_key = key_func(line_item) if key_func else _default_key(line_item)
        counters[base_key] = counters.get(base_key, 0) + 1
        key = base_key if counters[base_key] == 1 else f"{base_key}__{counters[base_key]}"
        result[key] = line_item
    return result


def _default_key(line_item: LineItem) -> str:
    sanitized = normalize_identifier(part_match_token(line_item.part_number))
    if sanitized:
        return f"part_number:{sanitized}"
    return f"row:{line_item.source_index}"


def compare_line_items(
    old: LineItem,
    new: LineItem,
    skip: frozenset[str] = frozenset(),
    fields: "list[str] | None" = None,
) -> tuple[FieldDelta, ...]:
    """Field-by-field comparison; baseline values are read from ``old`` by key.

    ``fields`` restricts which fields are compared (default: all). It is used to
    compare only columns present in BOTH documents, so a column that exists in one
    file but not the other is not reported as a content change. Fields in ``skip``
    are excluded (e.g. the part_number delta on healed pairs).
    """

    deltas: list[FieldDelta] = []
    for name in fields if fields is not None else COMPARE_FIELDS:
        if name in skip:
            continue
        old_value = old.raw.get(name)
        new_value = new.raw.get(name)
        if not values_equal(old_value, new_value, kind=field_kind(name)):
            deltas.append(FieldDelta(name, _str(old_value), _str(new_value)))
    return tuple(deltas)


# Minimum sanitized-prefix length for a high-confidence key heal.
_MIN_HEAL_PREFIX = 4


def _prefix_aligned(left: str, right: str) -> bool:
    """True if one key is a delimiter-bounded prefix of the other.

    "105-ENC" aligns with "105-ENC-AL-01" (boundary at '-'), but "ABC-1" does not
    align with "ABC-10" (the longer continues mid-token, not at a delimiter).
    """

    short, long_ = (left, right) if len(left) <= len(right) else (right, left)
    if short == long_ or not long_.startswith(short):
        return False
    return long_[len(short)] in "-_."


def _heal_prefix_matches(
    base_only: set[str],
    rev_only: set[str],
    baseline: Mapping[str, LineItem],
    revised: Mapping[str, LineItem],
) -> list[tuple[str, str]]:
    """Pair baseline-only and revised-only keys whose IPNs share a prefix.

    Resolves layout truncation (e.g. a wrapped "105-ENC-" baseline key healing to
    the complete "105-ENC-AL-01" revised key), nearest source position winning ties.
    """

    pairs: list[tuple[str, str]] = []
    used_revised: set[str] = set()
    for base_key in sorted(base_only):
        base_norm = normalize_identifier(baseline[base_key].part_number).rstrip("-_.")
        if len(base_norm) < _MIN_HEAL_PREFIX:
            continue
        best: tuple[int, str] | None = None
        for revised_key in sorted(rev_only):
            if revised_key in used_revised:
                continue
            revised_norm = normalize_identifier(revised[revised_key].part_number).rstrip("-_.")
            if len(revised_norm) < _MIN_HEAL_PREFIX or not _prefix_aligned(base_norm, revised_norm):
                continue
            score = abs(baseline[base_key].source_index - revised[revised_key].source_index)
            if best is None or score < best[0]:
                best = (score, revised_key)
        if best is not None:
            pairs.append((base_key, best[1]))
            used_revised.add(best[1])
    return pairs


# Minimum normalized description length and similarity for a content-based heal.
_DESC_MATCH_MIN_LEN = 8
_DESC_MATCH_THRESHOLD = 0.62


def _heal_by_description(
    base_only: set[str],
    rev_only: set[str],
    baseline: Mapping[str, LineItem],
    revised: Mapping[str, LineItem],
) -> list[tuple[str, str]]:
    """Pair leftover rows whose descriptions are highly similar.

    Layout differences (a table vs. labeled cards) can make the same item surface
    under different part-number keys; matching on description content recovers the
    alignment generically, without any template- or label-specific assumptions.
    Greedy one-to-one by best similarity, above a confidence threshold.
    """

    candidates: list[tuple[float, str, str]] = []
    for base_key in base_only:
        base_desc = baseline[base_key].description
        if len(_normalize_text(base_desc)) < _DESC_MATCH_MIN_LEN:
            continue
        for revised_key in rev_only:
            revised_desc = revised[revised_key].description
            if len(_normalize_text(revised_desc)) < _DESC_MATCH_MIN_LEN:
                continue
            score = text_similarity(base_desc, revised_desc)
            if score >= _DESC_MATCH_THRESHOLD:
                candidates.append((score, base_key, revised_key))

    candidates.sort(key=lambda candidate: candidate[0], reverse=True)
    used_base: set[str] = set()
    used_revised: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, base_key, revised_key in candidates:
        if base_key in used_base or revised_key in used_revised:
            continue
        used_base.add(base_key)
        used_revised.add(revised_key)
        pairs.append((base_key, revised_key))
    return pairs


# On a healed pair the two keys are the same item recovered under different
# extractions, so the part_number difference is the artifact we corrected, not a
# real change — suppress it.
_HEALED_SKIP = frozenset({"part_number"})


def _modified_delta(
    key: str,
    old: LineItem,
    new: LineItem,
    skip: frozenset[str] = frozenset(),
    fields: "list[str] | None" = None,
) -> Delta:
    field_changes = compare_line_items(old, new, skip=skip, fields=fields)
    old_blob = " ".join(old.value(name) for name in COMPARE_FIELDS)
    new_blob = " ".join(new.value(name) for name in COMPARE_FIELDS)
    return Delta(
        key,
        "MODIFIED" if field_changes else "UNCHANGED",
        new.part_number,
        field_changes,
        old,
        new,
        text_similarity(old_blob, new_blob),
    )


def full_outer_join(
    baseline: Mapping[str, LineItem],
    revised: Mapping[str, LineItem],
    compare_fields: "list[str] | None" = None,
) -> list[Delta]:
    """Relational full outer join over IPN-keyed maps, with prefix key healing.

    ADDED (revised-only), DELETED (baseline-only), MODIFIED/UNCHANGED (in both).
    Baseline values for modified items are fetched from ``baseline`` by key,
    structurally preventing the "OLD: ''" erasure bug. Before declaring an
    addition/deletion, a prefix-match heal aligns keys split by layout truncation
    so a single modified row is never reported as a delete + add pair.
    """

    base_keys = set(baseline)
    revised_keys = set(revised)
    exact = base_keys & revised_keys
    base_only = base_keys - exact
    rev_only = revised_keys - exact

    # Tier 1: delimiter-bounded prefix heal (truncated vs full key).
    healed = _heal_prefix_matches(base_only, rev_only, baseline, revised)
    matched_base = {base_key for base_key, _ in healed}
    matched_rev = {revised_key for _, revised_key in healed}
    # Tier 2: content heal — when two layouts extract the part number differently,
    # align the still-unmatched rows by description similarity so the documents are
    # actually compared instead of reported as all-added / all-removed.
    healed += _heal_by_description(
        base_only - matched_base, rev_only - matched_rev, baseline, revised
    )
    healed_base = {base_key for base_key, _ in healed}
    healed_rev = {revised_key for _, revised_key in healed}

    # Tier 3: substitution/swap detection — a removed item and an added item that
    # are the SAME line with a changed part number (quantity/manufacturer/etc. match)
    # are shown as one MODIFIED row whose part_number changed, not a delete + add.
    swaps = _heal_swaps(base_only - healed_base, rev_only - healed_rev, baseline, revised)
    swap_base = {base_key for base_key, _ in swaps}
    swap_rev = {revised_key for _, revised_key in swaps}

    deltas: list[Delta] = []
    for key in exact:
        deltas.append(_modified_delta(key, baseline[key], revised[key], fields=compare_fields))
    for base_key, revised_key in healed:
        # Keyed under the complete (revised) part string; the part_number delta is
        # suppressed because the key difference is the extraction artifact we healed.
        deltas.append(
            _modified_delta(revised_key, baseline[base_key], revised[revised_key], skip=_HEALED_SKIP, fields=compare_fields)
        )
    for base_key, revised_key in swaps:
        # A real part swap: keep the part_number delta so the substitution is visible.
        deltas.append(_modified_delta(revised_key, baseline[base_key], revised[revised_key], fields=compare_fields))
    for base_key in base_only - healed_base - swap_base:
        old = baseline[base_key]
        deltas.append(Delta(base_key, "DELETED", old.part_number, (), old, None, 1.0))
    for revised_key in rev_only - healed_rev - swap_rev:
        new = revised[revised_key]
        deltas.append(Delta(revised_key, "ADDED", new.part_number, (), None, new, 1.0))

    deltas.sort(key=lambda delta: delta.key)
    return deltas


# Non-key attributes that identify "the same line" across a part-number swap.
_SWAP_FIELDS = ("quantity", "unit", "manufacturer", "mpn", "unit_price", "total_price", "description")


def _swap_overlap(old: LineItem, new: LineItem) -> tuple[int, int]:
    comparable = 0
    equal = 0
    for name in _SWAP_FIELDS:
        old_value = old.raw.get(name)
        new_value = new.raw.get(name)
        if not (_str(old_value) or _str(new_value)):
            continue
        comparable += 1
        if values_equal(old_value, new_value, kind=field_kind(name)):
            equal += 1
    return comparable, equal


def _heal_swaps(
    base_only: set[str],
    rev_only: set[str],
    baseline: Mapping[str, LineItem],
    revised: Mapping[str, LineItem],
) -> list[tuple[str, str]]:
    """Pair a removed item with an added item that is the same line, part# changed.

    Requires several non-key attributes to be present and to match closely, so a
    coincidental add/remove is not mistaken for a substitution. Greedy best-match.
    """

    candidates: list[tuple[float, int, str, str]] = []
    for base_key in base_only:
        for revised_key in rev_only:
            comparable, equal = _swap_overlap(baseline[base_key], revised[revised_key])
            if comparable >= 3 and equal / comparable >= 0.8:
                candidates.append((equal / comparable, equal, base_key, revised_key))

    candidates.sort(key=lambda candidate: (candidate[0], candidate[1]), reverse=True)
    used_base: set[str] = set()
    used_revised: set[str] = set()
    pairs: list[tuple[str, str]] = []
    for _, _, base_key, revised_key in candidates:
        if base_key in used_base or revised_key in used_revised:
            continue
        used_base.add(base_key)
        used_revised.add(revised_key)
        pairs.append((base_key, revised_key))
    return pairs


def format_modification(old_value: Any, new_value: Any) -> str:
    """OLD/NEW serialization with explicit delimiters (no squashed tokens).

    A missing baseline (brand-new part) renders as OLD: 'None'.
    """

    old_text = "None" if old_value is None else _str(old_value)
    new_text = "None" if new_value is None else _str(new_value)
    return f"OLD: '{old_text}' || NEW: '{new_text}'"


def format_log_prefix(severity: str, change_type: str, part: str, field_name: str) -> str:
    """Pipe-delimited prefix, e.g. "HIGH | MODIFIED | Part: 220-BRK-15A | Field: quantity -> "."""

    return f"{(severity or '').upper()} | {(change_type or '').upper()} | Part: {part} | Field: {field_name} -> "
