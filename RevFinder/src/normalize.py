"""Shared sanitization helpers for keys, quantities, and prices.

These functions are the single source of truth for turning messy extracted
strings into stable comparison values. Keeping them here (instead of duplicated
across the parser and the diff engine) guarantees the extraction layer and the
differential layer normalize identically.
"""

from __future__ import annotations

import re
from typing import Any


# A part key is a greedy, space-bounded alphanumeric block that keeps every
# internal delimiter (hyphen, underscore, dot). This prevents truncation of
# strings such as "710-HARN-TLM-ALPH-TRK-01A_REV.3" or "1769-L30ERK-SERB".
PART_KEY_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)*")

# A "part-like" token additionally requires at least one delimiter group, which
# distinguishes real part numbers from ordinary words during line scanning.
PART_LIKE_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[._\-][A-Za-z0-9]+)+")


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).replace("\x00", "").strip()


def collapse_spaces(value: Any) -> str:
    """Collapse runs of white-space (including kerning gaps) to single spaces."""

    return re.sub(r"\s+", " ", _text(value)).strip()


def collapse_kerning(value: Any) -> str:
    """Repair PDF kerning fragmentation by rejoining split-apart short tokens.

    Layout shifts can fracture a single word into spaced single/double characters
    (e.g. "SYSTEMS C OR P" instead of "SYSTEMS CORP"). A run of two or more
    consecutive alphanumeric tokens of <=2 characters is treated as one fractured
    word and merged; lone short tokens (like "&" or "TO") are left untouched.
    """

    tokens = collapse_spaces(value).split(" ")
    if tokens == [""]:
        return ""

    result: list[str] = []
    run: list[str] = []

    def _flush_run() -> None:
        if not run:
            return
        if len(run) >= 2:
            result.append("".join(run))
        else:
            result.extend(run)
        run.clear()

    for token in tokens:
        if len(token) <= 2 and token.isalnum():
            run.append(token)
        else:
            _flush_run()
            result.append(token)
    _flush_run()
    return " ".join(result)


def extract_part_key(value: Any) -> str:
    """Return the first full part-key token without truncation.

    Case and delimiters are preserved exactly so the returned key is identical
    to the source token.
    """

    match = PART_KEY_PATTERN.search(_text(value))
    return match.group(0) if match else ""


def part_match_token(value: Any) -> str:
    """Return the part-number-like token used for keying, ignoring labels/prose.

    Picks the first token containing a delimiter (e.g. "930-PLC-044" out of
    "IPN: 930-PLC-044"), falling back to the whole cleaned value when none exists.
    Universal: keys align even when one layout prefixes a label and another does
    not — no template- or label-specific assumptions.
    """

    text = _text(value)
    match = PART_LIKE_PATTERN.search(text)
    return match.group(0) if match else text


def normalize_identifier(value: Any) -> str:
    """Collapse whitespace and upper-case a key for equality matching.

    Internal delimiters are preserved, so multi-hyphen part numbers stay intact.
    """

    return re.sub(r"\s+", "", _text(value)).upper()


def normalize_item_label(value: Any) -> str:
    """Canonicalize a line label so padding/prefix variants compare equal.

    "LN: 001", "Item #1", and "LN: 1" all normalize to "1".
    """

    text = _text(value)
    if not text:
        return ""
    match = re.search(r"\d+", text)
    if match:
        return str(int(match.group(0)))
    return normalize_identifier(text)


def _numeric_core(value: Any) -> str | None:
    """Strip currency, units, spaces, and thousands separators to bare digits."""

    text = _text(value)
    if not text:
        return None
    cleaned = re.sub(r"[^0-9.\-]", "", text)
    if cleaned in {"", "-", ".", "-.", "--"}:
        return None
    return cleaned


def normalize_quantity(value: Any) -> int | None:
    """Strip unit suffixes (UNIT, EA, pcs, ...) and cast to an integer.

    "12 UNITS" -> 12, "1,000.00 pcs" -> 1000, "002" -> 2.
    """

    cleaned = _numeric_core(value)
    if cleaned is None:
        return None
    try:
        return int(round(float(cleaned)))
    except ValueError:
        return None


# Known truncated / fragmented manufacturer blocks reconciled to a canonical form.
# Keyed by the space-normalized, case-folded leading token(s).
_MANUFACTURER_ALIASES = {
    "allen": "allen bradley",
    "ab": "allen bradley",
    "allen bradley": "allen bradley",
    "rockwell": "rockwell automation",
    "schneider": "schneider electric",
    "schneider electric": "schneider electric",
    "telemecanique": "schneider electric",
    "square d": "schneider electric",
    "ge": "general electric",
    "siemens": "siemens",
    "phoenix": "phoenix contact",
    "weidmuller": "weidmuller",
    "omron": "omron",
    "abb": "abb",
    "eaton": "eaton",
    "cutler": "eaton",
    "cutler hammer": "eaton",
}


def canonical_manufacturer(value: Any) -> str:
    """Reconcile fragmented / truncated manufacturer names for equality.

    Repairs kerning, normalizes spacing/case, then maps known truncated blocks
    ("ALLEN" -> "allen bradley", "SCHNEIDER" -> "schneider electric").
    """

    text = collapse_kerning(value).casefold()
    normalized = " ".join(re.sub(r"[^a-z0-9]+", " ", text).split())
    if not normalized:
        return ""
    if normalized in _MANUFACTURER_ALIASES:
        return _MANUFACTURER_ALIASES[normalized]
    first_token = normalized.split(" ", 1)[0]
    return _MANUFACTURER_ALIASES.get(first_token, normalized)


def implied_unit_price(total_price: Any, quantity: Any) -> float | None:
    """Derive a unit price from an aggregate total when the scalar is missing.

    Implied Unit Price = total_price / quantity. Returns ``None`` when either
    operand is missing or the quantity is zero.
    """

    total = normalize_price(total_price)
    qty = normalize_quantity(quantity)
    if total is None or not qty:
        return None
    return total / qty


def normalize_price(value: Any) -> float | None:
    """Strip currency markers, commas, and spaces, then cast to a float.

    "$1,420.00 USD" -> 1420.0, "$ 12,000.50" -> 12000.5.
    """

    cleaned = _numeric_core(value)
    if cleaned is None:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None
