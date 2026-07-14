"""Side-by-side PDF rendering with change-highlight overlays (PyMuPDF).

The heavy dependency (PyMuPDF / ``fitz``) is imported lazily so the rest of the
app and the test suite keep working even when it is not installed.
"""

from __future__ import annotations

import re
from typing import Any


# Structural colors (RGB 0-1): added lines and removed lines.
ADDED = (0.106, 0.478, 0.239)  # green
REMOVED = (0.690, 0.125, 0.227)  # red

# A rotating palette of visually distinct colors. Each modified line is assigned
# one color, and its OLD values (baseline page) and NEW values (amended page) are
# drawn in that SAME color so a reviewer can match a change across the two pages.
_MATCH_PALETTE = [
    (0.118, 0.314, 0.878),  # blue
    (0.902, 0.490, 0.129),  # orange
    (0.545, 0.235, 0.678),  # purple
    (0.000, 0.545, 0.545),  # teal
    (0.792, 0.153, 0.553),  # magenta
    (0.549, 0.353, 0.169),  # brown
    (0.235, 0.451, 0.204),  # olive
    (0.290, 0.290, 0.700),  # indigo
]

# Values distinctive enough to locate reliably on the page (bare quantities /
# long descriptions are skipped to avoid mis-boxing; the row anchor covers those).
_LOCATABLE_FIELDS = {"part_number", "unit_price", "total_price", "mpn"}


def _import_fitz():
    try:
        import fitz  # PyMuPDF
    except Exception:
        return None
    return fitz


def is_available() -> bool:
    """True when PyMuPDF is importable and visual review can render."""

    return _import_fitz() is not None


def page_count(pdf_bytes: bytes | None) -> int:
    fitz = _import_fitz()
    if fitz is None or not pdf_bytes:
        return 0
    try:
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            return doc.page_count
    except Exception:
        return 0


def render_page_with_highlights(
    pdf_bytes: bytes | None,
    page_index: int,
    highlights: list[tuple[str, tuple[float, float, float]]],
    zoom: float = 2.0,
) -> bytes | None:
    """Render one page to PNG bytes, drawing colored frames around matched text.

    ``highlights`` is a list of ``(search_text, rgb)`` pairs. Every occurrence of
    ``search_text`` on the page gets an outlined rectangle in ``rgb``.
    """

    fitz = _import_fitz()
    if fitz is None or not pdf_bytes:
        return None
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    except Exception:
        return None
    try:
        if not 0 <= page_index < doc.page_count:
            return None
        page = doc[page_index]
        for text, color in highlights:
            if not text:
                continue
            for rect in _locate(fitz, page, text):
                padded = rect + (-2.5, -2.5, 2.5, 2.5)
                # Translucent fill + bold border so changes are obvious on screen.
                page.draw_rect(padded, color=color, fill=color, fill_opacity=0.22, width=2.0)
        pixmap = page.get_pixmap(matrix=fitz.Matrix(zoom, zoom))
        return pixmap.tobytes("png")
    except Exception:
        return None
    finally:
        doc.close()


def _locate(fitz, page, needle: str) -> list:
    """Find rectangles for ``needle``, tolerating line-wrapped/hyphen-split tokens."""

    try:
        rects = list(page.search_for(needle, quads=False))
    except Exception:
        rects = []
    if rects:
        return rects
    return _reconstruct_rects(fitz, page, needle)


def _reconstruct_rects(fitz, page, needle: str) -> list:
    """Match a token that the PDF split across a line break (e.g. "105-ENC-\\nAL-01").

    Adjacent words are concatenated (ignoring spaces, hyphens, and the wrap) and
    compared to the normalized target; the union of the matching word boxes is
    returned so the overlay spans both lines.
    """

    target = re.sub(r"[^a-z0-9]", "", needle.lower())
    if len(target) < 4:
        return []
    try:
        words = page.get_text("words")
    except Exception:
        return []

    rects = []
    count = len(words)
    for start in range(count):
        accumulated = ""
        matched: list = []
        for index in range(start, min(start + 8, count)):
            token = re.sub(r"[^a-z0-9]", "", str(words[index][4]).lower())
            if not token:
                continue
            if not target.startswith(accumulated + token):
                break
            accumulated += token
            matched.append(words[index])
            if accumulated == target:
                rects.extend(_rects_per_line(fitz, matched))
                break
    return rects


def _rects_per_line(fitz, words: list) -> list:
    """Emit one rectangle per text line so a wrapped match never produces a single
    oversized box spanning unrelated content between the lines."""

    lines: dict[int, Any] = {}
    for word in words:
        band = int(round(((word[1] + word[3]) / 2) / 3.0))  # ~3pt vertical band
        rect = fitz.Rect(word[0], word[1], word[2], word[3])
        lines[band] = rect if band not in lines else (lines[band] | rect)
    return list(lines.values())


def split_annotation_targets(text: str) -> tuple[str, list[str]]:
    """Split a segment into its base description and its bracketed annotations."""

    annotations = [match.group(0) for match in re.finditer(r"\[[^\]]*\]", text or "")]
    base = re.sub(r"\[[^\]]*\]", " ", text or "")
    base = re.sub(r"\s+", " ", base).strip()
    return base, annotations


def slice_annotation_rects(fitz, page, segment: str) -> dict[str, list]:
    """Resolve coordinates for the base text and each bracketed annotation
    independently, so an inline remark never stretches the description's box.

    Returns ``{"base": [Rect...], "annotations": [Rect...]}``.
    """

    base, annotations = split_annotation_targets(segment)
    base_rects = _locate(fitz, page, base) if base else []
    annotation_rects: list = []
    for annotation in annotations:
        annotation_rects.extend(_locate(fitz, page, annotation))
    return {"base": base_rects, "annotations": annotation_rects}


def build_highlights(diff: Any) -> dict[str, list[tuple[str, tuple[float, float, float]]]]:
    """Translate a DiffResult into color-matched highlight tokens per page.

    Returns ``{"old": [...], "new": [...]}``. Each MODIFIED line is assigned a
    distinct color; the exact OLD values it boxes on the baseline page and the NEW
    values it boxes on the amended page use that SAME color, so a reviewer can match
    a change across the two pages. Added lines are green, removed lines are red.
    """

    old_side: list[tuple[str, tuple[float, float, float]]] = []
    new_side: list[tuple[str, tuple[float, float, float]]] = []
    color_index = 0

    for change in getattr(diff, "changes", []):
        if change.status == "added":
            new_side.append((_anchor_token(change.new_item), ADDED))
        elif change.status == "removed":
            old_side.append((_anchor_token(change.old_item), REMOVED))
        elif change.status == "modified":
            color = _MATCH_PALETTE[color_index % len(_MATCH_PALETTE)]
            color_index += 1
            # Row anchor (part number) so the line is locatable on both pages...
            old_side.append((_anchor_token(change.old_item), color))
            new_side.append((_anchor_token(change.new_item), color))
            # ...plus the exact changed values, same color on both sides.
            for field_change in getattr(change, "field_changes", []):
                if field_change.field not in _LOCATABLE_FIELDS:
                    continue
                old_value = str(field_change.old_value or "").strip()
                new_value = str(field_change.new_value or "").strip()
                if old_value:
                    old_side.append((old_value, color))
                if new_value:
                    new_side.append((new_value, color))

    return {
        "old": _dedupe(old_side),
        "new": _dedupe(new_side),
    }


def _dedupe(highlights: list) -> list:
    """Drop empty tokens and duplicate (token, color) pairs so a box is drawn once."""

    seen: set = set()
    result: list = []
    for token, color in highlights:
        if not token or (token, color) in seen:
            continue
        seen.add((token, color))
        result.append((token, color))
    return result


def _anchor_token(item: dict[str, Any] | None) -> str:
    item = item or {}
    for field in ("part_number", "mpn"):
        value = str(item.get(field) or "").strip()
        if value:
            return value
    description = str(item.get("description") or "").strip()
    if description:
        # A short, distinctive snippet is more likely to match than a wrapped line.
        return " ".join(description.split()[:4])
    return ""
