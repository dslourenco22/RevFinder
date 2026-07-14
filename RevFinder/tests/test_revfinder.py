"""Regression matrix for BoMination ingestion + differential engine.

Required verification conditions:
  1. Scalar vs. aggregate price validation (no field cross-contamination)
  2. Kerning noise / white-space collapsing
  3. Complete token preservation (stacked hyphens)
  4. Array mid-index injection without false deltas
  5. Metric unit-suffix truncation -> pure integers
  6. Unpriced context-matching resilience (lowmodified description delta)

Plus bonus regressions: layout inversion, column-level price isolation, implied
unit-price fallback, and manufacturer entity mapping.
"""

from __future__ import annotations

import re

from src.engine import compare_documents, discrepancy_log, format_modification, note_quantity_consistent
from src.extractor import PdfExtraction, TableExtraction
from src.llm_parser import (
    LINE_ITEM_FIELDS,
    ParsedDocument,
    fallback_parse,
    parse_price_fields,
)
from src.normalize import (
    canonical_manufacturer,
    collapse_kerning,
    extract_part_key,
    implied_unit_price,
    normalize_price,
    normalize_quantity,
)


def _item(part_number: str, **overrides) -> dict:
    item = {field: "" for field in LINE_ITEM_FIELDS}
    item["part_number"] = part_number
    item["confidence"] = 0.9
    item.update(overrides)
    return item


def _doc(name: str, items: list[dict], header: dict | None = None) -> ParsedDocument:
    return ParsedDocument(
        source_name=name,
        document_type="BOM",
        header=header or {},
        line_items=items,
    )


# ----------------------------------------------------------------------------
# Test Case 1: Scalar vs. Aggregate Price Validation
# ----------------------------------------------------------------------------
def test_scalar_vs_aggregate_price_validation():
    prices = parse_price_fields("UNIT PRICE: $85.50 | EXT PRICE: $427.50")
    assert prices["unit_price"] == 85.50
    assert prices["total_price"] == 427.50
    # No cross-contamination: the two slots never share a value.
    assert prices["unit_price"] != prices["total_price"]

    # End to end through the deterministic card parser.
    extraction = PdfExtraction(
        "eco.pdf", 1, "--- Page 1 ---\nPart: 105-ENC-AL-01\nUNIT PRICE: $85.50 | EXT PRICE: $427.50\n", (), (), {}
    )
    item = fallback_parse(extraction).line_items[0]
    assert normalize_price(item["unit_price"]) == 85.50
    assert normalize_price(item["total_price"]) == 427.50


# ----------------------------------------------------------------------------
# Test Case 2: Kerning Noise & Space Collapsing
# ----------------------------------------------------------------------------
def test_kerning_noise_and_space_collapsing():
    assert collapse_kerning("SYSTEMS C OR P") == "SYSTEMS CORP"
    assert collapse_kerning("NEXUS   AUTOMATION") == "NEXUS AUTOMATION"

    # The same vendor, fractured by kerning in the amended revision, must not
    # register as a document-level metadata shift.
    old = _doc("baseline.pdf", [], header={"vendor": "NEXUS AUTOMATION SYSTEMS & SYSTEMS CORP"})
    new = _doc("amended.pdf", [], header={"vendor": "NEXUS AUTOMATION SYSTEMS & SYSTEMS C OR P"})
    diff = compare_documents(old, new)
    assert diff.summary["document_changes"] == 0


# ----------------------------------------------------------------------------
# Test Case 3: Complete Token Preservation (Stacked Hyphens)
# ----------------------------------------------------------------------------
def test_complete_token_preservation():
    for source in ("105-ENC-AL-01", "1769-L30ERK-SERB"):
        assert extract_part_key(source) == source
        assert extract_part_key(f"{source} Enclosure 2 EA") == source

    # End to end: trailing revision identifiers survive parsing.
    extraction = PdfExtraction("eco.pdf", 1, "1769-L30ERK-SERB CompactLogix qty 1\n", (), (), {})
    parsed = fallback_parse(extraction)
    assert any(item["part_number"] == "1769-L30ERK-SERB" for item in parsed.line_items)


# ----------------------------------------------------------------------------
# Test Case 4: Array Mid-Index Injections & Shifting
# ----------------------------------------------------------------------------
def test_mid_index_injection_yields_single_added():
    baseline = [_item(f"P-{n}", description=f"Item {n}", quantity="1") for n in range(1, 6)]
    injected = _item("P-NEW", description="Injected Part", quantity="1")
    revised = baseline[:3] + [injected] + baseline[3:]

    diff = compare_documents(_doc("baseline.pdf", baseline), _doc("amended.pdf", revised))

    assert diff.summary["added"] == 1
    assert diff.summary["removed"] == 0
    assert diff.summary["modified"] == 0

    added = [change for change in diff.changes if change.status == "added"]
    assert len(added) == 1
    assert added[0].new_item["part_number"] == "P-NEW"


# ----------------------------------------------------------------------------
# Test Case 5: Metric Unit Suffix Truncation
# ----------------------------------------------------------------------------
def test_metric_unit_suffix_truncation():
    quantities = ["2 UNIT", "5 PC", "12 UNITS", "004"]
    assert [normalize_quantity(value) for value in quantities] == [2, 5, 12, 4]


# ----------------------------------------------------------------------------
# Bonus: Layout Inversion Resilience (table vs. vertical card blocks)
# ----------------------------------------------------------------------------
def test_layout_inversion_resilience():
    table = TableExtraction(
        page_number=1,
        table_index=1,
        rows=(
            ("Part Number", "Description", "Qty"),
            ("105-ENC-AL-01", "Aluminum Enclosure", "2"),
            ("1769-L30ERK-SERB", "CompactLogix Controller", "1"),
        ),
    )
    horizontal = PdfExtraction("baseline.pdf", 1, "", (), (table,), {})
    vertical_text = (
        "--- Page 1 ---\n"
        "Part: 105-ENC-AL-01\nDescription: Aluminum Enclosure\nQty: 2\n\n"
        "Part: 1769-L30ERK-SERB\nDescription: CompactLogix Controller\nQty: 1\n"
    )
    vertical = PdfExtraction("amended.pdf", 1, vertical_text, (), (), {})

    def identity(document):
        return [(i["part_number"], i["description"], i["quantity"]) for i in document.line_items]

    assert identity(fallback_parse(horizontal)) == identity(fallback_parse(vertical))


# ----------------------------------------------------------------------------
# Bonus: Column-level price isolation (Unit Price vs. Ext Price columns)
# ----------------------------------------------------------------------------
def test_column_level_price_isolation():
    table = TableExtraction(
        page_number=1,
        table_index=1,
        rows=(
            ("Part Number", "Description", "Qty", "Unit Price", "Ext Price"),
            ("105-ENC-AL-01", "Enclosure", "2", "$85.50", "$171.00"),
        ),
    )
    extraction = PdfExtraction("baseline.pdf", 1, "", (), (table,), {})
    item = fallback_parse(extraction).line_items[0]
    assert normalize_price(item["unit_price"]) == 85.50
    assert normalize_price(item["total_price"]) == 171.00


# ----------------------------------------------------------------------------
# Test Case 6: Unpriced Context-Matching Resilience
# ----------------------------------------------------------------------------
def test_unpriced_context_matching_resilience():
    old = _doc(
        "baseline.pdf",
        [
            _item(
                "990-NOTE-01",
                quantity="1 UNIT",
                unit_price="$0.00",
                total_price="$0.00",
                description="Reference Schematic Drawing Sheet 2",
            )
        ],
    )
    new = _doc(
        "amended.pdf",
        [
            _item(
                "990-NOTE-01",
                quantity="1 UNIT",
                unit_price="$0.00",
                total_price="$0.00",
                description="[REVISED] Reference Schematic Drawing Sheet 2 Rev B",
            )
        ],
    )

    diff = compare_documents(old, new)

    # The unpriced line is tracked, not skipped.
    assert diff.summary["added"] == 0
    assert diff.summary["removed"] == 0
    assert diff.summary["modified"] == 1

    log = discrepancy_log(diff)
    description_rows = log[(log["field"] == "description") & (log["status"] == "modified")]
    assert len(description_rows) == 1
    # severity "low" + status "modified" -> "lowmodified"
    assert description_rows.iloc[0]["severity"] == "low"


# ----------------------------------------------------------------------------
# Bonus: Implied unit-price fallback (total / quantity)
# ----------------------------------------------------------------------------
def test_implied_unit_price_fallback():
    assert implied_unit_price("$17.00", "2") == 8.50
    assert implied_unit_price("$0.00", "0") is None  # no divide-by-zero

    # Baseline dropped the unit price but the math reconciles -> no false delta.
    old = _doc("baseline.pdf", [_item("X-1", quantity="2", unit_price="", total_price="$17.00")])
    new = _doc("amended.pdf", [_item("X-1", quantity="2", unit_price="$8.50", total_price="$17.00")])
    diff = compare_documents(old, new)
    assert diff.summary["modified"] == 0


# ----------------------------------------------------------------------------
# Bonus: Manufacturer entity mapping
# ----------------------------------------------------------------------------
def test_manufacturer_entity_mapping():
    assert canonical_manufacturer("ALLEN") == canonical_manufacturer("ALLEN-BRADLEY")
    assert canonical_manufacturer("SCHNEIDER") == canonical_manufacturer("Schneider Electric")

    old = _doc("baseline.pdf", [_item("X-1", manufacturer="ALLEN", description="Relay", quantity="1")])
    new = _doc("amended.pdf", [_item("X-1", manufacturer="ALLEN-BRADLEY", description="Relay", quantity="1")])
    diff = compare_documents(old, new)
    assert diff.summary["modified"] == 0


# ============================================================================
# Semantic-scanning / dynamic-coordinate verification vectors
# ============================================================================
from src.llm_parser import cluster_contextual_notes  # noqa: E402
from src import visual  # noqa: E402


def _make_pdf(lines: list[tuple[float, float, str]]) -> bytes:
    """Build an in-memory PDF; lines are (x, y_baseline, text)."""

    import fitz

    doc = fitz.open()
    page = doc.new_page()
    for x, y, text in lines:
        page.insert_text((x, y), text)
    data = doc.tobytes()
    doc.close()
    return data


def _rects_for(pdf_bytes: bytes, needle: str) -> list:
    import fitz

    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return visual._locate(fitz, doc[0], needle)
    finally:
        doc.close()


# Vector 1: Loose engineering note detection & parent anchoring
def test_loose_note_parent_anchoring():
    base_text = "930-PLC-044 | Qty 2\n [CRITICAL NOTE: CONFORMAL COATING MANDATORY FOR MAIN ASSEMBLY FIELD]"
    rev_text = "930-PLC-044 | Qty 2\n [CRITICAL NOTE: CONFORMAL COATING OVERRIDE PER ECO-044]"

    # The clustering binds the bracketed remark to its parent IPN.
    clusters = cluster_contextual_notes(base_text)
    assert "930-PLC-044" in clusters
    assert "CRITICAL NOTE" in clusters["930-PLC-044"][0]

    old = fallback_parse(PdfExtraction("b.pdf", 1, base_text, (), (), {}))
    new = fallback_parse(PdfExtraction("a.pdf", 1, rev_text, (), (), {}))

    # The note is not discarded, and the revised bracket (with ECO-044) does not
    # become a bogus extra line item.
    assert len(old.line_items) == 1 and len(new.line_items) == 1
    assert "CRITICAL NOTE" in old.line_items[0]["contextual_notes"]

    diff = compare_documents(old, new)
    log = discrepancy_log(diff)
    note_rows = log[log["field"] == "contextual_notes"]
    assert len(note_rows) == 1
    assert note_rows.iloc[0]["status"] == "modified"
    assert note_rows.iloc[0]["severity"] in {"low", "medium"}


# Vector 2: Dynamic bounding box lookup under layout shifts
def test_dynamic_bbox_under_layout_shift():
    baseline = _make_pdf([(72, 120, "410-PWR-24V Power Supply")])
    shifted = _make_pdf([(72, 320, "410-PWR-24V Power Supply")])  # +200pt

    old_rects = _rects_for(baseline, "410-PWR-24V")
    new_rects = _rects_for(shifted, "410-PWR-24V")
    assert old_rects and new_rects

    # The box follows the text to its new physical placement (zero target drift).
    assert abs((new_rects[0].y0 - old_rects[0].y0) - 200) < 3


# Vector 3: Kerning noise & truncation elimination
def test_kerning_and_truncation_elimination():
    assert extract_part_key("710-HARN-TLM-ALPH-TRK-01A_REV.3") == "710-HARN-TLM-ALPH-TRK-01A_REV.3"
    assert collapse_kerning("SYSTEMS C OR P") == "SYSTEMS CORP"


# Vector 4: Unpriced element context tracking
def test_unpriced_element_context_tracking():
    old = _doc("baseline.pdf", [_item("REF-DWG-01", description="Reference Drawing Sheet 2", quantity="1")])
    new = _doc("amended.pdf", [_item("REF-DWG-01", description="Reference Drawing Sheet 2 Rev B", quantity="1")])

    diff = compare_documents(old, new)
    assert diff.summary["added"] == 0 and diff.summary["removed"] == 0
    assert diff.summary["modified"] == 1
    assert (discrepancy_log(diff)["field"] == "description").any()


# Bonus (Section 2B): a wrapped match yields one rect PER LINE, not a giant box.
def test_wrapped_match_produces_per_line_rects():
    pdf = _make_pdf([(72, 120, "105-ENC-"), (72, 160, "AL-01")])
    rects = _rects_for(pdf, "105-ENC-AL-01")
    assert len(rects) == 2
    assert all((rect.y1 - rect.y0) < 30 for rect in rects)


# ============================================================================
# State-leakage / serialization / annotation-isolation vectors
# ============================================================================
# Vector 1: Row iteration isolation & memory flush
def test_row_iteration_isolation_memory_flush():
    text = (
        "001 IPN: 410-PWR-24V DIN Rail Power Supply 5 PC [QTY DECREASED BY 1 PER ECO]\n"
        "002 IPN: 930-PLC-044 CompactLogix Controller 2 UNIT"
    )
    parsed = fallback_parse(PdfExtraction("d.pdf", 1, text, (), (), {}))
    by_pn = {item["part_number"]: item for item in parsed.line_items}

    # The note binds to its true parent (the power supply)...
    assert "[QTY DECREASED BY 1 PER ECO]" in by_pn["410-PWR-24V"]["contextual_notes"]
    # ...and does NOT leak onto the next row (the PLC) — loop state flushed.
    assert by_pn["930-PLC-044"]["contextual_notes"] == ""
    # The inline note is decoupled from the description, not fused into it.
    assert "[QTY DECREASED" not in by_pn["410-PWR-24V"]["description"]


# Vector 2: Delimited modification output serialization
def test_delimited_modification_serialization():
    assert format_modification("10", "Acti9") == "OLD: '10' || NEW: 'Acti9'"
    output = format_modification("5 PC", "Industrial Grade")
    assert "||" in output
    assert "5 PC" in output and "Industrial Grade" in output
    assert "5 PCIndustrial" not in output  # no squashed tokens


# Vector 3: Split coordinate generation for nested annotations
def test_split_coordinates_for_nested_annotations():
    import fitz

    segment = "DIN Rail Power Supply [QTY DECREASED BY 1 PER ECO]"
    pdf = _make_pdf([(72, 120, segment)])
    doc = fitz.open(stream=pdf, filetype="pdf")
    try:
        sliced = visual.slice_annotation_rects(fitz, doc[0], segment)
    finally:
        doc.close()

    assert sliced["base"] and sliced["annotations"]
    base_rect = sliced["base"][0]
    note_rect = sliced["annotations"][0]
    # Two distinct, non-overlapping boxes (description vs. bracketed annotation).
    assert not base_rect.intersects(note_rect)


# ============================================================================
# Baseline preservation / relaxed correlation / delimited-prefix vectors
# ============================================================================
# Test Case 1: Dual-state baseline preservation (no "OLD: ''" erasure)
def test_dual_state_baseline_preservation():
    old = _doc(
        "b.pdf",
        [_item("220-BRK-15A", quantity="10", description="Breaker", mpn="A9F-1", unit_price="$34.20", total_price="$342.00")],
    )
    new = _doc(
        "a.pdf",
        [_item("220-BRK-15A", quantity="12", description="Breaker", mpn="A9F-2", unit_price="$34.20", total_price="$410.40")],
    )
    log = discrepancy_log(compare_documents(old, new))

    qty = log[log["field"] == "quantity"].iloc[0]
    assert qty["old_value"] == "10" and qty["new_value"] == "12"
    assert "OLD: '10' || NEW: '12'" in qty["message"]

    # MPN and aggregate-price baselines are likewise preserved, not blanked.
    assert log[log["field"] == "mpn"].iloc[0]["old_value"] == "A9F-1"
    assert log[log["field"] == "total_price"].iloc[0]["old_value"] == "$342.00"


# Test Case 2: Proximity note assignment without veto
def test_proximity_note_assignment_without_veto():
    text = "002 IPN: 410-PWR-24V DIN Rail Power Supply 5 PC\n [QTY DECREASED BY 1 PER ECO]"
    parsed = fallback_parse(PdfExtraction("d.pdf", 1, text, (), (), {}))
    item = next(i for i in parsed.line_items if i["part_number"] == "410-PWR-24V")
    assert "[QTY DECREASED BY 1 PER ECO]" in item["contextual_notes"]

    # Even when the parent quantity did NOT change, proximity wins: the note is
    # retained (no absolute veto) rather than dropped.
    old = _doc("b.pdf", [_item("410-PWR-24V", quantity="5", description="PSU")])
    new = _doc(
        "a.pdf",
        [_item("410-PWR-24V", quantity="5", description="PSU", contextual_notes="[QTY DECREASED BY 1 PER ECO]")],
    )
    diff = compare_documents(old, new)
    assert (discrepancy_log(diff)["field"] == "contextual_notes").any()


# Test Case 3: Delimited header log verification
def test_delimited_header_log_verification():
    old = _doc("b.pdf", [_item("220-BRK-15A", quantity="10", description="Breaker")])
    new = _doc("a.pdf", [_item("220-BRK-15A", quantity="12", description="Breaker")])
    log = discrepancy_log(compare_documents(old, new))
    message = log[log["field"] == "quantity"].iloc[0]["message"]
    assert message.startswith("HIGH | MODIFIED | Part: 220-BRK-15A | Field: quantity -> ")
    assert "OLD: '10' || NEW: '12'" in message
    assert message.count("|") >= 3  # delimited, not a squashed continuous block


# Tie-breaker helper: signal only, never an absolute veto.
def test_note_quantity_consistency_tiebreaker():
    assert note_quantity_consistent("[QTY DECREASED BY 1 PER ECO]", 5, 4) is True
    assert note_quantity_consistent("[QTY DECREASED BY 1 PER ECO]", 2, 2) is False
    assert note_quantity_consistent("[CRITICAL NOTE: coating]", 5, 4) is None


# ============================================================================
# Object-oriented reconciliation (typed models + full outer join)
# ============================================================================
from src.models import (  # noqa: E402
    DocumentMetadata,
    LineItem,
    build_item_map,
    full_outer_join,
)
from src.models import format_modification as model_format_modification  # noqa: E402


# Test Case 1: Relational data mapping and baseline preservation
def test_relational_join_baseline_preservation():
    baseline = build_item_map(
        [LineItem.from_raw({"part_number": "220-BRK-15A", "quantity": "10", "total_price": "$342.00"}, 1)]
    )
    revised = build_item_map(
        [LineItem.from_raw({"part_number": "220-BRK-15A", "quantity": "12", "total_price": "$410.40"}, 1)]
    )

    deltas = full_outer_join(baseline, revised)
    modified = [delta for delta in deltas if delta.change_type == "MODIFIED"]
    assert len(modified) == 1

    quantity = next(fc for fc in modified[0].field_changes if fc.field == "quantity")
    assert quantity.old_value == "10" and quantity.new_value == "12"
    assert model_format_modification(quantity.old_value, quantity.new_value) == "OLD: '10' || NEW: '12'"


# Test Case 2: Memory isolation across rows (stateless models)
def test_memory_isolation_across_rows():
    row1 = LineItem.from_raw({"part_number": "A-1", "contextual_notes": ["[CRITICAL NOTE]"]}, 1)
    row2 = LineItem.from_raw({"part_number": "B-2"}, 2)
    assert row1.contextual_notes == ("[CRITICAL NOTE]",)
    assert row2.contextual_notes == ()  # empty, fully isolated from row 1


# Test Case 3: Complete suppression of legacy squashed strings in stdout
def test_legacy_log_format_absent_from_stdout(capsys):
    from src.engine import print_discrepancy_log

    old = _doc("b.pdf", [_item("220-BRK-15A", quantity="10", description="Breaker")])
    new = _doc("a.pdf", [_item("220-BRK-15A", quantity="12", description="Breaker")])
    print_discrepancy_log(compare_documents(old, new))

    out = capsys.readouterr().out
    assert out.strip()
    # Legacy squashed format like "part_number:220..." must be absent.
    assert not re.search(r"^[A-Za-z]+:[0-9]+", out, re.MULTILINE)
    assert "HIGH | MODIFIED | Part: 220-BRK-15A | Field: quantity ->" in out


# Bonus: ADDED / DELETED outer-join verbs and the typed metadata schema.
def test_full_outer_join_added_and_deleted():
    baseline = build_item_map([LineItem.from_raw({"part_number": "P-1"}, 1), LineItem.from_raw({"part_number": "P-2"}, 2)])
    revised = build_item_map([LineItem.from_raw({"part_number": "P-2"}, 1), LineItem.from_raw({"part_number": "P-3"}, 2)])
    verbs = {delta.part_number: delta.change_type for delta in full_outer_join(baseline, revised)}
    assert verbs["P-1"] == "DELETED"
    assert verbs["P-3"] == "ADDED"
    assert verbs["P-2"] == "UNCHANGED"


def test_match_ignores_label_prefix_on_part_number():
    # Baseline (table) left the "IPN:" label in the value; revised (cards) is clean.
    # They must still align as the SAME row, not all-added.
    old = _doc("o.pdf", [_item("IPN: 930-PLC-044", quantity="2", description="CompactLogix Controller")])
    new = _doc("n.pdf", [_item("930-PLC-044", quantity="3", description="CompactLogix Controller")])
    diff = compare_documents(old, new)
    assert diff.summary["added"] == 0
    assert diff.summary["removed"] == 0
    assert diff.summary["modified"] == 1


def test_compares_across_different_column_layouts():
    from src.llm_parser import parse_tabular

    # Baseline: 4 columns, title rows above the header, order Component/Description/Qty/UM.
    old = (
        b"Acme Controls - Bill of Material\nDwg 1002241 Rev A\n"
        b"Component,Description,Qty,UM\n"
        b"930-PLC-044,Controller,2,EA\n410-PWR-24V,Power Supply,5,EA\n620-NET-SW8,Switch,1,EA\n"
    )
    # Amended: 6 columns, different names/order, header on row 0; 410 qty 5->4, 620 gone, 710 new.
    new = (
        b"MPN,Part No,Manufacturer,Quantity,Unit Price,Extended\n"
        b"1769,930-PLC-044,AB,2,$1595.00,$3190.00\n"
        b"NDR,410-PWR-24V,MW,4,$85.50,$342.00\n710,710-HARN-TLM,Nexus,1,$185.00,$185.00\n"
    )
    diff = compare_documents(parse_tabular(old, "a.csv"), parse_tabular(new, "b.csv"))
    assert diff.summary["modified"] == 1
    assert diff.summary["removed"] == 1
    assert diff.summary["added"] == 1
    log = discrepancy_log(diff)
    # Only the shared column (quantity) is compared — not columns that exist in just one file.
    assert set(log[log["status"] == "modified"]["field"]) == {"quantity"}


def test_part_swap_shown_as_one_modified_row():
    # A component keeps all its attributes but its part number changes -> one
    # MODIFIED row (the swap), not a separate delete + add.
    old = _doc("b.pdf", [_item("OPN15670", quantity="10", unit="EA", manufacturer="SIEMENS")])
    new = _doc("a.pdf", [_item("OPN20000", quantity="10", unit="EA", manufacturer="SIEMENS")])
    diff = compare_documents(old, new)
    assert diff.summary["added"] == 0
    assert diff.summary["removed"] == 0
    assert diff.summary["modified"] == 1
    log = discrepancy_log(diff)
    row = log[log["field"] == "part_number"].iloc[0]
    assert "OLD: 'OPN15670' || NEW: 'OPN20000'" in row["message"]


def test_display_numbers_are_cleaned():
    old = _doc("b.pdf", [_item("X-1", quantity="10.00000000", unit_price="342.000000", description="d")])
    new = _doc("a.pdf", [_item("X-1", quantity="12.00000000", unit_price="410.400000", description="d")])
    rows = compare_documents(old, new).old_items.iloc[0]
    assert rows["quantity"] == "10"
    assert rows["unit_price"] == "$342.00"


def test_join_aligns_by_description_when_keys_differ():
    # Two layouts extracted the part number differently ("001" vs "930-PLC-044"),
    # but the descriptions match -> rows must align (MODIFIED), not add + delete.
    baseline = build_item_map(
        [LineItem.from_raw({"part_number": "001", "description": "CompactLogix 5370 L3 Controller Dual Ethernet", "quantity": "2"}, 1)]
    )
    revised = build_item_map(
        [LineItem.from_raw({"part_number": "930-PLC-044", "description": "CompactLogix 5370 L3 Controller Dual Ethernet", "quantity": "3"}, 1)]
    )
    deltas = full_outer_join(baseline, revised)
    assert [d.change_type for d in deltas] == ["MODIFIED"]
    assert any(fc.field == "quantity" for fc in deltas[0].field_changes)


def test_document_metadata_schema():
    meta = DocumentMetadata.from_header({"revision": "B", "vendor": "APEX INDUSTRIAL", "document_number": "PO-2026-7719"})
    assert meta.revision == "B"
    assert meta.vendor == "APEX INDUSTRIAL"
    assert meta.po_number == "PO-2026-7719"


# ============================================================================
# Multi-line hyphen-stitching + prefix-match key healing
# ============================================================================
from src.llm_parser import fallback_parse, stitch_hyphenated_lines  # noqa: E402


# Test Case 1: Multi-line IPN fragment stitching
def test_multiline_ipn_fragment_stitching():
    text = "IPN: 105-ENC-\n AL-01 | Qty: 3 | Price: 325.00"
    # The stitcher rebuilds the wrapped key...
    assert "105-ENC-AL-01" in stitch_hyphenated_lines(text)
    # ...and the stateless parser yields a consolidated part_number.
    parsed = fallback_parse(PdfExtraction("d.pdf", 1, text, (), (), {}))
    assert any(item["part_number"] == "105-ENC-AL-01" for item in parsed.line_items)
    # An ordinary hyphenated word at a wrap is NOT stitched.
    assert stitch_hyphenated_lines("high-\nefficiency drive") == "high-\nefficiency drive"


# Test Case 2: Financial delta verification on healed (prefix-matched) records
def test_financial_delta_on_healed_records():
    baseline = build_item_map(
        [LineItem.from_raw({"part_number": "105-ENC-", "quantity": "3", "unit_price": "$325.00", "total_price": "$975.00"}, 4)]
    )
    revised = build_item_map(
        [LineItem.from_raw({"part_number": "105-ENC-AL-01", "quantity": "3", "unit_price": "$350.00", "total_price": "$1,050.00"}, 4)]
    )
    deltas = full_outer_join(baseline, revised)
    modified = [delta for delta in deltas if delta.change_type == "MODIFIED"]
    assert len(modified) == 1

    fields = {fc.field: (fc.old_value, fc.new_value) for fc in modified[0].field_changes}
    assert model_format_modification(*fields["unit_price"]) == "OLD: '$325.00' || NEW: '$350.00'"
    assert model_format_modification(*fields["total_price"]) == "OLD: '$975.00' || NEW: '$1,050.00'"


# Test Case 3: Elimination of false deletion / addition records on healed keys
def test_elimination_of_false_deletion_records():
    baseline = build_item_map([LineItem.from_raw({"part_number": "105-ENC-", "unit_price": "$325.00"}, 4)])
    revised = build_item_map([LineItem.from_raw({"part_number": "105-ENC-AL-01", "unit_price": "$350.00"}, 4)])
    deltas = full_outer_join(baseline, revised)
    assert all(delta.change_type not in ("ADDED", "DELETED") for delta in deltas)
    assert any(delta.change_type == "MODIFIED" and delta.part_number == "105-ENC-AL-01" for delta in deltas)
    # A genuinely unrelated short key is NOT falsely healed.
    b2 = build_item_map([LineItem.from_raw({"part_number": "P-1"}, 1)])
    r2 = build_item_map([LineItem.from_raw({"part_number": "P-10"}, 1)])
    verbs = {delta.change_type for delta in full_outer_join(b2, r2)}
    assert verbs == {"ADDED", "DELETED"}


# Bonus: healed financial deltas surface as HIGH | MODIFIED in the engine log.
def test_healed_financial_delta_logged_high():
    old = _doc("b.pdf", [_item("105-ENC-", unit_price="$325.00", total_price="$975.00", quantity="3", description="Enclosure")])
    new = _doc("a.pdf", [_item("105-ENC-AL-01", unit_price="$350.00", total_price="$1,050.00", quantity="3", description="Enclosure")])
    log = discrepancy_log(compare_documents(old, new))

    assert log[log["status"].isin(["added", "removed"])].empty  # no false add/delete
    unit_row = log[log["field"] == "unit_price"].iloc[0]
    assert unit_row["severity"] == "high"
    assert "OLD: '$325.00' || NEW: '$350.00'" in unit_row["message"]
    total_row = log[log["field"] == "total_price"].iloc[0]
    assert "OLD: '$975.00' || NEW: '$1,050.00'" in total_row["message"]


# ============================================================================
# Universal multi-line stitching (layout-agnostic, no label coupling)
# ============================================================================
def test_universal_hyphen_stitch_is_layout_agnostic():
    # A part fragment dangling at a line-end is stitched regardless of any label.
    assert "ZX9-CTRL-REVB" in stitch_hyphenated_lines("ZX9-CTRL-\nREVB Some description")
    # Prefix-healing aligns a truncated baseline key to the full revised key with
    # no IPN/label assumption at all.
    baseline = build_item_map([LineItem.from_raw({"part_number": "105-ENC-", "unit_price": "$325.00"}, 4)])
    revised = build_item_map([LineItem.from_raw({"part_number": "105-ENC-AL-01", "unit_price": "$350.00"}, 4)])
    verbs = {d.change_type for d in full_outer_join(baseline, revised)}
    assert verbs == {"MODIFIED"}  # one aligned row, no ADDED/DELETED


# ============================================================================
# Universal money recovery (stacked unit/ext column, or dropped price fields)
# ============================================================================
def test_old_prices_recovered_from_stacked_field():
    # Baseline stacked "unit ext" in one field; revised separate. OLD prices recover.
    old = _doc("o.pdf", [_item("X-1", quantity="2", unit_price="$1,420.00 $2,840.00", description="Controller")])
    new = _doc("n.pdf", [_item("X-1", quantity="2", unit_price="$1,595.00", total_price="$3,190.00", description="Controller")])
    diff = compare_documents(old, new)
    old_row = diff.old_items.iloc[0]
    assert old_row["unit_price"] == "$1,420.00"
    assert old_row["total_price"] == "$2,840.00"

    log = discrepancy_log(diff)
    assert log[log["field"] == "unit_price"].iloc[0]["old_value"] == "$1,420.00"
    assert log[log["field"] == "total_price"].iloc[0]["old_value"] == "$2,840.00"


def test_old_prices_harvested_from_raw_text():
    # The model dropped the stacked price column, but the row's raw text carries it.
    old = _doc(
        "o.pdf",
        [_item("X-1", quantity="2", description="Controller", raw_text="1 X-1 Controller 2 UNIT $1,420.00 $2,840.00")],
    )
    new = _doc("n.pdf", [_item("X-1", quantity="2", unit_price="$1,595.00", total_price="$3,190.00", description="Controller")])
    diff = compare_documents(old, new)
    old_row = diff.old_items.iloc[0]
    assert old_row["unit_price"] == "$1,420.00"
    assert old_row["total_price"] == "$2,840.00"


def test_old_prices_recovered_from_stacked_total_field():
    # A combined "UNIT / EXT PRICE" column lands both values in total_price -> split.
    old = _doc("o.pdf", [_item("X-1", quantity="2", total_price="$1,420.00 $2,840.00", description="Controller")])
    new = _doc("n.pdf", [_item("X-1", quantity="2", unit_price="$1,595.00", total_price="$3,190.00", description="Controller")])
    diff = compare_documents(old, new)
    old_row = diff.old_items.iloc[0]
    assert old_row["unit_price"] == "$1,420.00"
    assert old_row["total_price"] == "$2,840.00"


def test_backfill_prices_from_deterministic_table():
    # The model returned the item but dropped the price column; pdfplumber's table
    # extraction has it, so backfill recovers the OLD prices.
    from src.llm_parser import backfill_from_deterministic

    table = TableExtraction(
        page_number=1,
        table_index=1,
        rows=(
            ("Part Number", "Description", "Qty", "Unit Price", "Ext Price"),
            ("930-PLC-044", "Controller", "2", "$1,420.00", "$2,840.00"),
        ),
    )
    extraction = PdfExtraction("o.pdf", 1, "", (), (table,), {})
    items = [{"part_number": "930-PLC-044", "description": "Controller", "quantity": "2", "unit_price": "", "total_price": ""}]
    out = backfill_from_deterministic(items, extraction)
    assert out[0]["unit_price"] == "$1,420.00"
    assert out[0]["total_price"] == "$2,840.00"


def test_block_scan_parses_table_layout_prices():
    # Mirrors the real baseline PO: a ruled table whose text wraps; pdfplumber's
    # table cells are unusable, so we parse the clean text stream into blocks.
    text = (
        "--- Page 1 ---\n"
        "LN PART DETAILS DESCRIPTION QTY UNIT / EXT PRICE\n"
        "001 IPN: 930-PLC-044 CompactLogix Controller 2 $1,420.00\n"
        "MFG: ALLEN- Ethernet ports UNIT $2,840.00\n"
        "MPN: 1769-L30ER-SERA\n"
        "002 IPN: 410-PWR-24V DIN Rail Power Supply 5 PC $85.50\n"
        "MFG: MEAN WELL $427.50\n"
    )
    items = fallback_parse(PdfExtraction("base.pdf", 1, text, (), (), {})).line_items
    by_pn = {i["part_number"]: i for i in items}
    assert by_pn["930-PLC-044"]["unit_price"] == "$1,420.00"
    assert by_pn["930-PLC-044"]["total_price"] == "$2,840.00"
    assert by_pn["930-PLC-044"]["quantity"] == "2"
    assert by_pn["410-PWR-24V"]["unit_price"] == "$85.50"
    assert by_pn["410-PWR-24V"]["total_price"] == "$427.50"
    assert by_pn["410-PWR-24V"]["quantity"] == "5"


def test_block_scan_parses_card_layout_prices():
    # Mirrors the real amended ECO: labeled cards.
    text = (
        "--- Page 1 ---\n"
        "Item #01 | Part Reference: 930-PLC-044\n"
        "Manufacturer: ALLEN-BRADLEY | Mfr Part Number: 1769-L30ERK-SERB\n"
        "Description: CompactLogix Controller\n"
        "Quantity Ordered: 2 • Price per Unit: $1,595.00 • Total Line Val: $3,190.00\n"
    )
    items = fallback_parse(PdfExtraction("rev.pdf", 1, text, (), (), {})).line_items
    item = next(i for i in items if i["part_number"] == "930-PLC-044")
    assert item["unit_price"] == "$1,595.00"
    assert item["total_price"] == "$3,190.00"
    assert item["quantity"] == "2"
    assert item["mpn"] == "1769-L30ERK-SERB"


def test_table_and_card_layouts_reconcile_old_to_new_prices():
    # The whole point: a table baseline vs a card amended must compare, OLD prices intact.
    base_text = "--- Page 1 ---\n001 IPN: 930-PLC-044 CompactLogix Controller 2 $1,420.00\nMFG: ALLEN $2,840.00\n"
    rev_text = (
        "--- Page 1 ---\nItem #01 | Part Reference: 930-PLC-044\n"
        "Quantity Ordered: 2 • Price per Unit: $1,595.00 • Total Line Val: $3,190.00\n"
    )
    old = fallback_parse(PdfExtraction("b.pdf", 1, base_text, (), (), {}))
    new = fallback_parse(PdfExtraction("a.pdf", 1, rev_text, (), (), {}))
    log = discrepancy_log(compare_documents(old, new))
    assert "OLD: '$1,420.00' || NEW: '$1,595.00'" in log[log["field"] == "unit_price"].iloc[0]["message"]
    assert "OLD: '$2,840.00' || NEW: '$3,190.00'" in log[log["field"] == "total_price"].iloc[0]["message"]


def test_csv_and_excel_inputs_parse_and_compare():
    from src.llm_parser import parse_tabular

    old_csv = b"Part Number,Description,Qty,Unit Price,Ext Price,MPN\n930-PLC-044,Controller,2,$1420.00,$2840.00,1769-L30ER-SERA\n"
    new_csv = b"Part Number,Description,Qty,Unit Price,Ext Price,MPN\n930-PLC-044,Controller,2,$1595.00,$3190.00,1769-L30ERK-SERB\n"
    old = parse_tabular(old_csv, "base.csv")
    new = parse_tabular(new_csv, "amended.csv")
    assert old.parser == "tabular"
    assert old.line_items[0]["part_number"] == "930-PLC-044"
    assert old.line_items[0]["unit_price"] == "$1420.00"

    log = discrepancy_log(compare_documents(old, new))
    # Prices are displayed in a consistent currency format.
    assert "OLD: '$1,420.00' || NEW: '$1,595.00'" in log[log["field"] == "unit_price"].iloc[0]["message"]

    # Excel input parses too.
    import io
    import openpyxl

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["Part Number", "Qty", "Unit Price"])
    ws.append(["620-NET-SW8", "1", "$610.00"])
    buffer = io.BytesIO()
    wb.save(buffer)
    xlsx = parse_tabular(buffer.getvalue(), "sheet.xlsx")
    assert xlsx.line_items[0]["part_number"] == "620-NET-SW8"
    assert xlsx.line_items[0]["unit_price"] == "$610.00"


def test_money_normalization_noop_on_unpriced():
    # No currency anywhere -> price fields stay empty (no false money).
    old = _doc("o.pdf", [_item("REF-1", quantity="1", description="Reference Drawing", raw_text="REF-1 Reference Drawing 1 EA")])
    new = _doc("n.pdf", [_item("REF-1", quantity="1", description="Reference Drawing Rev B")])
    diff = compare_documents(old, new)
    old_row = diff.old_items.iloc[0]
    assert old_row["unit_price"] == "" and old_row["total_price"] == ""


# ============================================================================
# Severity-coded PDF highlight overlays
# ============================================================================
def test_pdf_highlights_are_color_matched_per_line():
    from src import visual

    old = _doc(
        "o.pdf",
        [
            _item("A-1", quantity="2", unit_price="$10.00", description="Relay"),
            _item("B-2", quantity="1", unit_price="$20.00", description="Fuse"),
            _item("Z-9", quantity="1", description="dropped part"),
        ],
    )
    new = _doc(
        "n.pdf",
        [
            _item("A-1", quantity="2", unit_price="$12.00", description="Relay"),
            _item("B-2", quantity="1", unit_price="$25.00", description="Fuse"),
            _item("D-4", quantity="1", description="new part"),
        ],
    )
    highlights = visual.build_highlights(compare_documents(old, new))
    old_colors = dict(highlights["old"])
    new_colors = dict(highlights["new"])

    # Added / removed keep their structural colors.
    assert new_colors["D-4"] == visual.ADDED
    assert old_colors["Z-9"] == visual.REMOVED

    # Each modified line's OLD and NEW values share ONE distinct color (matched).
    assert old_colors["A-1"] == new_colors["A-1"]
    assert old_colors["$10.00"] == old_colors["A-1"]   # the old value is boxed in the line's color
    assert new_colors["$12.00"] == new_colors["A-1"]   # the matching new value too
    assert old_colors["B-2"] == new_colors["B-2"]
    # Different modified lines get different colors so they're distinguishable.
    assert old_colors["A-1"] != old_colors["B-2"]


# ============================================================================
# Baseline hydration / legacy-log purge / low-severity text channels
# ============================================================================
# Test Case 1: Baseline context map verification (no OLD:'' across the board)
def test_baseline_context_map_verification():
    old = _doc("b.pdf", [_item("105-ENC-AL-01", quantity="3", unit_price="$325.00", total_price="$975.00")])
    new = _doc("a.pdf", [_item("105-ENC-AL-01", quantity="3", unit_price="$350.00", total_price="$1,050.00")])
    log = discrepancy_log(compare_documents(old, new))

    # Every modified row recovers its historical baseline value.
    assert (log[log["status"] == "modified"]["old_value"] == "").sum() == 0
    unit_row = log[log["field"] == "unit_price"].iloc[0]
    assert "OLD: '$325.00' || NEW: '$350.00'" in unit_row["message"]


# Test Case 2: Complete text-stream cleansing (legacy block absent from stdout)
def test_complete_text_stream_cleansing(capsys):
    from src.engine import print_discrepancy_log

    old = _doc("b.pdf", [_item("105-ENC-AL-01", quantity="3", unit_price="$325.00")])
    new = _doc("a.pdf", [_item("105-ENC-AL-01", quantity="3", unit_price="$350.00")])
    print_discrepancy_log(compare_documents(old, new))

    out = capsys.readouterr().out
    assert "highmodifiedpart_number:" not in out
    assert not re.search(r"^[A-Za-z]+:[0-9]+", out, re.MULTILINE)
    assert "HIGH | MODIFIED | Part: 105-ENC-AL-01 | Field: unit_price ->" in out


# Test Case 3: Low-severity annotation inclusion (description + contextual_notes)
def test_low_severity_annotation_inclusion():
    old = _doc(
        "b.pdf",
        [_item("105-ENC-AL-01", quantity="3", description="Aluminum Enclosure", contextual_notes="[NOTE A]")],
    )
    new = _doc(
        "a.pdf",
        [_item("105-ENC-AL-01", quantity="3", description="Aluminum Enclosure Rev B", contextual_notes="[NOTE B]")],
    )
    log = discrepancy_log(compare_documents(old, new))

    description_row = log[log["field"] == "description"].iloc[0]
    assert description_row["severity"] == "low"
    assert description_row["message"].startswith("LOW | MODIFIED | Part: 105-ENC-AL-01 | Field: description -> ")

    notes_row = log[log["field"] == "contextual_notes"].iloc[0]
    assert notes_row["severity"] == "low"
    assert notes_row["message"].startswith("LOW | MODIFIED | Part: 105-ENC-AL-01 | Field: contextual_notes -> ")
