"""Streamlit interface for BoMination / RevFinder (OMNI Control Technology)."""

from __future__ import annotations

import base64
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import visual
from src.engine import compare_documents, discrepancy_log, filter_changes
from src.extractor import extract_pdf
from src.llm_parser import (
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    SUGGESTED_MODELS,
    LocalOllamaParser,
    parse_tabular,
)
from src.reporter import build_excel_report


# OMNI Control Technology brand palette.
PRIMARY = "#1E50E0"  # Cobalt Blue accent
INK = "#0F1B33"  # Deep-Navy typography
SURFACE = "#F5F7FB"  # Off-white containers / sidebar
WHITE = "#FFFFFF"  # Core content blocks
GREEN_BG, GREEN_FG = "#E4F4E9", "#1B7A3D"  # added / price drop
RED_BG, RED_FG = "#FBE6E9", "#B0203A"  # removed / price increase
AMBER_BG, AMBER_FG = "#FDF0D5", "#946200"  # high-severity / ECO updates
SLATE_BG, SLATE_FG = "#EAEEF6", INK  # neutral


st.set_page_config(
    page_title="RevFinder",
    page_icon="🛠️",
    layout="wide",
)


BRAND_CSS = f"""
<style>
  .stApp {{ background-color: {WHITE}; color: {INK}; }}
  section[data-testid="stSidebar"] {{ background-color: {SURFACE}; }}
  h1, h2, h3, h4, h5, h6, p, label, .stMarkdown {{ color: {INK}; }}

  /* Top corporate header */
  .omni-header {{
    display: flex; align-items: center; gap: 18px;
    background: {SURFACE}; border: 1px solid #DCE3F0; border-left: 6px solid {PRIMARY};
    border-radius: 10px; padding: 16px 20px; margin-bottom: 18px;
  }}
  .omni-logo-slot {{
    width: 88px; height: 56px; flex: 0 0 auto;
    display: flex; align-items: center; justify-content: center;
    border: 1px dashed {PRIMARY}; border-radius: 8px;
    color: {PRIMARY}; font-size: 11px; font-weight: 700; letter-spacing: .06em;
    background: {WHITE};
  }}
  .omni-logo {{
    width: 64px; height: 64px; flex: 0 0 auto;
    object-fit: contain; border-radius: 8px; background: {WHITE};
  }}
  .omni-title {{ font-size: 26px; font-weight: 800; color: {INK}; line-height: 1.1; }}
  .omni-sub {{ font-size: 14px; color: #44516B; margin-top: 2px; }}

  /* Primary controls -> Cobalt */
  .stButton > button[kind="primary"],
  button[data-testid="baseButton-primary"],
  button[data-testid="stBaseButton-primary"] {{
    background-color: {PRIMARY}; border-color: {PRIMARY}; color: {WHITE}; font-weight: 700;
  }}
  /* Upload dropzones */
  [data-testid="stFileUploaderDropzone"] {{
    background: {SURFACE}; border: 1.5px dashed {PRIMARY}; border-radius: 10px;
  }}

  /* Status badges */
  .omni-badge {{
    display: inline-block; padding: 2px 10px; border-radius: 999px;
    font-size: 12px; font-weight: 700; letter-spacing: .02em;
  }}
  .b-green {{ background: {GREEN_BG}; color: {GREEN_FG}; }}
  .b-red   {{ background: {RED_BG};   color: {RED_FG}; }}
  .b-amber {{ background: {AMBER_BG}; color: {AMBER_FG}; }}
  .b-slate {{ background: {SLATE_BG}; color: {SLATE_FG}; }}
</style>
"""


LOGO_PATH = PROJECT_ROOT / "logo.jpeg"


@st.cache_data(show_spinner=False)
def _extract_cached(file_bytes: bytes, file_name: str):
    """Cache deterministic PDF extraction keyed on the uploaded bytes."""

    return extract_pdf(file_bytes, source_name=file_name)


def _is_pdf(file_name: str) -> bool:
    return file_name.lower().endswith(".pdf")


def _parse_upload(file_bytes: bytes, file_name: str, parser):
    """Parse an uploaded document by type: PDF via the extractor+parser, CSV/Excel
    via the deterministic tabular parser."""

    if _is_pdf(file_name):
        return parser.parse(_extract_cached(file_bytes, file_name))
    return parse_tabular(file_bytes, file_name)


@st.cache_data(show_spinner=False)
def _logo_html() -> str:
    """Embed the corporate logo as a data URI, or fall back to a spacer slot."""

    try:
        encoded = base64.b64encode(LOGO_PATH.read_bytes()).decode("ascii")
    except OSError:
        return '<div class="omni-logo-slot">OMNI<br/>LOGO</div>'
    return f'<img class="omni-logo" src="data:image/jpeg;base64,{encoded}" alt="OMNI Control Technology logo" />'


def _render_header() -> None:
    st.markdown(
        f"""
        <div class="omni-header">
          {_logo_html()}
          <div>
            <div class="omni-title">RevFinder</div>
            <div class="omni-sub">OMNI Control Technology &mdash; internal engineering revision utility</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.markdown(BRAND_CSS, unsafe_allow_html=True)
    _render_header()

    with st.sidebar:
        st.header("Runtime")
        base_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
        model = st.selectbox(
            "Model",
            options=SUGGESTED_MODELS,
            index=0,
            help="Default is llama3.2. Stronger models map arbitrary layouts more reliably; pull one first, e.g. `ollama pull llama3.1:8b`.",
        )
        timeout_seconds = st.number_input("Timeout seconds", min_value=30, max_value=600, value=180, step=30)
        header_band_inches = st.slider(
            "Ignore top header band (inches)",
            min_value=0.0,
            max_value=3.0,
            value=1.5,
            step=0.25,
            help="Top margin excluded from notes parsing so company title and document metadata are not scraped as notes.",
        )
        check_health = st.button("Check Ollama")
        if check_health:
            ok, message = LocalOllamaParser(base_url=base_url, model=model).healthcheck()
            if ok:
                st.success(message)
            else:
                st.warning(message)

    st.subheader("Revision inputs")
    st.caption("Accepts PDF, CSV, or Excel (.xlsx/.xls). Visual side-by-side review is available for PDF inputs.")
    accepted = ["pdf", "csv", "xlsx", "xls", "xlsm"]
    left, right = st.columns(2)
    with left:
        old_file = st.file_uploader("Baseline (Old Revision)", type=accepted, key="old_pdf")
    with right:
        new_file = st.file_uploader("Amended (New ECO Revision)", type=accepted, key="new_pdf")

    run = st.button("Process Revisions", type="primary", disabled=not old_file or not new_file)
    if run and old_file and new_file:
        with st.status("Processing revisions", expanded=True) as status:
            parser = LocalOllamaParser(
                base_url=base_url,
                model=model,
                timeout_seconds=int(timeout_seconds),
                header_band_points=header_band_inches * 72.0,
            )
            st.write("Parsing both revisions in parallel")
            with ThreadPoolExecutor(max_workers=2) as pool:
                old_future = pool.submit(_parse_upload, old_file.getvalue(), old_file.name, parser)
                new_future = pool.submit(_parse_upload, new_file.getvalue(), new_file.name, parser)
                old_parsed = old_future.result()
                new_parsed = new_future.result()

            st.write("Calculating deterministic delta")
            diff = compare_documents(old_parsed, new_parsed)
            report = build_excel_report(diff)

            st.session_state["diff"] = diff
            st.session_state["report"] = report
            st.session_state["baseline_pdf"] = {
                "bytes": old_file.getvalue(), "name": old_file.name, "is_pdf": _is_pdf(old_file.name)
            }
            st.session_state["amended_pdf"] = {
                "bytes": new_file.getvalue(), "name": new_file.name, "is_pdf": _is_pdf(new_file.name)
            }
            st.session_state["viz_page"] = 1
            status.update(label="Comparison complete", state="complete", expanded=False)

    if "diff" in st.session_state:
        _render_results(st.session_state["diff"], st.session_state["report"])


_LEGEND_HTML = """
<div style="display:flex; gap:10px; flex-wrap:wrap; align-items:center; margin:4px 0 12px 0;
            position:sticky; top:0; z-index:5; padding:6px 0;">
  <span class="omni-badge b-slate">&#9632; Each changed line has its own color &mdash; find the same color on both pages to match OLD &rarr; NEW</span>
  <span class="omni-badge b-green">&#9632; Added</span>
  <span class="omni-badge b-red">&#9632; Removed</span>
</div>
"""


def _step_viz_page(delta: int, total_pages: int) -> None:
    """Move the synced page index, clamped — runs before the script re-executes."""

    current = int(st.session_state.get("viz_page", 1)) + delta
    st.session_state["viz_page"] = min(max(current, 1), total_pages)


def _render_pdf_pane(label: str, pdf: dict, page_index: int, total_for_doc: int, highlights) -> None:
    st.caption(label)
    if page_index >= total_for_doc:
        st.info(f"This document has {total_for_doc} page(s) — nothing on page {page_index + 1}.")
        return
    image = visual.render_page_with_highlights(pdf["bytes"], page_index, highlights, zoom=3.0)
    if image:
        st.image(image, use_column_width=True)
    else:
        st.info("Unable to render this page.")


_TABULAR_COLUMNS = ["part_number", "description", "quantity", "unit_price", "total_price", "manufacturer", "mpn"]


def _rgb_tint(rgb: tuple[float, float, float], alpha: float = 0.28) -> str:
    """Light background tint (blend the overlay color with white) for CSS."""

    r, g, b = (int((channel * alpha + (1 - alpha)) * 255) for channel in rgb)
    return f"rgb({r},{g},{b})"


def _change_color_map(diff):
    """part_number -> color for each side, matching the PDF overlay color scheme."""

    old_map: dict[str, tuple[float, float, float]] = {}
    new_map: dict[str, tuple[float, float, float]] = {}
    color_index = 0
    for change in diff.changes:
        if change.status == "added":
            new_map[str(change.new_item.get("part_number", ""))] = visual.ADDED
        elif change.status == "removed":
            old_map[str(change.old_item.get("part_number", ""))] = visual.REMOVED
        elif change.status == "modified":
            color = visual._MATCH_PALETTE[color_index % len(visual._MATCH_PALETTE)]
            color_index += 1
            old_map[str(change.old_item.get("part_number", ""))] = color
            new_map[str(change.new_item.get("part_number", ""))] = color
    return old_map, new_map


def _style_item_table(frame: pd.DataFrame, color_map: dict) -> pd.io.formats.style.Styler:
    columns = [c for c in _TABULAR_COLUMNS if c in frame.columns]
    view = frame[columns].copy()

    def _row_style(row: pd.Series) -> list[str]:
        color = color_map.get(str(row.get("part_number", "")))
        if not color:
            return [""] * len(row)
        return [f"background-color: {_rgb_tint(color)};"] * len(row)

    return view.style.apply(_row_style, axis=1)


def _render_tabular_review(diff) -> None:
    """Color-coded side-by-side line items for CSV/Excel inputs (no PDF pages)."""

    old_map, new_map = _change_color_map(diff)
    left, right = st.columns(2)
    with left:
        st.caption(f"Baseline (Old) — {diff.old_document.source_name}")
        _show_styled(_style_item_table(diff.old_items, old_map))
    with right:
        st.caption(f"Amended (New ECO) — {diff.new_document.source_name}")
        _show_styled(_style_item_table(diff.new_items, new_map))


def _show_styled(styler) -> None:
    if styler.data.empty:
        st.info("No line items parsed.")
        return
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _render_visual_review(diff) -> None:
    """Always-on, full-width side-by-side PDF review with synced page navigation."""

    old_pdf = st.session_state.get("baseline_pdf")
    new_pdf = st.session_state.get("amended_pdf")

    st.subheader("🔬 Visual Revision Review")
    if not old_pdf or not new_pdf:
        st.info("Run a comparison to enable visual review.")
        return
    if not (old_pdf.get("is_pdf") and new_pdf.get("is_pdf")):
        st.caption(
            "Side-by-side line items. Each changed line shares a color across both sides — "
            "match the color to see OLD → NEW (incl. part swaps). Added = green, removed = red."
        )
        _render_tabular_review(diff)
        return
    if not visual.is_available():
        st.warning("Visual review needs PyMuPDF. Install it with `pip install pymupdf`.")
        return

    st.markdown(_LEGEND_HTML, unsafe_allow_html=True)

    old_pages = visual.page_count(old_pdf["bytes"])
    new_pages = visual.page_count(new_pdf["bytes"])
    total_pages = max(old_pages, new_pages, 1)

    # Clamp first so the buttons below render their disabled state for the CURRENT
    # page. Page changes happen in on_click callbacks (before this run), so the
    # widgets and images here are always consistent with each other.
    current = min(max(int(st.session_state.get("viz_page", 1)), 1), total_pages)
    st.session_state["viz_page"] = current

    prev_col, next_col, label_col = st.columns([1, 1, 6])
    prev_col.button(
        "◀ Prev page",
        disabled=current <= 1,
        use_container_width=True,
        on_click=_step_viz_page,
        args=(-1, total_pages),
    )
    next_col.button(
        "Next page ▶",
        disabled=current >= total_pages,
        use_container_width=True,
        on_click=_step_viz_page,
        args=(1, total_pages),
    )
    label_col.markdown(
        f"<div style='padding-top:6px; font-weight:700;'>Page {current} of {total_pages} "
        "<span style='font-weight:400; color:#44516B;'>— both views switch together</span></div>",
        unsafe_allow_html=True,
    )

    page_index = current - 1
    highlights = visual.build_highlights(diff)

    left, right = st.columns(2)
    with left:
        _render_pdf_pane(f"Baseline (Old) — {old_pdf['name']}", old_pdf, page_index, old_pages, highlights["old"])
    with right:
        _render_pdf_pane(f"Amended (New ECO) — {new_pdf['name']}", new_pdf, page_index, new_pages, highlights["new"])


def _render_results(diff, report) -> None:
    doc_type, has_pricing = _document_context(diff)
    pricing_note = (
        "pricing detected" if has_pricing
        else "no pricing detected — comparing description, quantity & structural changes"
    )
    st.markdown(
        f"<div style='padding:8px 0 4px 0;'>Detected document type: "
        f"<span class='omni-badge b-slate'>{doc_type}</span> &nbsp;·&nbsp; "
        f"<span style='color:#44516B;'>{pricing_note}</span></div>",
        unsafe_allow_html=True,
    )

    summary = diff.summary
    metrics = st.columns(7)
    metrics[0].metric("Old Items", summary["old_items"])
    metrics[1].metric("New Items", summary["new_items"])
    metrics[2].metric("Added", summary["added"])
    metrics[3].metric("Removed", summary["removed"])
    metrics[4].metric("Modified", summary["modified"])
    metrics[5].metric("Unchanged", summary["unchanged"])
    metrics[6].metric("Doc Changes", summary.get("document_changes", 0))

    st.download_button(
        "Download Excel Report",
        data=report,
        file_name="revfinder_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    st.divider()
    _render_visual_review(diff)
    st.divider()

    _render_diagnostics(diff)

    warnings = [*diff.old_document.warnings, *diff.new_document.warnings]
    if warnings:
        with st.expander("Parser warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)

    tabs = st.tabs(
        [
            "Discrepancies",
            "Document Changes",
            "All Changes",
            "Added",
            "Removed",
            "Modified",
            "Raw Old",
            "Raw New",
        ]
    )
    with tabs[0]:
        _render_discrepancies(diff)
    with tabs[1]:
        _show_dataframe(diff.document_changes)
    with tabs[2]:
        _show_dataframe(diff.comparison)
    with tabs[3]:
        _show_dataframe(filter_changes(diff, "added"))
    with tabs[4]:
        _show_dataframe(filter_changes(diff, "removed"))
    with tabs[5]:
        _show_dataframe(filter_changes(diff, "modified"))
    with tabs[6]:
        _show_dataframe(diff.old_items)
    with tabs[7]:
        _show_dataframe(diff.new_items)


# Severity tiers, highest impact first.
_SEVERITY_TIERS = [
    ("High", "high"),
    ("Medium", "medium"),
    ("Low", "low"),
    ("Review", "review"),
]


def _render_discrepancies(diff) -> None:
    """Structured delta grid, grouped by severity tier with colored badges."""

    log = discrepancy_log(diff)
    if log.empty:
        st.info("No discrepancies")
        return

    for label, severity in _SEVERITY_TIERS:
        subset = log[log["severity"] == severity]
        if subset.empty:
            continue
        st.markdown(f"#### {label} severity &nbsp;<span class='omni-badge b-slate'>{len(subset)}</span>", unsafe_allow_html=True)
        st.dataframe(
            subset.style.apply(_discrepancy_row_style, axis=1),
            use_container_width=True,
            hide_index=True,
        )


def _render_diagnostics(diff) -> None:
    """Surface why a comparison may look empty: parser used, counts, match breakdown."""

    summary = diff.summary
    aligned = summary["modified"] + summary["unchanged"]
    old_parser = diff.old_document.parser or "unknown"
    new_parser = diff.new_document.parser or "unknown"
    # Deterministic parsing is the PRIMARY engine (accurate/consistent for
    # structured manifests) — not an error. Only warn if the LLM was actually
    # attempted and failed, which is recorded in the parser warnings.
    warnings = [*diff.old_document.warnings, *diff.new_document.warnings]
    ollama_failed = any("ollama" in str(w).lower() for w in warnings)

    with st.expander("Diagnostics", expanded=ollama_failed or aligned == 0):
        st.write(
            {
                "baseline parser": old_parser,
                "amended parser": new_parser,
                "baseline items": summary["old_items"],
                "amended items": summary["new_items"],
                "aligned rows (modified+unchanged)": aligned,
                "added": summary["added"],
                "removed": summary["removed"],
            }
        )
        st.caption(
            "RevFinder uses a deterministic parser as the primary engine for structured PO/BoM "
            "documents (more accurate and consistent than a small local LLM). A local model is only "
            "used when the deterministic parser cannot segment a layout."
        )
        if ollama_failed:
            st.warning(
                "A local-model parse was attempted and failed (Ollama unreachable or model not pulled). "
                "The deterministic parser was used instead."
            )
        elif aligned == 0 and (summary["added"] or summary["removed"]):
            st.warning(
                "No rows aligned between the two documents (everything shows as added/removed). The two "
                "layouts likely parsed part numbers differently; content matching is applied, but check the "
                "Raw Old / Raw New tabs to confirm both documents extracted line items correctly."
            )


def _document_context(diff) -> tuple[str, bool]:
    """Detected document type (PO/BoM/ECO) and whether any pricing is present."""

    new_type = str(getattr(diff.new_document, "document_type", "") or "").strip()
    old_type = str(getattr(diff.old_document, "document_type", "") or "").strip()
    doc_type = new_type if new_type and new_type.lower() != "unknown" else (old_type or "unknown")

    has_pricing = False
    for frame in (diff.old_items, diff.new_items):
        for column in ("unit_price", "total_price"):
            if column in frame.columns and frame[column].astype(str).str.strip().ne("").any():
                has_pricing = True
                break
    return doc_type.upper(), has_pricing


def _discrepancy_row_style(row: pd.Series) -> list[str]:
    background, foreground = _discrepancy_colors(row)
    return [f"background-color: {background}; color: {foreground};"] * len(row)


def _discrepancy_colors(row: pd.Series) -> tuple[str, str]:
    """Severity-coded row colors, matching the PDF overlay (high=red, med=amber, low=green)."""

    status = str(row.get("status", ""))
    severity = str(row.get("severity", ""))

    if status == "added":
        return GREEN_BG, GREEN_FG
    if status == "removed":
        return RED_BG, RED_FG
    if severity == "high":
        return RED_BG, RED_FG
    if severity == "medium":
        return AMBER_BG, AMBER_FG
    if severity == "low":
        return GREEN_BG, GREEN_FG
    return SLATE_BG, SLATE_FG


def _show_dataframe(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No rows")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
