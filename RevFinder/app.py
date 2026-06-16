"""Streamlit interface for RevFinder."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.engine import compare_documents, discrepancy_log, filter_changes
from src.extractor import extract_pdf
from src.llm_parser import DEFAULT_MODEL, DEFAULT_OLLAMA_URL, LocalOllamaParser
from src.reporter import build_excel_report


st.set_page_config(
    page_title="RevFinder",
    layout="wide",
)


def main() -> None:
    st.title("RevFinder")

    with st.sidebar:
        st.header("Runtime")
        base_url = st.text_input("Ollama URL", value=DEFAULT_OLLAMA_URL)
        model = st.selectbox("Model", options=[DEFAULT_MODEL, "llama3.2:1b"], index=0)
        timeout_seconds = st.number_input("Timeout seconds", min_value=30, max_value=600, value=180, step=30)
        check_health = st.button("Check Ollama")
        if check_health:
            ok, message = LocalOllamaParser(base_url=base_url, model=model).healthcheck()
            st.success(message) if ok else st.warning(message)

    left, right = st.columns(2)
    with left:
        old_file = st.file_uploader("Old revision PDF", type=["pdf"], key="old_pdf")
    with right:
        new_file = st.file_uploader("New revision PDF", type=["pdf"], key="new_pdf")

    run = st.button("Compare Revisions", type="primary", disabled=not old_file or not new_file)
    if run and old_file and new_file:
        with st.status("Processing revisions", expanded=True) as status:
            st.write("Extracting old revision")
            old_extraction = extract_pdf(old_file.getvalue(), source_name=old_file.name)
            st.write("Extracting new revision")
            new_extraction = extract_pdf(new_file.getvalue(), source_name=new_file.name)

            parser = LocalOllamaParser(base_url=base_url, model=model, timeout_seconds=int(timeout_seconds))
            st.write("Parsing old revision")
            old_parsed = parser.parse(old_extraction)
            st.write("Parsing new revision")
            new_parsed = parser.parse(new_extraction)

            st.write("Calculating deterministic delta")
            diff = compare_documents(old_parsed, new_parsed)
            report = build_excel_report(diff)

            st.session_state["diff"] = diff
            st.session_state["report"] = report
            status.update(label="Comparison complete", state="complete", expanded=False)

    if "diff" in st.session_state:
        _render_results(st.session_state["diff"], st.session_state["report"])


def _render_results(diff, report) -> None:
    summary = diff.summary
    metrics = st.columns(6)
    metrics[0].metric("Old Items", summary["old_items"])
    metrics[1].metric("New Items", summary["new_items"])
    metrics[2].metric("Added", summary["added"])
    metrics[3].metric("Removed", summary["removed"])
    metrics[4].metric("Modified", summary["modified"])
    metrics[5].metric("Unchanged", summary["unchanged"])

    st.download_button(
        "Download Excel Report",
        data=report,
        file_name="revfinder_comparison.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    warnings = [*diff.old_document.warnings, *diff.new_document.warnings]
    if warnings:
        with st.expander("Parser warnings", expanded=False):
            for warning in warnings:
                st.warning(warning)

    tabs = st.tabs(["Discrepancies", "All Changes", "Added", "Removed", "Modified", "Raw Old", "Raw New"])
    with tabs[0]:
        _show_dataframe(discrepancy_log(diff))
    with tabs[1]:
        _show_dataframe(diff.comparison)
    with tabs[2]:
        _show_dataframe(filter_changes(diff, "added"))
    with tabs[3]:
        _show_dataframe(filter_changes(diff, "removed"))
    with tabs[4]:
        _show_dataframe(filter_changes(diff, "modified"))
    with tabs[5]:
        _show_dataframe(diff.old_items)
    with tabs[6]:
        _show_dataframe(diff.new_items)


def _show_dataframe(frame: pd.DataFrame) -> None:
    if frame.empty:
        st.info("No rows")
        return
    st.dataframe(frame, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
