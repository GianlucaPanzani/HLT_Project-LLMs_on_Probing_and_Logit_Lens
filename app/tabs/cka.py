"""Tab 3: CKA and compression explorer (issue A3, owner Filippo)."""

from __future__ import annotations

import streamlit as st

from lib.paths import cka_result_path


def render() -> None:
    """Placeholder until issue A3; never raises if results are missing."""
    results_dir = cka_result_path("bert", "gpt2", "ewt").parent
    if not results_dir.is_dir() or not any(results_dir.glob("*.parquet")):
        st.info(f"No CKA results found in {results_dir}")
        return
    st.info("Tab under construction — see issue A3 (compression and CKA explorer).")