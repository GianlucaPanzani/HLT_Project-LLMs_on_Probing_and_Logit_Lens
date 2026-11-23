"""Tab 2: probing explorer (issue A2, owner Gianluca)."""

from __future__ import annotations

import streamlit as st

from lib.paths import probing_result_path


def render() -> None:
    """Placeholder until issue A2; never raises if results are missing."""
    results_dir = probing_result_path("gpt2", "ewt", "linear", "pretrained", "first").parent
    if not results_dir.is_dir() or not any(results_dir.glob("*.parquet")):
        st.info(f"No probing results found in {results_dir}")
        return
    st.info("Tab under construction — see issue A2 (probing curves and baselines explorer).")