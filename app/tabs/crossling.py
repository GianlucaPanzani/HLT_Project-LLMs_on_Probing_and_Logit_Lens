"""Tab 4: cross-linguistic dashboard (issue A4, owner Arcangelo)."""

from __future__ import annotations

import streamlit as st

from lib.paths import ROOT


def render() -> None:
    """Placeholder until issue A4; never raises if results are missing."""
    summary_dir = ROOT / "results" / "summary"
    if not summary_dir.is_dir() or not any(summary_dir.glob("*.parquet")):
        st.info(f"No cross-linguistic summary results found in {summary_dir}")
        return
    st.info("Tab under construction — see issue A4 (cross-linguistic dashboard).")