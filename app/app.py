"""HLT project interactive demo (issue A5). Run: PYTHONPATH=src streamlit run app/app.py."""

from __future__ import annotations

from pathlib import Path

import streamlit as st

st.set_page_config(page_title="LLMs: probing and lens", layout="wide")

_pages_dir = Path(__file__).parent / "pages"
_page_files = sorted(_pages_dir.glob("[0-9][0-9]_*.py"))
_pages = [st.Page(str(p)) for p in _page_files]
nav = st.navigation(_pages, position="sidebar")
nav.run()
