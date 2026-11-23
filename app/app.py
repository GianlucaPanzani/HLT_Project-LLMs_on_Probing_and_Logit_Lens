"""HLT project interactive demo (issue A1). Run: PYTHONPATH=src streamlit run app/app.py."""

from __future__ import annotations

import streamlit as st
from tabs import cka, crossling, lens_heatmap, probing

st.set_page_config(page_title="LLMs: probing and lens", layout="wide")

st.title("Probing and logit lens")
st.caption("Run locally with: `PYTHONPATH=src streamlit run app/app.py`")

tab_lens, tab_probing, tab_cka, tab_crossling = st.tabs(
    ["Lens heatmap", "Probing", "CKA", "Cross-linguistic"]
)

with tab_lens:
    lens_heatmap.render()

with tab_probing:
    probing.render()

with tab_cka:
    cka.render()

with tab_crossling:
    crossling.render()