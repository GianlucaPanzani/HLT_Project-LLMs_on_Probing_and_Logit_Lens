"""Page 01: live lens heatmap (issue A5, owner Matteo).

UI body for the lens heatmap page. Helpers and cached loaders live in app/tabs/lens_heatmap.py.
"""

from __future__ import annotations

import streamlit as st
from tabs.lens_heatmap import (
    DEFAULT_SENTENCES,
    LENSES,
    MAX_TOKENS,
    METRICS,
    MODELS,
    _build_heatmap,
    compute_lens_grid,
    load_lens_model,
    load_lens_params,
)

from lib.registry import get_model_spec

st.title("Lens Heatmap")
st.caption("Run locally with: `PYTHONPATH=src streamlit run app/app.py`")

col_model, col_lens, col_metric = st.columns(3)
with col_model:
    model_key = st.selectbox(
        "Model",
        MODELS,
        format_func=lambda k: k,
        key="lens_model",
    )
with col_lens:
    lens = st.radio(
        "Lens",
        LENSES,
        horizontal=True,
        key="lens_type",
    )
with col_metric:
    metric = st.radio(
        "Color",
        METRICS,
        format_func=lambda m: m.upper(),
        horizontal=True,
        key="lens_metric",
    )

default_text = DEFAULT_SENTENCES.get(model_key, "")
text = st.text_area(
    "Sentence (whitespace-separated words)",
    value=default_text,
    height=80,
    key="lens_text",
)

if not text.strip():
    st.info("Enter a sentence to run the lens.")
    st.stop()

words = text.strip().split()
if len(words) > MAX_TOKENS:
    st.warning(f"Input has {len(words)} words; only the first {MAX_TOKENS} are encoded.")

params, lens_error = load_lens_params(model_key, lens)
if lens_error:
    st.warning(lens_error)
    st.stop()

model, tokenizer, _meta = load_lens_model(model_key)

with st.spinner("Running lens…"):
    try:
        x_ticks, _input_toks, next_toks, top1_text, ent_grid, kl_grid, truncated = compute_lens_grid(
            model, tokenizer, words[:MAX_TOKENS], lens, params
        )
    except ValueError as err:
        st.error(str(err))
        st.stop()

spec = get_model_spec(model_key)
expected_rows = int(spec["num_layers"]) + 1
if ent_grid.shape[0] != expected_rows:
    st.warning(
        f"Expected {expected_rows} layers (0…L) but the lens returned {ent_grid.shape[0]} rows. "
        "Check model and checkpoint compatibility."
    )

if truncated:
    st.warning(f"Sequence truncated to {MAX_TOKENS} subword tokens (plus BOS).")

z_grid = ent_grid if metric == "entropy" else kl_grid
fig = _build_heatmap(
    x_ticks, next_toks, top1_text, z_grid, metric, kl_grid, ent_grid, lens=lens
)
st.plotly_chart(
    fig,
    width="stretch",
    config={"scrollZoom": True, "displaylogo": False},
    key="lens_heatmap_chart",
)

final_preds = top1_text[-1]
n_scored = sum(1 for g in next_toks if g != "—")
n_correct = sum(
    1 for pred, gold in zip(final_preds, next_toks, strict=True) if gold != "—" and pred == gold
)
if n_scored:
    st.caption(f"Final layer: **{n_correct}/{n_scored}** correct top-1 predictions.")
