"""Page 04: cross-linguistic probing dashboard (issue A5, owner Arcangelo).

UI body for the cross-linguistic page. Helpers and cached loaders live in app/tabs/crossling.py.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
from tabs.crossling import (
    EVAL_SETS,
    METRICS,
    PAIR_IDS,
    PROBE_DEFAULTS,
    _eval_treebanks,
    _format_delta_table,
    _load_curves,
    _load_delta_table,
    _load_probing_results,
    _pair_spec,
    _plot_en_it,
    _series_for_metric,
)

from lib.paths import probing_result_path
from viz.crossling_plots import plot_delta_cog, plot_selectivity_pairs

st.title("Cross-Linguistic")
st.caption("Run locally with: `PYTHONPATH=src streamlit run app/app.py`")

results = _load_probing_results()
if results is None:
    st.info(
        f"No probing results found in "
        f"{probing_result_path('gpt2', 'ewt', 'linear', 'pretrained', 'first').parent}"
    )
    st.stop()

delta = _load_delta_table()
if delta is None:
    st.info(
        "Missing Delta CoG summary. "
        "Run scripts/11_summarize_probing.py and scripts/40_crossling_report.py."
    )
    st.stop()

try:
    curves = _load_curves(results)
except ValueError as err:
    st.error(str(err))
    st.stop()

col_pair, col_eval, col_metric = st.columns(3)
with col_pair:
    pair_id = st.selectbox(
        "Model pair",
        PAIR_IDS,
        key="crossling_pair",
    )
with col_eval:
    eval_set = st.selectbox(
        "Eval set",
        list(EVAL_SETS.keys()),
        format_func=lambda k: EVAL_SETS[k],
        key="crossling_eval",
    )
with col_metric:
    metric = st.selectbox(
        "Metric",
        list(METRICS.keys()),
        format_func=lambda k: METRICS[k],
        key="crossling_metric",
    )

spec = _pair_spec(pair_id)
en_eval, it_eval = _eval_treebanks(spec, eval_set)
en_df = _series_for_metric(
    curves, results, metric,
    model=spec.en_model,
    train_treebank=spec.en_train_treebank,
    eval_treebank=en_eval,
)
it_df = _series_for_metric(
    curves, results, metric,
    model=spec.it_model,
    train_treebank=spec.it_train_treebank,
    eval_treebank=it_eval,
)

if en_df.empty and it_df.empty:
    st.warning("No curve data for this pair and eval set.")
else:
    title = f"{pair_id} · {EVAL_SETS[eval_set]} · {METRICS[metric]}"
    st.plotly_chart(
        _plot_en_it(en_df, it_df, metric, title=title),
        width="stretch",
        key="crossling_chart",
    )

st.subheader("Delta CoG")
delta_view = _format_delta_table(delta)
pair_delta = delta_view.loc[delta_view["pair"] == pair_id] if "pair" in delta_view.columns else delta_view
st.dataframe(
    pair_delta if not pair_delta.empty else delta_view,
    width="stretch",
    hide_index=True,
)

with st.expander("Overview figures (matplotlib)", expanded=False):
    with tempfile.TemporaryDirectory(prefix="crossling-demo-") as tmp:
        tmp_path = Path(tmp)
        pairs_path = tmp_path / "selectivity_pairs.png"
        delta_path = tmp_path / "delta_cog.png"
        try:
            plot_selectivity_pairs(curves, delta, pairs_path, **PROBE_DEFAULTS)
            plot_delta_cog(delta, delta_path)
        except (ValueError, KeyError) as err:
            st.warning(f"Could not build overview figures: {err}")
        else:
            col_a, col_b = st.columns(2)
            with col_a:
                st.image(str(pairs_path), caption="Selectivity vs τ (all pairs)", width="stretch")
            with col_b:
                st.image(str(delta_path), caption="Delta CoG by pair", width="stretch")
