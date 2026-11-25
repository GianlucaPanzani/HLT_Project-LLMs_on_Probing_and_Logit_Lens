"""Page 03: CKA and compression explorer (issue A5, owner Filippo).

UI body for the compression page. Helpers and cached loaders live in app/tabs/cka.py.
"""

from __future__ import annotations

import numpy as np
import streamlit as st
from tabs.cka import (
    CKA_DIR,
    PROBING_DIR,
    _format_mapping_table,
    _load_parquet,
    _parse_cka_filename,
)

from lib.paths import probing_result_path
from viz.cka_plots import (
    plot_cka_heatmap,
    plot_selectivity_comparison,
    plot_selectivity_comparison_plotly,
)
from ws2_compression.mapping import (
    build_layer_mapping_table,
    compute_cog_comparison,
)

st.title("Compression")
st.header("Model Compression & CKA Explorer: BERT vs DistilBERT")
st.markdown(
    "Investigating knowledge distillation between `bert-base-cased` ($L=12$, teacher) and "
    "`distilbert-base-cased` ($L=6$, student). Evaluate whether the rigid **1:2 layer mapping** "
    "($l \\leftrightarrow 2l$) holds geometrically via **Linear CKA** and functionally via "
    "**UPOS probing selectivity**."
)

# -------------------------------------------------------------------------
# 1. Check and Load CKA Results
# -------------------------------------------------------------------------
if not CKA_DIR.is_dir():
    st.info(
        f"No CKA results directory found at `{CKA_DIR}`. "
        "Run `PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert` "
        "to compute the CKA similarity matrix."
    )
    st.stop()

cka_files = sorted(CKA_DIR.glob("*.parquet"))
if not cka_files:
    st.info(
        f"No CKA results found in `{CKA_DIR}`. "
        "Run `PYTHONPATH=src python scripts/20_cka.py --model-a bert --model-b distilbert` "
        "to generate the CKA matrix."
    )
    st.stop()

col_file, col_opt1, col_opt2 = st.columns([2, 1, 1])

with col_file:
    if len(cka_files) > 1:
        selected_cka_file = st.selectbox(
            "CKA Matrix File",
            cka_files,
            index=0,
            format_func=lambda p: p.name,
            key="cka_selected_file",
        )
    else:
        selected_cka_file = cka_files[0]
        st.caption(f"Loaded CKA matrix: `{selected_cka_file.name}`")

model_a, model_b, cka_tb = _parse_cka_filename(selected_cka_file)

with col_opt1:
    mark_diagonal = st.checkbox(
        "Mark 1:2 diagonal",
        value=True,
        help="Highlight cells where Model A layer = 2 * Model B layer with a solid red border.",
        key="cka_mark_diag",
    )

with col_opt2:
    mark_best = st.checkbox(
        "Mark CKA-best",
        value=True,
        help="Highlight the Model A layer with highest CKA for each Model B layer with a dashed blue border.",
        key="cka_mark_best",
    )

try:
    cka_df = _load_parquet(selected_cka_file)
except Exception as err:
    st.error(f"Failed to read CKA parquet `{selected_cka_file.name}`: {err}")
    st.stop()

req_cka_cols = {"layer_a", "layer_b", "cka"}
if not req_cka_cols.issubset(cka_df.columns):
    st.error(f"CKA file missing required columns: {req_cka_cols - set(cka_df.columns)}")
    st.stop()

# -------------------------------------------------------------------------
# 2. CKA Heatmap
# -------------------------------------------------------------------------
st.subheader(f"Linear CKA Representation Similarity: {model_a.upper()} vs {model_b.upper()}")

try:
    fig_mpl, fig_plotly = plot_cka_heatmap(
        cka_df,
        model_a_name=model_a.upper(),
        model_b_name=model_b.upper(),
        mark_diagonal=mark_diagonal,
        mark_best=mark_best,
    )
except Exception as err:
    st.error(f"Error generating CKA heatmap: {err}")
    st.stop()

if fig_plotly is not None:
    st.plotly_chart(fig_plotly, use_container_width=True, key="cka_heatmap_plotly")
else:
    st.pyplot(fig_mpl, clear_figure=True)

st.caption(
    "**Legend**: Red solid border = rigid 1:2 mapping ($l \\leftrightarrow 2l$). "
    "Blue dashed border = empirical CKA-best mapping ($\\arg\\max_{l_a} \\text{CKA}$). "
    "If distillation preserved intermediate geometry linearly, highest values in column $l$ "
    "would fall exactly at row $2l$."
)

# -------------------------------------------------------------------------
# 3. Probing Results & Selectivity Curves
# -------------------------------------------------------------------------
st.divider()
st.subheader("Functional Alignment: Probing Selectivity & Center of Gravity")

probing_bert_path = probing_result_path(model_a, cka_tb, "linear", "pretrained", "first")
probing_distil_path = probing_result_path(model_b, cka_tb, "linear", "pretrained", "first")

if not probing_bert_path.is_file() or not probing_distil_path.is_file():
    alt_bert = sorted(PROBING_DIR.glob(f"{model_a}__*.parquet")) if PROBING_DIR.is_dir() else []
    alt_distil = sorted(PROBING_DIR.glob(f"{model_b}__*.parquet")) if PROBING_DIR.is_dir() else []

    if alt_bert and alt_distil:
        probing_bert_path = alt_bert[0]
        probing_distil_path = alt_distil[0]
    else:
        st.info(
            f"Probing results for `{model_a}` and/or `{model_b}` not found in `{PROBING_DIR}`. "
            "Run `PYTHONPATH=src python scripts/14_pull_probing_results.py` to download probing results "
            "or train probes to display selectivity curves and layer mapping comparison."
        )
        st.stop()

try:
    probing_bert = _load_parquet(probing_bert_path)
    probing_distil = _load_parquet(probing_distil_path)
except Exception as err:
    st.info(f"Could not load probing results: {err}")
    st.stop()

treebank = cka_tb if cka_tb in probing_bert["eval_treebank"].values else "ewt"
probe_type = "linear"
init_type = "pretrained"
pooling_type = "first"

try:
    cog = compute_cog_comparison(
        probing_bert,
        probing_distil,
        treebank=treebank,
        probe=probe_type,
        init=init_type,
        pooling=pooling_type,
    )
except (ValueError, KeyError) as err:
    st.info(f"Could not compute CoG comparison with current probing settings: {err}")
    st.stop()

m1, m2, m3, m4 = st.columns(4)
b_cog = cog["bert_cog_norm"]
d_cog = cog["distilbert_cog_norm"]
delta_cog = d_cog - b_cog if np.isfinite(d_cog) and np.isfinite(b_cog) else np.nan

m1.metric(
    f"{model_a.upper()} CoG_norm",
    f"{b_cog:.4f}" if np.isfinite(b_cog) else "N/A",
    help="Normalized Center of Gravity of selectivity across relative depth τ.",
)
m2.metric(
    f"{model_b.upper()} CoG_norm",
    f"{d_cog:.4f}" if np.isfinite(d_cog) else "N/A",
    help="Normalized Center of Gravity of selectivity across relative depth τ.",
)
m3.metric(
    "Δ CoG_norm (Student - Teacher)",
    f"{delta_cog:+.4f}" if np.isfinite(delta_cog) else "N/A",
    help="Positive indicates morphosyntactic peak shifted to deeper layers in the student.",
)
max_b = float(np.max(cog["bert_selectivity"])) if len(cog["bert_selectivity"]) > 0 else np.nan
max_d = float(np.max(cog["distilbert_selectivity"])) if len(cog["distilbert_selectivity"]) > 0 else np.nan
m4.metric(
    "Peak Selectivity (Teacher / Student)",
    f"{max_b:.3f} / {max_d:.3f}",
    help="Maximum UPOS selectivity reached by BERT and DistilBERT.",
)

fig_sel_plotly = plot_selectivity_comparison_plotly(
    cog["bert_selectivity"],
    cog["bert_tau"],
    cog["distilbert_selectivity"],
    cog["distilbert_tau"],
    bert_cog_norm=cog["bert_cog_norm"],
    distil_cog_norm=cog["distilbert_cog_norm"],
    model_a_name=f"{model_a.upper()} (L=12)",
    model_b_name=f"{model_b.upper()} (L=6)",
)

if fig_sel_plotly is not None:
    st.plotly_chart(fig_sel_plotly, use_container_width=True, key="cka_sel_plotly")
else:
    fig_sel_mpl = plot_selectivity_comparison(
        cog["bert_selectivity"],
        cog["bert_tau"],
        cog["distilbert_selectivity"],
        cog["distilbert_tau"],
        bert_cog_norm=cog["bert_cog_norm"],
        distil_cog_norm=cog["distilbert_cog_norm"],
    )
    st.pyplot(fig_sel_mpl, clear_figure=True)

st.caption(
    "Selectivity trajectories ($S = \\text{Acc}_{\\text{upos}} - \\text{Acc}_{\\text{control}}$) "
    "over relative layer depth $\\tau = l / L \\in [0, 1]$. "
    "Dashed vertical lines indicate normalized Center of Gravity ($\\text{CoG}_{\\text{norm}}$)."
)

# -------------------------------------------------------------------------
# 4. Layer Mapping Comparison Table
# -------------------------------------------------------------------------
st.subheader("Layer Mapping Comparison: Rigid 1:2 vs CKA-Best")

try:
    mapping_df = build_layer_mapping_table(
        probing_bert,
        probing_distil,
        cka_df,
        treebank=treebank,
        probe=probe_type,
        init=init_type,
        pooling=pooling_type,
    )
    formatted_table = _format_mapping_table(mapping_df)
    st.dataframe(formatted_table, use_container_width=True, hide_index=True)
except Exception as err:
    st.info(f"Could not build layer mapping table: {err}")

# -------------------------------------------------------------------------
# 5. Static Publication Figures Expander
# -------------------------------------------------------------------------
with st.expander("Static publication figures (matplotlib)", expanded=False):
    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        st.pyplot(fig_mpl, clear_figure=False)
    with col_fig2:
        fig_sel_mpl2 = plot_selectivity_comparison(
            cog["bert_selectivity"],
            cog["bert_tau"],
            cog["distilbert_selectivity"],
            cog["distilbert_tau"],
            bert_cog_norm=cog["bert_cog_norm"],
            distil_cog_norm=cog["distilbert_cog_norm"],
        )
        st.pyplot(fig_sel_mpl2, clear_figure=False)
