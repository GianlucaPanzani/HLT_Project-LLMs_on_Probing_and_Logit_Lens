"""P8 comparison UI, discovered by the A5 sidebar router."""

import streamlit as st
from tabs.probe_comparison import METRICS, MODELS, load_results, prepare_comparison, summarize_comparison

from viz.probing_plots import plot_probe_comparison

seed = 42

st.title("Linear vs MLP")

st.markdown("**BERT and GPT-2** :grey[pretrained · first pooling · current-word UPOS · trained on EWT · no retraining.]")

col1, _, col2, _ = st.columns([4,1,4,1])
with col1:
    eval_treebank = st.selectbox(
        label="Evaluation treebank",
        options=("ewt", "pud_en"),
        key="probe_comparison_eval",
    )
with col2:
    depth_type = st.radio(
        label="Depth axis",
        options=("layer", "tau"),
        horizontal=True,
        key="probe_comparison_axis",
    )

try:
    comparison = prepare_comparison(load_results(), eval_treebank=eval_treebank, seed=seed)
    summary = summarize_comparison(comparison)
except (FileNotFoundError, OSError, ValueError) as err:
    st.error(f"Cannot compare linear and MLP: {err}")
    st.stop()

figure = plot_probe_comparison(comparison, summary, x=depth_type)

st.plotly_chart(
    figure_or_data=figure,
    width="stretch",
    key="probe_comparison_curves",
    config={"toImageButtonOptions": {"filename": f"linear_vs_mlp_{eval_treebank}_seed{seed}", "scale": 2}},
)
st.caption(
    f"Seed {seed}; no seed averaging. Delta = MLP − linear, in percentage points (pp). "
    "Layer 0 is the embedding baseline. CoG uses layers 1…L and clips negative weights only "
    "for CoG; undefined CoG is empty. Peaks include layer 0, with ties resolved to the first layer.",
)

st.subheader("Peaks and center of gravity")
st.dataframe(
    data=summary[
        ["model", "probe", "max_acc", "acc_peak_layer", "max_control_acc", "control_acc_peak_layer",
        "max_selectivity", "selectivity_peak_layer", "cog", "cog_norm"]
    ],
    hide_index=True,
    width="stretch",
)
st.download_button(
    label="Download peaks and CoG (CSV)",
    data=summary.to_csv(index=False),
    file_name=f"linear_vs_mlp_{eval_treebank}_seed{seed}_summary.csv",
    mime="text/csv",
    key="probe_comparison_summary_download",
)

st.subheader("Layer-matched differences")
st.dataframe(
    data=comparison[
        ["model", "layer", "tau", "n_train", "n_eval"]
        + [f"delta_{metric}_pp" for metric in METRICS]
    ],
    hide_index=True,
    width="stretch",
)
st.download_button(
    label="Download matched scores and deltas (CSV)",
    data=comparison.to_csv(index=False),
    file_name=f"linear_vs_mlp_{eval_treebank}_seed{seed}_layers.csv",
    mime="text/csv",
    key="probe_comparison_layers_download",
)

st.markdown(
    "**Reading the comparison.** Compare task gains with control gains at the same layer. "
    "A larger control gain lowers selectivity and is consistent with greater capacity to memorize "
    "arbitrary word-type labels. Selectivity is not an exact decomposition of linguistic knowledge "
    "and memorization. A higher CoG describes a shift in the curve's depth distribution, not more information."
)
if eval_treebank == "ewt" and seed == 42:
    for model in MODELS:
        rows = comparison.loc[comparison["model"].eq(model)]
        lower_layers = int(rows["delta_selectivity_pp"].lt(0).sum())
        best_gain = 100 * (rows["mlp_acc"].max() - rows["linear_acc"].max())
        st.markdown(
            f"{model.upper()}: MLP best accuracy changes by {best_gain:+.2f} pp; "
            f"MLP selectivity is lower at {lower_layers}/{len(rows)} layers. "
            "The best accuracies can occur at different layers."
        )
st.caption(
    "Evidence: docs/results/ws1_probing.md, sections 5–6 (EWT, seed 42). "
    "Results concern these two fitted probes and this control task; they do not establish the absence "
    "of nonlinear linguistic information. No uncertainty or significance estimate is implied by one seed."
)
