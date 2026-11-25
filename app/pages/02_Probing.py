"""Page 02: probing curves and baselines explorer (issue A5, owner Gianluca).

UI body for the probing page. Helpers and cached loaders live in app/tabs/probing.py.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st
from tabs.probing import (
    CONFIG_COLUMNS,
    CONTROL_TASKS,
    INITIALIZATIONS,
    METRICS,
    PROBING_DIR,
    SUMMARY_PATH,
    _load_parquet,
    _plot_curves,
    _select_results,
)

from lib.registry import list_models

st.title("Probing")
st.caption("Run locally with: `PYTHONPATH=src streamlit run app/app.py`")

paths = sorted(PROBING_DIR.glob("*.parquet"))
if not paths:
    st.info(f"No probing results found in {PROBING_DIR}.")
    st.stop()

try:
    results = pd.concat([_load_parquet(path) for path in paths], ignore_index=True)
except FileNotFoundError as err:
    st.info(f"A probing result file is missing: {err}")
    st.stop()
except (OSError, ValueError) as err:
    st.error(f"Could not read probing results: {err}")
    st.stop()

if results.empty:
    st.info("The probing result files contain no rows.")
    st.stop()

required = CONFIG_COLUMNS + ["task", "layer", "L", "seed", "acc"]
missing = sorted(set(required) - set(results.columns))
if missing:
    st.error(f"Missing probing results columns: {missing}.")
    st.stop()

available_models = [model for model in list_models() if model in results["model"].values]
if not available_models:
    st.info("No results for registered models are available.")
    st.stop()

models = st.multiselect(
    "Models",
    available_models,
    default=[model for model in ("bert", "gpt2") if model in available_models] or available_models[:1],
    key="probing_models",
)

col_probe, col_init, col_pooling, col_eval = st.columns(4)
with col_probe:
    probe = st.selectbox(
        "Probe",
        ("linear", "mlp"),
        key="probing_probe",
    )
with col_init:
    init = st.selectbox(
        "Initialization",
        list(INITIALIZATIONS),
        key="probing_init",
    )
with col_pooling:
    pooling = st.selectbox(
        "Pooling",
        ("auto", "first", "mean", "last"),
        key="probing_pooling",
        format_func=lambda value: "Task default" if value == "auto" else value,
        help="Task default uses last pooling for decoder upos_next, first otherwise.",
    )
with col_eval:
    eval_treebank = st.selectbox(
        "Eval treebank",
        ("ewt", "pud_en", "isdt", "pud_it"),
        key="probing_eval_treebank",
    )

col_task, col_x, col_metric, _ = st.columns(4)
with col_task:
    task = st.selectbox(
        "Task",
        list(CONTROL_TASKS),
        key="probing_task",
    )
with col_x:
    x = st.selectbox(
        "X axis",
        ("tau", "layer"),
        key="probing_x",
    )
with col_metric:
    metric = st.selectbox(
        "Metric",
        list(METRICS),
        format_func=METRICS.get,
        key="probing_metric",
    )

show_layer_zero = st.checkbox(
    "Show layer-0 reference lines",
    key="probing_layer_zero",
)
if not models:
    st.info("Select at least one model to display probing curves.")
    st.stop()

selected = _select_results(
    results, models, probe=probe, initializations=INITIALIZATIONS[init],
    pooling=pooling, eval_treebank=eval_treebank, task=task,
)
for model in models:
    for initialization in INITIALIZATIONS[init]:
        if selected.loc[selected["model"].eq(model) & selected["init"].eq(initialization)].empty:
            st.info(
                f"No {task} results for {model} · {initialization} "
                "with the selected probe, pooling and treebank."
            )
if selected.empty:
    st.stop()

try:
    figure = _plot_curves(selected, task, metric, x, show_layer_zero)
except ValueError as err:
    st.error(f"Could not plot probing results: {err}")
    st.stop()

st.plotly_chart(
    figure,
    width="stretch",
    key="probing_curves",
)
st.caption(
    "Layer 0 is the embedding baseline. Solid curves: pretrained; dashed curves: random weights. "
    "Accuracies are averaged over the available paired seeds; negative selectivity is preserved."
)

# CoG summary table (UI body moved from former _render_summary helper)
st.subheader("Center of gravity")
try:
    summary = _load_parquet(SUMMARY_PATH)
except FileNotFoundError:
    st.info("Missing results/summary/cog.parquet. Run PYTHONPATH=src python scripts/11_summarize_probing.py.")
    st.stop()
except (OSError, ValueError) as err:
    st.error(f"Could not read the probing summary: {err}")
    st.stop()

keys = CONFIG_COLUMNS + ["task"]
missing_summary = sorted(set(keys + ["cog_norm"]) - set(summary.columns))
if missing_summary:
    st.error(f"Missing summary columns: {missing_summary}.")
    st.stop()

configurations = selected.loc[selected["task"].eq(task), keys].drop_duplicates()
try:
    merged = configurations.merge(
        summary,
        on=keys,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
except ValueError as err:
    st.error(f"Could not match the probing summary: {err}")
    st.stop()

if merged["_merge"].eq("left_only").any():
    st.info("Some selected configurations have no summary. Run PYTHONPATH=src python scripts/11_summarize_probing.py.")
merged = merged.loc[merged["_merge"].eq("both")]
if merged.empty:
    st.stop()

st.dataframe(
    merged[["model", "init", "pooling", "cog_norm"]],
    column_config={"cog_norm": st.column_config.NumberColumn("CoG_norm", format="%.4f")},
    hide_index=True,
    width="stretch",
)
st.caption("CoG_norm uses layers 1…L and clips negative selectivity to zero. An undefined CoG is shown as empty.")
