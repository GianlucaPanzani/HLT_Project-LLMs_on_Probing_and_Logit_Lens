"""Tab 2: probing curves and baselines explorer (issue A2)."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.paths import ROOT, probing_result_path
from lib.registry import get_model_spec, get_treebank_spec, list_models
from viz.probing_plots import plot_layer_curves

PROBING_DIR = probing_result_path("bert", "ewt", "linear", "pretrained", "first").parent
SUMMARY_PATH = ROOT / "results" / "summary" / "cog.parquet"
CONFIG_COLUMNS = ["model", "train_treebank", "eval_treebank", "probe", "init", "pooling"]
CONTROL_TASKS = {"upos": "control", "upos_next": "control_next"}
METRICS = {"selectivity": "Selectivity", "acc": "Accuracy", "control_acc": "Control accuracy"}
INITIALIZATIONS = {
    "Pretrained + random": ("pretrained", "random"),
    "Pretrained": ("pretrained",),
    "Random": ("random",),
}


@st.cache_data(show_spinner=False, max_entries=128)
def _read_parquet(path: Path, modified_ns: int, size: int) -> pd.DataFrame:
    """Cache a parquet file until its modification time or size changes.

    Parameters
    ----------
    path : Path
        Local result or summary file.
    modified_ns : int
        File modification time in nanoseconds, included in the cache key.
    size : int
        File size in bytes, included in the cache key.

    Returns
    -------
    pd.DataFrame
        File contents. Missing or unreadable files raise an error.
    """
    return pd.read_parquet(path)


def _load_parquet(path: Path) -> pd.DataFrame:
    """Read current file metadata before looking up the cached contents.

    Parameters
    ----------
    path : Path
        Local parquet file.

    Returns
    -------
    pd.DataFrame
        Current file contents.
    """
    metadata = path.stat()
    return _read_parquet(path, metadata.st_mtime_ns, metadata.st_size)


def _select_results(
    results: pd.DataFrame,
    models: list[str],
    *,
    probe: str,
    initializations: tuple[str, ...],
    pooling: str,
    eval_treebank: str,
    task: str,
) -> pd.DataFrame:
    """Select task/control rows with compatible training treebank and pooling.

    Parameters
    ----------
    results : pd.DataFrame
        Concatenated probing results, including all available seeds.
    models : list[str]
        Selected model registry keys.
    probe : str
        Linear or MLP probe.
    initializations : tuple[str, ...]
        Pretrained, random, or both.
    pooling : str
        Explicit pooling or 'auto': last for next-token decoder tasks,
        first otherwise.
    eval_treebank : str
        Evaluation treebank; its registry entry identifies the training set.
    task : str
        'upos' or 'upos_next'.

    Returns
    -------
    pd.DataFrame
        Matching rows, retaining both the task and its control.
    """
    train_treebank = get_treebank_spec(eval_treebank)["train_treebank"]
    selected = results.loc[
        results["model"].isin(models)
        & results["train_treebank"].eq(train_treebank)
        & results["eval_treebank"].eq(eval_treebank)
        & results["probe"].eq(probe)
        & results["init"].isin(initializations)
        & results["task"].isin([task, CONTROL_TASKS[task]])
    ]
    frames = []
    for model in models:
        model_pooling = pooling
        if pooling == "auto":
            kind = get_model_spec(model)["kind"]
            model_pooling = "last" if task == "upos_next" and kind == "decoder" else "first"
        frames.append(selected.loc[selected["model"].eq(model) & selected["pooling"].eq(model_pooling)])
    return pd.concat(frames, ignore_index=True)


def _plot_curves(
    results: pd.DataFrame,
    task: str,
    metric: str,
    x: str,
    show_layer_zero: bool,
) -> go.Figure:
    """Overlay configurations using the shared plotting function.

    Parameters
    ----------
    results : pd.DataFrame
        Nonempty results filtered to one configuration per model and init.
    task : str
        'upos' or 'upos_next'.
    metric : str
        Metric accepted by plot_layer_curves.
    x : str
        'layer' or 'tau'.
    show_layer_zero : bool
        Draw horizontal references at each curve's layer-0 value.

    Returns
    -------
    go.Figure
        Interactive curves, with stable model colors and dashed random runs.
        Invalid or incomplete task/control pairs raise ValueError.
    """
    figure = go.Figure()
    for (model, init), rows in results.groupby(["model", "init"], sort=False):
        curves = plot_layer_curves(rows, [model], task, metric, x=x)
        if not figure.data:
            figure.update_layout(curves.layout)
        trace = curves.data[0]
        trace.name = f"{model} · {init}"
        trace.legendgroup = trace.name
        trace.line.dash = "solid" if init == "pretrained" else "dash"
        trace.hovertemplate = (
            f"{trace.name}<br>{x}=%{{x}}<br>{METRICS[metric]}=%{{y:.4f}}<extra></extra>"
        )
        figure.add_trace(trace)
        if show_layer_zero:
            figure.add_trace(go.Scatter(
                x=[trace.x[0], trace.x[-1]],
                y=[trace.y[0], trace.y[0]],
                mode="lines",
                name=f"{trace.name} · layer 0",
                legendgroup=trace.legendgroup,
                line={
                    "color": trace.line.color,
                    "dash": "dot" if init == "pretrained" else "dashdot",
                    "width": 1,
                },
                hovertemplate=f"{trace.name} · layer 0<br>value=%{{y:.4f}}<extra></extra>",
            ))
    figure.update_layout(height=500, legend_title_text="Model · initialization")
    if x == "tau":
        figure.update_xaxes(range=[0, 1])
    if metric != "selectivity":
        figure.update_yaxes(range=[0, 1])
    return figure


def _render_summary(results: pd.DataFrame, task: str) -> None:
    """Display saved CoG values for exactly the selected configurations.

    Parameters
    ----------
    results : pd.DataFrame
        Results used in the displayed curves, without seed filtering.
    task : str
        Selected UPOS task pair.

    Returns
    -------
    None
        Renders a table or an informative message when the summary is missing.
    """
    st.subheader("Center of gravity")
    try:
        summary = _load_parquet(SUMMARY_PATH)
    except FileNotFoundError:
        st.info("Missing results/summary/cog.parquet. Run PYTHONPATH=src python scripts/11_summarize_probing.py.")
        return
    except (OSError, ValueError) as err:
        st.error(f"Could not read the probing summary: {err}")
        return

    keys = CONFIG_COLUMNS + ["task"]
    missing = sorted(set(keys + ["cog_norm"]) - set(summary.columns))
    if missing:
        st.error(f"Missing summary columns: {missing}.")
        return
    configurations = results.loc[results["task"].eq(task), keys].drop_duplicates()
    try:
        selected = configurations.merge(summary, on=keys, how="left", validate="one_to_one", indicator=True)
    except ValueError as err:
        st.error(f"Could not match the probing summary: {err}")
        return
    if selected["_merge"].eq("left_only").any():
        st.info("Some selected configurations have no summary. Run PYTHONPATH=src python scripts/11_summarize_probing.py.")
    selected = selected.loc[selected["_merge"].eq("both")]
    if selected.empty:
        return
    st.dataframe(
        selected[["model", "init", "pooling", "cog_norm"]],
        column_config={"cog_norm": st.column_config.NumberColumn("CoG_norm", format="%.4f")},
        hide_index=True,
        width="stretch",
    )
    st.caption("CoG_norm uses layers 1…L and clips negative selectivity to zero. An undefined CoG is shown as empty.")


def render() -> None:
    """Render the probing explorer called by the A1 Streamlit shell.

    Returns
    -------
    None
        Displays controls, curves and CoG values without loading any models.
    """
    paths = sorted(PROBING_DIR.glob("*.parquet"))
    if not paths:
        st.info(f"No probing results found in {PROBING_DIR}.")
        return
    
    try:
        results = pd.concat([_load_parquet(path) for path in paths], ignore_index=True)
    except FileNotFoundError as err:
        st.info(f"A probing result file is missing: {err}")
        return
    except (OSError, ValueError) as err:
        st.error(f"Could not read probing results: {err}")
        return
    
    if results.empty:
        st.info("The probing result files contain no rows.")
        return
    required = CONFIG_COLUMNS + ["task", "layer", "L", "seed", "acc"]
    missing = sorted(set(required) - set(results.columns))
    if missing:
        st.error(f"Missing probing results columns: {missing}.")
        return

    available_models = [model for model in list_models() if model in results["model"].values]
    if not available_models:
        st.info("No results for registered models are available.")
        return
    
    models = st.multiselect(
        "Models", available_models,
        default=[model for model in ("bert", "gpt2") if model in available_models] or available_models[:1],
        key="probing_models",
    )

    col_probe, col_init, col_pooling, col_eval = st.columns(4)
    with col_probe:
        probe = st.selectbox(
            "Probe",
            ("linear", "mlp"),
            key="probing_probe"
        )
    with col_init:
        init = st.selectbox(
            "Initialization",
            list(INITIALIZATIONS),
            key="probing_init"
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
            key="probing_task"
        )
    with col_x:
        x = st.selectbox(
            "X axis",
            ("tau", "layer"),
            key="probing_x"
        )
    with col_metric:
        metric = st.selectbox(
            "Metric",
            list(METRICS),
            format_func=METRICS.get,
            key="probing_metric"
        )

    show_layer_zero = st.checkbox(
        "Show layer-0 reference lines",
        key="probing_layer_zero"
    )
    if not models:
        st.info("Select at least one model to display probing curves.")
        return

    selected = _select_results(
        results, models, probe=probe, initializations=INITIALIZATIONS[init],
        pooling=pooling, eval_treebank=eval_treebank, task=task,
    )
    for model in models:
        for initialization in INITIALIZATIONS[init]:
            if selected.loc[selected["model"].eq(model) & selected["init"].eq(initialization)].empty:
                st.info(f"No {task} results for {model} · {initialization} with the selected probe, pooling and treebank.")
    if selected.empty:
        return
    
    try:
        figure = _plot_curves(selected, task, metric, x, show_layer_zero)
    except ValueError as err:
        st.error(f"Could not plot probing results: {err}")
        return
    st.plotly_chart(figure, width="stretch", key="probing_curves")
    st.caption(
        "Layer 0 is the embedding baseline. Solid curves: pretrained; dashed curves: random weights. "
        "Accuracies are averaged over the available paired seeds; negative selectivity is preserved."
    )
    
    _render_summary(selected, task)
    return


if __name__ == "__main__":
    st.set_page_config(page_title="Probing explorer", layout="wide")
    render()
