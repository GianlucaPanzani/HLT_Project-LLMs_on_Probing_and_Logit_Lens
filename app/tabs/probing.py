"""Tab 2: probing curves and baselines explorer (issue A2)."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.paths import ROOT, probing_result_path
from lib.registry import get_model_spec, get_treebank_spec
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
