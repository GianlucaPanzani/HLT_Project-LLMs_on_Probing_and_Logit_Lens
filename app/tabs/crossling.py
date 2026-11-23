"""Tab 4: cross-linguistic dashboard (issue A4)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from lib.paths import ROOT, probing_result_path
from viz.crossling_plots import plot_delta_cog, plot_selectivity_pairs
from ws3_crossling.delta_cog import PAIRS, ModelPair, delta_cog, selectivity_curves

PAIR_IDS: tuple[str, ...] = tuple(p.pair for p in PAIRS)
EVAL_SETS: dict[str, str] = {"own_test": "Own test (EWT / ISDT)", "pud": "PUD (EN / IT)"}
METRICS: dict[str, str] = {
    "selectivity": "Selectivity",
    "acc_task": "UPOS accuracy",
    "acc_control": "Control accuracy",
}
PROBE_DEFAULTS = {"probe": "linear", "init": "pretrained", "pooling": "first"}


def _pair_spec(pair_id: str) -> ModelPair:
    for spec in PAIRS:
        if spec.pair == pair_id:
            return spec
    raise KeyError(pair_id)


def _eval_treebanks(spec: ModelPair, eval_set: str) -> tuple[str, str]:
    if eval_set == "own_test":
        return spec.en_eval_own, spec.it_eval_own
    if eval_set == "pud":
        return spec.en_eval_pud, spec.it_eval_pud
    raise ValueError(f"Unknown eval_set: {eval_set}")


def _filter_curves(
    curves: pd.DataFrame,
    *,
    model: str,
    train_treebank: str,
    eval_treebank: str,
    probe: str,
    init: str,
    pooling: str,
) -> pd.DataFrame:
    mask = (
        (curves["model"] == model)
        & (curves["train_treebank"] == train_treebank)
        & (curves["eval_treebank"] == eval_treebank)
        & (curves["probe"] == probe)
        & (curves["init"] == init)
        & (curves["pooling"] == pooling)
    )
    return curves.loc[mask].sort_values("layer")


def _accuracy_curve(
    results: pd.DataFrame,
    *,
    model: str,
    train_treebank: str,
    eval_treebank: str,
    task: str,
    probe: str,
    init: str,
    pooling: str,
) -> pd.DataFrame:
    mask = (
        (results["model"] == model)
        & (results["train_treebank"] == train_treebank)
        & (results["eval_treebank"] == eval_treebank)
        & (results["probe"] == probe)
        & (results["init"] == init)
        & (results["pooling"] == pooling)
        & (results["task"] == task)
    )
    subset = results.loc[mask, ["layer", "L", "tau", "acc"]]
    if subset.empty:
        return pd.DataFrame()
    return subset.groupby("layer", as_index=False)[["tau", "acc", "L"]].mean(numeric_only=True)


@st.cache_data(show_spinner="Loading probing results…")
def _load_probing_results() -> pd.DataFrame | None:
    probing_dir = probing_result_path("gpt2", "ewt", "linear", "pretrained", "first").parent
    if not probing_dir.is_dir():
        return None
    paths = sorted(probing_dir.glob("*.parquet"))
    if not paths:
        return None
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


@st.cache_data(show_spinner="Building selectivity curves…")
def _load_curves(results: pd.DataFrame) -> pd.DataFrame:
    return selectivity_curves(results)


@st.cache_data(show_spinner="Loading Delta CoG summary…")
def _load_delta_table() -> pd.DataFrame | None:
    delta_path = ROOT / "results" / "summary" / "delta_cog.parquet"
    if delta_path.is_file():
        return pd.read_parquet(delta_path)
    cog_path = ROOT / "results" / "summary" / "cog.parquet"
    if not cog_path.is_file():
        return None
    summary = pd.read_parquet(cog_path)
    try:
        return delta_cog(summary, probe=PROBE_DEFAULTS["probe"])
    except ValueError:
        return None


def _series_for_metric(
    curves: pd.DataFrame,
    results: pd.DataFrame,
    metric: str,
    *,
    model: str,
    train_treebank: str,
    eval_treebank: str,
) -> pd.DataFrame:
    probe = PROBE_DEFAULTS["probe"]
    init = PROBE_DEFAULTS["init"]
    pooling = PROBE_DEFAULTS["pooling"]
    if metric == "selectivity":
        return _filter_curves(
            curves,
            model=model,
            train_treebank=train_treebank,
            eval_treebank=eval_treebank,
            probe=probe,
            init=init,
            pooling=pooling,
        )
    task = "upos" if metric == "acc_task" else "control"
    return _accuracy_curve(
        results,
        model=model,
        train_treebank=train_treebank,
        eval_treebank=eval_treebank,
        task=task,
        probe=probe,
        init=init,
        pooling=pooling,
    )


def _plot_en_it(
    en_df: pd.DataFrame,
    it_df: pd.DataFrame,
    metric: str,
    *,
    title: str,
) -> go.Figure:
    y_col = "selectivity" if metric == "selectivity" else "acc"
    fig = go.Figure()
    if not en_df.empty:
        fig.add_trace(
            go.Scatter(
                x=en_df["tau"],
                y=en_df[y_col],
                mode="lines",
                name="EN",
                line={"color": "#2563eb", "width": 2},
            )
        )
        if metric == "selectivity":
            cog = float(en_df["cog_norm"].iloc[0])
            fig.add_vline(x=cog, line_dash="dash", line_color="#2563eb", opacity=0.6)
    if not it_df.empty:
        fig.add_trace(
            go.Scatter(
                x=it_df["tau"],
                y=it_df[y_col],
                mode="lines",
                name="IT",
                line={"color": "#ea580c", "width": 2},
            )
        )
        if metric == "selectivity":
            cog = float(it_df["cog_norm"].iloc[0])
            fig.add_vline(x=cog, line_dash="dash", line_color="#ea580c", opacity=0.6)
    fig.update_layout(
        title=title,
        xaxis_title="τ (relative depth)",
        yaxis_title=METRICS[metric],
        xaxis={"range": [0, 1]},
        height=420,
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "x": 0},
    )
    if metric == "selectivity":
        fig.add_annotation(
            text="Dashed lines: CoG_norm",
            xref="paper",
            yref="paper",
            x=0,
            y=-0.18,
            showarrow=False,
            font={"size": 11},
        )
    return fig


def _render_matplotlib_overview(curves: pd.DataFrame, delta: pd.DataFrame) -> None:
    """Static overview figures (reuse viz.crossling_plots)."""
    with tempfile.TemporaryDirectory(prefix="crossling-demo-") as tmp:
        tmp_path = Path(tmp)
        pairs_path = tmp_path / "selectivity_pairs.png"
        delta_path = tmp_path / "delta_cog.png"
        try:
            plot_selectivity_pairs(curves, delta, pairs_path, **PROBE_DEFAULTS)
            plot_delta_cog(delta, delta_path)
        except (ValueError, KeyError) as err:
            st.warning(f"Could not build overview figures: {err}")
            return
        col_a, col_b = st.columns(2)
        with col_a:
            st.image(str(pairs_path), caption="Selectivity vs τ (all pairs)", use_container_width=True)
        with col_b:
            st.image(str(delta_path), caption="Delta CoG by pair", use_container_width=True)


def _format_delta_table(delta: pd.DataFrame) -> pd.DataFrame:
    if "delta_cog_mean" in delta.columns:
        cols = [
            "pair",
            "kind",
            "eval_set",
            "cog_norm_en_mean",
            "cog_norm_it_mean",
            "delta_cog_mean",
            "delta_cog_std",
            "n_seeds",
        ]
        view = delta[[c for c in cols if c in delta.columns]].copy()
        for col in ("cog_norm_en_mean", "cog_norm_it_mean", "delta_cog_mean", "delta_cog_std"):
            if col in view.columns:
                view[col] = view[col].map(lambda x: round(float(x), 4) if pd.notna(x) else None)
        return view
    cols = ["pair", "kind", "eval_set", "cog_norm_en", "cog_norm_it", "delta_cog"]
    view = delta[[c for c in cols if c in delta.columns]].copy()
    for col in ("cog_norm_en", "cog_norm_it", "delta_cog"):
        if col in view.columns:
            view[col] = view[col].map(lambda x: round(float(x), 4))
    return view


def render() -> None:
    """Cross-linguistic probing dashboard."""
    results = _load_probing_results()
    if results is None:
        st.info(f"No probing results found in {probing_result_path('gpt2', 'ewt', 'linear', 'pretrained', 'first').parent}")
        return

    delta = _load_delta_table()
    if delta is None:
        st.info("Missing Delta CoG summary. Run scripts/11_summarize_probing.py and scripts/40_crossling_report.py.")
        return

    try:
        curves = _load_curves(results)
    except ValueError as err:
        st.error(str(err))
        return

    col_pair, col_eval, col_metric = st.columns(3)
    with col_pair:
        pair_id = st.selectbox("Model pair", PAIR_IDS)
    with col_eval:
        eval_set = st.selectbox("Eval set", list(EVAL_SETS.keys()), format_func=lambda k: EVAL_SETS[k])
    with col_metric:
        metric = st.selectbox("Metric", list(METRICS.keys()), format_func=lambda k: METRICS[k])

    spec = _pair_spec(pair_id)
    en_eval, it_eval = _eval_treebanks(spec, eval_set)
    en_df = _series_for_metric(
        curves, results, metric, model=spec.en_model, train_treebank=spec.en_train_treebank, eval_treebank=en_eval
    )
    it_df = _series_for_metric(
        curves, results, metric, model=spec.it_model, train_treebank=spec.it_train_treebank, eval_treebank=it_eval
    )

    if en_df.empty and it_df.empty:
        st.warning("No curve data for this pair and eval set.")
    else:
        title = f"{pair_id} · {EVAL_SETS[eval_set]} · {METRICS[metric]}"
        st.plotly_chart(_plot_en_it(en_df, it_df, metric, title=title), use_container_width=True)

    st.subheader("Delta CoG")
    delta_view = _format_delta_table(delta)
    pair_delta = delta_view.loc[delta_view["pair"] == pair_id] if "pair" in delta_view.columns else delta_view
    st.dataframe(pair_delta if not pair_delta.empty else delta_view, use_container_width=True, hide_index=True)

    with st.expander("Overview figures (matplotlib)", expanded=False):
        _render_matplotlib_overview(curves, delta)