"""Generate the four WS1 probing figures using seed 42.

Run from the repository root:
    PYTHONPATH=src python scripts/12_plot_probing.py
"""

import argparse

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from lib.metrics import summarize_probing
from lib.paths import ROOT
from lib.registry import get_model_spec, get_treebank_spec, list_models
from viz.probing_plots import plot_layer_curves

SEED = 42
METRICS = ("acc", "control_acc", "selectivity")
METRIC_LABELS = ("Accuracy", "Control accuracy", "Selectivity")


def load_results(
    model_names: list[str],
    train_treebank: str,
    eval_treebank: str,
) -> pd.DataFrame:
    """Load local probing results for the requested experiment and seed.

    Parameters
    ----------
    model_names : list[str]
        Model registry keys.
    train_treebank : str
        Training treebank registry key.
    eval_treebank : str
        Evaluation treebank registry key.

    Returns
    -------
    pd.DataFrame
        Matching probing results, restricted to seed 42.
    """
    paths = sorted((ROOT / "results" / "probing").glob("*.parquet"))
    if not paths:
        raise FileNotFoundError("No parquet files found in results/probing/.")

    results = pd.concat(
        [pd.read_parquet(path) for path in paths],
        ignore_index=True,
    )
    selected = results.loc[
        results["model"].isin(model_names)
        & results["train_treebank"].eq(train_treebank)
        & results["eval_treebank"].eq(eval_treebank)
        & results["seed"].eq(SEED)
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No results for models={model_names}, "
            f"train={train_treebank}, eval={eval_treebank}, seed={SEED}."
        )
    return selected


def select_results(
    results: pd.DataFrame,
    model_names: list[str],
    task: str,
    probe: str,
    init: str,
) -> pd.DataFrame:
    """Select one configuration per model with task-appropriate pooling.

    Parameters
    ----------
    results : pd.DataFrame
        Results already filtered by treebanks and seed.
    model_names : list[str]
        Models to compare.
    task : str
        'upos' or 'upos_next'.
    probe : str
        'linear' or 'mlp'.
    init : str
        'pretrained' or 'random'.

    Returns
    -------
    pd.DataFrame
        Selected task/control pairs for each model.
    """
    control = {"upos": "control", "upos_next": "control_next"}[task]
    frames = []

    for model in model_names:
        kind = get_model_spec(model)["kind"]
        pooling = "last" if task == "upos_next" and kind == "decoder" else "first"

        selected = results.loc[
            results["model"].eq(model)
            & results["probe"].eq(probe)
            & results["init"].eq(init)
            & results["pooling"].eq(pooling)
            & results["task"].isin([task, control])
        ]

        if selected.empty:
            raise ValueError(
                f"Missing results: model={model}, task={task}, "
                f"probe={probe}, init={init}, pooling={pooling}."
            )
        frames.append(selected)

    return pd.concat(frames, ignore_index=True)


def add_panel(
    figure: go.Figure,
    curves: go.Figure,
    column: int,
    label: str,
    dash: str,
) -> None:
    """Copy curves and CoG markers into a subplot, stacking their labels.

    Parameters
    ----------
    figure : go.Figure
        Destination figure with one row of subplots.
    curves : go.Figure
        Figure returned by plot_layer_curves.
    column : int
        Destination column, starting at 1.
    label : str
        Configuration label appended to model names.
    dash : str
        Plotly line style.

    Returns
    -------
    None
        Updates the destination figure in place.
    """
    for trace in curves.data:
        copied = go.Scatter(trace.to_plotly_json())
        copied.name = f"{trace.name} · {label}"
        copied.legendgroup = copied.name
        copied.showlegend = column == 1
        copied.line.dash = dash
        copied.hovertemplate = (
            f"{copied.name}<br>τ=%{{x:.3f}}"
            "<br>value=%{y:.4f}<extra></extra>"
        )
        figure.add_trace(copied, row=1, col=column)

    # Keep the CoG lines at their data coordinates, but stack the labels
    # in the upper-left corner so nearby CoGs cannot overlap their text.
    axis_suffix = str(column) if column > 1 else ""
    xref = f"x{axis_suffix}"
    yref = f"y{axis_suffix}"
    label_index = sum(shape.xref == xref for shape in figure.layout.shapes)

    # plot_layer_curves adds one annotation for each CoG line.
    for shape, annotation in zip(
        curves.layout.shapes,
        curves.layout.annotations,
        strict=True,
    ):
        figure.add_vline(
            x=shape.x0,
            line_color=shape.line.color,
            line_dash="dot" if dash == "solid" else "dash",
            row=1,
            col=column,
        )
        figure.add_annotation(
            x=0.02,
            y=1,
            xref=f"{xref} domain",
            yref=f"{yref} domain",
            xanchor="left",
            yanchor="top",
            yshift=-20 * label_index,
            text=f"{annotation.text} · {label}",
            showarrow=False,
            font={"color": shape.line.color, "size": 11},
            bgcolor="rgba(255, 255, 255, 0.9)",
        )
        label_index += 1


def build_metric_figure(
    results: pd.DataFrame,
    model_names: list[str],
    task: str,
    probes: tuple[str, ...] = ("linear",),
) -> go.Figure:
    """Compare pretrained accuracy, control accuracy and selectivity.

    Parameters
    ----------
    results : pd.DataFrame
        Results filtered by treebanks and seed.
    model_names : list[str]
        Models to compare.
    task : str
        'upos' or 'upos_next'.
    probes : tuple[str, ...]
        Probe architectures to include.

    Returns
    -------
    go.Figure
        Three panels with model curves and selectivity CoG markers.
    """
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=METRIC_LABELS,
        horizontal_spacing=0.07,
    )

    for probe in probes:
        selected = select_results(
            results, model_names, task, probe, "pretrained"
        )
        summary = summarize_probing(selected)
        dash = "solid" if probe == "linear" else "dash"

        for column, metric in enumerate(METRICS, start=1):
            curves = plot_layer_curves(
                selected,
                model_names,
                task,
                metric,
                summary=summary,
                x="tau",
            )
            add_panel(figure, curves, column, probe, dash)
            figure.update_yaxes(
                title_text=METRIC_LABELS[column - 1],
                row=1,
                col=column,
            )

    figure.update_xaxes(title_text="Relative depth (τ)", range=[0, 1])
    figure.update_layout(
        template="plotly_white",
        title=f"{task} · pretrained · {' vs '.join(probes)}",
        hovermode="x unified",
        width=1500,
        height=550,
        margin={"t": 110, "b": 110},
        legend={"orientation": "h", "y": -0.25, "x": 0},
    )
    return figure


def build_baseline_figure(
    results: pd.DataFrame,
    model_names: list[str],
) -> go.Figure:
    """Compare pretrained and random UPOS accuracy with layer-0 references.

    Parameters
    ----------
    results : pd.DataFrame
        Results filtered by treebanks and seed.
    model_names : list[str]
        Models to compare.

    Returns
    -------
    go.Figure
        One accuracy panel per model, using linear probes and first pooling.
    """
    figure = make_subplots(
        rows=1,
        cols=len(model_names),
        subplot_titles=model_names,
        shared_yaxes=True,
    )

    for column, model in enumerate(model_names, start=1):
        for init, dash in (("pretrained", "solid"), ("random", "dash")):
            selected = select_results(
                results, [model], "upos", "linear", init
            )
            curves = plot_layer_curves(
                selected, [model], "upos", "acc", x="tau"
            )

            trace = go.Scatter(curves.data[0].to_plotly_json())
            trace.name = f"{model} · {init}"
            trace.line.dash = dash
            trace.hovertemplate = (
                f"{trace.name}<br>τ=%{{x:.3f}}"
                "<br>accuracy=%{y:.4f}<extra></extra>"
            )
            figure.add_trace(trace, row=1, col=column)

            # Curves are ordered by layer; the first value is layer 0.
            layer_zero = float(trace.y[0])
            figure.add_trace(
                go.Scatter(
                    x=[0, 1],
                    y=[layer_zero, layer_zero],
                    mode="lines",
                    name=f"{model} · {init} · layer 0",
                    line={
                        "color": trace.line.color,
                        "dash": "dot" if init == "pretrained" else "dashdot",
                        "width": 1,
                    },
                    hovertemplate=(
                        f"{model} · {init} · layer 0"
                        "<br>accuracy=%{y:.4f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

    figure.update_xaxes(title_text="Relative depth (τ)", range=[0, 1])
    figure.update_yaxes(title_text="UPOS accuracy", range=[0, 1])
    figure.update_layout(
        template="plotly_white",
        title="UPOS baselines · linear · first pooling",
        hovermode="x unified",
        width=1500,
        height=650,
        margin={"b": 160},
        legend={"orientation": "h", "y": -0.25, "x": 0},
    )
    return figure


def main() -> None:
    """Read results, build the four figures and export PNG and HTML."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list_models(),
        default=["bert", "gpt2"],
    )
    parser.add_argument("--train-treebank", default="ewt")
    parser.add_argument("--eval-treebank", default="ewt")
    args = parser.parse_args()

    if len(set(args.models)) != len(args.models):
        parser.error("--models must not contain duplicates.")

    train_spec = get_treebank_spec(args.train_treebank)
    eval_spec = get_treebank_spec(args.eval_treebank)
    if train_spec["train_treebank"] != args.train_treebank:
        parser.error("--train-treebank must identify a training treebank.")
    if eval_spec["train_treebank"] != args.train_treebank:
        parser.error("Evaluation and training treebanks are not compatible.")

    results = load_results(
        args.models,
        args.train_treebank,
        args.eval_treebank,
    )

    # Build every figure before starting the export, so missing data fail early.
    figures = {
        "upos_curves": build_metric_figure(
            results, args.models, "upos"
        ),
        "upos_next_curves": build_metric_figure(
            results, args.models, "upos_next"
        ),
        "upos_baselines": build_baseline_figure(
            results, args.models
        ),
        "upos_linear_vs_mlp": build_metric_figure(
            results, args.models, "upos", probes=("linear", "mlp")
        ),
    }

    output_dir = ROOT / "results" / "figures" / "ws1"
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = (
        f"{'_vs_'.join(args.models)}"
        f"__{args.train_treebank}__{args.eval_treebank}__seed{SEED}"
    )

    for name, figure in figures.items():
        figure.update_layout(
            title_text=(
                f"{figure.layout.title.text}"
                f"<br><sup>Train: {args.train_treebank} · "
                f"evaluation: {args.eval_treebank} test · seed {SEED}</sup>"
            )
        )

        html_path = output_dir / f"{prefix}__{name}.html"
        png_path = output_dir / f"{prefix}__{name}.png"

        figure.write_html(html_path, include_plotlyjs=True)
        print(f"Wrote {html_path}")

        figure.write_image(png_path, scale=2)
        print(f"Wrote {png_path}")


if __name__ == "__main__":
    main()
