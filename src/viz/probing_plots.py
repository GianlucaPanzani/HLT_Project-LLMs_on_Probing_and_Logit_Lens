"""Reusable probing figures built from result frames, without file access."""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative

from lib.metrics import relative_depth, selectivity
from lib.registry import list_models


def plot_layer_curves(
    results: pd.DataFrame,
    model_names: list[str],
    task: str,
    metric: str,
    *,
    summary: pd.DataFrame | None = None,
    x: str = "tau",
) -> go.Figure:
    """Plot one task metric across layers for each requested model.

    Parameters
    ----------
    results : pd.DataFrame
        Contracted probing results, filtered to one training/evaluation
        treebank, probe, initialization and pooling per model. Pooling may
        differ between models. Task and control must contain matching rows
        and all layers 0..L for each seed. Paired accuracies are averaged
        over the supplied seeds; filter to seed 42 for the main analysis.
    model_names : list[str]
        Unique registry keys in the desired legend order.
    task : str
        'upos' or 'upos_next'.
    metric : str
        'acc', 'control_acc' or 'selectivity'. Negative selectivity is preserved.
    summary : pd.DataFrame or None
        Summary produced from the same results and seeds. For selectivity,
        each configuration must have exactly one matching summary row.
        Its CoG is drawn as a vertical line; undefined (NaN) CoG is omitted.
        Ignored for accuracy plots.
    x : str
        'layer' for block indices or 'tau' for normalized depth.

    Returns
    -------
    go.Figure
        Plotly figure with one curve per model, ready for display or export.
        No files are read or written and no models are loaded.
    """
    if results.empty:
        raise ValueError("Probing results must not be empty.")
    registered_models = list_models()
    if not model_names or any(model not in registered_models for model in model_names):
        raise ValueError(f"model_names must contain registry keys from {registered_models}.")
    if len(set(model_names)) != len(model_names):
        raise ValueError("model_names must not contain duplicates.")
    if task not in ("upos", "upos_next"):
        raise ValueError("task must be 'upos' or 'upos_next'.")
    if metric not in ("acc", "control_acc", "selectivity"):
        raise ValueError("metric must be 'acc', 'control_acc' or 'selectivity'.")
    if x not in ("layer", "tau"):
        raise ValueError("x must be 'layer' or 'tau'.")

    configuration_columns = ["model", "train_treebank", "eval_treebank", "probe", "init", "pooling"]
    required = configuration_columns + ["task", "layer", "L", "seed", "acc"]
    missing = sorted(set(required) - set(results.columns))
    if missing:
        raise ValueError(f"Missing probing results columns: {missing}.")
    if summary is not None and metric == "selectivity":
        required_summary = configuration_columns + ["task", "L", "cog", "cog_norm"]
        missing = sorted(set(required_summary) - set(summary.columns))
        if missing:
            raise ValueError(f"Missing summary columns: {missing}.")

    control_tasks = {
        "upos": "control",
        "upos_next": "control_next",
    }
    control = control_tasks[task]
    figure = go.Figure()

    for model in model_names:
        rows = results.loc[(results["model"] == model) & results["task"].isin([task, control])]
        if rows.empty:
            raise ValueError(f"No {task}/{control} results for {model}.")
        if rows[configuration_columns].isna().any().any():
            raise ValueError(f"Missing configuration values for {model}.")
        if len(rows[configuration_columns].drop_duplicates()) != 1:
            raise ValueError(f"Filter results to one treebank/probe/init/pooling configuration for {model}.")
        if rows["L"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent L for {model}.")
        num_layers = rows["L"].iloc[0]
        relative_depth(rows["layer"].to_numpy(), num_layers)
        if rows["seed"].isna().any() or rows.duplicated(["task", "layer", "seed"]).any():
            raise ValueError(f"Missing seeds or duplicate task/layer/seed rows for {model}.")
        if not rows["acc"].between(0, 1).all():
            raise ValueError(f"Accuracies must be finite and between 0 and 1 for {model}.")

        pair_columns = ["layer", "seed", "acc"]
        paired = rows.loc[rows["task"] == task, pair_columns].merge(
            rows.loc[rows["task"] == control, pair_columns],
            on=["layer", "seed"],
            how="outer",
            suffixes=("_task", "_control"),
            indicator=True,
            validate="one_to_one",
        )
        if not paired["_merge"].eq("both").all():
            raise ValueError(f"Unmatched {task}/{control} layer/seed rows for {model}.")
        for seed, seed_rows in paired.groupby("seed"):
            if sorted(seed_rows["layer"].tolist()) != list(range(num_layers + 1)):
                raise ValueError(f"Expected all layers 0..{num_layers} for {model}, seed {seed}.")

        means = paired.groupby("layer", sort=True)[["acc_task", "acc_control"]].mean()
        values = {
            "acc": means["acc_task"].to_numpy(),
            "control_acc": means["acc_control"].to_numpy(),
            "selectivity": selectivity(means["acc_task"].to_numpy(), means["acc_control"].to_numpy()),
        }
        layers = means.index.to_numpy()
        positions = relative_depth(layers, num_layers) if x == "tau" else layers
        color = qualitative.Plotly[registered_models.index(model) % len(qualitative.Plotly)]
        figure.add_trace(go.Scatter(
            x=positions,
            y=values[metric],
            name=model,
            mode="lines+markers",
            line={"color": color},
            hovertemplate=f"{model}<br>{x}=%{{x}}<br>{metric}=%{{y:.4f}}<extra></extra>",
        ))

        if summary is not None and metric == "selectivity":
            matching = summary.loc[summary["task"] == task]
            for column in configuration_columns:
                matching = matching.loc[matching[column] == rows[column].iloc[0]]
            if len(matching) != 1 or matching["L"].iloc[0] != num_layers:
                raise ValueError(f"Expected exactly one matching summary row with L={num_layers} for {model}.")
            cog_column = "cog_norm" if x == "tau" else "cog"
            cog = matching[cog_column].iloc[0]
            if pd.isna(cog):
                continue
            if not np.isfinite(cog):
                raise ValueError(f"Summary {cog_column} must be finite or NaN for {model}.")
            figure.add_vline(
                x=float(cog),
                line_color=color,
                line_dash="dash",
                annotation_text=f"{model} {cog_column}={cog:.3f}",
                annotation_position="top left" if model_names.index(model) % 2 == 0 else "bottom right",
            )

    metric_labels = {
        "acc": "Accuracy",
        "control_acc": "Control accuracy",
        "selectivity": "Selectivity (task − control)",
    }
    figure.update_layout(
        template="plotly_white",
        title=f"{task}: {metric_labels[metric]}",
        xaxis_title="Relative depth (τ)" if x == "tau" else "Layer",
        yaxis_title=metric_labels[metric],
        legend_title_text="Model",
        hovermode="x unified",
    )
    if x == "layer":
        figure.update_xaxes(dtick=1)
    return figure
