"""Visualisation functions for the CKA and compression workstream.

Provides heatmaps for the CKA matrix and selectivity comparison plots
for the 1:2 layer mapping analysis (C3, C4).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    import matplotlib.pyplot as plt
    matplotlib.use("Agg")
except ImportError:
    matplotlib = None
    plt = None

try:
    import plotly.graph_objects as go
except ImportError:
    go = None

# Standard figure output directory
_FIG_DIR = Path(__file__).resolve().parents[2] / "results" / "figures" / "ws2"


def plot_cka_heatmap(
    df: pd.DataFrame,
    *,
    model_a_name: str = "BERT",
    model_b_name: str = "DistilBERT",
    save_prefix: str | Path | None = None,
    mark_diagonal: bool = True,
) -> tuple[plt.Figure, any]:
    """Generate a CKA heatmap (static + interactive).

    Parameters
    ----------
    df : pd.DataFrame
        Long-format CKA results with columns ``layer_a``, ``layer_b``,
        ``cka`` (CONTRACTS.md §8).
    model_a_name : str
        Display name for the first model (Y-axis).
    model_b_name : str
        Display name for the second model (X-axis).
    save_prefix : str or Path or None
        If given, save ``{prefix}_cka_heatmap.png`` and
        ``{prefix}_cka_heatmap.html`` (if plotly is installed).
    mark_diagonal : bool
        If ``True``, highlight the 1:2 mapping diagonal cells.

    Returns
    -------
    tuple[matplotlib.figure.Figure, Any]
        The matplotlib figure and Plotly figure (or None if plotly is not installed).
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_cka_heatmap, but is not installed.")

    pivot = df.pivot(index="layer_a", columns="layer_b", values="cka")
    layers_a = sorted(pivot.index)
    layers_b = sorted(pivot.columns)
    matrix = pivot.loc[layers_a, layers_b].to_numpy()

    # ---- Matplotlib static heatmap ------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 10))
    im = ax.imshow(matrix, cmap="viridis", aspect="auto", vmin=0, vmax=1, origin="lower")

    # Annotate cells
    for i, la in enumerate(layers_a):
        for j, lb in enumerate(layers_b):
            val = matrix[i, j]
            colour = "white" if val < 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=7, color=colour)

    # Mark 1:2 diagonal
    if mark_diagonal:
        for lb in layers_b:
            la_match = 2 * lb
            if la_match in layers_a:
                i = layers_a.index(la_match)
                j = layers_b.index(lb)
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, linewidth=2.5, edgecolor="red", facecolor="none")
                ax.add_patch(rect)

    ax.set_xticks(range(len(layers_b)))
    ax.set_xticklabels(layers_b)
    ax.set_yticks(range(len(layers_a)))
    ax.set_yticklabels(layers_a)
    ax.set_xlabel(f"{model_b_name} layer")
    ax.set_ylabel(f"{model_a_name} layer")
    ax.set_title(f"Linear CKA: {model_a_name} vs {model_b_name}")
    fig.colorbar(im, ax=ax, label="CKA", shrink=0.6)
    fig.tight_layout()

    # ---- Plotly interactive heatmap -----------------------------------------
    fig_plotly = None
    if go is not None:
        text_matrix = [[f"{matrix[i, j]:.3f}" for j in range(len(layers_b))] for i in range(len(layers_a))]

        fig_plotly = go.Figure(data=go.Heatmap(
            z=matrix,
            x=[str(lb) for lb in layers_b],
            y=[str(la) for la in layers_a],
            text=text_matrix,
            texttemplate="%{text}",
            colorscale="Viridis",
            zmin=0, zmax=1,
            colorbar={"title": "CKA"},
        ))
        fig_plotly.update_layout(
            title=f"Linear CKA: {model_a_name} vs {model_b_name}",
            xaxis_title=f"{model_b_name} layer",
            yaxis_title=f"{model_a_name} layer",
            width=600, height=750,
        )

    # ---- Save ---------------------------------------------------------------
    if save_prefix is not None:
        prefix = Path(save_prefix)
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(f"{prefix}_cka_heatmap.png", dpi=150, bbox_inches="tight")
        if fig_plotly is not None:
            fig_plotly.write_html(f"{prefix}_cka_heatmap.html")

    return fig, fig_plotly


def plot_selectivity_comparison(
    bert_sel: np.ndarray,
    bert_tau: np.ndarray,
    distil_sel: np.ndarray,
    distil_tau: np.ndarray,
    *,
    bert_cog_norm: float | None = None,
    distil_cog_norm: float | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot selectivity vs relative depth for BERT and DistilBERT.

    Parameters
    ----------
    bert_sel : np.ndarray
        BERT selectivity array (layers 0..L).
    bert_tau : np.ndarray
        BERT relative depth array.
    distil_sel : np.ndarray
        DistilBERT selectivity array (layers 0..L).
    distil_tau : np.ndarray
        DistilBERT relative depth array.
    bert_cog_norm : float or None
        If given, draw a vertical dashed line at this tau value.
    distil_cog_norm : float or None
        If given, draw a vertical dashed line at this tau value.
    save_path : str or Path or None
        If given, save the figure to this path.

    Returns
    -------
    matplotlib.figure.Figure
        The figure object.
    """
    if plt is None:
        raise ImportError("matplotlib is required for plot_selectivity_comparison, but is not installed.")

    fig, ax = plt.subplots(figsize=(8, 5))

    ax.plot(bert_tau, bert_sel, "o-", label="BERT (L=12)", color="tab:blue", markersize=4)
    ax.plot(distil_tau, distil_sel, "s-", label="DistilBERT (L=6)", color="tab:orange", markersize=4)

    if bert_cog_norm is not None and np.isfinite(bert_cog_norm):
        ax.axvline(bert_cog_norm, color="tab:blue", linestyle="--", alpha=0.7, label=f"BERT CoG_norm={bert_cog_norm:.3f}")
    if distil_cog_norm is not None and np.isfinite(distil_cog_norm):
        ax.axvline(distil_cog_norm, color="tab:orange", linestyle="--", alpha=0.7, label=f"DistilBERT CoG_norm={distil_cog_norm:.3f}")

    ax.set_xlabel("Relative depth τ = l / L")
    ax.set_ylabel("Selectivity")
    ax.set_title("UPOS Selectivity: BERT vs DistilBERT")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")

    return fig
