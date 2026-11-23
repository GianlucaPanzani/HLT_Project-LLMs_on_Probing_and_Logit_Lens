"""Overlay lens curves and next-token probing selectivity (Issue L8).

This module implements the comparative bridge between:
1. Probing selectivity for next-token UPOS (linear decodability from residual stream)
2. Lens predictions (model's own internal predictions via unembedding matrix)
across layers l in [0, L] over relative layer depth tau = l / L.

Implements commitment layer calculation (layer reaching >= 90% of maximum)
and publication-quality dual-panel figures for GPT-2 and GePpeTto.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from lib.metrics import center_of_gravity, selectivity  # noqa: E402
from lib.paths import ROOT, lens_result_path, probing_result_path  # noqa: E402

DEFAULT_TREEBANK_MAP: dict[str, tuple[str, str]] = {
    "gpt2": ("pud_en", "ewt"),
    "geppetto": ("pud_it", "isdt"),
}


def compute_commitment_layer(
    curve: np.ndarray | list[float],
    threshold_fraction: float = 0.90,
) -> tuple[int, float, float]:
    """Compute the commitment layer of a curve as the first layer reaching >= 90% of its maximum.

    Parameters
    ----------
    curve : np.ndarray or list[float]
        1D array of metric values over layers 0..L.
    threshold_fraction : float, default=0.90
        Fraction of maximum to reach (default: 0.90).

    Returns
    -------
    tuple[int, float, float]
        (commitment_layer, commitment_tau, max_value).
    """
    arr = np.asarray(curve, dtype=float)
    if arr.ndim != 1 or len(arr) == 0:
        raise ValueError("Curve must be a non-empty 1D array.")

    max_val = float(np.max(arr))
    n_layers = len(arr) - 1

    if max_val <= 0.0:
        return 0, 0.0, max_val

    threshold = threshold_fraction * max_val
    valid = np.where(arr >= threshold)[0]
    comm_layer = int(valid[0]) if len(valid) > 0 else n_layers
    comm_tau = float(comm_layer / n_layers) if n_layers > 0 else 0.0

    return comm_layer, comm_tau, max_val


def load_overlay_data(
    model: str,
    eval_treebank: str | None = None,
    train_treebank: str | None = None,
) -> dict[str, Any]:
    """Load aligned probing selectivity and lens metrics for a model and dataset.

    Parameters
    ----------
    model : str
        Model registry key ('gpt2' or 'geppetto').
    eval_treebank : str or None, optional
        Evaluation treebank key ('pud_en' for gpt2, 'pud_it' for geppetto).
    train_treebank : str or None, optional
        Train treebank key for probing ('ewt' for gpt2, 'isdt' for geppetto).

    Returns
    -------
    dict[str, Any]
        Dictionary containing layers, taus, probing selectivity and accuracies,
        and lens metrics (top1_acc, upos_top1_acc, entropy_mean, kl_mean).
    """
    if eval_treebank is None or train_treebank is None:
        if model not in DEFAULT_TREEBANK_MAP:
            raise ValueError(f"Unknown model '{model}'. Specify eval_treebank and train_treebank explicitly.")
        default_eval, default_train = DEFAULT_TREEBANK_MAP[model]
        eval_treebank = eval_treebank or default_eval
        train_treebank = train_treebank or default_train

    # 1. Probing results
    p_prob = probing_result_path(model, train_treebank, "linear", "pretrained", "last")
    if not p_prob.exists():
        raise FileNotFoundError(f"Probing results file not found at: {p_prob}")

    df_prob = pd.read_parquet(p_prob)
    df_eval = df_prob[df_prob["eval_treebank"] == eval_treebank]
    if df_eval.empty:
        raise ValueError(f"No probing records found for eval_treebank='{eval_treebank}' in {p_prob}")

    upos_next_df = df_eval[df_eval["task"] == "upos_next"].sort_values("layer")
    ctrl_next_df = df_eval[df_eval["task"] == "control_next"].sort_values("layer")

    if upos_next_df.empty or ctrl_next_df.empty:
        raise ValueError(f"Missing upos_next or control_next tasks in {p_prob} for {eval_treebank}")

    layers = upos_next_df["layer"].values
    taus = upos_next_df["tau"].values
    acc_task = upos_next_df["acc"].values
    acc_ctrl = ctrl_next_df["acc"].values
    sel_next = selectivity(acc_task, acc_ctrl)

    # 2. Lens results
    p_lens = lens_result_path(model, eval_treebank)
    if not p_lens.exists():
        raise FileNotFoundError(f"Lens results file not found at: {p_lens}")

    df_lens = pd.read_parquet(p_lens)
    lens_data: dict[str, dict[str, np.ndarray]] = {}

    for l_key in ("logit", "tuned", "jacobian"):
        sub = df_lens[df_lens["lens"] == l_key].sort_values("layer")
        if sub.empty:
            raise ValueError(f"Missing lens '{l_key}' in {p_lens}")
        lens_data[l_key] = {
            "top1_acc": sub["top1_acc"].values,
            "upos_top1_acc": sub["upos_top1_acc"].values,
            "entropy_mean": sub["entropy_mean"].values,
            "kl_mean": sub["kl_mean"].values,
        }

    return {
        "model": model,
        "eval_treebank": eval_treebank,
        "train_treebank": train_treebank,
        "layers": layers,
        "taus": taus,
        "probing_sel_next": sel_next,
        "probing_acc_upos_next": acc_task,
        "probing_acc_control_next": acc_ctrl,
        "lenses": lens_data,
    }


def build_commitment_table(
    models: tuple[str, ...] = ("gpt2", "geppetto"),
) -> pd.DataFrame:
    """Build the commitment layers table for probing selectivity, upos_top1_acc, and top1_acc.

    Parameters
    ----------
    models : tuple[str, ...], default=("gpt2", "geppetto")
        Tuple of model names to include.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: model, paradigm, metric, commitment_layer, commitment_tau, max_value.
    """
    records: list[dict[str, Any]] = []

    for model in models:
        eval_tb, train_tb = DEFAULT_TREEBANK_MAP[model]
        data = load_overlay_data(model, eval_tb, train_tb)

        # Probing selectivity
        c_lay, c_tau, max_v = compute_commitment_layer(data["probing_sel_next"])
        records.append({
            "model": model,
            "paradigm": "Probing (Linear)",
            "metric": "UPOS Next Selectivity",
            "commitment_layer": c_lay,
            "commitment_tau": round(c_tau, 3),
            "max_value": round(max_v, 4),
        })

        # Each Lens
        for l_key in ("logit", "tuned", "jacobian"):
            c_lay_u, c_tau_u, max_u = compute_commitment_layer(data["lenses"][l_key]["upos_top1_acc"])
            records.append({
                "model": model,
                "paradigm": f"Lens ({l_key.capitalize()})",
                "metric": "UPOS Top-1 Acc",
                "commitment_layer": c_lay_u,
                "commitment_tau": round(c_tau_u, 3),
                "max_value": round(max_u, 4),
            })

            c_lay_t, c_tau_t, max_t = compute_commitment_layer(data["lenses"][l_key]["top1_acc"])
            records.append({
                "model": model,
                "paradigm": f"Lens ({l_key.capitalize()})",
                "metric": "Token Top-1 Acc",
                "commitment_layer": c_lay_t,
                "commitment_tau": round(c_tau_t, 3),
                "max_value": round(max_t, 4),
            })

    return pd.DataFrame(records)


def plot_model_overlay(
    model: str,
    eval_treebank: str | None = None,
    train_treebank: str | None = None,
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generate and optionally save a publication-quality dual-panel overlay plot for a model.

    Parameters
    ----------
    model : str
        Model registry key ('gpt2' or 'geppetto').
    eval_treebank : str or None, optional
        Evaluation treebank key.
    train_treebank : str or None, optional
        Training treebank key.
    save_path : Path or str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
        Rendered matplotlib Figure.
    """
    data = load_overlay_data(model, eval_treebank, train_treebank)
    taus = data["taus"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=150)

    colors = {
        "probing": "#1f77b4",   # bold blue
        "logit": "#d62728",     # crimson
        "tuned": "#2ca02c",     # forest green
        "jacobian": "#9467bd",  # purple
    }

    # ------------------------------------------------------------------ #
    # Panel (a): Accuracies and Probing Selectivity vs tau
    # ------------------------------------------------------------------ #
    ax1.plot(
        taus,
        data["probing_sel_next"],
        label="Probing: Next UPOS Selectivity",
        color=colors["probing"],
        linewidth=2.5,
        marker="o",
        markersize=5,
    )
    c_prob, _, _ = compute_commitment_layer(data["probing_sel_next"])
    ax1.axvline(taus[c_prob], color=colors["probing"], linestyle=":", alpha=0.6)

    for l_key, l_name in (("logit", "Logit"), ("tuned", "Tuned"), ("jacobian", "Jacobian")):
        u_acc = data["lenses"][l_key]["upos_top1_acc"]
        ax1.plot(
            taus,
            u_acc,
            label=f"{l_name}: UPOS Acc",
            color=colors[l_key],
            linestyle="--",
            linewidth=2.0,
            marker="s",
            markersize=4,
        )
        w_acc = data["lenses"][l_key]["top1_acc"]
        ax1.plot(
            taus,
            w_acc,
            label=f"{l_name}: Token Top-1",
            color=colors[l_key],
            linestyle="-",
            linewidth=2.0,
            marker="^",
            markersize=4,
        )

    eval_label = data["eval_treebank"].upper()
    ax1.set_title(f"{model.upper()} ({eval_label}) — Probing & Lens Commitments", fontsize=12, fontweight="bold")
    ax1.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=11)
    ax1.set_ylabel("Accuracy / Selectivity", fontsize=11)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 0.48)
    ax1.grid(True, linestyle="--", alpha=0.4)
    ax1.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Panel (b): Information Dynamics (Entropy & KL to Final)
    # ------------------------------------------------------------------ #
    ax2_twin = ax2.twinx()
    lines: list[Any] = []
    labels: list[str] = []

    for l_key, l_name in (("logit", "Logit"), ("tuned", "Tuned"), ("jacobian", "Jacobian")):
        ent = data["lenses"][l_key]["entropy_mean"]
        kl = data["lenses"][l_key]["kl_mean"]

        line_ent, = ax2.plot(taus, ent, color=colors[l_key], linestyle="-", linewidth=1.8, marker="o", markersize=4)
        line_kl, = ax2_twin.plot(taus, kl, color=colors[l_key], linestyle=":", linewidth=1.8, marker="x", markersize=4)

        lines.extend([line_ent, line_kl])
        labels.extend([f"{l_name} Entropy (H)", f"{l_name} KL to Final"])

    ax2.set_title(f"{model.upper()} ({eval_label}) — Entropy & KL Dynamics", fontsize=12, fontweight="bold")
    ax2.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=11)
    ax2.set_ylabel("Shannon Entropy [nats] (Solid -)", fontsize=11)
    ax2_twin.set_ylabel(r"$D_{\mathrm{KL}}(P_L \parallel P_l)$ [nats] (Dotted :)", fontsize=11)
    ax2.set_xlim(-0.02, 1.02)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(lines, labels, loc="upper right", fontsize=8, framealpha=0.9, ncol=2)

    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def plot_crossling_comparison(
    save_path: Path | str | None = None,
) -> plt.Figure:
    """Generate side-by-side cross-linguistic comparison between GPT-2 (EN) and GePpeTto (IT).

    Compares:
    1. Probing Selectivity S^{(l)} for next UPOS with Center of Gravity markers.
    2. Tuned Lens UPOS Bridge Accuracy with commitment points.
    3. Tuned Lens Token Top-1 Accuracy with commitment points.
    4. Information Dynamics: Shannon Entropy H and KL Divergence to final layer.

    Parameters
    ----------
    save_path : Path or str or None, optional
        If provided, save the figure to this path.

    Returns
    -------
    plt.Figure
        Rendered matplotlib Figure.
    """
    d_en = load_overlay_data("gpt2", "pud_en", "ewt")
    d_it = load_overlay_data("geppetto", "pud_it", "isdt")

    taus = d_en["taus"]
    color_en = "#1f77b4"  # blue
    color_it = "#2ca02c"  # green

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), dpi=150)

    # ------------------------------------------------------------------ #
    # Panel (a): Probing Selectivity S^{(l)}
    # ------------------------------------------------------------------ #
    ax = axs[0, 0]
    ax.plot(
        taus,
        d_en["probing_sel_next"],
        label="GPT-2 (English, PUD EN)",
        color=color_en,
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    ax.plot(
        taus,
        d_it["probing_sel_next"],
        label="GePpeTto (Italian, PUD IT)",
        color=color_it,
        linewidth=2.2,
        marker="s",
        markersize=5,
    )

    cog_en = center_of_gravity(d_en["probing_sel_next"], normalize=True)
    cog_it = center_of_gravity(d_it["probing_sel_next"], normalize=True)

    ax.axvline(cog_en, color=color_en, linestyle=":", alpha=0.8, label=f"GPT-2 CoG ({cog_en:.3f})")
    ax.axvline(cog_it, color=color_it, linestyle=":", alpha=0.8, label=f"GePpeTto CoG ({cog_it:.3f})")

    ax.set_title("(a) Probing: Next-Token UPOS Selectivity", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=10)
    ax.set_ylabel("Selectivity (Task Acc - Control Acc)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.32)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Panel (b): Tuned Lens UPOS Bridge Accuracy
    # ------------------------------------------------------------------ #
    ax = axs[0, 1]
    u_en = d_en["lenses"]["tuned"]["upos_top1_acc"]
    u_it = d_it["lenses"]["tuned"]["upos_top1_acc"]
    c_upos_en, _, _ = compute_commitment_layer(u_en)
    c_upos_it, _, _ = compute_commitment_layer(u_it)

    ax.plot(
        taus,
        u_en,
        label=f"GPT-2 (Commit @ L{c_upos_en}, $\\tau$={taus[c_upos_en]:.3f})",
        color=color_en,
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    ax.plot(
        taus,
        u_it,
        label=f"GePpeTto (Commit @ L{c_upos_it}, $\\tau$={taus[c_upos_it]:.3f})",
        color=color_it,
        linewidth=2.2,
        marker="s",
        markersize=5,
    )

    ax.scatter([taus[c_upos_en]], [u_en[c_upos_en]], color="crimson", s=90, zorder=5)
    ax.scatter([taus[c_upos_it]], [u_it[c_upos_it]], color="crimson", s=90, zorder=5)

    ax.set_title("(b) Tuned Lens: Next-Token UPOS Bridge Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=10)
    ax.set_ylabel("UPOS Top-1 Accuracy", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.45)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Panel (c): Tuned Lens Exact Token Top-1 Accuracy
    # ------------------------------------------------------------------ #
    ax = axs[1, 0]
    t_en = d_en["lenses"]["tuned"]["top1_acc"]
    t_it = d_it["lenses"]["tuned"]["top1_acc"]
    c_tok_en, _, _ = compute_commitment_layer(t_en)
    c_tok_it, _, _ = compute_commitment_layer(t_it)

    ax.plot(
        taus,
        t_en,
        label=f"GPT-2 (Commit @ L{c_tok_en}, max={np.max(t_en):.3f})",
        color=color_en,
        linewidth=2.2,
        marker="o",
        markersize=5,
    )
    ax.plot(
        taus,
        t_it,
        label=f"GePpeTto (Commit @ L{c_tok_it}, max={np.max(t_it):.3f})",
        color=color_it,
        linewidth=2.2,
        marker="s",
        markersize=5,
    )

    ax.scatter([taus[c_tok_en]], [t_en[c_tok_en]], color="crimson", s=90, zorder=5)
    ax.scatter([taus[c_tok_it]], [t_it[c_tok_it]], color="crimson", s=90, zorder=5)

    ax.set_title("(c) Tuned Lens: Exact Token Top-1 Accuracy", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=10)
    ax.set_ylabel("Exact Token Top-1 Accuracy", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 0.30)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="upper left", fontsize=8.5, framealpha=0.9)

    # ------------------------------------------------------------------ #
    # Panel (d): Tuned Lens Entropy & KL to Final Layer
    # ------------------------------------------------------------------ #
    ax = axs[1, 1]
    ent_en = d_en["lenses"]["tuned"]["entropy_mean"]
    ent_it = d_it["lenses"]["tuned"]["entropy_mean"]
    kl_en = d_en["lenses"]["tuned"]["kl_mean"]
    kl_it = d_it["lenses"]["tuned"]["kl_mean"]

    ax.plot(taus, ent_en, label="GPT-2: Entropy (H)", color=color_en, linestyle="-", linewidth=2.0)
    ax.plot(taus, ent_it, label="GePpeTto: Entropy (H)", color=color_it, linestyle="-", linewidth=2.0)

    ax_twin = ax.twinx()
    ax_twin.plot(taus, kl_en, label=r"GPT-2: $D_{\mathrm{KL}}$ to Final", color=color_en, linestyle=":", linewidth=2.0)
    ax_twin.plot(taus, kl_it, label=r"GePpeTto: $D_{\mathrm{KL}}$ to Final", color=color_it, linestyle=":", linewidth=2.0)

    ax.set_title("(d) Tuned Lens: Prediction Uncertainty & Final Convergence", fontsize=11, fontweight="bold")
    ax.set_xlabel(r"Relative Layer Depth $\tau = l / L$", fontsize=10)
    ax.set_ylabel("Shannon Entropy [nats] (Solid -)", fontsize=10)
    ax_twin.set_ylabel(r"$D_{\mathrm{KL}}(P_L \parallel P_l)$ [nats] (Dotted :)", fontsize=10)
    ax.set_xlim(-0.02, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax_twin.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8, framealpha=0.9)

    plt.suptitle(
        "Cross-Linguistic Comparison: GPT-2 (English) vs. GePpeTto (Italian)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout()

    if save_path is not None:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches="tight")

    return fig


def plot_all_overlays(
    output_dir: Path | str | None = None,
) -> dict[str, Path]:
    """Generate and save overlay figures for all models into results/figures/ws4/.

    Parameters
    ----------
    output_dir : Path or str or None, optional
        Target directory (default: results/figures/ws4/).

    Returns
    -------
    dict[str, Path]
        Dictionary mapping model names to saved figure paths.
    """
    if output_dir is None:
        out_dir = ROOT / "results" / "figures" / "ws4"
    else:
        out_dir = Path(output_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: dict[str, Path] = {}

    for model in ("gpt2", "geppetto"):
        target_path = out_dir / f"{model}_overlay.png"
        plot_model_overlay(model, save_path=target_path)
        plt.close("all")
        saved_paths[model] = target_path

    # Cross-linguistic comparison figure
    cross_path = out_dir / "crossling_gpt2_vs_geppetto.png"
    plot_crossling_comparison(save_path=cross_path)
    plt.close("all")
    saved_paths["crossling"] = cross_path

    return saved_paths
