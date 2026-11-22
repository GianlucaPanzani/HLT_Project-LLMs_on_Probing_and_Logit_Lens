"""Cross-linguistic probing figures for workstream 3."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ws3_crossling.delta_cog import PAIRS, ModelPair


def _pair_spec(pair_id: str) -> ModelPair:
    for spec in PAIRS:
        if spec.pair == pair_id:
            return spec
    raise KeyError(pair_id)


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


def plot_selectivity_pairs(
    curves: pd.DataFrame,
    delta: pd.DataFrame,
    path: Path,
    *,
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
) -> Path:
    """Plot EN vs IT selectivity over tau for each pair and eval set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    pair_ids = [p.pair for p in PAIRS]
    eval_sets = ["own_test", "pud"]
    fig, axes = plt.subplots(len(pair_ids), len(eval_sets), figsize=(10, 12), sharex=True, sharey="row")
    if len(pair_ids) == 1:
        axes = [axes]
    eval_titles = {"own_test": "Own test (EWT / ISDT)", "pud": "PUD (EN / IT)"}

    for row, pair_id in enumerate(pair_ids):
        spec = _pair_spec(pair_id)
        for col, eval_set in enumerate(eval_sets):
            ax = axes[row][col]
            if eval_set == "own_test":
                en_eval, it_eval = spec.en_eval_own, spec.it_eval_own
            else:
                en_eval, it_eval = spec.en_eval_pud, spec.it_eval_pud

            en_df = _filter_curves(
                curves,
                model=spec.en_model,
                train_treebank=spec.en_train_treebank,
                eval_treebank=en_eval,
                probe=probe,
                init=init,
                pooling=pooling,
            )
            it_df = _filter_curves(
                curves,
                model=spec.it_model,
                train_treebank=spec.it_train_treebank,
                eval_treebank=it_eval,
                probe=probe,
                init=init,
                pooling=pooling,
            )
            if en_df.empty or it_df.empty:
                ax.set_visible(False)
                continue

            ax.plot(en_df["tau"], en_df["selectivity"], label="EN", color="C0", linewidth=1.5)
            ax.plot(it_df["tau"], it_df["selectivity"], label="IT", color="C1", linewidth=1.5)
            cog_en = float(en_df["cog_norm"].iloc[0])
            cog_it = float(it_df["cog_norm"].iloc[0])
            ax.axvline(cog_en, color="C0", linestyle="--", alpha=0.7, linewidth=1)
            ax.axvline(cog_it, color="C1", linestyle="--", alpha=0.7, linewidth=1)
            ax.set_xlim(0, 1)
            ax.grid(True, alpha=0.3)
            if row == 0:
                ax.set_title(eval_titles[eval_set])
            if col == 0:
                ax.set_ylabel(f"{pair_id}\nselectivity")
            if row == len(pair_ids) - 1:
                ax.set_xlabel("τ (relative depth)")

            delta_row = delta.loc[(delta["pair"] == pair_id) & (delta["eval_set"] == eval_set)]
            if not delta_row.empty:
                d = float(delta_row.iloc[0]["delta_cog"])
                ax.text(
                    0.03,
                    0.97,
                    f"ΔCoG={d:+.3f}",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                )
            if row == 0 and col == len(eval_sets) - 1:
                ax.legend(loc="upper right", fontsize=8)

    fig.suptitle("Selectivity vs τ (dashed: CoG_norm)", y=1.01)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_delta_cog(delta: pd.DataFrame, path: Path) -> Path:
    """Bar chart of Delta CoG per pair, grouped by eval set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    plot_df = delta.copy()
    order = [p.pair for p in PAIRS]
    plot_df["pair"] = pd.Categorical(plot_df["pair"], categories=order, ordered=True)
    plot_df = plot_df.sort_values(["pair", "eval_set"])

    x_labels = plot_df["pair"].astype(str) + "\n" + plot_df["eval_set"].astype(str)
    x_pos = range(len(plot_df))
    colors = ["C0" if es == "own_test" else "C2" for es in plot_df["eval_set"]]

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(x_pos, plot_df["delta_cog"], color=colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(list(x_pos), x_labels, rotation=0, fontsize=9)
    ax.set_ylabel("Δ CoG_norm (IT − EN)")
    ax.set_title("Delta CoG by model pair")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path