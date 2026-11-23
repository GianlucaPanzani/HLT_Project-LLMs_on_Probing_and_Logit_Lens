"""Cross-linguistic Delta CoG from probing summaries and selectivity curves."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from lib.metrics import center_of_gravity, relative_depth, selectivity, summarize_probing


@dataclass(frozen=True, slots=True)
class ModelPair:
    """English and Italian models compared for Delta CoG."""

    pair: str
    kind: Literal["multilingual", "paired"]
    en_model: str
    en_train_treebank: str
    it_model: str
    it_train_treebank: str
    en_eval_own: str
    it_eval_own: str
    en_eval_pud: str
    it_eval_pud: str


PAIRS: tuple[ModelPair, ...] = (
    ModelPair(
        "mbert",
        "multilingual",
        "mbert",
        "ewt",
        "mbert",
        "isdt",
        "ewt",
        "isdt",
        "pud_en",
        "pud_it",
    ),
    ModelPair(
        "mgpt",
        "multilingual",
        "mgpt",
        "ewt",
        "mgpt",
        "isdt",
        "ewt",
        "isdt",
        "pud_en",
        "pud_it",
    ),
    ModelPair(
        "bert",
        "paired",
        "bert",
        "ewt",
        "bert_it",
        "isdt",
        "ewt",
        "isdt",
        "pud_en",
        "pud_it",
    ),
    ModelPair(
        "gpt2",
        "paired",
        "gpt2",
        "ewt",
        "geppetto",
        "isdt",
        "ewt",
        "isdt",
        "pud_en",
        "pud_it",
    ),
)

DELTA_COG_COLUMNS = [
    "pair",
    "kind",
    "en_model",
    "it_model",
    "eval_set",
    "cog_norm_en",
    "cog_norm_it",
    "delta_cog",
    "L_en",
    "L_it",
]

DELTA_COG_BY_SEED_COLUMNS = [*DELTA_COG_COLUMNS, "seed"]

DELTA_COG_AGG_COLUMNS = [
    "pair",
    "kind",
    "en_model",
    "it_model",
    "eval_set",
    "cog_norm_en_mean",
    "cog_norm_it_mean",
    "delta_cog_mean",
    "delta_cog_std",
    "delta_cog_min",
    "delta_cog_max",
    "n_seeds",
]

SELECTIVITY_CURVE_COLUMNS = [
    "model",
    "train_treebank",
    "eval_treebank",
    "lang",
    "probe",
    "init",
    "pooling",
    "layer",
    "L",
    "tau",
    "selectivity",
    "cog_norm",
]


def _lookup_cog_norm(
    summary: pd.DataFrame,
    *,
    model: str,
    train_treebank: str,
    eval_treebank: str,
    probe: str,
    init: str,
    pooling: str,
    task: str,
) -> tuple[float, int]:
    """Return cog_norm and L for one summary row; raise if missing or ambiguous."""
    mask = (
        (summary["model"] == model)
        & (summary["train_treebank"] == train_treebank)
        & (summary["eval_treebank"] == eval_treebank)
        & (summary["probe"] == probe)
        & (summary["init"] == init)
        & (summary["pooling"] == pooling)
        & (summary["task"] == task)
    )
    rows = summary.loc[mask]
    if rows.empty:
        raise ValueError(
            f"No summary row for model={model!r}, train={train_treebank!r}, "
            f"eval={eval_treebank!r}, probe={probe!r}, task={task!r}."
        )
    if len(rows) > 1:
        raise ValueError(
            f"Duplicate summary rows for model={model!r}, train={train_treebank!r}, "
            f"eval={eval_treebank!r}, probe={probe!r}, task={task!r}."
        )
    row = rows.iloc[0]
    return float(row["cog_norm"]), int(row["L"])


def delta_cog(
    summary: pd.DataFrame,
    pairs: tuple[ModelPair, ...] = PAIRS,
    *,
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
    task: str = "upos",
) -> pd.DataFrame:
    """Compute Delta CoG = cog_norm(IT) - cog_norm(EN) for each pair and eval set.

    Parameters
    ----------
    summary
        Output of ``lib.metrics.summarize_probing`` (``results/summary/cog.parquet``).
    pairs
        Model pairs to compare.
    probe, init, pooling, task
        Filters on the summary table.

    Returns
    -------
    DataFrame with columns in ``DELTA_COG_COLUMNS``.
    """
    required = {"model", "train_treebank", "eval_treebank", "probe", "init", "pooling", "task", "cog_norm", "L"}
    missing = sorted(required - set(summary.columns))
    if missing:
        raise ValueError(f"Missing summary columns: {missing}.")

    rows: list[dict[str, object]] = []
    for spec in pairs:
        for eval_set, en_eval, it_eval in (
            ("own_test", spec.en_eval_own, spec.it_eval_own),
            ("pud", spec.en_eval_pud, spec.it_eval_pud),
        ):
            cog_en, l_en = _lookup_cog_norm(
                summary,
                model=spec.en_model,
                train_treebank=spec.en_train_treebank,
                eval_treebank=en_eval,
                probe=probe,
                init=init,
                pooling=pooling,
                task=task,
            )
            cog_it, l_it = _lookup_cog_norm(
                summary,
                model=spec.it_model,
                train_treebank=spec.it_train_treebank,
                eval_treebank=it_eval,
                probe=probe,
                init=init,
                pooling=pooling,
                task=task,
            )
            rows.append(
                {
                    "pair": spec.pair,
                    "kind": spec.kind,
                    "en_model": spec.en_model,
                    "it_model": spec.it_model,
                    "eval_set": eval_set,
                    "cog_norm_en": cog_en,
                    "cog_norm_it": cog_it,
                    "delta_cog": cog_it - cog_en,
                    "L_en": l_en,
                    "L_it": l_it,
                }
            )

    return pd.DataFrame(rows, columns=DELTA_COG_COLUMNS)


def delta_cog_by_seed(
    results: pd.DataFrame,
    pairs: tuple[ModelPair, ...] = PAIRS,
    *,
    probe: str = "linear",
    init: str = "pretrained",
    pooling: str = "first",
    task: str = "upos",
) -> pd.DataFrame:
    """Compute Delta CoG separately for each probing seed.

    Parameters
    ----------
    results
        Raw probing rows (possibly several seeds per experiment).

    Returns
    -------
    One row per pair, eval set and seed. Pairs without both languages for that
    seed are skipped (e.g. mGPT with only seed 42).
    """
    if "seed" not in results.columns:
        raise ValueError("Missing probing results columns: ['seed'].")
    if results.empty:
        return pd.DataFrame(columns=DELTA_COG_BY_SEED_COLUMNS)

    rows: list[dict[str, object]] = []
    for seed in sorted(results["seed"].unique()):
        seed_results = results.loc[results["seed"] == seed]
        summary = summarize_probing(seed_results)
        for spec in pairs:
            try:
                part = delta_cog(
                    summary,
                    pairs=(spec,),
                    probe=probe,
                    init=init,
                    pooling=pooling,
                    task=task,
                )
            except ValueError:
                continue
            for _, row in part.iterrows():
                rows.append({**row.to_dict(), "seed": int(seed)})

    return pd.DataFrame(rows, columns=DELTA_COG_BY_SEED_COLUMNS)


def aggregate_delta_cog(per_seed: pd.DataFrame) -> pd.DataFrame:
    """Mean and spread of Delta CoG across seeds."""
    if per_seed.empty:
        return pd.DataFrame(columns=DELTA_COG_AGG_COLUMNS)

    group_cols = ["pair", "kind", "en_model", "it_model", "eval_set"]
    missing = sorted(set(group_cols + ["delta_cog", "cog_norm_en", "cog_norm_it", "seed"]) - set(per_seed.columns))
    if missing:
        raise ValueError(f"Missing per-seed columns: {missing}.")

    grouped = per_seed.groupby(group_cols, sort=False)
    agg = grouped.agg(
        cog_norm_en_mean=("cog_norm_en", "mean"),
        cog_norm_it_mean=("cog_norm_it", "mean"),
        delta_cog_mean=("delta_cog", "mean"),
        delta_cog_min=("delta_cog", "min"),
        delta_cog_max=("delta_cog", "max"),
        n_seeds=("seed", "nunique"),
    ).reset_index()
    std = grouped["delta_cog"].std(ddof=1).reset_index(name="delta_cog_std")
    std.loc[std["delta_cog_std"].isna(), "delta_cog_std"] = np.nan
    out = agg.merge(std, on=group_cols, how="left")
    out.loc[out["n_seeds"] < 2, "delta_cog_std"] = np.nan
    return out[DELTA_COG_AGG_COLUMNS]


def selectivity_curves(results: pd.DataFrame) -> pd.DataFrame:
    """Layer-wise selectivity and CoG markers from raw probing results.

    Parameters
    ----------
    results
        Concatenated probing parquet (CONTRACTS.md section 6).

    Returns
    -------
    One row per model group and layer, with ``tau`` and ``cog_norm``.
    """
    group_columns = ["model", "train_treebank", "eval_treebank", "probe", "init", "pooling"]
    required = group_columns + ["lang", "task", "L", "layer", "seed", "acc"]
    missing = sorted(set(required) - set(results.columns))
    if missing:
        raise ValueError(f"Missing probing results columns: {missing}.")
    if results.empty:
        return pd.DataFrame(columns=SELECTIVITY_CURVE_COLUMNS)

    curve_rows: list[dict[str, object]] = []
    for group_key, group in results.groupby(group_columns, sort=False, dropna=False):
        if group["L"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent L in group {group_key}.")
        if group["lang"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent lang in group {group_key}.")
        num_layers = int(group["L"].iloc[0])

        task_rows = group.loc[group["task"] == "upos", ["layer", "seed", "acc"]]
        control_rows = group.loc[group["task"] == "control", ["layer", "seed", "acc"]]
        if task_rows.empty or control_rows.empty:
            continue

        paired = task_rows.merge(
            control_rows,
            on=["layer", "seed"],
            how="inner",
            suffixes=("_task", "_control"),
        )
        if paired.empty:
            continue
        if paired.duplicated(["layer", "seed"]).any():
            raise ValueError(f"Duplicate layer/seed rows in group {group_key}.")

        means = paired.groupby("layer", sort=True)[["acc_task", "acc_control"]].mean()
        if not np.array_equal(means.index.to_numpy(), np.arange(num_layers + 1)):
            raise ValueError(f"Expected all layers 0..{num_layers} in group {group_key}.")

        sel = selectivity(means["acc_task"].to_numpy(), means["acc_control"].to_numpy())
        cog_norm = center_of_gravity(sel, normalize=True)

        lang = group["lang"].iloc[0]
        for layer_idx, sel_val in enumerate(sel):
            curve_rows.append(
                {
                    **dict(zip(group_columns, group_key)),
                    "lang": lang,
                    "layer": layer_idx,
                    "L": num_layers,
                    "tau": relative_depth(layer_idx, num_layers),
                    "selectivity": float(sel_val),
                    "cog_norm": cog_norm,
                }
            )

    return pd.DataFrame(curve_rows, columns=SELECTIVITY_CURVE_COLUMNS)