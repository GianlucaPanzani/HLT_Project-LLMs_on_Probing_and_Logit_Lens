"""Matched linear/MLP comparison using existing current-word UPOS results (P8)."""

import numpy as np
import pandas as pd

from lib.metrics import center_of_gravity, peak_layer, relative_depth, selectivity
from lib.paths import probing_result_path
from lib.registry import get_model_spec

MODELS = ("bert", "gpt2")
PROBES = ("linear", "mlp")
METRICS = ("acc", "control_acc", "selectivity")
REQUIRED_COLUMNS = (
    "model", "train_treebank", "eval_treebank", "lang", "kind", "probe", "task",
    "init", "pooling", "layer", "L", "tau", "acc", "n_train", "n_eval", "hparams", "seed",
)


def load_results() -> pd.DataFrame:
    """Read the four P8 parquets without loading features or models.

    Returns
    -------
    pd.DataFrame
        Existing results. Missing files or incompatible file metadata raise
        FileNotFoundError or ValueError, respectively.
    """
    frames = []
    for model in MODELS:
        for probe in PROBES:
            path = probing_result_path(model, "ewt", probe, "pretrained", "first")
            if not path.is_file():
                raise FileNotFoundError(f"Missing {model}/{probe} results: {path}")
            rows = pd.read_parquet(path)
            missing = sorted(set(REQUIRED_COLUMNS) - set(rows.columns))
            if missing:
                raise ValueError(f"{path.name}: missing columns {missing}.")
            expected = {
                "model": model, "probe": probe, "train_treebank": "ewt",
                "init": "pretrained", "pooling": "first",
            }
            for column, value in expected.items():
                if not rows[column].eq(value).all():
                    raise ValueError(f"{path.name}: expected {column}={value} on every row.")
            frames.append(rows)
    return pd.concat(frames, ignore_index=True)


def prepare_comparison(
    results: pd.DataFrame,
    *,
    eval_treebank: str = "ewt",
    seed: int = 42,
) -> pd.DataFrame:
    """Validate and align both probes and controls for one evaluation set/seed.

    Parameters
    ----------
    results : pd.DataFrame
        The four pretrained, first-pooling BERT/GPT-2 result files.
    eval_treebank : str
        EWT test or PUD English test; training always uses EWT.
    seed : int
        One explicit seed. Seeds are never averaged or silently substituted.

    Returns
    -------
    pd.DataFrame
        One row per model/layer, with task/control/selectivity for both probes
        and MLP-minus-linear deltas in percentage points. Incomplete pairs,
        incompatible configurations or sample counts raise ValueError.
    """
    missing = sorted(set(REQUIRED_COLUMNS) - set(results.columns))
    if missing:
        raise ValueError(f"Missing probing columns: {missing}.")
    if eval_treebank not in ("ewt", "pud_en"):
        raise ValueError("Evaluation treebank must be ewt or pud_en.")
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)):
        raise ValueError("Seed must be an integer.")

    rows = results.loc[
        results["eval_treebank"].eq(eval_treebank)
        & results["seed"].eq(seed)
        & results["task"].isin(("upos", "control"))
    ].copy()
    if rows.empty:
        raise ValueError(f"No UPOS/control results for eval={eval_treebank}, seed={seed}.")
    if rows[list(REQUIRED_COLUMNS)].isna().any().any():
        raise ValueError("Missing values in the selected probing results.")
    for column, expected in (("train_treebank", "ewt"), ("init", "pretrained"), ("pooling", "first"), ("lang", "en")):
        if not rows[column].eq(expected).all():
            raise ValueError(f"Incompatible {column}: expected {expected}.")
    if not rows["model"].isin(MODELS).all() or not rows["probe"].isin(PROBES).all():
        raise ValueError("Expected only BERT/GPT-2 and linear/MLP results.")
    if rows.duplicated(["model", "probe", "task", "layer"]).any():
        raise ValueError("Duplicate model/probe/task/layer/seed rows.")
    for column in ("layer", "L", "seed", "n_train", "n_eval"):
        if not pd.api.types.is_integer_dtype(rows[column]) or pd.api.types.is_bool_dtype(rows[column]):
            raise ValueError(f"{column} must contain integers.")
    if not pd.api.types.is_numeric_dtype(rows["acc"]) or not rows["acc"].between(0, 1).all():
        raise ValueError("Accuracies must be finite numbers between 0 and 1.")
    if (rows[["n_train", "n_eval"]] <= 0).any().any():
        raise ValueError("n_train and n_eval must be positive.")

    frames = []
    for model in MODELS:
        model_rows = rows.loc[rows["model"].eq(model)]
        spec = get_model_spec(model)
        num_layers = spec["num_layers"]
        if not model_rows["L"].eq(num_layers).all() or not model_rows["kind"].eq(spec["kind"]).all():
            raise ValueError(f"Incompatible L or kind for {model}.")
        for probe in PROBES:
            for task in ("upos", "control"):
                rows_mask = model_rows["probe"].eq(probe) & model_rows["task"].eq(task)
                layers = model_rows.loc[rows_mask, "layer"]
                if sorted(layers.tolist()) != list(range(num_layers + 1)):
                    raise ValueError(
                        f"Missing or invalid {model}/{probe}/{task} layers: "
                        f"expected 0..{num_layers}, eval={eval_treebank}, seed={seed}."
                    )
        expected_tau = relative_depth(model_rows["layer"].to_numpy(), num_layers)
        if not pd.api.types.is_numeric_dtype(model_rows["tau"]) or not np.allclose(model_rows["tau"], expected_tau):
            raise ValueError(f"Inconsistent tau for {model}: expected layer / L.")
        counts = model_rows.groupby("layer")[["n_train", "n_eval"]]
        if counts.nunique().ne(1).any().any():
            raise ValueError(f"Incompatible n_train/n_eval for {model}: task/control and both probes must match.")

        # Pivot by explicit keys: input row order must never affect the pairing.
        paired = model_rows.pivot(index="layer", columns=["probe", "task"], values="acc").sort_index()
        table = counts.first().reset_index()
        for probe in PROBES:
            table[f"{probe}_acc"] = paired[(probe, "upos")].to_numpy()
            table[f"{probe}_control_acc"] = paired[(probe, "control")].to_numpy()
            table[f"{probe}_selectivity"] = selectivity(
                table[f"{probe}_acc"].to_numpy(), table[f"{probe}_control_acc"].to_numpy(),
            )
        for metric in METRICS:
            table[f"delta_{metric}_pp"] = 100 * (table[f"mlp_{metric}"] - table[f"linear_{metric}"])
        table["model"] = model
        table["L"] = num_layers
        table["tau"] = relative_depth(table["layer"].to_numpy(), num_layers)
        frames.append(table)

    comparison = pd.concat(frames, ignore_index=True)
    comparison["train_treebank"] = "ewt"
    comparison["eval_treebank"] = eval_treebank
    comparison["seed"] = seed
    comparison["init"] = "pretrained"
    comparison["pooling"] = "first"
    comparison["task"] = "upos"
    return comparison


def summarize_comparison(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compute peaks and CoG from the same validated curves used in the figure.

    Parameters
    ----------
    comparison : pd.DataFrame
        Layer-matched table returned by prepare_comparison, for one seed.

    Returns
    -------
    pd.DataFrame
        Four model/probe summaries. Peaks include layer 0; CoG excludes it
        and clips negative weights only for that calculation.
    """
    summaries = []
    metadata = ["model", "train_treebank", "eval_treebank", "init", "pooling", "task", "seed", "L"]
    for _, rows in comparison.groupby("model", sort=False):
        rows = rows.sort_values("layer")
        for probe in PROBES:
            summary = rows.iloc[0][metadata].to_dict()
            summary["probe"] = probe
            for metric in METRICS:
                values = rows[f"{probe}_{metric}"].to_numpy()
                peak = peak_layer(values)
                summary[f"max_{metric}"] = float(values[peak])
                summary[f"{metric}_peak_layer"] = peak
                summary[f"{metric}_peak_tau"] = relative_depth(peak, int(summary["L"]))
            sel = rows[f"{probe}_selectivity"].to_numpy()
            summary["cog"] = center_of_gravity(sel, normalize=False)
            summary["cog_norm"] = center_of_gravity(sel)
            summaries.append(summary)
    return pd.DataFrame(summaries)
