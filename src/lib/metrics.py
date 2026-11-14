import warnings

import numpy as np
import pandas as pd


def selectivity(acc_task: np.ndarray, acc_control: np.ndarray) -> np.ndarray:
    '''
    Compute the elementwise difference between task and control accuracy.
    Negative selectivity values are preserved.

    Parameters
    ----------
    - acc_task (np.ndarray): Task accuracies, aligned with acc_control.
    - acc_control (np.ndarray): Control accuracies with the same shape as
    acc_task. A shape mismatch raises ValueError.

    Returns
    -------
    sel (np.ndarray): Task accuracy minus control accuracy, with the same
    shape as the input arrays.
    '''
    if acc_task.shape != acc_control.shape:
        raise ValueError("Accuracy arrays must have the same shape.")
    # Floating-point arithmetic also preserves negative differences for unsigned inputs.
    return acc_task.astype(float) - acc_control.astype(float)


def relative_depth(layer: int | np.ndarray, num_layers: int) -> float | np.ndarray:
    '''
    Compute layer depth relative to the number of Transformer blocks.

    Parameters
    ----------
    - layer (int | np.ndarray): Integer layer index or array of integer
    indices. The contract includes layers 0 through num_layers, with 0
    representing the embedding output.
    - num_layers (int): Positive number of Transformer blocks, excluding
    the embedding layer.

    Returns
    -------
    tau (float | np.ndarray): layer / num_layers, preserving the shape
    when layer is an array.
    '''

    if isinstance(num_layers, (bool, np.bool_)) or not isinstance(num_layers, (int, np.integer)):
        raise TypeError("num_layers must be an integer.")
    if num_layers <= 0:
        raise ValueError("num_layers must be greater than zero.")

    if isinstance(layer, (bool, np.bool_)) or not isinstance(layer, (int, np.ndarray, np.integer)):
        raise TypeError("layer must be an integer or a NumPy array.")

    if isinstance(layer, np.ndarray):
        if not np.issubdtype(layer.dtype, np.integer):
            raise TypeError("Variable \"layer\" must contain integer values.")

    if np.any(layer < 0) or np.any(layer > num_layers):
        raise ValueError("Each layer must satisfy 0 <= layer <= num_layers.")

    return layer / num_layers


def center_of_gravity(sel: np.ndarray, *, normalize: bool = True, clip_negative: bool = True) -> float:
    '''
    Compute the selectivity-weighted mean layer, excluding the embedding layer.
    The contract defines CoG as sum(l * sel[l]) / sum(sel[l]) over layers 1..L,
    optionally divided by L for normalized depth.

    Parameters
    ----------
    - sel (np.ndarray): One-dimensional finite selectivity array with L + 1
    entries for layers 0..L and L >= 1. The entry at index 0 is excluded
    from the calculation. Invalid shapes or nonfinite values raise ValueError.
    - normalize (bool): Whether to divide CoG by L. Defaults to True.
    - clip_negative (bool): Whether to replace negative selectivity values
    with zero before calculating CoG. Defaults to True.

    Returns
    -------
    cog (float): Selectivity-weighted layer depth, normalized when requested,
    or NaN if the sum of the included selectivities is zero after optional
    clipping.
    '''
    if sel.ndim != 1 or sel.size < 2:
        raise ValueError("sel must be one-dimensional and include layer 0 and at least one block.")
    if not np.all(np.isfinite(sel)):
        raise ValueError("sel must contain only finite values.")

    num_layers = len(sel) - 1
    # Copy to float before clipping so the caller's array is never modified.
    sel = sel[1:].astype(float)
    if clip_negative:
        sel = np.clip(sel, 0, None)

    # Denominator of the Center of Gravity (CoG)
    cog_denominator = np.sum(sel)
    if cog_denominator == 0:
        return np.nan  # Return NaN to avoid division by zero
    
    layers = np.arange(1, num_layers + 1)
    cog = np.sum(sel * layers) / cog_denominator
    if normalize:
        cog /= num_layers
    return float(cog)


def peak_layer(sel: np.ndarray) -> int:
    '''
    Find the layer with maximum selectivity. The contract includes all
    layers 0..L and chooses the lowest layer index when maxima are tied.

    Parameters
    ----------
    - sel (np.ndarray): Nonempty one-dimensional finite selectivity array
    ordered by layer, beginning with the embedding layer at index 0.
    Invalid shapes or nonfinite values raise ValueError.

    Returns
    -------
    layer (int): Index of the first occurrence of the maximum selectivity.
    '''
    if sel.ndim != 1 or sel.size == 0:
        raise ValueError("sel must be a nonempty one-dimensional array.")
    if not np.all(np.isfinite(sel)):
        raise ValueError("sel must contain only finite values.")
    return int(np.argmax(sel))


def summarize_probing(results: pd.DataFrame) -> pd.DataFrame:
    '''
    Summarize probing results by model, training and evaluation treebanks,
    probe, initialization and pooling. The function pairs upos/control and 
    upos_next/control_next on layer and seed, averages paired accuracies over
    seeds, and computes layer selectivity, CoG and peak statistics.
    Task pairs without the matching control task are skipped with a warning.
    Duplicate or unmatched layer/seed rows, incomplete layer coverage and
    inconsistent group metadata raise ValueError.

    Parameters
    ----------
    - results (pd.DataFrame): Concatenated probing results following
    CONTRACTS.md section 6, with one row per evaluation treebank, task,
    layer and seed for each experiment.

    Returns
    -------
    summary (pd.DataFrame): Output following CONTRACTS.md section 7,
    with one row per experiment group and task pair. Columns are model,
    train_treebank, eval_treebank, lang, probe, init, pooling, task, L, cog,
    cog_norm, peak_layer, peak_tau and max_selectivity.
    '''
    columns = [
        "model", "train_treebank", "eval_treebank", "lang", "probe", "init", "pooling",
        "task", "L", "cog", "cog_norm", "peak_layer", "peak_tau", "max_selectivity"
    ]
    group_columns = ["model", "train_treebank", "eval_treebank", "probe", "init", "pooling"]
    required = group_columns + ["lang", "task", "L", "layer", "seed", "acc"]

    # Case of emty DataFrame
    if results.empty:
        return pd.DataFrame(columns=columns)

    # Case of missing columns in the input DataFrame
    missing = sorted(set(required) - set(results.columns))
    if missing:
        raise ValueError(f"Missing probing results columns: {missing}.")
    
    def are_valid_task_control_rows(task_rows, control_rows, group_key) -> bool:
        if task_rows.empty:
            return False
        if control_rows.empty:
            warnings.warn(f"Skipping upos in group {group_key}: missing control.", UserWarning, stacklevel=2)
            return False
        if task_rows.duplicated(["layer", "seed"]).any() or control_rows.duplicated(["layer", "seed"]).any():
            raise ValueError(f"Duplicate layer/seed rows for upos in group {group_key}.")
        return True
    
    def validate_pairs(paired_rows, group_key, task) -> None:
        # Case of an existing unmatch between task and control rows on seed and layer
        matched_rows = paired_rows["_merge"] == "both"
        if not matched_rows.all():
            raise ValueError(f"Unmatched layer/seed rows for {task} in group {group_key}.")
        # Case of non-finite accuracy values in the paired rows
        if not np.isfinite(paired_rows[["acc_task", "acc_control"]].to_numpy()).all():
            raise ValueError(f"Accuracies must be finite for {task} in group {group_key}.")
        return
    
    def validate_means(means_rows, group_key, task) -> None:
        if not np.array_equal(means_rows.index.to_numpy(), np.arange(num_layers + 1)):
            raise ValueError(f"Expected all layers 0..{num_layers} for {task} in group {group_key}.")
        return

    summarizing_rows = []
    for group_key, results_group in results.groupby(group_columns, sort=False, dropna=False):

        # Case of non-unique values for L (number of layers of the model
        if results_group["L"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent L in group {group_key}.")
        # Case of non-unique values for lang (language of the treebank)
        if results_group["lang"].nunique(dropna=False) != 1:
            raise ValueError(f"Inconsistent lang in group {group_key}.")
        
        num_layers = results_group["L"].iloc[0]

        # Validate both L and all layer indices using the same rules as relative_depth.
        relative_depth(results_group["layer"].to_numpy(), num_layers)

        # Process task/control pairs for current-word and next-word UPOS prediction.
        for task, control in [("upos", "control"), ("upos_next", "control_next")]:

            # Select the rows with that task, filtering on the columns: layer, seed and acc.
            task_rows = results_group.loc[results_group["task"] == task, ["layer", "seed", "acc"]]
            control_rows = results_group.loc[results_group["task"] == control, ["layer", "seed", "acc"]]
            if not are_valid_task_control_rows(task_rows, control_rows, group_key):
                continue

            # Merge together the the task rows with the control rows, having the same layer 
            # and the same seed, before averaging.
            paired_rows = task_rows.merge(
                control_rows,
                on=["layer", "seed"],
                how="outer",
                suffixes=("_task", "_control"),
                indicator=True,
                validate="one_to_one",
            )
            validate_pairs(paired_rows, group_key, task)

            # Compute the average of the accuracy for each layer (and for each seed)
            means_rows = paired_rows.groupby("layer", sort=True)[["acc_task", "acc_control"]].mean()
            validate_means(means_rows, group_key, task)

            # Metrics
            sel = selectivity(means_rows["acc_task"].to_numpy(), means_rows["acc_control"].to_numpy())
            sel_peak_layer = peak_layer(sel)
            sel_peak_tau = relative_depth(sel_peak_layer, num_layers)
            cog = center_of_gravity(sel, normalize=False)
            cog_norm = center_of_gravity(sel)

            summarizing_rows.append({
                **dict(zip(group_columns, group_key)),
                "lang": results_group["lang"].iloc[0],
                "task": task,
                "L": int(num_layers),
                "cog": cog,
                "cog_norm": cog_norm,
                "peak_layer": sel_peak_layer,
                "peak_tau": sel_peak_tau,
                "max_selectivity": float(sel[sel_peak_layer]),
            })

    return pd.DataFrame(summarizing_rows, columns=columns)
