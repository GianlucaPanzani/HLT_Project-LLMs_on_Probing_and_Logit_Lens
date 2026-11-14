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
    return acc_task - acc_control


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

    if num_layers <= 0:
        raise ValueError(f"Variable \"num_layers\" should be grater than 0.")
    
    if not isinstance(layer, (int, np.ndarray, np.integer)):
        raise TypeError(f"Variable \"layer\" should be an integer or a NumPy array.")
    
    if isinstance(layer, np.ndarray):
        if not np.issubdtype(layer.dtype, np.integer):
            raise TypeError("Variable \"layer\" must contain integer values.")
        
    if np.any(layer <= 0) or np.any(layer > num_layers):  # common check for both int and NumPy arrays
        raise ValueError("Each layer has to satisfy the condition: 1 <= layer <= num_layers.")

    return layer / num_layers


def center_of_gravity(sel: np.ndarray, *, normalize: bool = True, clip_negative: bool = True) -> float:
    '''
    Compute the selectivity-weighted mean layer, excluding the embedding layer.
    The contract defines CoG as sum(l * sel[l]) / sum(sel[l]) over layers 1..L,
    optionally divided by L for normalized depth.

    Parameters
    ----------
    - sel (np.ndarray): One-dimensional selectivity array with L + 1 entries
    for layers 0..L. The entry at index 0 is excluded from the calculation.
    - normalize (bool): Whether to divide CoG by L. Defaults to True.
    - clip_negative (bool): Whether to replace negative selectivity values
    with zero before calculating CoG. Defaults to True.

    Returns
    -------
    cog (float): Selectivity-weighted layer depth, normalized when requested,
    or NaN if the sum of the included selectivities is zero after optional
    clipping.
    '''
    if sel.ndim == 0:
        raise ValueError("Input array must have at least one dimension.")

    # The clip function pushes the negative values to 0
    sel = sel[1:] if not clip_negative else np.clip(sel[1:], 0, None)

    # Denominator of the Center of Gravity (CoG)
    cog_denominator = np.sum(sel)
    if cog_denominator == 0:
        return np.nan  # Return NaN to avoid division by zero
    
    # Case of normalization
    if normalize:
        sel = sel / cog_denominator

    # Numerator of the Center of Gravity (CoG)
    cog_numerator = np.sum(sel * np.arange(1, len(sel)))

    return cog_numerator / cog_denominator


def peak_layer(sel: np.ndarray) -> int:
    '''
    Find the layer with maximum selectivity. The contract includes all
    layers 0..L and chooses the lowest layer index when maxima are tied.

    Parameters
    ----------
    - sel (np.ndarray): Nonempty one-dimensional selectivity array with
    entries ordered by layer, beginning with the embedding layer at index 0.

    Returns
    -------
    layer (int): Index of the first occurrence of the maximum selectivity.
    '''
    if sel.ndim == 0:
        raise ValueError("Input array must have at least one dimension.")
    return int(np.argmax(sel[1:])) + 1  # +1 because we skip layer 0


def summarize_probing(results: pd.DataFrame) -> pd.DataFrame:
    '''
    Summarize probing results by model, training and evaluation treebanks,
    probe, initialization and pooling. The function pairs upos/control and 
    upos_next/control_next on layer and seed, averages paired accuracies over
    seeds, and computes layer selectivity, CoG and peak statistics.
    Groups without the matching control task are skipped with a warning.

    Parameters
    ----------
    - results (pd.DataFrame): Concatenated probing results following
    CONTRACTS.md section 6, with one row per evaluation treebank, task,
    layer and seed for each experiment.

    Returns
    -------
    summary (pd.DataFrame): Planned output following CONTRACTS.md section 7,
    with one row per experiment group and task pair. Columns are model,
    train_treebank, eval_treebank, lang, probe, init, pooling, task, L, cog,
    cog_norm, peak_layer, peak_tau and max_selectivity.
    '''
    """summary = {
        "model": None,
        "train_treebank": None,
        "eval_treebank": None,
        "lang": None,
        "probe": None,
        "init": None,
        "pooling": None,
        "task": None,
        "L": None,
        "cog": None,
        "cog_norm": None,
        "peak_layer": None,
        "peak_tau": None,
        "max_selectivity": None
    }"""
    summary = results.copy()

    summary["selectivity"] = summary.apply(
        lambda x: selectivity(x["acc_task"].values, x["acc_control"].values)
    )
    summary["cog"] = summary.apply(
        lambda x: center_of_gravity(x["selectivity"].values, normalize=False)
    )
    summary["cog_norm"] = summary.apply(
        lambda x: center_of_gravity(x["selectivity"].values, normalize=True)
    )
    summary["peak_layer"] = summary.apply(
        lambda x: peak_layer(x["selectivity"].values)
    )
    summary["peak_tau"] = summary.apply(
        lambda x: float(relative_depth(x["peak_layer"], x["L"]))
    )
    summary["max_selectivity"] = summary.apply(
        lambda x: x["selectivity"].max()
    )

    return summary[[
        "model", "train_treebank", "eval_treebank", "lang", "probe", "init", "pooling",
        "task", "L", "cog", "cog_norm", "peak_layer", "peak_tau", "max_selectivity"
    ]]
