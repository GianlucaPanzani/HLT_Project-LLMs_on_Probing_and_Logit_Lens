import warnings

import numpy as np
import pandas as pd
import pytest

from lib.metrics import center_of_gravity, peak_layer, relative_depth, selectivity, summarize_probing

SUMMARY_COLUMNS = [
    "model", "train_treebank", "eval_treebank", "lang", "probe", "init", "pooling",
    "task", "L", "cog", "cog_norm", "peak_layer", "peak_tau", "max_selectivity",
]


def test_selectivity() -> None:
    '''
    Check elementwise subtraction, negative results and protection from broadcasting.

    Returns
    -------
    None: Assertions verify the differences and the shape validation.
    '''
    np.testing.assert_allclose(selectivity(np.array([0.8, 0.4]), np.array([0.3, 0.6])), [0.5, -0.2])
    # Unsigned subtraction must not wrap negative results to a large positive integer.
    np.testing.assert_allclose(selectivity(np.array([0], dtype="uint8"), np.array([1], dtype="uint8")), [-1])
    with pytest.raises(ValueError, match="same shape"):
        selectivity(np.ones((3, 1)), np.ones(3))


def test_relative_depth() -> None:
    '''
    Check embeddings, intermediate layers, final layers and NumPy integer scalars.

    Returns
    -------
    None: Assertions verify scalar and vector normalized depths.
    '''
    assert relative_depth(0, 12) == 0.0
    assert relative_depth(np.int64(6), np.int64(12)) == 0.5
    np.testing.assert_allclose(relative_depth(np.array([0, 6, 12]), 12), [0, 0.5, 1])


@pytest.mark.parametrize("num_layers,error", [(0, ValueError), (-1, ValueError), (12.5, TypeError), (True, TypeError)])
def test_invalid_num_layers(num_layers, error) -> None:
    '''
    Reject block counts that are not positive integers.

    Parameters
    ----------
    - num_layers (object): Invalid block count.
    - error (type[Exception]): Expected exception class.

    Returns
    -------
    None: The test fails unless the specified exception is raised.
    '''
    with pytest.raises(error):
        relative_depth(0, num_layers)


@pytest.mark.parametrize("layer,error", [
    (-1, ValueError), (13, ValueError), (np.array([0, 13]), ValueError),
    (1.5, TypeError), (np.array([0.0, 1.5]), TypeError), (True, TypeError),
])
def test_invalid_layer(layer, error) -> None:
    '''
    Reject noninteger layer indices and indices outside the model depth.

    Parameters
    ----------
    - layer (object): Invalid scalar or array of layer indices.
    - error (type[Exception]): Expected exception class.

    Returns
    -------
    None: The test fails unless invalid indices are rejected.
    '''
    with pytest.raises(error):
        relative_depth(layer, 12)


@pytest.mark.parametrize("values,clip,expected", [
    ([0.0, 0.1, 0.2, 0.3], True, 7 / 3),
    ([0.5, -0.2, 0.4], True, 2.0),
    ([0.5, -0.2, 0.4], False, 3.0),
    ([100.0, 0.3], True, 1.0),
])
def test_center_of_gravity(values: list[float], clip: bool, expected: float) -> None:
    '''
    Check P1 examples, normalization, embedding exclusion and input preservation.

    Parameters
    ----------
    - values (list[float]): Synthetic selectivity curve including layer 0.
    - clip (bool): Whether negative selectivity should be clipped.
    - expected (float): Hand-calculated unnormalized center of gravity.

    Returns
    -------
    None: Assertions compare both CoG variants against analytical values.
    '''
    sel = np.array(values)
    before = sel.copy()
    assert center_of_gravity(sel, normalize=False, clip_negative=clip) == pytest.approx(expected)
    assert center_of_gravity(sel, clip_negative=clip) == pytest.approx(expected / (len(sel) - 1))
    np.testing.assert_array_equal(sel, before)


@pytest.mark.parametrize("values,clip", [([0, 0, 0], True), ([0.5, -0.2, -0.4], True), ([0, -1, 1], False)])
def test_zero_weight_returns_nan(values: list[float], clip: bool) -> None:
    '''
    Check that zero denominators return NaN without division warnings.

    Parameters
    ----------
    - values (list[float]): Selectivity curve with zero total usable weight.
    - clip (bool): Whether clipping is applied before checking the denominator.

    Returns
    -------
    None: Assertions require NaN and treat any warning as a failure.
    '''
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert np.isnan(center_of_gravity(np.array(values), clip_negative=clip))


@pytest.mark.parametrize("values,expected", [([0.9, 0.2, 0.1], 0), ([0.4, 0.4, 0.2], 0), ([0.1, 0.4, 0.4], 1), ([0.3], 0)])
def test_peak_layer(values: list[float], expected: int) -> None:
    '''
    Check embedding peaks and the lowest-index rule for equal maxima.

    Parameters
    ----------
    - values (list[float]): Synthetic selectivity curve.
    - expected (int): Expected peak index including layer 0.

    Returns
    -------
    None: Assertions verify the peak index and Python integer return type.
    '''
    result = peak_layer(np.array(values))
    assert isinstance(result, int)
    assert result == expected


@pytest.mark.parametrize("values", [np.array(1), np.ones((2, 2)), np.array([]), np.array([0, np.nan]), np.array([0, np.inf])])
def test_invalid_selectivity_curve(values: np.ndarray) -> None:
    '''
    Reject invalid dimensions, empty curves and nonfinite selectivity values.

    Parameters
    ----------
    - values (np.ndarray): Invalid selectivity input.

    Returns
    -------
    None: Both curve metrics must raise clear errors.
    '''
    with pytest.raises(ValueError):
        center_of_gravity(values)
    with pytest.raises(ValueError):
        peak_layer(values)


def test_cog_requires_transformer_block() -> None:
    '''
    Reject a CoG curve containing only an embedding value.

    Returns
    -------
    None: A curve without Transformer blocks must raise ValueError.
    '''
    with pytest.raises(ValueError):
        center_of_gravity(np.array([0.5]))


@pytest.fixture
def probing_results() -> pd.DataFrame:
    '''
    Build two synthetic experiments with two seeds and both task/control pairs.

    Returns
    -------
    results (pd.DataFrame): Small input in the probing-results contract format.
    '''
    rows = []
    for model, curve in (("bert", [0.0, 0.1, 0.2, 0.3]), ("gpt2", [0.5, -0.2, 0.4])):
        num_layers = len(curve) - 1
        for task, control, values in (("upos", "control", curve), ("upos_next", "control_next", curve[::-1])):
            for seed, offset in ((42, -0.02), (43, 0.02)):
                for layer, value in enumerate(values):
                    # Seed effects cancel when averaged; reversing row order tests alignment.
                    control_acc = 0.3 + offset
                    task_acc = control_acc + value + offset * (layer + 1)
                    for label, accuracy in ((task, task_acc), (control, control_acc)):
                        rows.append({
                            "model": model, "train_treebank": "ewt", "eval_treebank": "ewt",
                            "lang": "en", "kind": "encoder" if model == "bert" else "decoder",
                            "probe": "linear", "task": label, "init": "pretrained", "pooling": "first",
                            "layer": layer, "L": num_layers, "tau": layer / num_layers,
                            "acc": accuracy, "n_train": 100, "n_eval": 20, "hparams": "{}", "seed": seed,
                        })
    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


def test_summarize_probing(probing_results: pd.DataFrame) -> None:
    '''
    Verify the summary schema and hand-calculated metrics for both task pairs.

    Parameters
    ----------
    - probing_results (pd.DataFrame): Unsorted synthetic probing observations.

    Returns
    -------
    None: Assertions verify grouping, seed averaging and unchanged input data.
    '''
    before = probing_results.copy(deep=True)
    summary = summarize_probing(probing_results)
    assert list(summary.columns) == SUMMARY_COLUMNS
    assert len(summary) == 4
    expected = [
        ("bert", "upos", 7 / 3, 7 / 9, 3, 1.0, 0.3),
        ("bert", "upos_next", 4 / 3, 4 / 9, 0, 0.0, 0.3),
        ("gpt2", "upos", 2.0, 1.0, 0, 0.0, 0.5),
        ("gpt2", "upos_next", 2.0, 1.0, 2, 1.0, 0.5),
    ]
    for model, task, cog, norm, peak, tau, maximum in expected:
        row = summary.loc[(summary["model"] == model) & (summary["task"] == task)].iloc[0]
        assert row["cog"] == pytest.approx(cog)
        assert row["cog_norm"] == pytest.approx(norm)
        assert row["peak_layer"] == peak
        assert row["peak_tau"] == pytest.approx(tau)
        assert row["max_selectivity"] == pytest.approx(maximum)
    pd.testing.assert_frame_equal(probing_results, before)


@pytest.mark.parametrize("column,value", [
    ("train_treebank", "isdt"), ("eval_treebank", "pud_en"), ("probe", "mlp"),
    ("init", "random"), ("pooling", "last"),
])
def test_summary_separates_experiments(probing_results: pd.DataFrame, column: str, value: str) -> None:
    '''
    Ensure experiments with different grouping fields are summarized independently.

    Parameters
    ----------
    - probing_results (pd.DataFrame): Original experiment observations.
    - column (str): Grouping field to change in a second set of observations.
    - value (str): Distinct value identifying the second experiment set.

    Returns
    -------
    None: Assertions require separate summary rows for both experiment sets.
    '''
    extra = probing_results.copy()
    extra[column] = value
    summary = summarize_probing(pd.concat([probing_results, extra], ignore_index=True))
    assert len(summary) == 8
    assert (summary[column] == value).sum() == 4


def test_missing_control_warns(probing_results: pd.DataFrame) -> None:
    '''
    Skip only the affected task pair when its entire control task is missing.

    Parameters
    ----------
    - probing_results (pd.DataFrame): Synthetic results from two models.

    Returns
    -------
    None: Assertions require a warning and preservation of the other summaries.
    '''
    keep = ~((probing_results["model"] == "bert") & (probing_results["task"] == "control"))
    with pytest.warns(UserWarning, match="missing control"):
        summary = summarize_probing(probing_results.loc[keep])
    assert len(summary) == 3
    assert not ((summary["model"] == "bert") & (summary["task"] == "upos")).any()


@pytest.mark.parametrize("problem,match", [
    ("duplicate", "Duplicate"), ("unmatched", "Unmatched"), ("missing_layer", "Expected all layers"),
    ("metadata", "Inconsistent"), ("nonfinite", "finite"), ("missing_column", "Missing probing"),
])
def test_invalid_summary_input(probing_results: pd.DataFrame, problem: str, match: str) -> None:
    '''
    Reject malformed results instead of silently reporting misleading statistics.

    Parameters
    ----------
    - probing_results (pd.DataFrame): Valid synthetic observations to modify.
    - problem (str): Invalid-input scenario to construct.
    - match (str): Expected fragment of the ValueError message.

    Returns
    -------
    None: Every malformed input must raise the expected error.
    '''
    results = probing_results.copy()
    if problem == "duplicate":
        results = pd.concat([results, results.iloc[[0]]], ignore_index=True)
    elif problem == "unmatched":
        results = results.iloc[1:]
    elif problem == "missing_layer":
        results = results.loc[results["layer"] != 1]
    elif problem == "metadata":
        results.loc[0, "L"] = 99
    elif problem == "nonfinite":
        results.loc[0, "acc"] = np.nan
    elif problem == "missing_column":
        results = results.drop(columns="acc")
    with pytest.raises(ValueError, match=match):
        summarize_probing(results)


def test_empty_summary() -> None:
    '''
    Preserve the complete output schema when no probing results are available.

    Returns
    -------
    None: Assertions require an empty DataFrame with the contract columns.
    '''
    summary = summarize_probing(pd.DataFrame())
    assert summary.empty
    assert list(summary.columns) == SUMMARY_COLUMNS
