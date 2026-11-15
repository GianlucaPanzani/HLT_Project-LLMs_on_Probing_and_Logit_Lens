"""Exercise linear probes with small, reproducible synthetic datasets."""

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression

from ws1_probing.probes import LinearProbe


@pytest.fixture
def synthetic_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate independent train/dev/test samples for three separable classes.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Float16 feature arrays, like saved features, and integer UPOS indices.
    """
    rng = np.random.default_rng(42)
    centers = np.array([[-4.0, -4.0], [4.0, -4.0], [0.0, 4.0]])
    classes = np.array([0, 7, 15])
    splits = {}
    for split, n_per_class in (("train", 30), ("dev", 10), ("test", 10)):
        X = np.vstack([rng.normal(center, 0.3, size=(n_per_class, 2)) for center in centers])
        y = np.repeat(classes, n_per_class)
        order = rng.permutation(len(y))
        splits[split] = (X[order].astype(np.float16), y[order])
    return splits


@pytest.fixture
def regularization_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build data where strong regularization predicts only the majority class.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        Imbalanced training data and a balanced development set.
    """
    X_train = np.array([[0.0]] * 12 + [[1.0]] * 4, dtype=np.float32)
    y_train = np.array([0] * 12 + [1] * 4)
    X_dev = np.array([[0.0], [1.0]] * 4, dtype=np.float32)
    y_dev = np.array([0, 1] * 4)
    return X_train, y_train, X_dev, y_dev


def test_training_and_prediction(synthetic_splits: dict) -> None:
    """Learn three classes and generalize to independent dev and test samples."""
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    X_test, y_test = synthetic_splits["test"]
    original_splits = {
        split: (X.copy(), y.copy()) for split, (X, y) in synthetic_splits.items()
    }
    probe = LinearProbe()

    assert probe.fit(X_train, y_train, X_dev, y_dev) is probe
    predictions = probe.predict(X_test)

    assert predictions.shape == y_test.shape
    assert np.issubdtype(predictions.dtype, np.integer)
    assert np.mean(predictions == y_test) > 0.95
    assert np.mean(probe.predict(X_dev) == y_dev) > 0.95
    np.testing.assert_array_equal(probe.best_estimator_.classes_, [0, 7, 15])
    for split, (X, y) in synthetic_splits.items():
        np.testing.assert_array_equal(X, original_splits[split][0])
        np.testing.assert_array_equal(y, original_splits[split][1])


def test_all_upos_classes() -> None:
    """Train and predict all 17 contracted class indices on synthetic features.

    Returns
    -------
    None
        Every class must be recognized on independent development and test data.
    """
    rng = np.random.default_rng(42)
    centers = 4.0 * np.eye(17, dtype=np.float32)
    splits = []
    for n_per_class in (8, 2, 2):
        y = np.repeat(np.arange(17), n_per_class)
        X = centers[y] + rng.normal(0.0, 0.1, size=(len(y), 17))
        splits.append((X.astype(np.float32), y))
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = splits

    probe = LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)

    np.testing.assert_array_equal(probe.best_estimator_.classes_, np.arange(17))
    np.testing.assert_array_equal(probe.predict(X_dev), y_dev)
    np.testing.assert_array_equal(probe.predict(X_test), y_test)


@pytest.mark.parametrize("constant_dev_labels,expected_C", [(False, 100.0), (True, 0.001)])
@pytest.mark.parametrize("Cs", [(0.001, 100.0), (100.0, 0.001)])
def test_C_is_selected_using_dev_accuracy(
    regularization_data: tuple, constant_dev_labels: bool, expected_C: float, Cs: tuple[float, ...],
) -> None:
    """Changing only dev labels must change the selected regularization value."""
    X_train, y_train, X_dev, y_dev = regularization_data
    if constant_dev_labels:
        y_dev = np.zeros_like(y_dev)
    probe = LinearProbe(Cs=Cs)
    probe.fit(X_train, y_train, X_dev, y_dev)

    assert probe.hparams == {"C": expected_C}
    assert probe.best_estimator_.C == expected_C
    assert np.mean(probe.predict(X_dev) == y_dev) == 1.0

    # The selected classifier must still be trained only on training examples.
    reference = LogisticRegression(C=expected_C, solver="lbfgs", max_iter=1000, random_state=42)
    reference.fit(X_train, y_train)
    np.testing.assert_allclose(probe.best_estimator_.coef_, reference.coef_)
    np.testing.assert_allclose(probe.best_estimator_.intercept_, reference.intercept_)


@pytest.mark.parametrize("Cs", [(10.0, 1.0), (1.0, 10.0)])
def test_tied_accuracy_keeps_first_candidate(synthetic_splits: dict, Cs: tuple[float, ...]) -> None:
    """Resolve equal development accuracies in the configured candidate order."""
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    # Establish that both candidates really tie, so this tests tie-breaking.
    for C in Cs:
        candidate = LinearProbe(Cs=(C,)).fit(X_train, y_train, X_dev, y_dev)
        assert np.mean(candidate.predict(X_dev) == y_dev) == 1.0
    probe = LinearProbe(Cs=Cs)
    probe.fit(X_train, y_train, X_dev, y_dev)

    assert probe.hparams == {"C": Cs[0]}


@pytest.mark.parametrize("Cs", [(1.0,), (10.0,), (1.0, 10.0)])
def test_zero_dev_accuracy_still_selects_a_model(Cs: tuple[float, ...]) -> None:
    """Save a fitted classifier even when every candidate gets zero accuracy."""
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])
    X_dev = np.array([[-2.0], [2.0]])
    y_dev = np.array([1, 0])
    probe = LinearProbe(Cs=Cs)
    probe.fit(X_train, y_train, X_dev, y_dev)

    assert probe.hparams == {"C": Cs[0]}
    assert np.mean(probe.predict(X_dev) == y_dev) == 0.0


def test_refit_replaces_previous_selection(regularization_data: tuple) -> None:
    """A second fit must select afresh rather than keep the previous winner."""
    X_train, y_train, X_dev, y_dev = regularization_data
    probe = LinearProbe(Cs=(100.0, 0.001))
    probe.fit(X_train, y_train, X_dev, y_dev)
    assert probe.hparams == {"C": 100.0}

    probe.fit(X_train, y_train, X_dev, np.zeros_like(y_dev))
    assert probe.hparams == {"C": 0.001}
    np.testing.assert_array_equal(probe.predict(X_dev), np.zeros_like(y_dev))


def test_reproducible_training(synthetic_splits: dict) -> None:
    """Identical data and seeds produce identical model selections and weights."""
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    first = LinearProbe(seed=42).fit(X_train, y_train, X_dev, y_dev)
    second = LinearProbe(seed=42).fit(X_train, y_train, X_dev, y_dev)

    assert first.hparams == second.hparams
    np.testing.assert_allclose(first.best_estimator_.coef_, second.best_estimator_.coef_)
    np.testing.assert_array_equal(first.predict(X_dev), second.predict(X_dev))


@pytest.mark.parametrize("Cs", [
    (), (0.0,), (-1.0,), (np.nan,), (np.inf,), (-np.inf,), (1.0, 0.0), (1.0, np.nan),
])
def test_invalid_Cs(Cs: tuple[float, ...]) -> None:
    """Reject empty, nonpositive and nonfinite regularization candidates."""
    with pytest.raises(ValueError, match="Cs"):
        LinearProbe(Cs=Cs)


@pytest.mark.parametrize("max_iter", [0, -1, 1.5, True, np.nan])
def test_invalid_max_iter(max_iter: object) -> None:
    """Require a positive integer iteration limit."""
    with pytest.raises(ValueError, match="max_iter"):
        LinearProbe(max_iter=max_iter)


def test_predict_before_fit() -> None:
    """Report a clear error when no classifier has been trained."""
    with pytest.raises(NotFittedError, match="fitted"):
        LinearProbe().predict(np.zeros((2, 3)))


@pytest.mark.parametrize("problem", [
    "nan_train", "inf_dev", "label_count", "dev_label_count", "feature_count",
    "one_class", "empty_train", "empty_dev",
])
def test_invalid_training_data(regularization_data: tuple, problem: str) -> None:
    """Reject unusable data instead of silently dropping or misaligning rows."""
    X_train, y_train, X_dev, y_dev = regularization_data
    if problem == "nan_train":
        X_train[0, 0] = np.nan
    elif problem == "inf_dev":
        X_dev[0, 0] = np.inf
    elif problem == "label_count":
        y_train = y_train[:-1]
    elif problem == "dev_label_count":
        y_dev = y_dev[:-1]
    elif problem == "feature_count":
        X_dev = np.zeros((len(y_dev), 2))
    elif problem == "one_class":
        y_train = np.zeros_like(y_train)
    elif problem == "empty_train":
        X_train, y_train = X_train[:0], y_train[:0]
    elif problem == "empty_dev":
        X_dev, y_dev = X_dev[:0], y_dev[:0]

    with pytest.raises(ValueError):
        LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)


@pytest.mark.parametrize("split", ["train", "dev"])
@pytest.mark.parametrize("value,dtype,message", [
    (-1, np.int64, r"range \[0,16\]"),
    (17, np.int64, r"range \[0,16\]"),
    (0.0, np.float64, "integer labels"),
    (0.5, np.float64, "integer labels"),
    (np.nan, np.float64, "integer labels"),
    (np.inf, np.float64, "integer labels"),
    (True, np.bool_, "integer labels"),
    ("NOUN", str, "integer labels"),
])
def test_invalid_labels(
    regularization_data: tuple, split: str, value: object, dtype: object, message: str,
) -> None:
    """Reject invalid label values and types in either training or development data.

    Parameters
    ----------
    regularization_data : tuple
        Valid features and labels for both splits.
    split : str
        Split whose labels are corrupted.
    value : object
        Invalid value placed at the end of the label array.
    dtype : object
        NumPy dtype used to represent the invalid labels.
    message : str
        Expected validation error pattern.

    Returns
    -------
    None
        Fitting must fail with the label validation error.
    """
    X_train, y_train, X_dev, y_dev = regularization_data
    labels = (y_train if split == "train" else y_dev).astype(dtype)
    labels[-1] = value
    if split == "train":
        y_train = labels
    else:
        y_dev = labels

    with pytest.raises(ValueError, match=message):
        LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)


@pytest.mark.parametrize("input_index", [0, 1, 2, 3], ids=["X_train", "y_train", "X_dev", "y_dev"])
@pytest.mark.parametrize("shape_kind", ["scalar", "wrong_dimensions"])
def test_invalid_fit_dimensions(regularization_data: tuple, input_index: int, shape_kind: str) -> None:
    """Require two-dimensional features and one-dimensional labels in both splits.

    Parameters
    ----------
    regularization_data : tuple
        Valid training and development arrays.
    input_index : int
        Position of the input to replace in the fit arguments.
    shape_kind : str
        Whether to supply a scalar or an array with incorrect dimensions.

    Returns
    -------
    None
        Fitting must report the expected feature or label dimension error.
    """
    inputs = list(regularization_data)
    is_feature = input_index in (0, 2)
    if shape_kind == "scalar":
        inputs[input_index] = np.array(0)
    elif is_feature:
        inputs[input_index] = inputs[input_index].ravel()
    else:
        inputs[input_index] = inputs[input_index].reshape(-1, 1)

    message = "must be 2D arrays" if is_feature else "must be 1D arrays"
    with pytest.raises(ValueError, match=message):
        LinearProbe(Cs=(1.0,)).fit(*inputs)


@pytest.mark.parametrize("X", [
    np.array([[np.nan]]), np.array([[np.inf]]), np.zeros((2, 2)), np.empty((0, 1)), np.zeros(2),
])
def test_invalid_prediction_data(regularization_data: tuple, X: np.ndarray) -> None:
    """Reject nonfinite, empty or incorrectly shaped prediction features.

    Parameters
    ----------
    regularization_data : tuple
        Valid training and development arrays.
    X : np.ndarray
        Malformed prediction features for a classifier trained on one feature.

    Returns
    -------
    None
        Prediction must raise ValueError for each invalid input.
    """
    X_train, y_train, X_dev, y_dev = regularization_data
    probe = LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)
    with pytest.raises(ValueError):
        probe.predict(X)
