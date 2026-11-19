"""Check P2 and P3 requirements with small synthetic datasets.

Run the separate fit benchmark with:
PYTHONPATH=src python -m pytest -q -s tests/tests_probing_fit_benchmark.py
"""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# probes imports torch directly, so skip before importing it when unavailable.
torch = pytest.importorskip("torch")

from ws1_probing.probes import LinearProbe, MLPProbe  # noqa: E402


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


@pytest.mark.parametrize("classes", [
    np.arange(17),
    np.array([tag for tag in range(17) if tag not in (6, 9)]),
    np.array([0, 7, 16]),
    np.array([7, 16]),
], ids=["all_tags", "without_intj_part", "nonconsecutive_tags", "two_tags_without_zero"])
def test_present_upos_classes(classes: np.ndarray) -> None:
    """Learn present tags without renumbering them or requiring all 17 classes.

    Parameters
    ----------
    classes : np.ndarray
        Global UPOS indices present in the synthetic dataset.

    Returns
    -------
    None
        Predictions preserve global indices and perfect accuracy is one even
        when some tags are absent from the dataset.
    """
    rng = np.random.default_rng(42)
    centers = 4.0 * np.eye(17, dtype=np.float32)
    splits = []
    for n_per_class in (8, 2, 2):
        y = np.repeat(classes, n_per_class)
        X = centers[y] + rng.normal(0.0, 0.1, size=(len(y), 17))
        splits.append((X.astype(np.float32), y))
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = splits

    probe = LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)

    np.testing.assert_array_equal(probe.best_estimator_.classes_, classes)
    np.testing.assert_array_equal(probe.predict(X_dev), y_dev)
    np.testing.assert_array_equal(probe.predict(X_test), y_test)
    assert np.mean(probe.predict(X_dev) == y_dev) == 1.0
    assert np.mean(probe.predict(X_test) == y_test) == 1.0


@pytest.mark.parametrize("y_dev,expected_accuracy", [
    (np.array([0, 16, 7]), 2 / 3),
    (np.array([7, 7, 7]), 0.0),
], ids=["one_unseen_tag", "only_unseen_tag"])
def test_unseen_dev_tags_are_counted_as_errors(
    monkeypatch: pytest.MonkeyPatch, y_dev: np.ndarray, expected_accuracy: float,
) -> None:
    """Keep unseen dev tags in every candidate's accuracy calculation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Records calls to the real classifier score method without replacing
        training or the accuracy calculation.
    y_dev : np.ndarray
        Valid global labels including tag 7, absent from training.
    expected_accuracy : float
        Accuracy over all three development words.

    Returns
    -------
    None
        Each candidate scores all dev rows and retains only training classes.
    """
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
    y_train = np.array([0, 0, 16, 16])
    X_dev = np.array([[-3.0], [3.0], [0.0]], dtype=np.float32)
    original_y_dev = y_dev.copy()
    expected_dev = StandardScaler().fit(X_train).transform(X_dev)
    original_score = LogisticRegression.score
    evaluations = []

    def record_score(model, X, y):
        accuracy = original_score(model, X, y)
        evaluations.append((X.copy(), y.copy(), accuracy))
        return accuracy

    monkeypatch.setattr(LogisticRegression, "score", record_score)
    probe = LinearProbe(Cs=(0.1, 1.0)).fit(X_train, y_train, X_dev, y_dev)

    assert len(evaluations) == 2
    for evaluated_X, evaluated_y, accuracy in evaluations:
        np.testing.assert_allclose(evaluated_X, expected_dev)
        np.testing.assert_array_equal(evaluated_y, original_y_dev)
        assert accuracy == pytest.approx(expected_accuracy)
    predictions = probe.predict(X_dev)
    assert predictions.shape == y_dev.shape
    assert np.mean(predictions == y_dev) == pytest.approx(expected_accuracy)
    np.testing.assert_array_equal(probe.best_estimator_.classes_, [0, 16])
    np.testing.assert_array_equal(y_dev, original_y_dev)


@pytest.mark.parametrize("tag", [0, 7, 16])
def test_single_training_class_is_rejected(regularization_data: tuple, tag: int) -> None:
    """Require at least two training classes, even when dev has other tags.

    Parameters
    ----------
    regularization_data : tuple
        Valid features and development labels.
    tag : int
        The sole training tag, still within the valid global range.

    Returns
    -------
    None
        A one-class training set must raise ValueError.
    """
    X_train, y_train, X_dev, y_dev = regularization_data
    with pytest.raises(ValueError, match="at least 2 classes"):
        LinearProbe(Cs=(1.0,)).fit(X_train, np.full_like(y_train, tag), X_dev, y_dev)


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

    assert probe.hparams == {"C": expected_C, "converged": True}
    assert probe.best_estimator_.C == expected_C
    assert np.mean(probe.predict(X_dev) == y_dev) == 1.0

    # The selected classifier must still be trained only on training examples.
    reference = LogisticRegression(C=expected_C, solver="lbfgs", max_iter=1000, random_state=42)
    reference.fit(StandardScaler().fit_transform(X_train), y_train)
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

    assert probe.hparams["C"] == Cs[0]


@pytest.mark.parametrize("Cs", [(1.0,), (10.0,), (1.0, 10.0)])
def test_zero_dev_accuracy_still_selects_a_model(Cs: tuple[float, ...]) -> None:
    """Save a fitted classifier even when every candidate gets zero accuracy."""
    X_train = np.array([[-2.0], [-1.0], [1.0], [2.0]])
    y_train = np.array([0, 0, 1, 1])
    X_dev = np.array([[-2.0], [2.0]])
    y_dev = np.array([1, 0])
    probe = LinearProbe(Cs=Cs)
    probe.fit(X_train, y_train, X_dev, y_dev)

    assert probe.hparams["C"] == Cs[0]
    assert np.mean(probe.predict(X_dev) == y_dev) == 0.0


def test_refit_replaces_previous_selection(regularization_data: tuple) -> None:
    """A second fit must select afresh rather than keep the previous winner."""
    X_train, y_train, X_dev, y_dev = regularization_data
    probe = LinearProbe(Cs=(100.0, 0.001))
    probe.fit(X_train, y_train, X_dev, y_dev)
    assert probe.hparams["C"] == 100.0

    probe.fit(X_train, y_train, X_dev, np.zeros_like(y_dev))
    assert probe.hparams["C"] == 0.001
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


def test_scaler_fits_train_only_and_transforms_predictions(
    synthetic_splits: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train-only scaling and reuse of the fitted transform at prediction.

    Parameters
    ----------
    synthetic_splits : dict
        Independent synthetic training, development and test samples.
    monkeypatch : pytest.MonkeyPatch
        Records the real StandardScaler.fit calls.

    Returns
    -------
    None
        Scaling must use only training data and match a reference classifier.
    """
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    X_test, _ = synthetic_splits["test"]
    scale = np.array([0.1, 100.0], dtype=np.float32)
    offset = np.array([10.0, -200.0], dtype=np.float32)
    X_train = X_train.astype(np.float32) * scale + offset
    X_dev = X_dev.astype(np.float32) * scale + offset + scale * 2
    X_test = X_test.astype(np.float32) * scale + offset - scale

    reference_scaler = StandardScaler().fit(X_train)
    reference_model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, random_state=42)
    reference_model.fit(reference_scaler.transform(X_train), y_train)
    original_fit = StandardScaler.fit
    fitted_inputs = []

    def record_fit(scaler, X, y=None, **kwargs):
        fitted_inputs.append(np.asarray(X).copy())
        return original_fit(scaler, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    probe = LinearProbe(Cs=(1.0,)).fit(X_train, y_train, X_dev, y_dev)
    assert fitted_inputs, "fit must learn a StandardScaler from the training features."
    for fitted_X in fitted_inputs:
        np.testing.assert_array_equal(fitted_X, X_train)
    fit_count = len(fitted_inputs)
    for X in (X_dev, X_test):
        np.testing.assert_array_equal(probe.predict(X), reference_model.predict(reference_scaler.transform(X)))
    assert len(fitted_inputs) == fit_count, "predict must reuse the scaler without fitting again."
    np.testing.assert_allclose(probe.best_estimator_.coef_, reference_model.coef_)
    np.testing.assert_allclose(probe.best_estimator_.intercept_, reference_model.intercept_)


def test_shuffled_labels_do_not_generalize() -> None:
    """Check the issue's majority-rate bound on independent shuffled-label data.

    Returns
    -------
    None
        Test accuracy must not exceed the majority-class rate plus 0.05.
    """
    rng = np.random.default_rng(42)
    splits = []
    centers = 4.0 * np.eye(3, dtype=np.float32)
    for n_per_class in (300, 200, 1000):
        groups = np.repeat(np.arange(3), n_per_class)
        X = centers[groups] + rng.normal(0.0, 0.3, size=(len(groups), 3))
        # Shuffle independently in each split to remove the feature/label relation.
        y = rng.permutation(np.array([0, 7, 16])[groups])
        splits.append((X.astype(np.float32), y))
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = splits
    probe = LinearProbe().fit(X_train, y_train, X_dev, y_dev)
    test_accuracy = np.mean(probe.predict(X_test) == y_test)
    majority_rate = np.bincount(y_test, minlength=17).max() / len(y_test)
    assert test_accuracy <= majority_rate + 0.05


def test_nonconvergence_is_recorded_without_warning(synthetic_splits: dict) -> None:
    """Record a real iteration-limit warning in the selected hyperparameters.

    Parameters
    ----------
    synthetic_splits : dict
        Separable multiclass data that needs more than one optimizer iteration.

    Returns
    -------
    None
        A fitted but unconverged classifier is usable and its warning is captured.
    """
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = LinearProbe(Cs=(1.0,), max_iter=1).fit(X_train, y_train, X_dev, y_dev)
    assert probe.hparams.get("converged") is False
    assert not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)
    assert probe.predict(X_dev).shape == y_dev.shape


@pytest.mark.parametrize("warning_C", [0.001, 100.0])
def test_convergence_status_belongs_to_selected_candidate(
    regularization_data: tuple, monkeypatch: pytest.MonkeyPatch, warning_C: float,
) -> None:
    """Keep convergence metadata for the winner, including when it is not last.

    Parameters
    ----------
    regularization_data : tuple
        Data where C=100 has strictly better development accuracy.
    monkeypatch : pytest.MonkeyPatch
        Adds a controlled warning after real classifier training.
    warning_C : float
        Candidate that emits a ConvergenceWarning.

    Returns
    -------
    None
        The selected candidate determines the flag and warnings do not escape.
    """
    X_train, y_train, X_dev, y_dev = regularization_data
    original_fit = LogisticRegression.fit

    def fit_with_warning(model, X, y, **kwargs):
        result = original_fit(model, X, y, **kwargs)
        if model.C == warning_C:
            warnings.warn("Controlled convergence warning for the test.", ConvergenceWarning, stacklevel=2)
        return result

    monkeypatch.setattr(LogisticRegression, "fit", fit_with_warning)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        probe = LinearProbe(Cs=(100.0, 0.001)).fit(X_train, y_train, X_dev, y_dev)
    assert probe.hparams == {"C": 100.0, "converged": warning_C != 100.0}
    assert not any(issubclass(warning.category, ConvergenceWarning) for warning in caught)


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
    "empty_train", "empty_dev",
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


@pytest.fixture
def mlp_cpu():
    """Limit CPU threads for tiny networks and restore the previous setting.

    Returns
    -------
    Iterator[None]
        A test context using one CPU thread without changing other tests.
    """
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture
def xor_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Generate balanced XOR clusters with independent train and dev samples.

    Returns
    -------
    dict[str, tuple[np.ndarray, np.ndarray]]
        Float32 features and global UPOS labels 7 and 16. Reflections keep
        each class mean at zero so a linear boundary cannot solve the task.
    """
    rng = np.random.default_rng(42)
    splits = {}
    for split, count in (("train", 512), ("dev", 128)):
        magnitudes = rng.uniform(0.8, 1.2, size=(count, 2))
        X = np.vstack([magnitudes * signs for signs in ((1, 1), (-1, -1), (1, -1), (-1, 1))])
        y = np.repeat([7, 7, 16, 16], count)
        order = rng.permutation(len(y))
        splits[split] = (X[order].astype(np.float32), y[order])
    return splits


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_learns_xor_while_linear_stays_near_chance(xor_splits: dict) -> None:
    """Verify the P3 nonlinear accuracy criterion using default hyperparameters."""
    X_train, y_train = xor_splits["train"]
    X_dev, y_dev = xor_splits["dev"]
    probe = MLPProbe(device="cpu")
    assert probe.fit(X_train, y_train, X_dev, y_dev) is probe
    linear = LinearProbe().fit(X_train, y_train, X_dev, y_dev)
    predictions = probe.predict(X_dev)

    assert predictions.shape == y_dev.shape
    assert np.issubdtype(predictions.dtype, np.integer)
    assert np.mean(predictions == y_dev) > 0.9
    assert abs(np.mean(linear.predict(X_dev) == y_dev) - 0.5) <= 0.1
    assert {"hidden", "lr", "epochs_trained", "best_dev_acc"} <= probe.hparams.keys()
    assert probe.hparams["hidden"] == 256
    assert probe.hparams["lr"] == 1e-3
    assert 1 <= probe.hparams["epochs_trained"] <= probe.max_epochs
    assert np.mean(predictions == y_dev) == pytest.approx(probe.hparams["best_dev_acc"])


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_shuffled_labels_do_not_generalize() -> None:
    """Check the same majority-rate bound as P2 on independent shuffled labels."""
    rng = np.random.default_rng(42)
    splits = []
    centers = 4.0 * np.eye(3, dtype=np.float32)
    for n_per_class in (300, 200, 1000):
        groups = np.repeat(np.arange(3), n_per_class)
        X = centers[groups] + rng.normal(0.0, 0.3, size=(len(groups), 3))
        y = rng.permutation(np.array([0, 7, 16])[groups])
        splits.append((X.astype(np.float32), y))
    (X_train, y_train), (X_dev, y_dev), (X_test, y_test) = splits
    probe = MLPProbe(device="cpu").fit(X_train, y_train, X_dev, y_dev)

    accuracy = np.mean(probe.predict(X_test) == y_test)
    majority_rate = np.bincount(y_test, minlength=17).max() / len(y_test)
    assert accuracy <= majority_rate + 0.05


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_stops_early_on_noisy_labels() -> None:
    """Stop before max_epochs with patience=1 and labels independent of features."""
    rng = np.random.default_rng(42)
    X_train = rng.normal(size=(256, 4)).astype(np.float32)
    X_dev = rng.normal(size=(128, 4)).astype(np.float32)
    y_train = rng.integers(0, 17, size=len(X_train))
    y_dev = rng.integers(0, 17, size=len(X_dev))
    probe = MLPProbe(patience=1, device="cpu").fit(X_train, y_train, X_dev, y_dev)

    assert 1 < probe.hparams["epochs_trained"] < probe.max_epochs
    assert np.mean(probe.predict(X_dev) == y_dev) == pytest.approx(probe.hparams["best_dev_acc"])


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_restores_best_weights_and_counts_global_improvements(
    synthetic_splits: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Restore an independent checkpoint even when accuracy recovers below its best.

    Parameters
    ----------
    synthetic_splits : dict
        Small feature arrays used for real gradient updates.
    monkeypatch : pytest.MonkeyPatch
        Controls dev predictions to produce accuracies 0.8, 0.4, 0.6.

    Returns
    -------
    None
        Patience=2 stops after three epochs and restores epoch one's weights.
    """
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    probe = MLPProbe(hidden=16, patience=2, max_epochs=10, device="cpu")
    states = []

    def controlled_predictions(X_scaled):
        states.append({name: value.detach().clone() for name, value in probe.model_.state_dict().items()})
        correct = (24, 12, 18)[min(len(states) - 1, 2)]
        predictions = (y_dev + 1) % 17
        predictions[:correct] = y_dev[:correct]
        return predictions

    monkeypatch.setattr(probe, "_predict_scaled", controlled_predictions)
    probe.fit(X_train, y_train, X_dev, y_dev)

    assert probe.hparams["epochs_trained"] == 3
    assert probe.hparams["best_dev_acc"] == pytest.approx(0.8)
    assert any(not torch.equal(states[0][name], states[-1][name]) for name in states[0])
    for name, value in probe.model_.state_dict().items():
        torch.testing.assert_close(value, states[0][name], rtol=0, atol=0)


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_reproducible_training(synthetic_splits: dict) -> None:
    """Identical seeds reproduce CPU training with shuffled mini-batches."""
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    first = MLPProbe(hidden=16, batch_size=17, max_epochs=4, seed=42, device="cpu")
    second = MLPProbe(hidden=16, batch_size=17, max_epochs=4, seed=42, device="cpu")
    first.fit(X_train, y_train, X_dev, y_dev)
    second.fit(X_train, y_train, X_dev, y_dev)

    assert first.hparams == second.hparams
    for name, value in first.model_.state_dict().items():
        torch.testing.assert_close(value, second.model_.state_dict()[name], rtol=0, atol=0)
    np.testing.assert_array_equal(first.predict(X_dev), second.predict(X_dev))


@pytest.mark.usefixtures("mlp_cpu")
def test_mlp_train_only_scaling_and_batched_prediction(
    synthetic_splits: dict, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify train-only statistics and prediction equivalence across batch sizes.

    Parameters
    ----------
    synthetic_splits : dict
        Independent training, development and test arrays.
    monkeypatch : pytest.MonkeyPatch
        Records the real scaler fits to detect development or prediction leakage.

    Returns
    -------
    None
        Only train statistics are fitted; partial batches preserve predictions.
    """
    X_train, y_train = synthetic_splits["train"]
    X_dev, y_dev = synthetic_splits["dev"]
    X_test, _ = synthetic_splits["test"]
    X_train = X_train.astype(np.float32) * [0.1, 100.0] + [10.0, -200.0]
    X_dev = X_dev.astype(np.float32) + [20.0, 50.0]
    original_train = X_train.copy()
    original_fit = StandardScaler.fit
    fitted_inputs = []

    def record_fit(scaler, X, y=None, **kwargs):
        fitted_inputs.append(X.copy())
        return original_fit(scaler, X, y, **kwargs)

    monkeypatch.setattr(StandardScaler, "fit", record_fit)
    probe = MLPProbe(hidden=16, batch_size=17, max_epochs=2, patience=5, device="cpu")
    probe.fit(X_train, y_train, X_dev, y_dev)
    assert probe.hparams["epochs_trained"] == 2
    predictions = probe.predict(X_test)
    probe.batch_size = 100
    np.testing.assert_array_equal(probe.predict(X_test), predictions)
    with torch.inference_mode():
        scaled = probe.scaler.transform(X_test.astype(np.float32))
        expected = probe.model_(torch.from_numpy(scaled)).argmax(dim=1).numpy()
    np.testing.assert_array_equal(predictions, expected)
    assert len(fitted_inputs) == 1
    np.testing.assert_array_equal(fitted_inputs[0], X_train.astype(np.float32))
    np.testing.assert_array_equal(X_train, original_train)
