from typing import Protocol

import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression


class Probe(Protocol):
    def fit(
        self, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray,
    ) -> "Probe": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...
    hparams: dict


class LinearProbe(Probe):
    """L2-regularized multinomial probe, with C selected on development data."""

    def __init__(
        self,
        Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        max_iter: int = 1000,
        seed: int = 42,
    ) -> None:
        """
        Configure the probe without training a classifier.

        Parameters
        ----------
        Cs : tuple[float, ...]
            Positive, finite regularization candidates, evaluated in this order.
        max_iter : int
            Positive maximum number of optimization iterations per candidate.
        seed : int
            Random state passed to the classifier.

        Returns
        -------
        None
            The probe is configured and must be fitted before prediction.
        """
        if len(Cs) == 0:
            raise ValueError("Cs must contain at least one candidate value.")
        if any(not np.isfinite(C) or C <= 0 for C in Cs):
            raise ValueError("Cs must contain only finite values greater than zero.")
        if isinstance(max_iter, (bool, np.bool_)) or not isinstance(max_iter, (int, np.integer)) or max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        self.Cs = Cs
        self.max_iter = max_iter
        self.seed = seed
        self.hparams: dict = {}
    
    def _validate_fit_inputs(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> None:
        """
        Validate the input shapes and types for the fit method.

        Raises
        ------
        ValueError
            If any of the inputs are not 2D arrays or if the number of samples does not match.
        """
        if X_train.ndim != 2 or X_dev.ndim != 2:
            raise ValueError("X_train and X_dev must be 2D arrays.")
        if y_train.ndim != 1 or y_dev.ndim != 1:
            raise ValueError("y_train and y_dev must be 1D arrays.")
        if X_train.shape[0] != y_train.shape[0]:
            raise ValueError("Number of samples in X_train and y_train must match.")
        if X_dev.shape[0] != y_dev.shape[0]:
            raise ValueError("Number of samples in X_dev and y_dev must match.")
        for labels in [y_train, y_dev]:
            labels = np.asarray(labels)
            if not np.issubdtype(labels.dtype, np.integer):
                raise ValueError("y_train and y_dev must contain integer labels.")
            if np.any((labels < 0) | (labels > 16)):
                raise ValueError("y_train and y_dev must contain integer labels in range [0,16].")

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> "LinearProbe":
        """
        Train candidates on training data and select the highest dev accuracy.

        Ties keep the first candidate in Cs. The selected model is not retrained
        on the development set. Missing feature rows must be removed together
        with their labels before calling this method.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, with shape (n_train, hidden_size).
        y_train : np.ndarray
            Integer training labels, with shape (n_train,).
        X_dev : np.ndarray
            Development features, with shape (n_dev, hidden_size).
        y_dev : np.ndarray
            Integer development labels, with shape (n_dev,).

        Returns
        -------
        LinearProbe
            This probe, containing the selected classifier.
        """
        self._validate_fit_inputs(X_train, y_train, X_dev, y_dev)

        X_train = np.asarray(X_train, dtype=np.float32)
        X_dev = np.asarray(X_dev, dtype=np.float32)

        best_acc = -1.0
        best_estimator = None
        for C in self.Cs:
            model = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            dev_acc = model.score(X_dev, y_dev)

            # Store the best model
            if dev_acc > best_acc:
                best_estimator = model
                best_acc = dev_acc

        if best_estimator is None:
            raise ValueError("No valid model was trained. Check your training data.")
        self.best_estimator_ = best_estimator
        self.hparams = {"C": float(best_estimator.C)}
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict integer class labels using the selected classifier.

        Parameters
        ----------
        X : np.ndarray
            Features with shape (n_samples, hidden_size).

        Returns
        -------
        np.ndarray
            One predicted class index per sample.
        """
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None:
            raise NotFittedError("LinearProbe must be fitted before calling predict.")
        return self.best_estimator_.predict(np.asarray(X, dtype=np.float32))


class MLPProbe(Probe): ...  # P3: to be implement separately.
