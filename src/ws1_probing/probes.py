from typing import Protocol
import numpy as np
from sklearn.linear_model import LogisticRegression



class Probe(Protocol):
    def fit(self, X_train, y_train, X_dev, y_dev) -> "Probe": ...
    def predict(self, X) -> np.ndarray: ...
    hparams: dict


class LinearProbe(Probe):

    def __init__(
        self,
        Cs: tuple[float, ...] = (0.01, 0.1, 1.0, 10.0),
        max_iter: int = 1000,
        seed: int = 42,
    ) -> None:
        if len(Cs) == 0:
            raise ValueError("Cs must contain at least one candidate value.")
        if any([1 for C in Cs if C <= 0]):
            raise ValueError("Cs must contain values higher than 0.")
        if max_iter <= 0:
            raise ValueError("max_iter must be a positive integer.")
        self.Cs = Cs
        self.max_iter = max_iter
        self.seed = seed
        self.hparams: dict = {}
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> "LinearProbe":
        """
        Train candidates and select the highest dev accuracy.

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
        X_train = np.asarray(X_train, dtype=np.float32)
        X_dev = np.asarray(X_dev, dtype=np.float32)
        
        best_acc = 0
        for C in self.Cs:
            model = LogisticRegression(
                penalty="l2",
                solver="liblinear",
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            model.fit(X_train, y_train)
            dev_acc = model.score(X_dev, y_dev)

            # Store the best model
            if dev_acc > best_acc:
                self.best_estimator_ = model
                best_acc = dev_acc
                self.hparams = {"C": float(C)}

        return self
    
    def predict(self, X_test) -> np.ndarray:
        return self.best_estimator_.predict(X_test)


class MLPProbe: ...      # __init__(self, hidden=256, lr=1e-3, weight_decay=0.0, batch_size=512, max_epochs=50, patience=5, seed=42, device=None)
    