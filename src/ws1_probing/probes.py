import warnings
from typing import Protocol

import numpy as np
from sklearn.exceptions import ConvergenceWarning, NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import torch


class Probe(Protocol):
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, X_dev: np.ndarray, y_dev: np.ndarray,) -> "Probe": ...
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
    
    def _validate_inputs(
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
        self._validate_inputs(X_train, y_train, X_dev, y_dev)

        X_train = np.asarray(X_train, dtype=np.float32)
        X_dev = np.asarray(X_dev, dtype=np.float32)
        
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_dev_scaled = self.scaler.transform(X_dev)

        best_acc = -1.0
        best_estimator = None
        for C in self.Cs:
            model = LogisticRegression(
                C=C,
                solver="lbfgs",
                max_iter=self.max_iter,
                random_state=self.seed,
            )
            with warnings.catch_warnings(record=True) as captured:
                warnings.simplefilter("always", ConvergenceWarning)
                model.fit(X_train_scaled, y_train)
            dev_acc = model.score(X_dev_scaled, y_dev)

            # Store the best model
            if dev_acc > best_acc:
                best_estimator = model
                best_captured_warnings = captured
                best_acc = dev_acc

        if best_estimator is None:
            raise ValueError("No valid model was trained. Check your training data.")
        self.best_estimator_ = best_estimator

        converged = not [
            str(w.message) for w in best_captured_warnings
            if issubclass(w.category, ConvergenceWarning)
        ]

        self.hparams = {"C": float(best_estimator.C), "converged": converged}
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
        if not hasattr(self, "best_estimator_") or self.best_estimator_ is None or self.scaler is None:
            raise NotFittedError("LinearProbe must be fitted before calling predict.")
        
        X = np.asarray(X, dtype=np.float32)

        X_scaled = self.scaler.transform(X)
        return self.best_estimator_.predict(np.asarray(X_scaled, dtype=np.float32))


class MLPProbe(Probe):
    """A simple MLP probe with one hidden layer and ReLU activation."""

    def __init__(
        self,
        hidden: int = 256,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 512,
        max_epochs: int = 50,
        patience: int = 5,
        seed: int = 42,
        device: str | None = None,
    ) -> None:
        """
        Configure the probe without training a classifier.

        Parameters
        ----------
        hidden : int
            Number of neurons in the hidden layer.
        max_epochs : int
            Positive maximum number of optimization iterations.
        seed : int
            Random state passed to the classifier.

        Returns
        -------
        None
            The probe is configured and must be fitted before prediction.
        """
        
        if not (isinstance(hidden, (int, np.integer)) and hidden > 0):
            raise ValueError("hidden must be a positive integer.")
        if not (isinstance(lr, (float, np.floating)) and lr > 0):
            raise ValueError("lr must be a positive float.")
        if not (isinstance(weight_decay, (float, np.floating)) and weight_decay >= 0):
            raise ValueError("weight_decay must be a positive or equal zero float.")
        if not (isinstance(batch_size, (int, np.integer)) and batch_size > 0):
            raise ValueError("batch_size must be a positive integer.")
        if not (isinstance(max_epochs, (int, np.integer)) and max_epochs > 0):
            raise ValueError("max_epochs must be a positive integer.")
        if not (isinstance(patience, (int, np.integer)) and patience > 0):
            raise ValueError("patience must be a positive integer.")
        if not (isinstance(seed, (int, np.integer)) and seed > 0):
            raise ValueError("seed must be a positive integer.")
        
        self.hidden = hidden
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.seed = seed
        self.device = device
        self.hparams: dict = {}
        self.model_: torch.nn.Module | None = None

    def _validate_inputs(
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
    
    def _predict_scaled(self, X_scaled: np.ndarray) -> np.ndarray:
        """Predict from standardised features in batches.

        Parameters
        ----------
        X_scaled : np.ndarray
            Standardised float32 features of shape (n_samples, d).

        Returns
        -------
        np.ndarray
            Integer predictions on CPU, in input row order.
        """
        if self.model_ is None:
            raise NotFittedError("MLPProbe must be fitted before calling predict.")
        
        self.model_.eval()

        predictions = np.empty(len(X_scaled), dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, len(X_scaled), self.batch_size):
                stop = start + self.batch_size
                X_batch = torch.from_numpy(X_scaled[start:stop]).to(self.device_)
                logits = self.model_(X_batch)
                predictions[start:stop] = logits.argmax(dim=1).cpu().numpy()

        return predictions

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
    ) -> "MLPProbe":
        """
        Train the MLP probe on training data.

        Parameters
        ----------
        X_train : np.ndarray
            Training features, with shape (n_train, hidden).
        y_train : np.ndarray
            Integer training labels, with shape (n_train,).
        X_dev : np.ndarray
            Development features, with shape (n_dev, hidden).
        y_dev : np.ndarray
            Integer development labels, with shape (n_dev,).

        Returns
        -------
        MLPProbe
            This probe, containing the trained classifier.
        """
        X_train = np.asarray(X_train, dtype=np.float32)
        X_dev = np.asarray(X_dev, dtype=np.float32)
        y_train = np.asarray(y_train)
        y_dev = np.asarray(y_dev)

        self._validate_inputs(X_train, y_train, X_dev, y_dev)

        self.hparams = {}
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_dev_scaled = self.scaler.transform(X_dev)

        if self.device is not None:
            self.device_ = torch.device(self.device)
        elif torch.cuda.is_available():
            self.device_ = torch.device("cuda")
        elif torch.backends.mps.is_available():
            self.device_ = torch.device("mps")
        else:
            self.device_ = torch.device("cpu")

        torch.manual_seed(self.seed)

        self.model_ = torch.nn.Sequential(
            torch.nn.Linear(X_train_scaled.shape[1], self.hidden),
            torch.nn.ReLU(),
            torch.nn.Linear(self.hidden, 17),
        ).to(self.device_)

        optimizer = torch.optim.Adam(
            self.model_.parameters(),
            lr=self.lr,
            weight_decay=self.weight_decay
        )
        loss_fun = torch.nn.CrossEntropyLoss()

        dataset = torch.utils.data.TensorDataset(
            torch.from_numpy(X_train_scaled),
            torch.from_numpy(y_train.astype(np.int64)),
        )
        generator = torch.Generator().manual_seed(self.seed)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            generator=generator,
        )
        
        best_dev_acc = -1.0
        best_state = None
        counter = 0
        epochs_trained = 0
        for epoch in range(self.max_epochs):
            self.model_.train()

            for X_batch, y_batch in loader:
                X_batch = X_batch.to(self.device_)
                y_batch = y_batch.to(self.device_)

                optimizer.zero_grad()
                logits = self.model_(X_batch)
                loss = loss_fun(logits, y_batch)
                loss.backward()
                optimizer.step()

            epochs_trained = epoch + 1
            predictions = self._predict_scaled(X_dev_scaled)
            dev_acc = np.mean(predictions == y_dev)

            # Case of improvement in development accuracy
            if dev_acc > best_dev_acc:
                counter = 0
                best_dev_acc = dev_acc
                best_state = {
                    name: tensor.detach().cpu().clone()
                    for name, tensor in self.model_.state_dict().items()
                }
            else:
                counter += 1

            # Early stopping based on development accuracy
            if counter >= self.patience:
                break
        
        if best_state is None:
            raise ValueError("No valid model was trained. Check your training data.")
        self.model_.load_state_dict(best_state)
        self.model_.eval()

        self.hparams = {
            "hidden": int(self.hidden),
            "lr": float(self.lr),
            "epochs_trained": epochs_trained,
            "best_dev_acc": best_dev_acc,
            "weight_decay": float(self.weight_decay),
            "batch_size": int(self.batch_size),
            "max_epochs": int(self.max_epochs),
            "patience": int(self.patience),
            "seed": int(self.seed),
            "device": str(self.device_),
        }
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict integer class labels using the trained MLP classifier.

        Parameters
        ----------
        X : np.ndarray
            Features with shape (n_samples, hidden_size).

        Returns
        -------
        np.ndarray
            One integer class index per sample.
        """
        if not self.hparams or self.model_ is None or self.scaler is None:
            raise NotFittedError("MLPProbe must be fitted before calling predict.")

        X = np.asarray(X, dtype=np.float32)
        if X.ndim != 2 or X.size == 0:
            raise ValueError("X must be a nonempty 2D array.")
        if not np.isfinite(X).all():
            raise ValueError("X must contain only finite values.")

        X_scaled = self.scaler.transform(X)
        return self._predict_scaled(X_scaled)
