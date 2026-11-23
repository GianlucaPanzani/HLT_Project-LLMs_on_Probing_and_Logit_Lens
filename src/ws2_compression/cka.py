"""Linear CKA (Centered Kernel Alignment) for representation similarity.

Implements Linear CKA between two representation matrices using the
feature-space formulation with column mean centering (PROPOSAL.md §WS2.3,
CONTRACTS.md §10).
"""

from __future__ import annotations

import warnings

import numpy as np


def linear_cka(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute Linear Centered Kernel Alignment between two representation matrices.

    Uses the feature-space formulation with mean-centred activations, cast to
    float64 for numerical stability (PROPOSAL.md §WS2.3).

    Parameters
    ----------
    X : np.ndarray
        Representation matrix of shape ``[N, d1]``.
    Y : np.ndarray
        Representation matrix of shape ``[N, d2]``.  ``d2`` may differ
        from ``d1`` but ``N`` must match.

    Returns
    -------
    float
        CKA similarity in [0, 1].

    Raises
    ------
    ValueError
        If inputs are not 2-D, have mismatched row counts, ``N < 2``,
        or contain NaN / Inf values.
    """
    # --- validation -----------------------------------------------------------
    if X.ndim != 2:
        raise ValueError(f"X must be 2-D, got {X.ndim}-D.")
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2-D, got {Y.ndim}-D.")
    if X.shape[0] != Y.shape[0]:
        raise ValueError(
            f"Row counts must match: X has {X.shape[0]} rows, Y has {Y.shape[0]}."
        )
    if X.shape[0] < 2:
        raise ValueError(f"Need N >= 2, got N = {X.shape[0]}.")
    if not np.all(np.isfinite(X)):
        raise ValueError("X contains NaN or Inf values.")
    if not np.all(np.isfinite(Y)):
        raise ValueError("Y contains NaN or Inf values.")

    # --- computation (float64) ------------------------------------------------
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    norm_X = np.linalg.norm(X)
    norm_Y = np.linalg.norm(Y)
    if norm_X == 0.0 or norm_Y == 0.0:
        return 0.0

    # Centre columns (subtract column means)
    X_c = X - X.mean(axis=0)
    Y_c = Y - Y.mean(axis=0)

    # For columns with zero sample variance (constant columns), zero out residuals
    # to avoid spurious tiny floating-point noise from mean subtraction.
    X_const = (np.ptp(X, axis=0) == 0.0) | np.all(X == X[0:1, :], axis=0)
    Y_const = (np.ptp(Y, axis=0) == 0.0) | np.all(Y == Y[0:1, :], axis=0)
    X_c[:, X_const] = 0.0
    Y_c[:, Y_const] = 0.0

    # Check if either representation has zero sample variance relative to its scale.
    # Uses scale-relative threshold to preserve scale invariance for legitimate small-valued inputs.
    norm_Xc = np.linalg.norm(X_c)
    norm_Yc = np.linalg.norm(Y_c)

    eps = np.finfo(np.float64).eps
    tol_X = eps * X.shape[0] * norm_X
    tol_Y = eps * Y.shape[0] * norm_Y

    if norm_Xc <= tol_X or norm_Yc <= tol_Y:
        return 0.0

    # Feature-space formulation: ||Y^T X||_F^2 / (||X^T X||_F * ||Y^T Y||_F)
    YtX = Y_c.T @ X_c
    XtX = X_c.T @ X_c
    YtY = Y_c.T @ Y_c

    numerator = np.sum(YtX ** 2)  # ||Y^T X||_F^2
    denom_a = np.sqrt(np.sum(XtX ** 2))  # ||X^T X||_F
    denom_b = np.sqrt(np.sum(YtY ** 2))  # ||Y^T Y||_F
    denominator = denom_a * denom_b

    if denominator == 0.0:
        return 0.0

    return float(numerator / denominator)


def sample_rows(valid_mask: np.ndarray, n: int, seed: int = 42) -> np.ndarray:
    """Sample a fixed subset of valid (non-NaN) row indices.

    Used to select aligned tokens shared between two models before
    computing CKA across all layer pairs (CONTRACTS.md §10).

    Parameters
    ----------
    valid_mask : np.ndarray
        1-D boolean array where ``True`` marks a valid (non-NaN) row.
    n : int
        Desired number of rows to sample.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Sorted 1-D integer array of sampled row indices. Its length is
        ``min(n, n_valid)`` where ``n_valid`` is the number of ``True``
        entries in *valid_mask*.
    """
    if valid_mask.ndim != 1:
        raise ValueError(f"valid_mask must be 1-D, got {valid_mask.ndim}-D.")

    valid_indices = np.flatnonzero(valid_mask)
    n_valid = len(valid_indices)

    if n_valid < n:
        warnings.warn(
            f"Requested {n} rows but only {n_valid} are valid; using all {n_valid}.",
            UserWarning,
            stacklevel=2,
        )

    k = min(n, n_valid)
    rng = np.random.default_rng(seed)
    chosen = rng.choice(valid_indices, size=k, replace=False)
    return np.sort(chosen)

