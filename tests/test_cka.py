"""Tests for ws2_compression.cka – linear_cka (issue C1, #15).

All tests use synthetic data and require no downloads or torch.
"""

import numpy as np
import pytest

from ws2_compression.cka import linear_cka

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #

@pytest.fixture()
def rng():
    return np.random.default_rng(42)


@pytest.fixture()
def X(rng):
    return rng.standard_normal((200, 50))


@pytest.fixture()
def Y(rng):
    return rng.standard_normal((200, 30))


# ------------------------------------------------------------------ #
# linear_cka: core properties
# ------------------------------------------------------------------ #

class TestLinearCkaSelfSimilarity:
    """CKA(X, X) must be 1."""

    def test_self_similarity(self, X):
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)

    def test_self_similarity_different_dims(self, rng):
        X = rng.standard_normal((300, 768))
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


class TestLinearCkaInvariance:
    """Orthogonal and isotropic scale/shift invariance."""

    def test_orthogonal_invariance(self, rng, X):
        # Random orthogonal Q via QR decomposition
        M = rng.standard_normal((X.shape[1], X.shape[1]))
        Q, _ = np.linalg.qr(M)
        assert linear_cka(X, X @ Q) == pytest.approx(1.0, abs=1e-10)

    def test_isotropic_scale_shift(self, X):
        assert linear_cka(X, 3.0 * X + 5.0) == pytest.approx(1.0, abs=1e-10)

    def test_negative_scale(self, X):
        assert linear_cka(X, -2.0 * X) == pytest.approx(1.0, abs=1e-10)


class TestLinearCkaSymmetry:
    """CKA(X, Y) == CKA(Y, X)."""

    def test_symmetry(self, X, Y):
        assert linear_cka(X, Y) == pytest.approx(linear_cka(Y, X), abs=1e-14)


class TestLinearCkaIndependence:
    """Two independent random matrices should have CKA near 0."""

    def test_independent_low(self):
        rng = np.random.default_rng(99)
        A = rng.standard_normal((2000, 50))
        B = rng.standard_normal((2000, 50))
        assert linear_cka(A, B) < 0.1


class TestLinearCkaDifferentDims:
    """d1 != d2 should work."""

    def test_d1_768_d2_2048(self):
        rng = np.random.default_rng(7)
        X = rng.standard_normal((500, 768))
        Y = rng.standard_normal((500, 2048))
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0


class TestLinearCkaRange:
    """Output should be in [0, 1]."""

    def test_range(self, X, Y):
        val = linear_cka(X, Y)
        assert 0.0 <= val <= 1.0 + 1e-10


# ------------------------------------------------------------------ #
# linear_cka: constant representations & scale invariance
# ------------------------------------------------------------------ #

class TestLinearCkaConstantRepresentations:
    """Zero-variance and constant input handling without breaking scale invariance."""

    @pytest.mark.parametrize("val", [0.0, 1.0, 0.1, -0.7, 3.14159, 1e-20, 1e20])
    def test_constant_matrices_return_zero(self, val):
        """Constant matrices have zero sample variance and must return 0.0."""
        X_const = np.full((10, 5), val)
        assert linear_cka(X_const, X_const) == 0.0

    def test_constant_vs_varying_returns_zero(self, rng):
        X_const = np.full((50, 10), 0.1)
        Y_var = rng.standard_normal((50, 10))
        assert linear_cka(X_const, Y_var) == 0.0
        assert linear_cka(Y_var, X_const) == 0.0

    def test_scale_invariance_small_valued_inputs(self, rng):
        """Scale invariance must hold for legitimate small-valued representations."""
        X_tiny = 1e-20 * rng.standard_normal((50, 10))
        assert linear_cka(X_tiny, X_tiny) == pytest.approx(1.0, abs=1e-10)
        assert linear_cka(X_tiny, 3.0 * X_tiny) == pytest.approx(1.0, abs=1e-10)

    def test_partially_constant_columns_preserved(self, rng):
        """If some columns are constant but others vary, CKA should measure varying columns."""
        X = rng.standard_normal((50, 5))
        X[:, 0] = 0.1  # Column 0 is constant
        assert linear_cka(X, X) == pytest.approx(1.0, abs=1e-10)


# ------------------------------------------------------------------ #
# linear_cka: error handling
# ------------------------------------------------------------------ #

class TestLinearCkaErrors:
    """Validation errors."""

    def test_mismatched_rows(self):
        with pytest.raises(ValueError, match="Row counts must match"):
            linear_cka(np.ones((10, 5)), np.ones((11, 5)))

    def test_n_less_than_2(self):
        with pytest.raises(ValueError, match="N >= 2"):
            linear_cka(np.ones((1, 5)), np.ones((1, 5)))

    def test_1d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(np.ones(10), np.ones(10))

    def test_3d_input(self):
        with pytest.raises(ValueError, match="2-D"):
            linear_cka(np.ones((2, 3, 4)), np.ones((2, 3, 4)))

    def test_nan_in_X(self):
        X = np.ones((10, 5))
        X[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            linear_cka(X, np.ones((10, 5)))

    def test_nan_in_Y(self):
        Y = np.ones((10, 5))
        Y[0, 0] = np.nan
        with pytest.raises(ValueError, match="NaN"):
            linear_cka(np.ones((10, 5)), Y)

    def test_inf_in_X(self):
        X = np.ones((10, 5))
        X[0, 0] = np.inf
        with pytest.raises(ValueError, match="NaN or Inf"):
            linear_cka(X, np.ones((10, 5)))
